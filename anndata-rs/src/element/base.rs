use crate::{
    backend::{Backend, GroupOp, LocationOp, DataContainer, DataType, BackendData},
    //iterator::{ChunkedMatrix, StackedChunkedMatrix},
    data::*,
};

use anyhow::{bail, ensure, Ok, Result};
use either::Either;
use indexmap::set::IndexSet;
use itertools::Itertools;
use ndarray::{Ix1, Array1};
use parking_lot::{Mutex, MutexGuard};
use polars::{frame::DataFrame, series::Series};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    boxed::Box,
    collections::HashMap,
    ops::{Deref, DerefMut},
    sync::Arc,
};

/// Slot stores an optional object wrapped by Arc and Mutex.
/// Encapsulating an object inside a slot allows us to drop the object from all references.
pub struct Slot<T>(pub(crate) Arc<Mutex<Option<T>>>);

impl<T> Clone for Slot<T> {
    fn clone(&self) -> Self {
        Slot(self.0.clone())
    }
}

impl<T> std::fmt::Display for Slot<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            write!(f, "Empty or closed slot")
        } else {
            write!(f, "{}", self.inner().deref())
        }
    }
}

impl<T> Slot<T> {
    /// Create a slot from data.
    pub fn new(x: T) -> Self {
        Slot(Arc::new(Mutex::new(Some(x))))
    }

    /// Create an empty slot.
    pub fn empty() -> Self {
        Slot(Arc::new(Mutex::new(None)))
    }

    pub fn is_empty(&self) -> bool {
        self.0.lock().is_none()
    }

    pub fn lock(&self) -> MutexGuard<'_, Option<T>> {
        self.0.lock()
    }

    pub fn inner(&self) -> Inner<'_, T> {
        Inner(self.0.lock())
    }

    /// Insert data to the slot, and return the old data.
    pub fn insert(&self, data: T) -> Option<T> {
        std::mem::replace(self.0.lock().deref_mut(), Some(data))
    }

    /// Extract the data from the slot. The slot becomes empty after this operation.
    pub fn extract(&self) -> Option<T> {
        std::mem::replace(self.0.lock().deref_mut(), None)
    }

    /// Remove the data from the slot.
    pub fn drop(&self) {
        let _ = self.extract();
    }

    pub fn swap(&self, other: &Self) {
        let mut self_lock = self.0.lock();
        let mut other_lock = other.0.lock();
        std::mem::swap(self_lock.deref_mut(), other_lock.deref_mut());
    }
}

pub struct Inner<'a, T>(pub MutexGuard<'a, Option<T>>);

impl<T> Deref for Inner<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &self.0.deref() {
            None => panic!("accessing an empty slot"),
            Some(x) => x,
        }
    }
}

impl<T> DerefMut for Inner<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self.0.deref_mut() {
            None => panic!("accessing an empty slot"),
            Some(ref mut x) => x,
        }
    }
}

pub struct InnerDataFrameElem<B: Backend> {
    element: Option<DataFrame>,
    container: DataContainer<B>,
    column_names: IndexSet<String>,
    pub index: DataFrameIndex,
}

impl<B: Backend> InnerDataFrameElem<B> {
    pub fn new<G: GroupOp<Backend = B>>(
        location: &G,
        name: &str,
        index: DataFrameIndex,
        df: &DataFrame,
    ) -> Result<Self> {
        ensure!(
            df.height() == 0 || index.len() == df.height(),
            "cannot create dataframe element as lengths of index and dataframe differ"
        );
        let container = index.overwrite(df.write(location, name)?)?;
        let column_names = df.get_column_names_owned().into_iter().collect();
        Ok(Self {
            element: None,
            container,
            column_names,
            index,
        })
    }
}

impl<B: Backend> std::fmt::Display for InnerDataFrameElem<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dataframe element")
    }
}

impl<B: Backend> InnerDataFrameElem<B> {
    pub fn height(&self) -> usize {
        self.index.len()
    }

    pub fn set_index(&mut self, index: DataFrameIndex) -> Result<()> {
        ensure!(
            self.index.len() == index.len(),
            "cannot change the index as the lengths differ"
        );
        self.index = index;
        replace_with::replace_with_or_abort(&mut self.container, |x| self.index.overwrite(x).unwrap());
        Ok(())
    }

    pub fn data(&mut self) -> Result<DataFrame> {
        match self.element {
            Some(ref df) => Ok(df.clone()),
            None => {
                let df = DataFrame::read(&self.container)?;
                self.element = Some(df.clone());
                Ok(df)
            }
        }
    }

    pub fn export(&self, location: &B::Group, name: &str) -> Result<()> {
        let df = match self.element {
            Some(ref df) => df.clone(),
            None => DataFrame::read(&self.container)?,
        };
        self.index.overwrite(df.write(location, name)?)?;
        Ok(())
    }

    pub fn save(&mut self, data: DataFrame) -> Result<()> {
        let num_recs = data.height();
        ensure!(
            num_recs == 0 || self.index.len() == num_recs,
            "cannot update dataframe as lengths differ"
        );
        replace_with::replace_with_or_abort(&mut self.container, |x| data.overwrite(x).unwrap());
        self.column_names = data.get_column_names_owned().into_iter().collect();
        if self.element.is_some() {
            self.element = Some(data);
        }
        Ok(())
    }

    pub fn subset_rows<S: AsRef<SelectInfoElem>>(&mut self, selection: S) -> Result<()> {
        todo!()
    }
}


pub type DataFrameElem<B> = Slot<InnerDataFrameElem<B>>;

impl<B: Backend> TryFrom<DataContainer<B>> for DataFrameElem<B> {
    type Error = anyhow::Error;

    fn try_from(container: DataContainer<B>) -> Result<Self> {
        match container.encoding_type()? {
            DataType::DataFrame => {
                //let grp = container.as_group()?;
                let index = DataFrameIndex::read(&container)?;
                let column_names = container.read_str_arr_attr::<Ix1>("column_order")?.into_raw_vec().into_iter().collect();
                let df = InnerDataFrameElem {
                    element: None,
                    container,
                    column_names,
                    index,
                };
                Ok(Slot::new(df))
            },
            ty => bail!("Expecting a dataframe but found: '{}'", ty),
        }
    }
}

/*

    pub fn column(&self, name: &str) -> Result<Series> {
        self.with_data_ref(|x| x.unwrap().1.column(name).unwrap().clone())
    }

    pub fn get_column_names(&self) -> Option<IndexSet<String>> {
        self.lock().as_ref().map(|x| x.column_names.clone())
    }

    pub fn subset_rows(&self, idx: &[usize]) -> Result<()> {
        let subset = self
            .with_data_ref(|x| {
                x.map(|(index, df)| {
                    let new_df = df.take_iter(idx.into_iter().map(|x| *x))?;
                    let new_index: DataFrameIndex =
                        idx.into_iter().map(|i| index.names[*i].clone()).collect();
                    Ok::<_, anyhow::Error>((new_index, new_df))
                })
            })?
            .transpose()?;

        if let Some((index, df)) = subset {
            let mut inner = self.inner();
            inner.container = df.update(&inner.container)?;
            index.write(&inner.container)?;
            inner.column_names = df.get_column_names_owned().into_iter().collect();
            if inner.element.is_some() {
                inner.element = Some(df);
            }
            inner.index = index;
        }
        Ok(())
    }
}
*/

/// Container holding general data types.
pub struct InnerElem<B: Backend, T> {
    dtype: DataType,
    shape: Option<Shape>,
    cache_enabled: bool,
    container: DataContainer<B>,
    element: Option<T>,
}

impl<B: Backend, T> std::fmt::Display for InnerElem<B, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} element, cache_enabled: {}, cached: {}",
            self.dtype,
            if self.cache_enabled { "yes" } else { "no" },
            if self.element.is_some() { "yes" } else { "no" },
        )
    }
}

impl<B: Backend, T> InnerElem<B, T> {
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    pub fn enable_cache(&mut self) {
        self.cache_enabled = true;
    }

    pub fn disable_cache(&mut self) {
        if self.element.is_some() {
            self.element = None;
        }
        self.cache_enabled = false;
    }
}

impl<B: Backend, T: ReadData + WriteData + Clone> InnerElem<B, T> {
    pub fn data(&mut self) -> Result<T> {
        match self.element.as_ref() {
            Some(data) => Ok(data.clone()),
            None => {
                let data = T::read(&self.container)?;
                if self.cache_enabled {
                    self.element = Some(data.clone());
                }
                Ok(data)
            }
        }
    }

    pub fn export(&mut self, location: &B::Group, name: &str) -> Result<()> {
        match self.element.as_ref() {
            Some(data) => data.write(location, name)?,
            None => T::read(&self.container)?.write(location, name)?,
        };
        Ok(())
    }

    pub(crate) fn save<D: WriteData + Into<T>>(&mut self, data: D) -> Result<()> {
        replace_with::replace_with_or_abort(&mut self.container, |x| data.overwrite(x).unwrap());
        if self.element.is_some() {
            self.element = Some(data.into());
        }
        Ok(())
    }
}

impl<B: Backend, T: ReadArrayData + WriteArrayData + Clone> InnerElem<B, T> {
    pub fn shape(&self) -> &Shape {
        self.shape.as_ref().unwrap()
    }

    pub fn select<S, E>(&mut self, selection: S) -> Result<T>
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
    {
        if select_all(&selection) {
            self.data()
        } else {
            match self.element.as_ref() {
                Some(data) => Ok(data.select(selection)),
                None => T::read_select(&self.container, selection),
            }
        }
    }

    pub fn export_select<S, E>(
        &mut self,
        selection: S,
        location: &B::Group,
        name: &str,
    ) -> Result<()>
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
    {
        if select_all(&selection) {
            self.export(location, name)
        } else {
            self.select(selection)?.write(location, name)?;
            Ok(())
        }
    }

    pub(crate) fn subset<S, E>(&mut self, selection: S) -> Result<()>
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
    {
        let data = match self.element.as_ref() {
            Some(data) => data.select(selection),
            None => T::read_select(&self.container, selection)?,
        };

        self.shape = Some(data.shape());
        replace_with::replace_with_or_abort(&mut self.container, |x| data.overwrite(x).unwrap());
        if self.element.is_some() {
            self.element = Some(data);
        }
        Ok(())
    }
}

pub type Elem<B> = Slot<InnerElem<B, Data>>;

impl<B: Backend> TryFrom<DataContainer<B>> for Elem<B> {
    type Error = anyhow::Error;

    fn try_from(container: DataContainer<B>) -> Result<Self> {
        let dtype = container.encoding_type()?;
        let elem = InnerElem {
            dtype,
            shape: None,
            cache_enabled: false,
            element: None,
            container,
        };
        Ok(Slot::new(elem))
    }
}

pub type ArrayElem<B> = Slot<InnerElem<B, ArrayData>>;

/// Container holding matrix data types.
impl<B: Backend> TryFrom<DataContainer<B>> for ArrayElem<B> {
    type Error = anyhow::Error;

    fn try_from(container: DataContainer<B>) -> Result<Self> {
        let dtype = container.encoding_type()?;
        let elem = InnerElem {
            dtype,
            shape: Some(ArrayData::get_shape(&container)?),
            cache_enabled: false,
            element: None,
            container,
        };
        Ok(Slot::new(elem))
    }
}

/*
pub fn chunked(&self, chunk_size: usize) -> ChunkedMatrix {
    ChunkedMatrix::new(self.clone(), chunk_size)
}
*/

/*
/// This struct is used to perform index lookup for Vectors of Vectors.
pub struct VecVecIndex(Vec<usize>);

impl VecVecIndex {
    /// Find the outer and inner index for a given index corresponding to the
    /// flattened view.
    pub fn ix(&self, i: &usize) -> (usize, usize) {
        let j = self.outer_ix(i);
        (j, i - self.0[j])
    }

    /// The inverse of ix.
    pub fn inv_ix(&self, idx: (usize, usize)) -> usize {
        self.0[idx.0] + idx.1
    }

    /// Find the outer index for a given index corresponding to the flattened view.
    pub fn outer_ix(&self, i: &usize) -> usize {
        match self.0.binary_search(i) {
            Ok(i_) => i_,
            Err(i_) => i_ - 1,
        }
    }

    pub fn ix_group_by_outer<'a, I>(
        &self,
        indices: I,
    ) -> std::collections::HashMap<usize, (Vec<usize>, Vec<usize>)>
    where
        I: Iterator<Item = &'a usize>,
    {
        indices
            .map(|x| self.ix(x))
            .enumerate()
            .sorted_by_key(|(_, (x, _))| *x)
            .into_iter()
            .group_by(|(_, (x, _))| *x)
            .into_iter()
            .map(|(outer, inner)| (outer, inner.map(|(i, (_, x))| (x, i)).unzip()))
            .collect()
    }

    /// The total number of elements
    pub fn len(&self) -> usize {
        *self.0.last().unwrap_or(&0)
    }
}

impl FromIterator<usize> for VecVecIndex {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = usize>,
    {
        let index: Vec<usize> = std::iter::once(0)
            .chain(iter.into_iter().scan(0, |state, x| {
                *state = *state + x;
                Some(*state)
            }))
            .collect();
        VecVecIndex(index)
    }
}
*/

/*
impl<T> From<&[T]> for VecVecIndex {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = usize>,
    {
        let accum: Vec<usize> = std::iter::once(0).chain(
            iter.into_iter().scan(0, |state, x| {
                *state = *state + x;
                Some(*state)
            })
        ).collect();
        VecVecIndex(accum)
    }
}
*/

/*
// TODO: remove Arc.
#[derive(Clone)]
pub struct StackedArrayElem<B: Backend> {
    nrows: Arc<Mutex<usize>>,
    ncols: Arc<Mutex<usize>>,
    pub elems: Arc<Vec<ArrayElem<B>>>,
    index: Arc<Mutex<VecVecIndex>>,
}

impl<B: Backend> std::fmt::Display for StackedArrayElem<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.elems.len() == 0 {
            write!(f, "empty stacked elements")
        } else {
            write!(
                f,
                "{} x {} stacked elements ({}) with {}",
                *self.nrows.lock(),
                *self.ncols.lock(),
                self.elems.len(),
                self.elems[0].dtype().unwrap(),
            )
        }
    }
}

/*
impl<B: Backend> StackedArrayElem<B> {
    pub(crate) fn new(
        elems: Vec<ArrayElem<B>>,
        nrows: Arc<Mutex<usize>>,
        ncols: Arc<Mutex<usize>>,
        index: Arc<Mutex<VecVecIndex>>,
    ) -> Result<Self> {
        if !elems.iter().map(|x| x.dtype()).all_equal() {
            bail!("dtype not equal")
        } else {
            Ok(Self {
                nrows,
                ncols,
                elems: Arc::new(elems),
                index,
            })
        }
    }

    pub fn nrows(&self) -> usize {
        *self.nrows.lock()
    }
    pub fn ncols(&self) -> usize {
        *self.ncols.lock()
    }

    pub fn read(
        &self,
        ridx_: Option<&[usize]>,
        cidx: Option<&[usize]>,
    ) -> Result<ArrayData> {
        match ridx_ {
            Some(ridx) => {
                let index = self.index.lock();
                let (ori_idx, rows): (Vec<_>, Vec<_>) = ridx
                    .iter()
                    .map(|x| index.ix(x))
                    .enumerate()
                    .sorted_by_key(|x| x.1 .0)
                    .into_iter()
                    .group_by(|x| x.1 .0)
                    .into_iter()
                    .map(|(key, grp)| {
                        let (ori_idx, (_, inner_idx)): (Vec<_>, (Vec<_>, Vec<_>)) = grp.unzip();
                        (
                            ori_idx,
                            self.elems[key].read(Some(inner_idx.as_slice()), cidx),
                        )
                    })
                    .unzip();
                Ok(rstack_with_index(
                    ori_idx.into_iter().flatten().collect::<Vec<_>>().as_slice(),
                    rows.into_iter().collect::<Result<_>>()?,
                )?)
            }
            None => {
                let mats: Result<Vec<_>> =
                    self.elems.par_iter().map(|x| x.read(None, cidx)).collect();
                Ok(rstack(mats?)?)
            }
        }
    }

    pub fn chunked(&self, chunk_size: usize) -> StackedChunkedMatrix {
        StackedChunkedMatrix::new(self.elems.iter().map(|x| x.clone()), chunk_size)
    }

    pub fn enable_cache(&self) {
        self.elems.iter().for_each(|x| x.enable_cache())
    }

    pub fn disable_cache(&self) {
        self.elems.iter().for_each(|x| x.disable_cache())
    }
}

#[derive(Clone)]
pub struct StackedDataFrame {
    pub column_names: IndexSet<String>,
    pub elems: Arc<Vec<DataFrameElem>>,
}

impl std::fmt::Display for StackedDataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "stacked dataframe with columns: '{}'",
            self.column_names.iter().join("', '")
        )
    }
}

impl StackedDataFrame {
    pub fn new(elems: Vec<DataFrameElem>) -> Self {
        let column_names = elems
            .iter()
            .map(|x| x.get_column_names().unwrap_or(IndexSet::new()))
            .reduce(|shared_keys, next_keys| {
                shared_keys
                    .intersection(&next_keys)
                    .map(|x| x.to_owned())
                    .collect()
            })
            .unwrap_or(IndexSet::new());
        Self {
            column_names,
            elems: Arc::new(elems),
        }
    }

    pub fn read(&self) -> Result<DataFrame> {
        let mut merged = DataFrame::empty();
        self.elems.iter().for_each(|el| {
            el.with_data_mut_ref(|x| {
                if let Some((_, df)) = x {
                    merged.vstack_mut(df).unwrap();
                }
            })
            .unwrap();
        });
        merged.rechunk();
        Ok(merged)
    }

    pub fn column(&self, name: &str) -> Result<Series> {
        if self.column_names.contains(name) {
            Ok(self.read()?.column(name)?.clone())
        } else {
            bail!("key is not present");
        }
    }
}

*/
*/
