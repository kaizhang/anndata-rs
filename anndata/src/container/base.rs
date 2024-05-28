use crate::{
    traits::ArrayElemOp,
    backend::{Backend, DataContainer, DataType, GroupOp, AttributeOp},
    data::*,
    data::index::VecVecIndex,
};

use anyhow::{bail, ensure, Result};
use indexmap::set::IndexSet;
use itertools::Itertools;
use ndarray::Ix1;
use num::integer::div_rem;
use parking_lot::{Mutex, MutexGuard};
use polars::{
    frame::DataFrame,
    prelude::{concat, IntoLazy, UnionArgs},
    series::{Series, IntoSeries},
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use smallvec::SmallVec;
use std::{ops::{Deref, DerefMut}, sync::Arc};

/// Slot stores an optional object wrapped by Arc and Mutex.
/// Encapsulating an object inside a slot allows us to drop the object from all references.
#[derive(Debug)]
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
        if self.is_none() {
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
    pub fn none() -> Self {
        Slot(Arc::new(Mutex::new(None)))
    }

    pub fn is_none(&self) -> bool {
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

#[derive(Debug)]
pub struct InnerDataFrameElem<B: Backend> {
    element: Option<DataFrame>,
    container: DataContainer<B>,
    column_names: IndexSet<String>,
    pub index: DataFrameIndex,
}

impl<B: Backend> InnerDataFrameElem<B> {
    pub fn new<G: GroupOp<B>>(
        location: &G,
        name: &str,
        index: DataFrameIndex,
        df: &DataFrame,
    ) -> Result<Self> {
        ensure!(
            df.height() == 0 || index.len() == df.height(),
            "cannot create dataframe element as lengths of index and dataframe differ"
        );
        let container = df.overwrite(index.write(location, name)?)?;
        let column_names = df.get_column_names().into_iter().map(|x| x.to_string()).collect();
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
    pub fn width(&self) -> usize {
        self.column_names.len()
    }

    pub fn height(&self) -> usize {
        self.index.len()
    }

    pub fn column(&mut self, name: &str) -> Result<&Series> {
        self.data().and_then(|x| Ok(x.column(name)?))
    }

    pub fn get_column_names(&self) -> &IndexSet<String> {
        &self.column_names
    }

    /// Set a column with a Series.
    //TODO: this is not efficient. We should be able to replace a column without reading the whole dataframe.
    pub fn set_column<S: IntoSeries>(&mut self, name: &str, new_col: S) -> Result<()> {
        let mut df = self.data()?.clone();
        df.replace_or_add(name, new_col)?;
        self.save(df)
    }

    pub fn set_index(&mut self, index: DataFrameIndex) -> Result<()> {
        ensure!(
            self.index.len() == index.len(),
            "cannot change the index as the lengths differ"
        );
        self.index = index;
        replace_with::replace_with_or_abort(&mut self.container, |x| {
            self.index.overwrite(x).unwrap()
        });
        Ok(())
    }

    pub fn data(&mut self) -> Result<&DataFrame> {
        match self.element {
            Some(ref df) => Ok(df),
            None => {
                let df = DataFrame::read(&self.container)?;
                self.element = Some(df);
                Ok(&self.element.as_ref().unwrap())
            }
        }
    }

    pub fn export<O: Backend, G: GroupOp<O>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<()> {
        let df = match self.element {
            Some(ref df) => df.clone(),
            None => DataFrame::read(&self.container)?,
        };
        df.overwrite(self.index.write(location, name)?)?;
        Ok(())
    }

    pub fn export_select<O, G>(
        &mut self,
        selection: &[&SelectInfoElem],
        location: &G,
        name: &str,
    ) -> Result<()>
    where
        O: Backend,
        G: GroupOp<O>,
    {
        if selection.as_ref().into_iter().all(|x| x.is_full()) {
            self.export::<O, _>(location, name)
        } else {
            self.select(selection)?.overwrite(
                self.index.select(&selection[0]).write(location, name)?
            )?;
            Ok(())
        }
    }

    pub fn export_axis<O, S, G>(
        &mut self,
        axis: usize,
        selection: S,
        location: &G,
        name: &str,
    ) -> Result<()>
    where
        O: Backend,
        S: AsRef<SelectInfoElem>,
        G: GroupOp<O>,
    {
        let full = SelectInfoElem::full();
        let slice = selection.as_ref().set_axis(axis, 2, &full);
        self.export_select(slice.as_slice(), location, name)
    }

    pub fn select<S>(&mut self, selection: &[S]) -> Result<DataFrame>
    where
        S: AsRef<SelectInfoElem>,
    {
        Ok(ArrayOp::select(self.data()?, selection))
    }

    pub fn select_axis<S>(&mut self, axis: usize, selection: S) -> Result<DataFrame>
    where
        S: AsRef<SelectInfoElem>,
    {
        Ok(ArrayOp::select_axis(self.data()?, axis, selection))
    }

    pub fn save(&mut self, data: DataFrame) -> Result<()> {
        let num_recs = data.height();
        ensure!(
            num_recs == 0 || self.index.len() == num_recs,
            "cannot update dataframe as lengths differ"
        );
        replace_with::replace_with_or_abort(&mut self.container, |x| data.overwrite(x).unwrap());
        self.column_names = data.get_column_names().into_iter().map(|x| x.to_string()).collect();
        if self.element.is_some() {
            self.element = Some(data);
        }
        Ok(())
    }

    pub fn subset<S>(&mut self, selection: &[S]) -> Result<()>
    where
        S: AsRef<SelectInfoElem>,
    {
        self.index = self.index.select(selection[0].as_ref());
        replace_with::replace_with_or_abort(&mut self.container, |x| {
            self.index.overwrite(x).unwrap()
        });
        let df = self.select(selection)?;
        self.save(df)
    }

    pub fn subset_axis<S>(&mut self, axis: usize, selection: S) -> Result<()>
    where
        S: AsRef<SelectInfoElem>,
    {
        let full = SelectInfoElem::full();
        let slice = selection.as_ref().set_axis(axis, 2, &full);
        self.subset(slice.as_slice())
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
                let column_names = container
                    .read_array_attr::<String, Ix1>("column-order")?
                    .into_raw_vec()
                    .into_iter()
                    .collect();
                let df = InnerDataFrameElem {
                    element: None,
                    container,
                    column_names,
                    index,
                };
                Ok(Slot::new(df))
            }
            ty => bail!("Expecting a dataframe but found: '{}'", ty),
        }
    }
}

impl<B: Backend> DataFrameElem<B> {
    /// Delete and Remove the data from the element.
    pub fn clear(&self) -> Result<()> {
        if let Some(elem) = self.extract() {
            DataContainer::delete(elem.container)?;
        }
        Ok(())
    }
}

/// Container holding general data types.
#[derive(Debug)]
pub struct InnerElem<B: Backend, T> {
    dtype: DataType,
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

    pub(crate) fn save<D: WriteData + Into<T>>(&mut self, data: D) -> Result<()> {
        replace_with::replace_with_or_abort(&mut self.container, |x| data.overwrite(x).unwrap());
        self.dtype = data.data_type();
        if self.element.is_some() {
            self.element = Some(data.into());
        }
        Ok(())
    }
}

impl<B: Backend, T: Clone> InnerElem<B, T> {
    pub fn data<D>(&mut self) -> Result<D>
    where
        D: Into<T> + ReadData + Clone + TryFrom<T>,
        <D as TryFrom<T>>::Error: Into<anyhow::Error>,
    {
        match self.element.as_ref() {
            Some(data) => Ok(data.clone().try_into().map_err(Into::into)?),
            None => {
                let data = D::read(&self.container)?;
                if self.cache_enabled {
                    self.element = Some(data.clone().into());
                }
                Ok(data)
            }
        }
    }
}

impl<B: Backend, T: ReadData + WriteData + Clone> InnerElem<B, T> {
    pub fn export<O: Backend, G: GroupOp<O>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<()> {
        match self.element.as_ref() {
            Some(data) => data.write(location, name)?,
            None => T::read(&self.container)?.write(location, name)?,
        };
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
            cache_enabled: false,
            element: None,
            container,
        };
        Ok(Slot::new(elem))
    }
}

impl<B: Backend> Elem<B> {
    /// Delete and Remove the data from the element.
    pub fn clear(&self) -> Result<()> {
        if let Some(elem) = self.extract() {
            DataContainer::delete(elem.container)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct InnerArrayElem<B: Backend, T> {
    dtype: DataType,
    shape: Shape,
    cache_enabled: bool,
    container: DataContainer<B>,
    element: Option<T>,
}

impl<B: Backend, T> std::fmt::Display for InnerArrayElem<B, T> {
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

impl<B: Backend, T> InnerArrayElem<B, T> {
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
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

    pub(crate) fn save<D: HasShape + WriteArrayData + Into<T>>(&mut self, data: D) -> Result<()> {
        replace_with::replace_with_or_abort(&mut self.container, |x| data.overwrite(x).unwrap());
        self.dtype = data.data_type();
        self.shape = data.shape();
        if self.element.is_some() {
            self.element = Some(data.into());
        }
        Ok(())
    }
}

impl<B: Backend, T: Clone> InnerArrayElem<B, T> {
    pub fn data<D>(&mut self) -> Result<D>
    where
        D: Into<T> + ReadData + Clone + TryFrom<T>,
        <D as TryFrom<T>>::Error: Into<anyhow::Error>,
    {
        match self.element.as_ref() {
            Some(data) => Ok(data.clone().try_into().map_err(Into::into)?),
            None => {
                let data = D::read(&self.container)?;
                if self.cache_enabled {
                    self.element = Some(data.clone().into());
                }
                Ok(data)
            }
        }
    }
}

impl<B: Backend, T: ReadArrayData + WriteArrayData + Clone> InnerArrayElem<B, T> {
    pub fn export<O: Backend, G: GroupOp<O>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<()> {
        match self.element.as_ref() {
            Some(data) => data.write(location, name)?,
            None => T::read(&self.container)?.write(location, name)?,
        };
        Ok(())
    }
}

impl<B: Backend, T: ArrayOp + Clone> InnerArrayElem<B, T> {
    pub fn select<D, S>(&mut self, selection: &[S]) -> Result<D>
    where
        D: Into<T> + TryFrom<T> + ReadArrayData + Clone,
        S: AsRef<SelectInfoElem>,
        <D as TryFrom<T>>::Error: Into<anyhow::Error>,
    {
        if selection.as_ref().iter().all(|x| x.as_ref().is_full()) {
            self.data()
        } else {
            match self.element.as_ref() {
                Some(data) => Ok(data.select(selection).try_into().map_err(Into::into)?),
                None => D::read_select(&self.container, selection),
            }
        }
    }

    pub fn select_axis<D, S>(&mut self, axis: usize, selection: S) -> Result<D>
    where
        D: Into<T> + TryFrom<T> + ReadArrayData + Clone,
        S: AsRef<SelectInfoElem>,
        <D as TryFrom<T>>::Error: Into<anyhow::Error>,
    {
        let full = SelectInfoElem::full();
        let slice = selection
            .as_ref()
            .set_axis(axis, self.shape().ndim(), &full);
        self.select(slice.as_slice())
    }
}

impl<B: Backend, T: ReadArrayData + WriteArrayData + ArrayOp + Clone> InnerArrayElem<B, T> {
    pub fn export_select<O, G>(
        &mut self,
        selection: &[&SelectInfoElem],
        location: &G,
        name: &str,
    ) -> Result<()>
    where
        O: Backend,
        G: GroupOp<O>,
    {
        if selection.as_ref().into_iter().all(|x| x.is_full()) {
            self.export::<O, _>(location, name)
        } else {
            self.select::<T, _>(selection)?.write(location, name)?;
            Ok(())
        }
    }

    pub fn export_axis<O, G>(
        &mut self,
        axis: usize,
        selection: &SelectInfoElem,
        location: &G,
        name: &str,
    ) -> Result<()>
    where
        O: Backend,
        G: GroupOp<O>,
    {
        let full = SelectInfoElem::full();
        let slice = selection
            .as_ref()
            .set_axis(axis, self.shape().ndim(), &full);
        self.export_select::<O, _>(slice.as_slice(), location, name)
    }

    pub(crate) fn subset<S>(&mut self, selection: &[S]) -> Result<()>
    where
        S: AsRef<SelectInfoElem>,
    {
        let data = match self.element.as_ref() {
            Some(data) => data.select(selection),
            None => T::read_select(&self.container, selection)?,
        };

        self.shape = data.shape();
        replace_with::replace_with_or_abort(&mut self.container, |x| data.overwrite(x).unwrap());
        if self.element.is_some() {
            self.element = Some(data);
        }
        Ok(())
    }

    pub(crate) fn subset_axis<S>(&mut self, axis: usize, selection: S) -> Result<()>
    where
        S: AsRef<SelectInfoElem>,
    {
        let full = SelectInfoElem::full();
        let slice = selection
            .as_ref()
            .set_axis(axis, self.shape().ndim(), &full);
        self.subset(slice.as_slice())
    }
}

pub type ArrayElem<B> = Slot<InnerArrayElem<B, ArrayData>>;

/// Container holding matrix data types.
impl<B: Backend> TryFrom<DataContainer<B>> for ArrayElem<B> {
    type Error = anyhow::Error;

    fn try_from(container: DataContainer<B>) -> Result<Self> {
        let dtype = container.encoding_type()?;
        let elem = InnerArrayElem {
            dtype,
            shape: ArrayData::get_shape(&container)?,
            cache_enabled: false,
            element: None,
            container,
        };
        Ok(Slot::new(elem))
    }
}

impl<B: Backend> ArrayElemOp for ArrayElem<B> {
    type ArrayIter<T> = ChunkedArrayElem<B, T>
        where
            T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
            <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;
    
    fn shape(&self) -> Option<Shape> {
        self.lock().as_ref().map(|x| x.shape().clone())
    }

    fn get<D>(&self) -> Result<Option<D>>
        where
            D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
            <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>
    {
        self.lock().as_mut().map(|x| x.data()).transpose()
    }

    fn slice<D, S>(&self, slice: S) -> Result<Option<D>>
        where
            D: ReadArrayData + Into<ArrayData> + TryFrom<ArrayData> + ArrayOp + Clone,
            S: AsRef<[SelectInfoElem]>,
            <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error> {
        self.lock().as_mut().map(|x| x.select(slice.as_ref())).transpose()
    }

    fn iter<T>(&self, chunk_size: usize) -> Self::ArrayIter<T>
        where
            T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
            <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>
    {
        self.chunked(chunk_size)
    }
}

impl<B: Backend> ArrayElem<B> {
    /// Delete and Remove the data from the element.
    pub fn clear(&self) -> Result<()> {
        if let Some(elem) = self.extract() {
            DataContainer::delete(elem.container)?;
        }
        Ok(())
    }

    pub fn chunked<T>(&self, chunk_size: usize) -> ChunkedArrayElem<B, T>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
    {
        ChunkedArrayElem::new(self.clone(), chunk_size)
    }
}

/// Horizontal concatenated dataframe elements.
pub struct StackedDataFrame<B: Backend> {
    column_names: IndexSet<String>,
    elems: Arc<Vec<DataFrameElem<B>>>,
    index: VecVecIndex,
}

impl<B: Backend> Clone for StackedDataFrame<B> {
    fn clone(&self) -> Self {
        Self {
            column_names: self.column_names.clone(),
            elems: self.elems.clone(),
            index: self.index.clone(),
        }
    }
}

impl<B: Backend> std::fmt::Display for StackedDataFrame<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "stacked dataframe with columns: '{}'",
            self.column_names.iter().join("', '")
        )
    }
}

impl<B: Backend> StackedDataFrame<B> {
    pub fn is_empty(&self) -> bool {
        self.elems.iter().all(|x| x.is_none())
    }

    pub fn width(&self) -> usize {
        self.column_names.len()
    }

    pub fn height(&self) -> usize {
        self.elems.iter().map(|x| x.inner().height()).sum()
    }

    pub fn new(elems: Vec<DataFrameElem<B>>) -> Result<Self> {
        let index = elems
            .iter()
            .map(|x| x.lock().as_ref().map(|x| x.height()).unwrap_or(0))
            .collect();
        if elems.iter().all(|x| x.is_none()) {
            Ok(Self {
                column_names: IndexSet::new(),
                elems: Arc::new(elems),
                index,
            })
        } else if elems.iter().all(|x| !x.is_none()) {
            let column_names = elems
                .iter()
                .map(|x| x.inner().get_column_names().clone())
                .reduce(|shared_keys, next_keys| {
                    shared_keys
                        .intersection(&next_keys)
                        .map(|x| x.to_owned())
                        .collect()
                })
                .unwrap_or(IndexSet::new());
            Ok(Self {
                column_names,
                elems: Arc::new(elems),
                index,
            })
        } else {
            bail!("slots must be either all empty or all full");
        }
    }

    pub fn get_column_names(&self) -> &IndexSet<String> {
        &self.column_names
    }

    pub fn data(&self) -> Result<DataFrame> {
        let df = if self.column_names.is_empty() || self.elems.is_empty() {
            DataFrame::empty()
        } else {
            let _sc = polars::datatypes::string_cache::StringCacheHolder::hold();
            let mut elems = self.elems.iter();
            let mut columns = elems.next().unwrap().inner().data()?
                .columns(self.column_names.iter())?.into_iter().cloned().collect::<Vec<_>>();
            elems.try_for_each(|el| {
                let mut inner = el.inner();
                let col = inner.data()?.columns(self.column_names.iter())?;
                columns.iter_mut().zip(col.into_iter()).try_for_each(|(a, b)| {
                    a.append(b)?;
                    Ok::<_, anyhow::Error>(())
                })
            })?;
            DataFrame::new(columns)?
        };
        Ok(df)
    }

    pub fn select<S>(&self, selection: &[S]) -> Result<DataFrame>
    where
        S: AsRef<SelectInfoElem>,
    {
        let (indices, mapping) = self.index.split_select(selection.as_ref()[0].as_ref());
        let dfs = self
            .elems
            .iter()
            .enumerate()
            .flat_map(|(i, el)| {
                indices.get(&i).and_then(|idx| {
                    let select: SmallVec<[_; 3]> = std::iter::once(idx)
                        .chain(selection.as_ref()[1..].iter().map(|x| x.as_ref()))
                        .collect();
                    el.lock()
                        .as_mut()
                        .map(|x| x.select(select.as_slice()).unwrap().lazy())
                })
            })
            .collect::<Vec<_>>();
        let df = concat(
            &dfs,
            UnionArgs{parallel: true, rechunk: true, ..Default::default()}
            )?.collect()?;
        if let Some(m) = mapping {
            let select: SmallVec<[SelectInfoElem; 3]> = std::iter::once(m.into())
                .chain(std::iter::repeat((..).into()).take(selection.as_ref().len() - 1))
                .collect();
            Ok(ArrayOp::select(&df, select.as_slice()))
        } else {
            Ok(df)
        }
    }

    // TODO: this is not efficient, we should use the index to select the columns
    pub fn column(&self, name: &str) -> Result<Series> {
        if self.column_names.contains(name) {
            Ok(self.data()?.column(name)?.clone())
        } else {
            bail!("key is not present");
        }
    }
}

pub struct InnerStackedArrayElem<B: Backend> {
    shape: Option<Shape>,
    elems: SmallVec<[ArrayElem<B>; 96]>,
    index: VecVecIndex,
}

impl<B: Backend> std::fmt::Display for InnerStackedArrayElem<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.elems.len() == 0 {
            write!(f, "empty stacked elements")
        } else {
            write!(
                f,
                "{} stacked elements ({}) with {}",
                self.shape.as_ref().unwrap(),
                self.elems.len(),
                self.elems[0].inner().dtype(),
            )
        }
    }
}

impl<B: Backend> InnerStackedArrayElem<B> {
    pub(crate) fn get_index(&self) -> &VecVecIndex {
        &self.index
    }

    pub fn is_empty(&self) -> bool {
        self.elems.is_empty() || self.elems.iter().all(|x| x.is_none())
    }

    pub fn dtype(&self) -> DataType {
        self.elems[0].inner().dtype()
    }

    pub fn shape(&self) -> &Option<Shape> {
        &self.shape
    }

    pub fn data<D>(&self) -> Result<Option<D>>
    where
        D: Into<ArrayData> + ReadData + Clone + TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let data = if self.is_empty() {
            None
        } else {
            let array = self
                .elems
                .iter()
                .flat_map(|x| x.lock().as_mut().map(|i| i.data::<ArrayData>()))
                .process_results(|x| ArrayOp::vstack(x).unwrap())?;
            Some(array.try_into().map_err(Into::into)?)
        };
        Ok(data)
    }

    pub fn par_data<D>(&self) -> Result<Option<D>>
    where
        D: Into<ArrayData> + ReadData + Clone + TryFrom<ArrayData>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let data = if self.is_empty() {
            None
        } else {
            let array = self
                .elems
                .par_iter()
                .flat_map(|x| x.lock().as_mut().map(|i| i.data::<ArrayData>()))
                .collect::<Vec<_>>()
                .into_iter()
                .process_results(|x| ArrayOp::vstack(x).unwrap())?;
            Some(array.try_into().map_err(Into::into)?)
        };
        Ok(data)
    }

    pub fn select_axis<D, S>(&self, axis: usize, selection: S) -> Result<Option<D>>
    where
        D: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        S: AsRef<SelectInfoElem>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.shape.as_ref().map(|s| {
            let n = s.ndim();
            let full = SelectInfoElem::full();
            let slice = selection
                .as_ref()
                .set_axis(axis, n, &full);
            self.select(slice.as_slice()).map(|x| x.unwrap())
        }).transpose()
    }
 
    pub fn select<D, S>(&self, selection: &[S]) -> Result<Option<D>>
    where
        D: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        S: AsRef<SelectInfoElem>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let data = if self.is_empty() {
            None
        } else {
            let (indices, mapping) = self.index.split_select(selection.as_ref()[0].as_ref());
            let array: ArrayData = self
                .elems
                .iter()
                .enumerate()
                .flat_map(|(i, el)| {
                    indices.get(&i).map(|idx| {
                        let select: SmallVec<[_; 3]> = std::iter::once(idx)
                            .chain(selection.as_ref()[1..].iter().map(|x| x.as_ref()))
                            .collect();
                        el.inner().select(select.as_slice())
                    })
                })
                .process_results(|x| ArrayOp::vstack(x).unwrap())?;
            if let Some(m) = mapping {
                Some(array
                    .select_axis(0, SelectInfoElem::from(reverse_mapping(m)))
                    .try_into().map_err(Into::into)?)
            } else {
                Some(array.try_into().map_err(Into::into)?)
            }
        };
        Ok(data)
    }

    pub fn par_select<D, S>(&self, selection: &[S]) -> Result<Option<D>>
    where
        D: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        S: AsRef<SelectInfoElem> + Sync,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let data = if self.is_empty() {
            None
        } else {
            let (indices, mapping) = self.index.split_select(selection.as_ref()[0].as_ref());
            let array: ArrayData = self
                .elems
                .par_iter()
                .enumerate()
                .flat_map(|(i, el)| {
                    indices.get(&i).map(|idx| {
                        let select: SmallVec<[_; 3]> = std::iter::once(idx)
                            .chain(selection.as_ref()[1..].iter().map(|x| x.as_ref()))
                            .collect();
                        el.inner().select(select.as_slice())
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .process_results(|x| ArrayOp::vstack(x).unwrap())?;
            if let Some(m) = mapping {
                Some(array
                    .select_axis(0, SelectInfoElem::from(reverse_mapping(m)))
                    .try_into()
                    .map_err(Into::into)?)
            } else {
                Some(array.try_into().map_err(Into::into)?)
            }
        };
        Ok(data)
    }

    /// Activate the cache for all elements.
    pub fn enable_cache(&self) {
        for el in self.elems.iter() {
            if let Some(x) = el.lock().as_mut() {
                x.enable_cache();
            }
        }
    }

    /// Deactivate the cache for all elements.
    pub fn disable_cache(&self) {
        for el in self.elems.iter() {
            if let Some(x) = el.lock().as_mut() {
                x.disable_cache();
            }
        }
    }
}

pub struct StackedArrayElem<B: Backend>(Arc<InnerStackedArrayElem<B>>);

impl<B: Backend> Clone for StackedArrayElem<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<B: Backend> std::fmt::Display for StackedArrayElem<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.deref().fmt(f)
    }
}

impl<B: Backend> Deref for StackedArrayElem<B> {
    type Target = InnerStackedArrayElem<B>;
    fn deref(&self) -> &InnerStackedArrayElem<B> {
        &self.0
    }
}

impl<B: Backend> ArrayElemOp for StackedArrayElem<B> {
    type ArrayIter<T> = StackedChunkedArrayElem<B, T>
        where
            T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
            <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;
    
    fn shape(&self) -> Option<Shape> {
        self.shape.clone()
    }

    fn get<D>(&self) -> Result<Option<D>>
        where
            D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
            <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>
    {
        self.data()
    }

    fn slice<D, S>(&self, slice: S) -> Result<Option<D>>
        where
            D: ReadArrayData + Into<ArrayData> + TryFrom<ArrayData> + ArrayOp + Clone,
            S: AsRef<[SelectInfoElem]>,
            <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error> {
        self.select(slice.as_ref())
    }

    fn iter<T>(&self, chunk_size: usize) -> Self::ArrayIter<T>
        where
            T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
            <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>
    {
        self.chunked(chunk_size)
    }
}


impl<B: Backend> StackedArrayElem<B> {
    pub fn empty() -> Self {
        Self(Arc::new(InnerStackedArrayElem {
            shape: None,
            elems: SmallVec::new(),
            index: std::iter::empty().collect(),
        }))
    }

    pub(crate) fn new(elems: SmallVec<[ArrayElem<B>; 96]>) -> Result<Self> {
        ensure!(
            elems
                .iter()
                .map(|x| x.lock().as_ref().map(|x| x.dtype()))
                .all_equal(),
            "all elements must have the same dtype"
        );

        let shapes: Vec<_> = elems
            .iter()
            .map(|x| x.lock().as_ref().map(|x| x.shape().clone()))
            .collect();
        ensure!(
            shapes.iter().map(|x| x.as_ref().map(|s| &s.as_ref()[1..])).all_equal(),
            "all elements must have the same shape except for the first axis"
        );
        let index: VecVecIndex = shapes.iter().flatten().map(|x| x.as_ref()[0]).collect();
        let shape = shapes.get(0).and_then(|x| x.as_ref().map(|s| {
            let mut ss = s.clone();
            ss[0] = index.len();
            ss
        }));
        Ok(Self(Arc::new(InnerStackedArrayElem { shape, elems, index })))
    }

    pub fn chunked<T>(&self, chunk_size: usize) -> StackedChunkedArrayElem<B, T>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
    {
        StackedChunkedArrayElem::new(self.elems.iter().map(|x| x.clone()), chunk_size)
    }
}

/// Chunked Arrays
pub struct ChunkedArrayElem<B: Backend, T> {
    /// The underlying array element.
    elem: ArrayElem<B>,
    /// The chunk size.
    chunk_size: usize,
    num_items: usize,
    current_position: usize,
    type_marker: std::marker::PhantomData<T>,
}

impl<B: Backend, T> ChunkedArrayElem<B, T> {
    pub fn new(elem: ArrayElem<B>, chunk_size: usize) -> Self {
        let num_items = elem.inner().shape()[0];
        Self {
            elem,
            chunk_size,
            num_items,
            current_position: 0,
            type_marker: std::marker::PhantomData,
        }
    }
}

impl<B, T> Iterator for ChunkedArrayElem<B, T>
where
    B: Backend,
    T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
    <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
{
    type Item = (T, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_position >= self.num_items {
            None
        } else {
            let i = self.current_position;
            let j = std::cmp::min(self.num_items, self.current_position + self.chunk_size);
            self.current_position = j;
            let data = self.elem.inner().select_axis(0, SelectInfoElem::from(i..j)).unwrap();
            Some((data, i, j))
        }
    }
}

impl<B, T> ExactSizeIterator for ChunkedArrayElem<B, T>
where
    B: Backend,
    T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
    <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
{
    fn len(&self) -> usize {
        let (n, remain) = div_rem(self.num_items, self.chunk_size);
        if remain == 0 {
            n
        } else {
            n + 1
        }
    }
}

pub struct StackedChunkedArrayElem<B: Backend, T> {
    arrays: SmallVec<[ChunkedArrayElem<B, T>; 96]>,
    current_position: usize,
    current_array: usize,
}

impl<B: Backend, T> StackedChunkedArrayElem<B, T> {
    pub(crate) fn new<I: Iterator<Item = ArrayElem<B>>>(elems: I, chunk_size: usize) -> Self {
        Self {
            arrays: elems
                .map(|x| ChunkedArrayElem::new(x, chunk_size))
                .collect(),
            current_position: 0,
            current_array: 0,
        }
    }
}

impl<B, T> Iterator for StackedChunkedArrayElem<B, T>
where
    B: Backend,
    T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
    <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
{
    type Item = (T, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(mat) = self.arrays.get_mut(self.current_array) {
            if let Some((data, start, stop)) = mat.next() {
                let new_start = self.current_position;
                let new_stop = new_start + stop - start;
                self.current_position = new_stop;
                Some((data, new_start, new_stop))
            } else {
                self.current_array += 1;
                self.next()
            }
        } else {
            None
        }
    }
}

impl<B, T> ExactSizeIterator for StackedChunkedArrayElem<B, T>
where
    B: Backend,
    T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
    <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
{
    fn len(&self) -> usize {
        self.arrays.iter().map(|x| x.len()).sum()
    }
}

fn reverse_mapping(mapping: Vec<usize>) -> Vec<usize> {
    let mut res = vec![0; mapping.len()];
    for (i, x) in mapping.into_iter().enumerate() {
        res[x] = i;
    }
    res
}