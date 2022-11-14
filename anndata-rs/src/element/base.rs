use crate::{
    data::*, iterator::{ChunkedMatrix, StackedChunkedMatrix},
    utils::hdf5::{create_str_attr, read_str_attr, read_str_vec_attr, read_str_vec, create_dataset},
};

use std::{boxed::Box, collections::HashMap, sync::Arc, ops::{Deref, DerefMut}};
use hdf5::Group; 
use anyhow::{Result, ensure, bail};
use ndarray::Array1;
use polars::{frame::DataFrame, series::Series};
use parking_lot::{Mutex, MutexGuard};
use indexmap::set::IndexSet;
use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// Slot stores an optional object wrapped by Arc and Mutex.
pub struct Slot<T>(pub(crate) Arc<Mutex<Option<T>>>);

impl<T> Clone for Slot<T> {
    fn clone(&self) -> Self { Slot(self.0.clone()) }
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
    pub fn new(x: T) -> Self { Slot(Arc::new(Mutex::new(Some(x)))) }

    /// Create an empty slot.
    pub fn empty() -> Self { Slot(Arc::new(Mutex::new(None))) }

    pub fn is_empty(&self) -> bool { self.0.lock().is_none() }

    pub fn lock(&self) -> MutexGuard<'_, Option<T>> { self.0.lock() }

    pub fn inner(&self) -> Inner<'_, T> { Inner(self.0.lock()) }

    /// Insert data to the slot, and return the old data.
    pub fn insert(&self, data: T) -> Option<T> {
        std::mem::replace(self.0.lock().deref_mut(), Some(data))
    }

    /// Extract the data from the slot. The slot becomes empty after this operation.
    pub fn extract(&self) -> Option<T> {
        std::mem::replace(self.0.lock().deref_mut(), None)
    }

    /// Remove the data from the slot.
    pub fn drop(&self) { let _ = self.extract(); } 
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

pub struct DataFrameIndex {
    pub index_name: String,
    pub names: Vec<String>,
    pub index_map: HashMap<String, usize>,
}

impl DataFrameIndex {
    pub fn len(&self) -> usize { self.names.len() }

    pub fn get(&self, k: &String) -> Option<usize> { self.index_map.get(k).map(|x| *x) }

    pub fn is_empty(&self) -> bool { self.names.is_empty() }

    pub fn write(&self, container: &DataContainer) -> hdf5::Result<()> {
        let group = container.get_group_ref()?;
        create_str_attr(group, "_index", &self.index_name)?;

        if group.link_exists(&self.index_name) { group.unlink(&self.index_name)?; }

        let names: Array1<hdf5::types::VarLenUnicode> =
            self.names.iter().map(|x| x.parse().unwrap()).collect();
        create_dataset(group, &self.index_name, &names)?;
        Ok(())
    }
}

impl From<Vec<String>> for DataFrameIndex {
    fn from(names: Vec<String>) -> Self {
        let index_map = names.clone().into_iter().enumerate().map(|(a, b)| (b, a)).collect();
        Self { index_name: "index".to_owned(), names, index_map }
    }
}

impl From<usize> for DataFrameIndex {
    fn from(size: usize) -> Self { (0..size).map(|x| x.to_string()).collect() }
}

impl FromIterator<String> for DataFrameIndex {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = String>,
    {
        let names: Vec<_> = iter.into_iter().collect();
        let index_map = names.clone().into_iter().enumerate().map(|(a, b)| (b, a)).collect();
        Self { index_name: "index".to_owned(), names, index_map }
    }
}

impl<'a> FromIterator<&'a str> for DataFrameIndex {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a str>,
    {
        let names: Vec<_> = iter.into_iter().map(|x| x.to_owned()).collect();
        let index_map = names.clone().into_iter().enumerate().map(|(a, b)| (b, a)).collect();
        Self { index_name: "index".to_owned(), names, index_map }
    }
}

pub struct InnerDataFrameElem {
    element: Option<DataFrame>,
    container: DataContainer,
    column_names: IndexSet<String>,
    pub index: DataFrameIndex,
}

impl InnerDataFrameElem {
    pub fn new(location: &Group, name: &str, index: DataFrameIndex, df: &DataFrame) -> Result<Self> {
        ensure!(
            df.height() == 0 || index.len() == df.height(),
            "cannot create dataframe element as lengths of index and dataframe differ"
        );
        let container = df.write(location, name)?;
        index.write(&container)?;
        let column_names = df.get_column_names_owned().into_iter().collect();
        Ok(Self { element: None, container, column_names, index })
    }
}

impl std::fmt::Display for InnerDataFrameElem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dataframe element")
    }
}

pub type DataFrameElem = Slot<InnerDataFrameElem>;

impl TryFrom<DataContainer> for DataFrameElem {
    type Error = anyhow::Error;

    fn try_from(container: DataContainer) -> Result<Self> {
        match container.get_encoding_type()? {
            DataType::DataFrame => {
                let grp = container.get_group_ref()?;
                let index_name = read_str_attr(grp, "_index")?;
                let mut index: DataFrameIndex = read_str_vec(&grp.dataset(&index_name)?)?.into();
                index.index_name = index_name;
                let column_names = read_str_vec_attr(grp, "column-order")?.into_iter().collect();
                let df = InnerDataFrameElem { element: None, container, column_names, index };
                Ok(Slot::new(df))
            },
            ty => bail!("Expecting a dataframe but found: {}", ty),
        }
    }
}

impl DataFrameElem {
    pub fn nrows(&self) -> usize { self.lock().as_ref().map(|x| x.index.len()).unwrap_or(0) }

    pub fn with_data_ref<F, O>(&self, f: F) -> Result<O>
    where
        F: Fn(Option<(&DataFrameIndex, &DataFrame)>) -> O,
    {
        if self.is_empty() {
            Ok(f(None))
        } else {
            let mut inner = self.inner();
            if let None = inner.element {
                inner.element = Some(DataFrame::read(&inner.container)?);
            }
            Ok(f(Some(
                (&inner.index, inner.element.as_ref().unwrap())
            )))
        }
    }

    pub fn with_data_mut_ref<F, O>(&self, mut f: F) -> Result<O>
    where
        F: FnMut(Option<(&DataFrameIndex, &DataFrame)>) -> O,
    {
        if self.is_empty() {
            Ok(f(None))
        } else {
            let mut inner = self.inner();
            if let None = inner.element {
                inner.element = Some(DataFrame::read(&inner.container)?);
            }
            Ok(f(Some(
                (&inner.index, inner.element.as_ref().unwrap())
            )))
        }
    }

    pub fn read(&self) -> Result<DataFrame> {
        self.with_data_ref(|x| x.map(|(_, df)| df.clone()).unwrap_or(DataFrame::empty()))
    }

    pub fn write(&self, idx_: Option<&[usize]>, location: &Group, name: &str) -> Result<()> {
        self.with_data_ref(|x| {
            if let Some((index, df)) = x {
                match idx_ {
                    None => index.write(&df.write(location, name)?),
                    Some(idx) => {
                        let new_df = df.take_iter(idx.into_iter().map(|x| *x))?;
                        let new_index: DataFrameIndex = idx.into_iter().map(|i| index.names[*i].clone()).collect();
                        new_index.write(&new_df.write(location, name)?)
                    },
                }?;
            }
            Ok(())
        })?
    }

    pub fn update(&self, data: DataFrame) -> Result<()> {
        ensure!(!self.is_empty(), "cannot update an empty DataFrameElem");
        let num_recs = data.height();
        let mut inner = self.inner();
        ensure!(
            num_recs == 0 || inner.index.len() == num_recs,
            "cannot update dataframe as lengths differ"
        );
        inner.container = data.update(&inner.container)?;  // Note updating the container removes the index
        inner.index.write(&inner.container)?; // So we need to add back the index here
        inner.column_names = data.get_column_names_owned().into_iter().collect();
        if inner.element.is_some() { inner.element = Some(data); }
        Ok(())
    }

    pub fn column(&self, name: &str) -> Result<Series> {
        self.with_data_ref(|x| x.unwrap().1.column(name).unwrap().clone())
    }

    pub fn get_column_names(&self) -> Option<IndexSet<String>> {
        self.lock().as_ref().map(|x| x.column_names.clone())
    }

    pub fn set_index(&self, index: DataFrameIndex) -> Result<()> {
        let mut inner = self.inner();
        ensure!(inner.index.len() == index.len(), "cannot change the index as the lengths differ");
        inner.index = index;
        inner.index.write(&inner.container)?;
        Ok(())
    }

    pub fn subset_rows(&self, idx: &[usize]) -> Result<()> {
        let subset = self.with_data_ref(|x| x.map(|(index, df)| {
            let new_df = df.take_iter(idx.into_iter().map(|x| *x))?;
            let new_index: DataFrameIndex = idx.into_iter().map(|i| index.names[*i].clone()).collect();
            Ok::<_, anyhow::Error>((new_index, new_df))
        }))?.transpose()?;

        if let Some((index, df)) = subset {
            let mut inner = self.inner();
            inner.container = df.update(&inner.container)?;
            index.write(&inner.container)?;
            inner.column_names = df.get_column_names_owned().into_iter().collect();
            if inner.element.is_some() { inner.element = Some(df); }
            inner.index = index;
        }
        Ok(())
    }
}


/// Container holding general data types.
pub struct InnerElem {
    dtype: DataType,
    cache_enabled: bool,
    container: DataContainer,
    element: Option<Box<dyn Data>>,
}

impl std::fmt::Display for InnerElem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} element, cache_enabled: {}, cached: {}",
            self.dtype,
            if self.cache_enabled { "yes" } else { "no" },
            if self.element.is_some() { "yes" } else { "no" },
        )
    }
}

#[derive(Clone)]
pub struct Elem(Arc<Mutex<InnerElem>>);

impl std::fmt::Display for Elem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.lock().fmt(f)
    }
}

impl TryFrom<DataContainer> for Elem {
    type Error = anyhow::Error;

    fn try_from(container: DataContainer) -> Result<Self> {
        let dtype = container.get_encoding_type()?;
        let elem = InnerElem { dtype, cache_enabled: false, element: None, container };
        Ok(Elem(Arc::new(Mutex::new(elem))))
    }
}

impl Elem {
    pub fn dtype(&self) -> DataType { self.0.lock().dtype.clone() }

    pub fn read(&self) -> Result<Box<dyn Data>> {
        let mut inner = self.0.lock();
        match inner.element.as_ref() {
            Some(data) => Ok(data.clone()),
            None => {
                let data = <Box<dyn Data>>::read(&inner.container)?;
                if inner.cache_enabled { inner.element = Some(data.clone()); }
                Ok(data)
            }
        }
    }

    pub fn write(&self, location: &Group, name: &str) -> Result<()> {
        let inner = self.0.lock();
        match inner.element.as_ref() {
            Some(data) => data.write(location, name)?,
            None => <Box<dyn Data>>::read(&inner.container)?.write(location, name)?,
        };
        Ok(())
    }

    pub fn update<D: Data>(&self, data: D) -> Result<()> {
        let mut inner = self.0.lock();
        inner.container = data.update(&inner.container)?;
        if inner.element.is_some() { inner.element = Some(data.into_dyn_data()); }
        Ok(())
    }

    pub fn enable_cache(&self) { self.0.lock().cache_enabled = true; }

    pub fn disable_cache(&self) {
        let mut inner = self.0.lock();
        if inner.element.is_some() { inner.element = None; }
        inner.cache_enabled = false;
    }
}

/// Container holding matrix data types.
pub struct InnerMatrixElem {
    dtype: DataType,
    cache_enabled: bool,
    container: DataContainer,
    nrows: usize,
    ncols: usize,
    element: Option<Box<dyn MatrixData>>,
}

impl std::fmt::Display for InnerMatrixElem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} x {} {} element, cache_enabled: {}, cached: {}",
            self.nrows, self.ncols, self.dtype,
            if self.cache_enabled { "yes" } else { "no" },
            if self.element.is_some() { "yes" } else { "no" },
        )
    }
}

impl TryFrom<DataContainer> for InnerMatrixElem {
    type Error = anyhow::Error;

    fn try_from(container: DataContainer) -> Result<Self> {
        let dtype = container.get_encoding_type()?;
        let nrows = <Box<dyn MatrixData>>::get_nrows(&container);
        let ncols = <Box<dyn MatrixData>>::get_ncols(&container);
        Ok(Self { dtype, cache_enabled: false, element: None, container, nrows, ncols })
    }
}

pub type MatrixElem = Slot<InnerMatrixElem>;

impl TryFrom<DataContainer> for MatrixElem {
    type Error = anyhow::Error;
    fn try_from(container: DataContainer) -> Result<Self> { Ok(Slot::new(InnerMatrixElem::try_from(container)?)) }
}

impl MatrixElem {
    pub fn dtype(&self) -> Option<DataType> { self.lock().as_ref().map(|x| x.dtype.clone()) }

    pub fn enable_cache(&self) { self.inner().cache_enabled = true; }
    pub fn disable_cache(&self) {
        let mut inner = self.inner();
        if inner.element.is_some() { inner.element = None; }
        inner.cache_enabled = false;
    }

    pub fn nrows(&self) -> usize { self.inner().nrows }
    pub fn ncols(&self) -> usize { self.inner().ncols }

    pub fn read(&self, ridx: Option<&[usize]>, cidx: Option<&[usize]>) -> Result<Box<dyn MatrixData>> {
        ensure!(!self.is_empty(), "cannot read an empty MatrixElem");
        let mut inner = self.inner();
        match inner.element.as_ref() {
            Some(data) => match (ridx, cidx) {
                (None, None) => Ok(data.clone()),
                (Some(i), Some(j)) => Ok(data.subset(i, j)),
                (Some(i), None) => Ok(data.get_rows(i)),
                (None, Some(j)) => Ok(data.get_columns(j)),
            }
            None => match (ridx, cidx) {
                (None, None) => {
                    let data = <Box<dyn MatrixData>>::read(&inner.container)?;
                    if inner.cache_enabled {
                        inner.element = Some(data.clone());
                    }
                    Ok(data)
                },
                (Some(i), Some(j)) => Ok(<Box<dyn MatrixData>>::read_partial(&inner.container, i, j)),
                (Some(i), None) => Ok(<Box<dyn MatrixData>>::read_rows(&inner.container, i)),
                (None, Some(j)) => Ok(<Box<dyn MatrixData>>::read_columns(&inner.container, j)),
            }
        }
    }

    // TODO: use in-memory data when possible
    pub fn read_row_slice(&self, slice: std::ops::Range<usize>) -> Result<Box<dyn MatrixData>> {
        ensure!(!self.is_empty(), "cannot read rows from an empty MatrixElem");
        let inner = self.inner();
        Ok(<Box<dyn MatrixData>>::read_row_slice(&inner.container, slice)?)
    }
    
    pub fn write(&self, ridx: Option<&[usize]>, cidx: Option<&[usize]>, location: &Group, name: &str) -> Result<()> {
        if !self.is_empty() {
            self.read(ridx, cidx)?.write(location, name)?;
        }
        Ok(())
    }

    pub fn subset(&self, ridx: Option<&[usize]>, cidx: Option<&[usize]>) -> Result<()> {
        if !self.is_empty() {
            let mut inner = self.inner();
            let sub = match inner.element.as_ref() {
                Some(data) => match (ridx, cidx) {
                    (None, None) => None,
                    (Some(i), Some(j)) => Some(data.subset(i, j)),
                    (Some(i), None) => Some(data.get_rows(i)),
                    (None, Some(j)) => Some(data.get_columns(j)),
                }
                None => match (ridx, cidx) {
                    (None, None) => None,
                    (Some(i), Some(j)) => Some(<Box<dyn MatrixData>>::read_partial(&inner.container, i, j)),
                    (Some(i), None) => Some(<Box<dyn MatrixData>>::read_rows(&inner.container, i)),
                    (None, Some(j)) => Some(<Box<dyn MatrixData>>::read_columns(&inner.container, j)),
                },
            };
            if let Some(data) = sub {
                inner.container = data.update(&inner.container)?;
                if inner.element.is_some() { inner.element = Some(data); }
                if let Some(i) = ridx { inner.nrows = i.len(); }
                if let Some(j) = cidx { inner.ncols = j.len(); }
            }
        }
        Ok(())
    }

    pub fn update<D: MatrixData>(&self, data: D) -> Result<()> {
        ensure!(!self.is_empty(), "cannot update an empty MatrixElem");
        let mut inner = self.inner();
        inner.nrows = data.nrows();
        inner.ncols = data.ncols();
        inner.container = data.update(&inner.container)?;
        if inner.element.is_some() { inner.element = Some(data.into_dyn_matrix()); }
        Ok(())
    }

    pub fn chunked(&self, chunk_size: usize) -> ChunkedMatrix { ChunkedMatrix::new(self.clone(), chunk_size) }
}

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
    pub fn inv_ix(&self, idx: (usize, usize)) -> usize { self.0[idx.0] + idx.1 }

    /// Find the outer index for a given index corresponding to the flattened view. 
    pub fn outer_ix(&self, i: &usize) -> usize { 
        match self.0.binary_search(i) {
            Ok(i_) => i_,
            Err(i_) => i_ - 1,
        }
    }

    pub fn ix_group_by_outer<'a, I>(&self, indices: I) -> std::collections::HashMap<usize, (Vec<usize>, Vec<usize>)>
    where I: Iterator<Item = &'a usize>
    {
        indices.map(|x| self.ix(x)).enumerate().sorted_by_key(|(_, (x,_))| *x).into_iter()
            .group_by(|(_, (x,_))| *x).into_iter()
            .map(|(outer, inner)| (outer, inner.map(|(i, (_, x))| (x, i)).unzip())).collect()
    }

    /// The total number of elements
    pub fn len(&self) -> usize { *self.0.last().unwrap_or(&0) }
}

impl FromIterator<usize> for VecVecIndex {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = usize>,
    {
        let index : Vec<usize> = std::iter::once(0).chain(
            iter.into_iter().scan(0, |state, x| {
                *state = *state + x;
                Some(*state)
            })
        ).collect();
        VecVecIndex(index)
    }
}

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

// TODO: remove Arc.
#[derive(Clone)]
pub struct StackedMatrixElem {
    nrows: Arc<Mutex<usize>>,
    ncols: Arc<Mutex<usize>>,
    pub elems: Arc<Vec<MatrixElem>>,
    index: Arc<Mutex<VecVecIndex>>,
}

impl std::fmt::Display for StackedMatrixElem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.elems.len() == 0 {
            write!(f, "empty stacked elements")
        } else {
            write!(f, "{} x {} stacked elements ({}) with {}",
                *self.nrows.lock(),
                *self.ncols.lock(),
                self.elems.len(),
                self.elems[0].dtype().unwrap(),
            )
        }
    }
}

impl StackedMatrixElem
{
    pub(crate) fn new(
        elems: Vec<MatrixElem>,
        nrows: Arc<Mutex<usize>>,
        ncols: Arc<Mutex<usize>>,
        index: Arc<Mutex<VecVecIndex>>,
    ) -> Result<Self> {
        if !elems.iter().map(|x| x.dtype()).all_equal() {
            bail!("dtype not equal")
        } else {
            Ok(Self { nrows, ncols, elems: Arc::new(elems), index })
        }
    }

    pub fn nrows(&self) -> usize { *self.nrows.lock() }
    pub fn ncols(&self) -> usize { *self.ncols.lock() }

    pub fn read(&self, ridx_: Option<&[usize]>, cidx: Option<&[usize]>) -> Result<Box<dyn MatrixData>> {
        match ridx_ {
            Some(ridx) => {
                let index = self.index.lock();
                let (ori_idx, rows): (Vec<_>, Vec<_>) = ridx.iter().map(|x| index.ix(x))
                    .enumerate().sorted_by_key(|x| x.1.0).into_iter()
                    .group_by(|x| x.1.0).into_iter().map(|(key, grp)| {
                        let (ori_idx, (_, inner_idx)): (Vec<_>, (Vec<_>, Vec<_>)) = grp.unzip();
                        (ori_idx, self.elems[key].read(Some(inner_idx.as_slice()), cidx))
                    }).unzip();
                Ok(rstack_with_index(
                    ori_idx.into_iter().flatten().collect::<Vec<_>>().as_slice(),
                    rows.into_iter().collect::<Result<_>>()?
                )?)
            },
            None => {
                let mats: Result<Vec<_>> = self.elems.par_iter().map(|x| x.read(None, cidx)).collect();
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
        write!(f, "stacked dataframe with columns: '{}'", self.column_names.iter().join("', '"))
    }
}

impl StackedDataFrame {
    pub fn new(elems: Vec<DataFrameElem>) -> Self {
        let column_names = elems.iter().map(|x| x.get_column_names().unwrap_or(IndexSet::new()))
            .reduce(|shared_keys, next_keys| shared_keys.intersection(&next_keys).map(|x| x.to_owned()).collect())
            .unwrap_or(IndexSet::new());
        Self { column_names, elems: Arc::new(elems) }
    }

    pub fn read(&self) -> Result<DataFrame> {
        let mut merged = DataFrame::empty();
        self.elems.iter().for_each(|el| {
            el.with_data_mut_ref(|x| 
                if let Some((_, df)) = x { merged.vstack_mut(df).unwrap(); }
            ).unwrap();
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