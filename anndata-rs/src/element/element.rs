use crate::{
    anndata_trait::*,
    iterator::{ChunkedMatrix, StackedChunkedMatrix},
    element::{RawMatrixElem, RawElem},
    utils::hdf5::{read_str_vec_attr, read_str_attr, read_str_vec},
};

use polars::{frame::DataFrame, series::Series};
use std::collections::HashSet;
use hdf5::Group; 
use anyhow::{anyhow, Result};
use std::sync::Arc;
use parking_lot::{Mutex, MutexGuard};
use itertools::Itertools;
use std::ops::{Deref, DerefMut};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

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
    pub fn new(x: T) -> Self { Slot(Arc::new(Mutex::new(Some(x)))) }

    pub fn empty() -> Self { Slot(Arc::new(Mutex::new(None))) }

    pub fn is_empty(&self) -> bool { self.0.lock().is_none() }

    pub fn inner(&self) -> Inner<'_, T> { Inner(self.0.lock()) }

    pub fn insert(&self, data: T) -> Option<T> {
        std::mem::replace(self.0.lock().deref_mut(), Some(data))
    }

    pub fn extract(&self) -> Option<T> {
        std::mem::replace(self.0.lock().deref_mut(), None)
    } 

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


pub trait ElemTrait {
    type Data;

    fn dtype(&self) -> DataType;

    fn is_empty(&self) -> bool;

    fn new_elem(container: DataContainer) -> Result<Self> where Self: Sized;

    fn read(&self) -> Result<Self::Data>;

    fn write(&self, location: &Group, name: &str) -> Result<()>;

    fn update(&self, data: &Self::Data);

    fn enable_cache(&self);

    fn disable_cache(&self);
}

pub type Elem = Slot<RawElem<dyn DataIO>>;

impl ElemTrait for Elem {
    type Data = Box<dyn DataIO>;

    fn dtype(&self) -> DataType { self.inner().dtype.clone() }

    fn is_empty(&self) -> bool { self.inner().0.is_none() }

    fn new_elem(container: DataContainer) -> Result<Self> {
        let elem = RawElem::new(container)?;
        Ok(Self(Arc::new(Mutex::new(Some(elem)))))
    }

    fn read(&self) -> Result<Self::Data> { self.inner().read() }

    fn write(&self, location: &Group, name: &str) -> Result<()> {
        self.inner().write(location, name)
    }

    fn update(&self, data: &Self::Data) {
        self.inner().update(data).unwrap();
    }

    fn enable_cache(&self) { self.inner().enable_cache(); }

    fn disable_cache(&self) { self.inner().disable_cache(); }
}

pub type MatrixElem = Slot<RawMatrixElem<dyn DataPartialIO>>;

impl MatrixElem {
    pub fn nrows(&self) -> usize { self.inner().nrows() }

    pub fn ncols(&self) -> usize { self.inner().ncols() }

    pub fn subset_rows(&self, idx: &[usize]) {
        self.inner().subset_rows(idx).unwrap();
    }

    pub fn subset_cols(&self, idx: &[usize]) {
        self.inner().subset_cols(idx).unwrap();
    }

    pub fn subset(&self, ridx: &[usize], cidx: &[usize]) {
        self.inner().subset(ridx, cidx).unwrap();
    }

    pub fn chunked(&self, chunk_size: usize) -> ChunkedMatrix {
        ChunkedMatrix {
            elem: self.clone(),
            chunk_size,
            size: self.nrows(),
            current_index: 0,
        }
    }
}

pub type DataFrameElem = Slot<RawMatrixElem<DataFrame>>;

impl DataFrameElem {
    pub fn new_elem(container: DataContainer) -> Result<Self> {
        let mut elem = RawMatrixElem::<DataFrame>::new_elem(container)?;
        elem.enable_cache();
        Ok(Slot::new(elem))
    }

    pub fn enable_cache(&self) {
        self.inner().enable_cache();
    }

    pub fn disable_cache(&self) {
        self.inner().disable_cache();
    }

    pub fn read(&self) -> Result<DataFrame> {
        self.inner().read()
    }

    pub fn write(&self, location: &Group, name: &str) -> Result<()> {
        self.inner().write(location, name)
    }

    pub fn update(&self, data: &DataFrame) {
        self.inner().update(data).unwrap()
    }

    pub fn column(&self, name: &str) -> Result<Series> {
        let elem = self.inner();
        match &elem.inner.element {
            Some(el) => Ok(el.column(name)?.clone()),
            None => {
                let grp = elem.inner.container.get_group_ref()?;
                Ok(ReadData::read(&DataContainer::open(grp, name)?)?)
                /*
                let mut r = read_str_vec_attr(grp, "column-order")?;
                r.insert(0, read_str_attr(grp, "_index")?);
                */
            }
        }
    }

    pub fn get_column_names(&self) -> Result<Vec<String>> {
        let elem = self.inner();
        match &elem.inner.element {
            Some(el) => Ok(el.get_column_names_owned()),
            None => {
                let grp = elem.inner.container.get_group_ref()?;
                let mut r = read_str_vec_attr(grp, "column-order")?;
                r.insert(0, read_str_attr(grp, "_index")?);
                Ok(r)
            }
        }
    }

    pub fn get_index(&self) -> Result<Vec<String>> {
        let elem = self.inner();
        match &elem.inner.element {
            Some(el) => Ok(el[0].utf8().unwrap().into_iter()
                .map(|s| s.unwrap().to_string()).collect()),
            None => {
                let grp = elem.inner.container.get_group_ref()?;
                let index = read_str_attr(grp, "_index")?;
                Ok(read_str_vec(&grp.dataset(index.as_str())?)?)
            },
        }
    }

    pub fn nrows(&self) -> usize {
        self.inner().nrows()
    }

    pub fn ncols(&self) -> usize {
        self.inner().ncols()
    }

    pub fn subset_rows(&self, idx: &[usize]) {
        self.inner().0.as_mut().map(|x| x.subset_rows(idx).unwrap());
    }

    pub fn subset_cols(&self, idx: &[usize]) {
        self.inner().0.as_mut().map(|x| x.subset_cols(idx).unwrap());
    }

    pub fn subset(&self, ridx: &[usize], cidx: &[usize]) {
        self.inner().0.as_mut().map(|x| x.subset(ridx, cidx).unwrap());
    }
}

/// This struct stores the accumulated lengths of objects in a vector
pub struct AccumLength(Vec<usize>);

impl AccumLength {
    /// convert index to adata index and inner element index
    pub fn normalize_index(&self, i: usize) -> (usize, usize) {
        match self.0.binary_search(&i) {
            Ok(i_) => (i_, 0),
            Err(i_) => (i_ - 1, i - self.0[i_ - 1]),
        }
    }

    /// Reorder indices such that consecutive indices come from the same underlying
    /// element.
    pub fn sort_index_by_buckets(&self, indices: &mut [usize]) {
        todo!()
    }

    pub fn normalize_indices(&self, indices: &[usize]) -> std::collections::HashMap<usize, Vec<usize>> {
        indices.iter().map(|x| self.normalize_index(*x))
            .sorted_by_key(|x| x.0).into_iter()
            .group_by(|x| x.0).into_iter().map(|(key, grp)|
                (key, grp.map(|x| x.1).collect())
            ).collect()
    }

    pub fn size(&self) -> usize { *self.0.last().unwrap_or(&0) }
}

impl FromIterator<usize> for AccumLength {
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
        AccumLength(accum)
    }
}

#[derive(Clone)]
pub struct Stacked<T> {
    nrows: Arc<Mutex<usize>>,
    ncols: Arc<Mutex<usize>>,
    pub elems: Arc<Vec<T>>,
    pub(crate) accum: Arc<Mutex<AccumLength>>,
}

impl std::fmt::Display for Stacked<MatrixElem> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} x {} stacked elements ({}) with {}",
            *self.nrows.lock(),
            *self.ncols.lock(),
            self.elems.len(),
            self.elems[0].dtype(),
        )
    }
}

impl Stacked<MatrixElem>
{
    pub(crate) fn new(
        elems: Vec<MatrixElem>,
        nrows: Arc<Mutex<usize>>,
        ncols: Arc<Mutex<usize>>,
        accum: Arc<Mutex<AccumLength>>,
    ) -> Result<Self> {
        if !elems.iter().map(|x| x.dtype()).all_equal() {
            return Err(anyhow!("dtype not equal"));
        }
        Ok(Self { nrows, ncols, elems: Arc::new(elems), accum })
    }

    pub fn nrows(&self) -> usize { *self.nrows.lock() }
    pub fn ncols(&self) -> usize { *self.ncols.lock() }

    pub fn read(&self) -> Result<Box<dyn DataPartialIO>> {
        let mats: Result<Vec<_>> = self.elems.par_iter().map(|x| x.read()).collect();
        Ok(rstack(mats?)?)
    }

    pub fn read_rows(&self, idx: &[usize]) -> Result<Box<dyn DataPartialIO>> {
        let accum = self.accum.lock();
        let (ori_idx, rows): (Vec<_>, Vec<_>) = idx.iter().map(|x| accum.normalize_index(*x))
            .enumerate().sorted_by_key(|x| x.1.0).into_iter()
            .group_by(|x| x.1.0).into_iter().map(|(key, grp)| {
                let (ori_idx, (_, inner_idx)): (Vec<_>, (Vec<_>, Vec<_>)) = grp.unzip();
                (ori_idx, self.elems[key].inner().read_rows(inner_idx.as_slice()))
            }).unzip();
        Ok(rstack_with_index(
            ori_idx.into_iter().flatten().collect::<Vec<_>>().as_slice(),
            rows.into_iter().collect::<Result<_>>()?
        )?)
    }

    pub fn chunked(&self, chunk_size: usize) -> StackedChunkedMatrix {
        StackedChunkedMatrix {
            matrices: self.elems.iter().map(|x| x.chunked(chunk_size)).collect(),
            current_matrix_index: 0,
            n_mat: self.elems.len(),
        }
    }

    pub fn enable_cache(&self) {
        self.elems.iter().for_each(|x| x.enable_cache())
    }

    pub fn disable_cache(&self) {
        self.elems.iter().for_each(|x| x.disable_cache())
    }
}

impl ElemTrait for MatrixElem {
    type Data = Box<dyn DataPartialIO>;

    fn dtype(&self) -> DataType {
        self.inner().inner.dtype.clone()
    }

    fn is_empty(&self) -> bool { todo!() }

    fn new_elem(container: DataContainer) -> Result<Self> {
        let elem = RawMatrixElem::new(container)?;
        Ok(Slot::new(elem))
    }

    fn enable_cache(&self) { self.inner().enable_cache(); }

    fn disable_cache(&self) { self.inner().disable_cache(); }

    fn read(&self) -> Result<Self::Data> {
        self.inner().read()
    }

    fn write(&self, location: &Group, name: &str) -> Result<()> {
        self.inner().write(location, name)
    }

    fn update(&self, data: &Self::Data) {
        self.inner().update(data).unwrap();
    }
}


#[derive(Clone)]
pub struct StackedDataFrame {
    pub keys: HashSet<String>,
    pub elems: Arc<Vec<DataFrameElem>>,
}

impl std::fmt::Display for StackedDataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let keys: String = self.keys.iter().map(|x| x.as_str()).intersperse(", ").collect();
        write!(f, "stacked dataframe with columns: {}", keys)
    }
}

impl StackedDataFrame {
    pub fn new(elems: Vec<DataFrameElem>) -> Result<Self> {
        let keys: Result<HashSet<String>> = elems.iter()
            .map(|x| Ok(x.get_column_names()?.into_iter().collect::<HashSet<String>>()))
            .reduce(|accum, item| match (accum, item) {
                (Ok(mut x), Ok(y)) => {
                    x.retain(|e| y.contains(e));
                    Ok(x)
                },
                (Err(e), _) => Err(e),
                (_, Err(e)) => Err(e),
            }).unwrap_or(Ok(HashSet::new()));
        Ok(Self { keys: keys?, elems: Arc::new(elems) })
    }

    pub fn column(&self, name: &str) -> Result<Series> {
        if self.keys.contains(name) {
            let mut series = self.elems.iter().map(|x| x.column(name))
                .collect::<Result<Vec<_>>>()?;
            Ok(series.iter_mut().reduce(|accum, item| {
                accum.append(item).unwrap();
                accum
            }).unwrap().clone())
        } else {
            return Err(anyhow!("key is not present"));
        }
    }
}
