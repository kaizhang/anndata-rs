use crate::{
    data::*,
    iterator::{ChunkedMatrix, StackedChunkedMatrix},
    element::{RawMatrixElem, RawElem, Slot, DataFrameElem},
};

use polars::{frame::DataFrame, series::Series};
use hdf5::Group; 
use anyhow::{bail, Result};
use std::sync::Arc;
use parking_lot::Mutex;
use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use indexmap::set::IndexSet;

pub type Elem = Slot<RawElem<dyn DataIO>>;

impl Elem {
    pub fn dtype(&self) -> Option<DataType> {
        if self.is_empty() { None } else { Some(self.inner().dtype.clone()) }
    }

    pub fn new_elem(container: DataContainer) -> Result<Self> {
        let elem = RawElem::new(container)?;
        Ok(Self(Arc::new(Mutex::new(Some(elem)))))
    }

    pub fn read(&self) -> Result<Box<dyn DataIO>> { self.inner().read() }

    pub fn write(&self, location: &Group, name: &str) -> Result<()> { self.inner().write(location, name) }

    pub fn update<D: DataIO>(&self, data: &D) { self.inner().update(data).unwrap(); }

    pub fn enable_cache(&self) { self.inner().enable_cache(); }
    pub fn disable_cache(&self) { self.inner().disable_cache(); }
}

pub type MatrixElem = Slot<RawMatrixElem<dyn DataPartialIO>>;

impl MatrixElem {
    pub fn dtype(&self) -> Option<DataType> {
        if self.is_empty() { None } else { Some(self.inner().inner.dtype.clone()) }
    }

    pub fn new_elem(container: DataContainer) -> Result<Self> {
        let elem = RawMatrixElem::new(container)?;
        Ok(Slot::new(elem))
    }

    pub fn enable_cache(&self) { self.inner().enable_cache(); }
    pub fn disable_cache(&self) { self.inner().disable_cache(); }

    pub fn nrows(&self) -> usize { self.inner().nrows() }
    pub fn ncols(&self) -> usize { self.inner().ncols() }

    pub fn read(&self) -> Result<Box<dyn DataPartialIO>> { self.inner().read() }

    pub fn write(&self, location: &Group, name: &str) -> Result<()> { self.inner().write(location, name) }

    pub fn update<D: DataPartialIO>(&self, data: &D) { self.inner().update(data).unwrap(); }

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
    pub fn sort_index_to_buckets(&self, indices: &[usize]) -> Vec<usize> {
        indices.into_iter().map(|x| *x)
            .sorted_by_cached_key(|x| self.normalize_index(*x).0).collect()
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

impl Stacked<MatrixElem>
{
    pub(crate) fn new(
        elems: Vec<MatrixElem>,
        nrows: Arc<Mutex<usize>>,
        ncols: Arc<Mutex<usize>>,
        accum: Arc<Mutex<AccumLength>>,
    ) -> Result<Self> {
        if !elems.iter().map(|x| x.dtype()).all_equal() {
            bail!("dtype not equal")
        } else {
            Ok(Self { nrows, ncols, elems: Arc::new(elems), accum })
        }
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

    pub fn read_columns(&self, idx: &[usize]) -> Result<Box<dyn DataPartialIO>> {
        let mats: Result<Vec<_>> = self.elems.par_iter().map(|x| x.inner().read_columns(idx)).collect();
        Ok(rstack(mats?)?)
    }

    pub fn read_partial(&self, ridx: &[usize], cidx: &[usize]) -> Result<Box<dyn DataPartialIO>> {
        let accum = self.accum.lock();
        let (ori_idx, rows): (Vec<_>, Vec<_>) = ridx.iter().map(|x| accum.normalize_index(*x))
            .enumerate().sorted_by_key(|x| x.1.0).into_iter()
            .group_by(|x| x.1.0).into_iter().map(|(key, grp)| {
                let (ori_idx, (_, inner_idx)): (Vec<_>, (Vec<_>, Vec<_>)) = grp.unzip();
                (ori_idx, self.elems[key].inner().read_partial(inner_idx.as_slice(), cidx))
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


#[derive(Clone)]
pub struct StackedDataFrame {
    pub column_names: IndexSet<String>,
    pub elems: Arc<Vec<DataFrameElem>>,
}

impl std::fmt::Display for StackedDataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let column_names: String = self.column_names.iter().map(|x| x.as_str()).intersperse(", ").collect();
        write!(f, "stacked dataframe with columns: {}", column_names)
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