use crate::{
    anndata_trait::*,
    iterator::{ChunkedMatrix, StackedChunkedMatrix},
    element::{RawMatrixElem, RawElem},
    utils::hdf5::{read_str_vec_attr, read_str_attr},
    utils::macros::{proc_csr_data, proc_arr_data},
};

use polars::frame::DataFrame;
use hdf5::{Result, Group}; 
use std::sync::{Arc, Mutex};
use itertools::Itertools;
use ndarray::{ArrayD, Axis};
use nalgebra_sparse::CsrMatrix;

#[derive(Clone)]
pub struct Elem(pub Arc<Mutex<RawElem<dyn DataIO>>>);

impl std::fmt::Display for Elem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let elem = self.0.lock().unwrap();
        write!(f, "Elem with {}, cache_enabled: {}, cached: {}",
            elem.dtype,
            if elem.cache_enabled { "yes" } else { "no" },
            if elem.element.is_some() { "yes" } else { "no" },
        )
    }
}

#[derive(Clone)]
pub struct MatrixElem(pub Arc<Mutex<RawMatrixElem<dyn DataPartialIO>>>);

impl MatrixElem {
    pub fn nrows(&self) -> usize { self.0.lock().unwrap().nrows }

    pub fn ncols(&self) -> usize { self.0.lock().unwrap().ncols }

    pub fn subset_rows(&self, idx: &[usize]) {
        self.0.lock().unwrap().subset_rows(idx).unwrap();
    }

    pub fn subset_cols(&self, idx: &[usize]) {
        self.0.lock().unwrap().subset_cols(idx).unwrap();
    }

    pub fn subset(&self, ridx: &[usize], cidx: &[usize]) {
        self.0.lock().unwrap().subset(ridx, cidx).unwrap();
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

impl std::fmt::Display for MatrixElem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let elem = self.0.lock().unwrap();
        write!(f, "{} x {} MatrixElem with {}, cache_enabled: {}, cached: {}",
            elem.nrows,
            elem.ncols,
            elem.inner.dtype,
            if elem.inner.cache_enabled { "yes" } else { "no" },
            if elem.inner.element.is_some() { "yes" } else { "no" },
        )
    }
}

#[derive(Clone)]
pub struct DataFrameElem(pub Arc<Mutex<RawMatrixElem<DataFrame>>>);

impl DataFrameElem {
    pub fn new(container: DataContainer) -> Result<Self> {
        let mut elem = RawMatrixElem::<DataFrame>::new_elem(container)?;
        elem.enable_cache();
        Ok(Self(Arc::new(Mutex::new(elem))))
    }

    pub fn enable_cache(&self) {
        self.0.lock().unwrap().enable_cache();
    }

    pub fn disable_cache(&self) {
        self.0.lock().unwrap().disable_cache();
    }

    pub fn read(&self) -> Result<DataFrame> {
        self.0.lock().unwrap().read_elem()
    }

    pub fn write(&self, location: &Group, name: &str) -> Result<()> {
        self.0.lock().unwrap().write_elem(location, name)
    }

    pub fn update(&self, data: &DataFrame) {
        self.0.lock().unwrap().update(data).unwrap()
    }

    pub fn get_column_names(&self) -> Result<Vec<String>> {
        let elem = self.0.lock().unwrap();
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

    pub fn nrows(&self) -> usize {
        self.0.lock().unwrap().nrows
    }

    pub fn ncols(&self) -> usize {
        self.0.lock().unwrap().ncols
    }

    pub fn subset_rows(&self, idx: &[usize]) {
        self.0.lock().unwrap().subset_rows(idx).unwrap();
    }

    pub fn subset_cols(&self, idx: &[usize]) {
        self.0.lock().unwrap().subset_cols(idx).unwrap();
    }

    pub fn subset(&self, ridx: &[usize], cidx: &[usize]) {
        self.0.lock().unwrap().subset(ridx, cidx).unwrap();
    }
}

impl std::fmt::Display for DataFrameElem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let elem = self.0.lock().unwrap();
        write!(f, "DataFrameElem, cache_enabled: {}, cached: {}",
            if elem.inner.cache_enabled { "yes" } else { "no" },
            if elem.inner.element.is_some() { "yes" } else { "no" },
        )
    }
}

#[derive(Clone)]
pub struct Stacked<T> {
    pub nrows: usize,
    pub ncols: usize,
    pub elems: Arc<Vec<T>>,
    accum: Vec<usize>,
}

impl std::fmt::Display for Stacked<MatrixElem> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} x {} stacked elements ({}) with {}",
            self.nrows,
            self.ncols,
            self.elems.len(),
            self.elems[0].dtype(),
        )
    }
}

impl Stacked<MatrixElem>
{
    /// convert index to adata index and inner element index
    fn normalize_index(&self, i: usize) -> (usize, usize) {
        match self.accum.binary_search(&i) {
            Ok(i_) => (i_, 0),
            Err(i_) => (i_ - 1, i - self.accum[i_ - 1]),
        }
    }

    pub fn new(elems: Vec<MatrixElem>) -> Result<Self> {
        if !elems.iter().map(|x| x.dtype()).all_equal() {
            panic!("dtype not equal");
        }
        if !elems.iter().map(|x| x.ncols()).all_equal() {
            panic!("num cols mismatch");
        }
        let ncols = elems.iter().next().map(|x| x.ncols()).unwrap_or(0);
        let accum: Vec<usize> = std::iter::once(0).chain(elems.iter().scan(0, |state, x| {
            *state = *state + x.nrows();
            Some(*state)
        })).collect();
        let nrows = *accum.last().unwrap();
        Ok(Self { nrows, ncols, elems: Arc::new(elems), accum })
    }

    pub fn read_rows(&self, idx: &[usize]) -> Result<Box<dyn DataPartialIO>> {
        let (ori_idx, rows): (Vec<_>, Vec<_>) = idx.iter().map(|x| self.normalize_index(*x))
            .enumerate().sorted_by_key(|x| x.1.0).into_iter()
            .group_by(|x| x.1.0).into_iter().map(|(key, grp)| {
                let (ori_idx, (_, inner_idx)): (Vec<_>, (Vec<_>, Vec<_>)) = grp.unzip();
                (ori_idx, self.elems[key].0.lock().unwrap().read_rows(inner_idx.as_slice()))
            }).unzip();
        concat_matrices(
            ori_idx.into_iter().flatten().collect(),
            rows.into_iter().collect::<Result<_>>()?
        )
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

fn concat_matrices(index: Vec<usize>, mats: Vec<Box<dyn DataPartialIO>>) -> Result<Box<dyn DataPartialIO>> {
    if !mats.iter().map(|x| x.get_dtype()).all_equal() {
        panic!("type mismatch");
    }
    match mats[0].get_dtype() {
        DataType::Array(ty) => {
            proc_arr_data!(ty, concat_array(
                index.as_slice(),
                mats.into_iter().map(|x| x.into_any().downcast().unwrap()).collect(),
            ))
        },
        DataType::CsrMatrix(ty) => {
            proc_csr_data!(ty, concat_csr(
                index.as_slice(),
                mats.into_iter().map(|x| x.into_any().downcast().unwrap()).collect(),
            ))
        },
        x => panic!("{}", x),
    }
}

fn concat_array<T: Clone>(index: &[usize], mats: Vec<Box<ArrayD<T>>>) -> ArrayD<T> {
    let merged = mats.into_iter().reduce(|mut accum, other| {
        accum.as_mut().append(Axis(0), other.view()).unwrap();
        accum
    }).unwrap();
    let new_idx: Vec<_> = index.iter().enumerate().sorted_by_key(|x| *x.1)
        .map(|x| x.0).collect();
    merged.select(Axis(0), new_idx.as_slice())
}

fn concat_csr<T: Clone>(index: &[usize], mats: Vec<Box<CsrMatrix<T>>>) -> CsrMatrix<T> {
    if !mats.iter().map(|x| x.ncols()).all_equal() {
        panic!("num cols mismatch");
    }
    let num_rows = mats.iter().map(|x| x.nrows()).sum();
    let num_cols = mats.iter().next().map(|x| x.ncols()).unwrap_or(0);
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_offsets = Vec::new();
    let nnz = mats.iter().map(|x| x.row_iter()).flatten()
        .zip(index).sorted_by_key(|x| *x.1).fold(0, |acc, x| {
            row_offsets.push(acc);
            values.extend_from_slice(x.0.values());
            col_indices.extend_from_slice(x.0.col_indices());
            acc + x.0.nnz()
        });
    row_offsets.push(nnz);
    CsrMatrix::try_from_csr_data(num_rows, num_cols, row_offsets, col_indices, values).unwrap()
}

pub trait ElemTrait {
    type Data;

    fn dtype(&self) -> DataType;

    fn new(container: DataContainer) -> Result<Self> where Self: Sized;

    fn read(&self) -> Result<Self::Data>;

    fn write(&self, location: &Group, name: &str) -> Result<()>;

    fn update(&self, data: &Self::Data);

    fn enable_cache(&self);

    fn disable_cache(&self);
}

impl ElemTrait for Elem {
    type Data = Box<dyn DataIO>;

    fn dtype(&self) -> DataType {
        self.0.lock().unwrap().dtype.clone()
    }

    fn new(container: DataContainer) -> Result<Self> {
        let elem = RawElem::new(container)?;
        Ok(Self(Arc::new(Mutex::new(elem))))
    }

    fn read(&self) -> Result<Self::Data> {
        self.0.lock().unwrap().read_dyn_elem()
    }

    fn write(&self, location: &Group, name: &str) -> Result<()> {
        self.0.lock().unwrap().write_elem(location, name)
    }

    fn update(&self, data: &Self::Data) {
        self.0.lock().unwrap().update(data).unwrap();
    }

    fn enable_cache(&self) { self.0.lock().unwrap().enable_cache(); }

    fn disable_cache(&self) { self.0.lock().unwrap().disable_cache(); }
}

impl ElemTrait for MatrixElem {
    type Data = Box<dyn DataPartialIO>;

    fn dtype(&self) -> DataType {
        self.0.lock().unwrap().inner.dtype.clone()
    }

    fn new(container: DataContainer) -> Result<Self> {
        let elem = RawMatrixElem::new(container)?;
        Ok(Self(Arc::new(Mutex::new(elem))))
    }

    fn enable_cache(&self) { self.0.lock().unwrap().enable_cache(); }

    fn disable_cache(&self) { self.0.lock().unwrap().disable_cache(); }

    fn read(&self) -> Result<Self::Data> {
        self.0.lock().unwrap().read_dyn_elem()
    }

    fn write(&self, location: &Group, name: &str) -> Result<()> {
        self.0.lock().unwrap().write_elem(location, name)
    }

    fn update(&self, data: &Self::Data) {
        self.0.lock().unwrap().update(data).unwrap();
    }
}