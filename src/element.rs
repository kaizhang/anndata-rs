mod raw;

pub use raw::{RawMatrixElem, RawElem};
use crate::anndata_trait::*;

use polars::frame::DataFrame;
use hdf5::{Result, Group}; 
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Elem(pub Arc<Mutex<RawElem<dyn DataIO>>>);

impl Elem {

    pub fn new(container: DataContainer) -> Result<Self> {
        let elem = RawElem::new(container)?;
        Ok(Self(Arc::new(Mutex::new(elem))))
    }

    pub fn read(&self) -> Result<Box<dyn DataIO>> {
        self.0.lock().unwrap().read_dyn_elem()
    }

    pub fn write(&self, location: &Group, name: &str) -> Result<()> {
        self.0.lock().unwrap().write_elem(location, name)
    }

    pub fn enable_cache(&self) { self.0.lock().unwrap().enable_cache(); }

    pub fn disable_cache(&self) { self.0.lock().unwrap().disable_cache(); }
}

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
    pub fn new(container: DataContainer) -> Result<Self> {
        let elem = RawMatrixElem::new(container)?;
        Ok(Self(Arc::new(Mutex::new(elem))))
    }

    pub fn enable_cache(&self) { self.0.lock().unwrap().enable_cache(); }

    pub fn disable_cache(&self) { self.0.lock().unwrap().disable_cache(); }

    pub fn read(&self) -> Result<Box<dyn DataPartialIO>> {
        self.0.lock().unwrap().read_dyn_elem()
    }

    pub fn write(&self, location: &Group, name: &str) -> Result<()> {
        self.0.lock().unwrap().write_elem(location, name)
    }

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

    pub fn enable_cache(&self) { self.0.lock().unwrap().enable_cache(); }

    pub fn disable_cache(&self) { self.0.lock().unwrap().disable_cache(); }

    pub fn read(&self) -> Result<DataFrame> {
        self.0.lock().unwrap().read_elem()
    }

    pub fn write(&self, location: &Group, name: &str) -> Result<()> {
        self.0.lock().unwrap().write_elem(location, name)
    }

    pub fn update(&self, data: &DataFrame) {
        self.0.lock().unwrap().update(data).unwrap();
    }

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