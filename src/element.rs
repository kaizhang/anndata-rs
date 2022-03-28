mod raw;

use raw::RawMatrixElem;
use crate::anndata_trait::*;

use hdf5::{Result, Group}; 
use std::sync::Arc;

#[derive(Clone)]
pub struct MatrixElem(pub Arc<RawMatrixElem<dyn DataPartialIO>>);

impl MatrixElem {
    pub fn new(container: DataContainer) -> Result<Self> {
        let elem = RawMatrixElem::new(container)?;
        Ok(Self(Arc::new(elem)))
    }

    pub fn write(&self, location: &Group, name: &str) -> Result<()> {
        self.0.write_elem(location, name)
    }

    pub fn nrows(&self) -> usize { self.0.nrows }

    pub fn ncols(&self) -> usize { self.0.ncols }

    pub fn subset_rows(&self, idx: &[usize]) -> Self {
        Self(Arc::new(self.0.subset_rows(idx)))
    }

    pub fn subset_cols(&self, idx: &[usize]) -> Self {
        Self(Arc::new(self.0.subset_cols(idx)))
    }

    pub fn subset(&self, ridx: &[usize], cidx: &[usize]) -> Self {
        Self(Arc::new(self.0.subset(ridx, cidx)))
    }
}