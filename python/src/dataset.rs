use crate::PyAnnData;
use crate::iterator::PyStackedChunkedMatrix;

use anndata_rs::base;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
#[repr(transparent)]
pub struct AnnDataSet(pub base::AnnDataSet);

#[pymethods]
impl AnnDataSet {
    #[new]
    fn new(adatas: HashMap<String, PyAnnData>) -> Self {
        AnnDataSet(base::AnnDataSet::new(adatas.into_iter().map(|(k, v)| (k, v.0)).collect()).unwrap())
    }

    fn chunked_X(&self, chunk_size: usize) -> PyStackedChunkedMatrix {
        PyStackedChunkedMatrix(self.0.chunked_x(chunk_size))
    }

    #[getter]
    fn n_obs(&self) -> usize { self.0.n_obs() }

    #[getter]
    fn n_vars(&self) -> usize { self.0.n_vars() }
}