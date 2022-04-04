use crate::PyAnnData;
use crate::iterator::PyStackedChunkedMatrix;

use anndata_rs::{
    base::AnnDataSet,
};
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
#[repr(transparent)]
pub struct PyAnnDataSet(pub AnnDataSet);

#[pymethods]
impl PyAnnDataSet {
    #[new]
    fn new(adatas: HashMap<String, PyAnnData>) -> Self {
        PyAnnDataSet(AnnDataSet::new(adatas.into_iter().map(|(k, v)| (k, v.0)).collect()).unwrap())
    }

    fn chunked_X(&self, chunk_size: usize) -> PyStackedChunkedMatrix {
        PyStackedChunkedMatrix(self.0.chunked_x(chunk_size))
    }

    #[getter]
    fn n_obs(&self) -> usize { self.0.n_obs() }

    #[getter]
    fn n_vars(&self) -> usize { self.0.n_vars() }
}