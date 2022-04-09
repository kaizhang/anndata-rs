use crate::utils::conversion::to_py_data2;
use anndata_rs::iterator::{ChunkedMatrix, StackedChunkedMatrix};

use pyo3::prelude::*;

#[pyclass]
pub struct PyChunkedMatrix(pub ChunkedMatrix);

#[pymethods]
impl PyChunkedMatrix {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> { slf }

    fn __next__<'py>(mut slf: PyRefMut<Self>, py: Python<'py>) -> Option<PyObject> {
        slf.0.next().map(|data| to_py_data2(py, data).unwrap())
    }
}

#[pyclass]
pub struct PyStackedChunkedMatrix(pub StackedChunkedMatrix);

#[pymethods]
impl PyStackedChunkedMatrix {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> { slf }

    fn __next__<'py>(mut slf: PyRefMut<Self>, py: Python<'py>) -> Option<PyObject> {
        slf.0.next().map(|data| to_py_data2(py, data).unwrap())
    }
}