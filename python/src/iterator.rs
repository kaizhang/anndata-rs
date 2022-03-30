use pyo3::prelude::*;

use crate::to_py_data2;

use hdf5::H5Type;
use anndata_rs::{
    base::AnnData,
    element::{Elem, RawMatrixElem, MatrixElem, DataFrameElem},
    anndata_trait::{DataType, WritePartialData},
    iterator::{CsrRowIterator, IntoRowIterator},
};

#[pyclass]
pub struct ChunkedMatrix {
    pub elem: MatrixElem,
    pub chunk_size: usize,
    pub size: usize,
    pub current_index: usize,
}

#[pymethods]
impl ChunkedMatrix {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<Self>) -> Option<PyObject> {
        if slf.current_index >= slf.size {
            None
        } else {
            let i = slf.current_index;
            let j = slf.current_index + slf.chunk_size;
            slf.current_index = j;
            Python::with_gil(|py|
                Some(to_py_data2(py, slf.elem.0.read_dyn_row_slice(i .. j)).unwrap())
            )
        }
    }
}

/*
pub fn xxx<'py, 'a>(
    py: Python<'py>,
    elem: &'a RawMatrixElem<CsrMatrix<f64>>
) -> RowIterator {
    RowIterator { iter: Box::new(iter_row(elem).map(|x| x.into_py(py))) }
}
*/