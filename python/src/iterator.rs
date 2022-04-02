use crate::to_py_data2;
use anndata_rs::element::{MatrixElem, MatrixElemOptional};

use pyo3::prelude::*;

pub enum MatrixElemLike {
    M1(MatrixElem),
    M2(MatrixElemOptional),
}

#[pyclass]
pub struct ChunkedMatrix {
    pub elem: MatrixElemLike,
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
            let j = std::cmp::min(slf.size, slf.current_index + slf.chunk_size);
            slf.current_index = j;
            let data = match &slf.elem {
                MatrixElemLike::M1(m) => m.0.lock().unwrap().read_dyn_row_slice(i..j).unwrap(),
                MatrixElemLike::M2(m) => m.0.lock().unwrap().as_ref().unwrap()
                    .read_dyn_row_slice(i..j).unwrap(),
            };
            Python::with_gil(|py| Some(to_py_data2(py, data).unwrap()))
        }
    }
}