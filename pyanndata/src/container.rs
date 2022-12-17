mod traits;

use crate::data::{PyData, PyArrayData};

use pyo3::prelude::*;
use anndata::{Elem, Backend};
use traits::{ElemTrait, ArrayElemTrait};
use anyhow::Result;

#[pyclass]
#[repr(transparent)]
pub struct PyElem(Box<dyn ElemTrait>);

impl<T: ElemTrait + 'static> From<T> for PyElem {
    fn from(elem: T) -> Self {
        Self(Box::new(elem))
    }
}

#[pymethods]
impl PyElem {
    fn enable_cache(&self) {
        self.0.enable_cache();
    }

    fn disable_cache(&self) {
        self.0.disable_cache();
    }

    fn is_scalar(&self) -> bool {
        self.0.is_scalar()
    }

    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> Result<PyData> {
        self.0.data(py, subscript)
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.0.show()
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyArrayElem(Box<dyn ArrayElemTrait>);

impl<T: ArrayElemTrait + 'static> From<T> for PyArrayElem {
    fn from(elem: T) -> Self {
        Self(Box::new(elem))
    }
}

#[pymethods]
impl PyArrayElem {
    fn enable_cache(&self) {
        self.0.enable_cache();
    }

    fn disable_cache(&self) {
        self.0.disable_cache();
    }

    fn is_scalar(&self) -> bool {
        self.0.is_scalar()
    }

    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> Result<PyArrayData> {
        self.0.data(py, subscript)
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.0.show()
    }
}

