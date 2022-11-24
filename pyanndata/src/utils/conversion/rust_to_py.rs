use crate::utils::conversion::dataframe::{to_py_df, to_py_series};

use anndata_rs::{
    data::{Data, DataType, Mapping, MatrixData, Scalar},
    proc_numeric_data,
};
use hdf5::types::FloatSize;
use hdf5::types::IntSize;
use hdf5::types::TypeDescriptor::*;
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::ArrayD;
use numpy::IntoPyArray;
use polars::{frame::DataFrame, series::Series};
use pyo3::{prelude::*, types::PyModule, PyResult, Python};
use std::collections::HashMap;

macro_rules! to_py_scalar_macro {
    ($py:expr, $data:expr, $dtype:expr) => {
        match $dtype {
            Unsigned(IntSize::U1) => Ok(PyModule::import($py, "numpy")?
                .call_method1(
                    "uint8",
                    ($data.downcast_ref::<Scalar<u8>>().unwrap().0.to_object($py),),
                )?
                .to_object($py)),
            Unsigned(IntSize::U2) => Ok(PyModule::import($py, "numpy")?
                .call_method1(
                    "uint16",
                    ($data
                        .downcast_ref::<Scalar<u16>>()
                        .unwrap()
                        .0
                        .to_object($py),),
                )?
                .to_object($py)),
            Unsigned(IntSize::U4) => Ok(PyModule::import($py, "numpy")?
                .call_method1(
                    "uint32",
                    ($data
                        .downcast_ref::<Scalar<u32>>()
                        .unwrap()
                        .0
                        .to_object($py),),
                )?
                .to_object($py)),
            Unsigned(IntSize::U8) => Ok(PyModule::import($py, "numpy")?
                .call_method1(
                    "uint64",
                    ($data
                        .downcast_ref::<Scalar<u64>>()
                        .unwrap()
                        .0
                        .to_object($py),),
                )?
                .to_object($py)),
            Integer(IntSize::U1) => Ok(PyModule::import($py, "numpy")?
                .call_method1(
                    "int8",
                    ($data.downcast_ref::<Scalar<i8>>().unwrap().0.to_object($py),),
                )?
                .to_object($py)),
            Integer(IntSize::U2) => Ok(PyModule::import($py, "numpy")?
                .call_method1(
                    "int16",
                    ($data
                        .downcast_ref::<Scalar<i16>>()
                        .unwrap()
                        .0
                        .to_object($py),),
                )?
                .to_object($py)),
            Integer(IntSize::U4) => Ok(PyModule::import($py, "numpy")?
                .call_method1(
                    "int32",
                    ($data
                        .downcast_ref::<Scalar<i32>>()
                        .unwrap()
                        .0
                        .to_object($py),),
                )?
                .to_object($py)),
            Integer(IntSize::U8) => Ok(PyModule::import($py, "numpy")?
                .call_method1(
                    "int64",
                    ($data
                        .downcast_ref::<Scalar<i64>>()
                        .unwrap()
                        .0
                        .to_object($py),),
                )?
                .to_object($py)),
            Float(FloatSize::U4) => Ok(PyModule::import($py, "numpy")?
                .call_method1(
                    "float32",
                    ($data
                        .downcast_ref::<Scalar<f32>>()
                        .unwrap()
                        .0
                        .to_object($py),),
                )?
                .to_object($py)),
            Float(FloatSize::U8) => Ok(PyModule::import($py, "numpy")?
                .call_method1(
                    "float64",
                    ($data
                        .downcast_ref::<Scalar<f64>>()
                        .unwrap()
                        .0
                        .to_object($py),),
                )?
                .to_object($py)),
            Boolean => Ok($data
                .downcast_ref::<Scalar<bool>>()
                .unwrap()
                .0
                .to_object($py)),
            ty => panic!("converting scalar type \"{}\" is not supported", ty),
        }
    };
}

pub trait RustToPy {
    fn rust_into_py<'py>(self, py: Python<'py>) -> PyResult<PyObject>;
}

impl<T: numpy::Element> RustToPy for CsrMatrix<T> {
    fn rust_into_py<'py>(self, py: Python<'py>) -> PyResult<PyObject> {
        let n = self.nrows();
        let m = self.ncols();
        let (intptr, indices, data) = self.disassemble();
        let scipy = PyModule::import(py, "scipy.sparse")?;
        Ok(scipy
            .getattr("csr_matrix")?
            .call1((
                (
                    data.into_pyarray(py),
                    indices.into_pyarray(py),
                    intptr.into_pyarray(py),
                ),
                (n, m),
            ))?
            .to_object(py))
    }
}

impl<T: numpy::Element> RustToPy for ArrayD<T> {
    fn rust_into_py<'py>(self, py: Python<'py>) -> PyResult<PyObject> {
        Ok(self.into_pyarray(py).to_object(py))
    }
}

impl RustToPy for DataFrame {
    fn rust_into_py<'py>(self, py: Python<'py>) -> PyResult<PyObject> {
        to_py_df(py, self)
    }
}

impl RustToPy for Series {
    fn rust_into_py<'py>(self, py: Python<'py>) -> PyResult<PyObject> {
        to_py_series(py, &self)
    }
}

impl RustToPy for Box<dyn Data> {
    fn rust_into_py<'py>(self, py: Python<'py>) -> PyResult<PyObject> {
        macro_rules! _into {
            ($m:expr) => {
                $m.rust_into_py(py)
            };
        }
        match self.get_dtype() {
            DataType::CsrMatrix(dtype) => {
                proc_numeric_data!(dtype, *self.downcast().unwrap(), _into, CsrMatrix)
            }
            DataType::Array(dtype) => {
                proc_numeric_data!(dtype, *self.downcast().unwrap(), _into, ArrayD)
            }
            DataType::DataFrame => (*self.downcast::<DataFrame>().unwrap()).rust_into_py(py),
            DataType::String => Ok(self.downcast::<String>().unwrap().to_object(py)),
            DataType::Scalar(dtype) => to_py_scalar_macro!(py, self, dtype),
            DataType::Mapping => Ok((*self.downcast::<Mapping>().unwrap())
                .0
                .into_iter()
                .map(|(k, v)| Ok((k, v.rust_into_py(py)?)))
                .collect::<PyResult<HashMap<_, _>>>()?
                .to_object(py)),
            ty => panic!("Cannot convert Rust element \"{}\" to Python object", ty),
        }
    }
}

impl RustToPy for Box<dyn MatrixData> {
    fn rust_into_py<'py>(self, py: Python<'py>) -> PyResult<PyObject> {
        macro_rules! _into {
            ($m:expr) => {
                $m.rust_into_py(py)
            };
        }
        match self.get_dtype() {
            DataType::CsrMatrix(dtype) => {
                proc_numeric_data!(dtype, *self.downcast().unwrap(), _into, CsrMatrix)
            }
            DataType::Array(dtype) => {
                proc_numeric_data!(dtype, *self.downcast().unwrap(), _into, ArrayD)
            }
            DataType::DataFrame => (*self.downcast::<DataFrame>().unwrap()).rust_into_py(py),
            ty => panic!("Cannot convert Rust element \"{}\" to Python object", ty),
        }
    }
}
