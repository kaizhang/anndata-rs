use crate::utils::conversion::to_py_df;

use pyo3::{
    prelude::*,
    types::PyModule, PyResult, Python,
};
use numpy::IntoPyArray;
use nalgebra_sparse::csr::CsrMatrix;
use std::collections::HashMap;
use hdf5::types::TypeDescriptor::*;
use hdf5::types::IntSize;
use hdf5::types::FloatSize;
use ndarray::ArrayD;
use polars::frame::DataFrame;
use anndata_rs::{
    proc_numeric_data,
    anndata_trait::{Mapping, DataType, Scalar, DataIO, DataPartialIO},
};

macro_rules! to_py_scalar_macro {
    ($py:expr, $data:expr, $dtype:expr) => {
        match $dtype {
            Unsigned(IntSize::U1) => Ok(PyModule::import($py, "numpy")?.call_method1(
                "uint8", ($data.into_any().downcast::<Scalar<u8>>().unwrap().0.to_object($py),)
                )?.to_object($py)),
            Unsigned(IntSize::U2) => Ok(PyModule::import($py, "numpy")?.call_method1(
                "uint16", ($data.into_any().downcast::<Scalar<u16>>().unwrap().0.to_object($py),)
                )?.to_object($py)),
            Unsigned(IntSize::U4) => Ok(PyModule::import($py, "numpy")?.call_method1(
                "uint32", ($data.into_any().downcast::<Scalar<u32>>().unwrap().0.to_object($py),)
                )?.to_object($py)),
            Unsigned(IntSize::U8) => Ok(PyModule::import($py, "numpy")?.call_method1(
                "uint64", ($data.into_any().downcast::<Scalar<u64>>().unwrap().0.to_object($py),)
                )?.to_object($py)),
            Integer(IntSize::U1) => Ok(PyModule::import($py, "numpy")?.call_method1(
                "int8", ($data.into_any().downcast::<Scalar<i8>>().unwrap().0.to_object($py),)
                )?.to_object($py)),
            Integer(IntSize::U2) => Ok(PyModule::import($py, "numpy")?.call_method1(
                "int16", ($data.into_any().downcast::<Scalar<i16>>().unwrap().0.to_object($py),)
                )?.to_object($py)),
            Integer(IntSize::U4) => Ok(PyModule::import($py, "numpy")?.call_method1(
                "int32", ($data.into_any().downcast::<Scalar<i32>>().unwrap().0.to_object($py),)
                )?.to_object($py)),
            Integer(IntSize::U8) => Ok(PyModule::import($py, "numpy")?.call_method1(
                "int64", ($data.into_any().downcast::<Scalar<i64>>().unwrap().0.to_object($py),)
                )?.to_object($py)),
            Float(FloatSize::U4) => Ok(PyModule::import($py, "numpy")?.call_method1(
                "float32", ($data.into_any().downcast::<Scalar<f32>>().unwrap().0.to_object($py),)
                )?.to_object($py)),
            Float(FloatSize::U8) => Ok(PyModule::import($py, "numpy")?.call_method1(
                "float64", ($data.into_any().downcast::<Scalar<f64>>().unwrap().0.to_object($py),)
                )?.to_object($py)),
            Boolean => Ok($data.into_any().downcast::<Scalar<bool>>().unwrap().0.to_object($py)),
            ty => panic!("converting scalar type \"{}\" is not supported", ty)
        }
    }
}

pub fn to_py_data1<'py>(
    py: Python<'py>,
    data: Box<dyn DataIO>,
) -> PyResult<PyObject>
{
    macro_rules! _csr { ($m:expr) => { csr_to_scipy(py, $m) }; }
    macro_rules! _arr { ($m:expr) => { Ok($m.into_pyarray(py).to_object(py)) }; }
    match data.as_ref().get_dtype() {
        DataType::CsrMatrix(dtype) => proc_numeric_data!(
            dtype, *data.into_any().downcast().unwrap(), _csr, CsrMatrix
        ),
        DataType::Array(dtype) => proc_numeric_data!(
            dtype, *data.into_any().downcast().unwrap(), _arr, ArrayD
        ),
        DataType::DataFrame => to_py_df(*data.into_any().downcast::<DataFrame>().unwrap()),
        DataType::String => Ok(data.into_any().downcast::<String>().unwrap().to_object(py)),
        DataType::Scalar(dtype) => to_py_scalar_macro!(py, data, dtype),
        DataType::Mapping => Ok(
            (*data.into_any().downcast::<Mapping>().unwrap()).0
                .into_iter().map(|(k, v)| Ok((k, to_py_data1(py, v)?)))
                .collect::<PyResult<HashMap<_, _>>>()?.to_object(py)
        ),
        ty => panic!("Cannot convert Rust element \"{}\" to Python object", ty)
    }
}

pub fn to_py_data2<'py>(
    py: Python<'py>,
    data: Box<dyn DataPartialIO>,
) -> PyResult<PyObject>
{
    macro_rules! _csr { ($m:expr) => { csr_to_scipy(py, $m) }; }
    macro_rules! _arr { ($m:expr) => { Ok($m.into_pyarray(py).to_object(py)) }; }
    match data.as_ref().get_dtype() {
        DataType::CsrMatrix(dtype) => proc_numeric_data!(
            dtype, *data.into_any().downcast().unwrap(), _csr, CsrMatrix
        ),
        DataType::Array(dtype) => proc_numeric_data!(
            dtype, *data.into_any().downcast().unwrap(), _arr, ArrayD
        ),
        DataType::DataFrame => to_py_df(*data.into_any().downcast::<DataFrame>().unwrap()),
        ty => panic!("Cannot convert Rust element \"{}\" to Python object", ty)
    }
}

fn csr_to_scipy<'py, T>(
    py: Python<'py>,
    mat: CsrMatrix<T>
) -> PyResult<PyObject>
where T: numpy::Element
{
    let n = mat.nrows();
    let m = mat.ncols();
    let (intptr, indices, data) = mat.disassemble();

    let scipy = PyModule::import(py, "scipy.sparse")?;
    Ok(scipy.getattr("csr_matrix")?.call1((
        (data.into_pyarray(py), indices.into_pyarray(py), intptr.into_pyarray(py)),
        (n, m),
    ))?.to_object(py))
}

