use crate::to_py_df;

use pyo3::{
    prelude::*,
    types::PyModule, PyResult, Python,
};
use numpy::IntoPyArray;
use nalgebra_sparse::csr::CsrMatrix;
use hdf5::types::TypeDescriptor::*;
use hdf5::types::IntSize;
use hdf5::types::FloatSize;
use ndarray::ArrayD;
use polars::frame::DataFrame;

use anndata_rs::{
    anndata_trait::DataType,
    anndata_trait::{DataIO, DataPartialIO},
};

macro_rules! to_py_data_macro {
    ($py:expr, $data:expr) => {
        match $data.as_ref().get_dtype() {
            DataType::CsrMatrix(Unsigned(IntSize::U4)) =>
                csr_to_scipy::<u32>($py, *$data.into_any().downcast().unwrap()),
            DataType::CsrMatrix(Unsigned(IntSize::U8)) =>
                csr_to_scipy::<u64>($py, *$data.into_any().downcast().unwrap()),
            DataType::CsrMatrix(Integer(IntSize::U4)) =>
                csr_to_scipy::<i32>($py, *$data.into_any().downcast().unwrap()),
            DataType::CsrMatrix(Integer(IntSize::U8)) =>
                csr_to_scipy::<i64>($py, *$data.into_any().downcast().unwrap()),
            DataType::CsrMatrix(Float(FloatSize::U4)) =>
                csr_to_scipy::<f32>($py, *$data.into_any().downcast().unwrap()),
            DataType::CsrMatrix(Float(FloatSize::U8)) =>
                csr_to_scipy::<f64>($py, *$data.into_any().downcast().unwrap()),

            DataType::Array(Unsigned(IntSize::U4)) => Ok((
                &*$data.into_any().downcast::<ArrayD<u32>>().unwrap().into_pyarray($py)
            ).to_object($py)),
            DataType::Array(Unsigned(IntSize::U8)) => Ok((
                &*$data.into_any().downcast::<ArrayD<u64>>().unwrap().into_pyarray($py)
            ).to_object($py)),
            DataType::Array(Integer(IntSize::U4)) => Ok((
                &*$data.into_any().downcast::<ArrayD<i32>>().unwrap().into_pyarray($py)
            ).to_object($py)),
            DataType::Array(Integer(IntSize::U8)) => Ok((
                &*$data.into_any().downcast::<ArrayD<i64>>().unwrap().into_pyarray($py)
            ).to_object($py)),
            DataType::Array(Float(FloatSize::U4)) => Ok((
                &*$data.into_any().downcast::<ArrayD<f32>>().unwrap().into_pyarray($py)
            ).to_object($py)),
            DataType::Array(Float(FloatSize::U8)) => Ok((
                &*$data.into_any().downcast::<ArrayD<f64>>().unwrap().into_pyarray($py)
            ).to_object($py)),

            DataType::DataFrame =>
                to_py_df(*$data.into_any().downcast::<DataFrame>().unwrap()),

            ty => panic!("Cannot convert Rust element \"{:?}\" to Python object", ty)
        }
    }
}

pub fn to_py_data1<'py>(
    py: Python<'py>,
    data: Box<dyn DataIO>,
) -> PyResult<PyObject> {
    to_py_data_macro!(py, data)
}

pub fn to_py_data2<'py>(
    py: Python<'py>,
    data: Box<dyn DataPartialIO>,
) -> PyResult<PyObject> {
    to_py_data_macro!(py, data)
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

