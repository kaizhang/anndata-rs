use crate::utils::instance::{isinstance_of_csr, isinstance_of_arr};

use pyo3::{
    prelude::*,
    PyResult, Python,
};
use numpy::PyReadonlyArrayDyn;
use nalgebra_sparse::csr::CsrMatrix;

use anndata_rs::{
    anndata_trait::{Scalar, DataIO, DataPartialIO},
};

macro_rules! to_rust_arr_macro {
    ($obj:expr) => {
        Ok(match $obj.getattr("dtype")?.getattr("name")?.extract::<&str>()? {
            "float64" => Box::new($obj.extract::<PyReadonlyArrayDyn<f64>>()?.to_owned_array()),
            "float32" => Box::new($obj.extract::<PyReadonlyArrayDyn<f32>>()?.to_owned_array()),
            "int64" => Box::new($obj.extract::<PyReadonlyArrayDyn<i64>>()?.to_owned_array()),
            "int32" => Box::new($obj.extract::<PyReadonlyArrayDyn<i32>>()?.to_owned_array()),
            dtype => panic!("Converting array type {} is not implemented", dtype),
        })
    }
}

macro_rules! to_rust_csr_macro {
    ($obj:expr) => {{
        let shape: Vec<usize> = $obj.getattr("shape")?.extract()?;
        let indices = $obj.getattr("indices")?
            .extract::<PyReadonlyArrayDyn<i32>>()?.as_array().iter()
            .map(|x| (*x).try_into().unwrap()).collect();
        let indptr = $obj.getattr("indptr")?
            .extract::<PyReadonlyArrayDyn<i32>>()?.as_array().iter()
            .map(|x| (*x).try_into().unwrap()).collect();

        Ok(match $obj.getattr("dtype")?.getattr("name")?.extract::<&str>()? {
            "float64" => {
                let data = $obj.getattr("data")?
                    .extract::<PyReadonlyArrayDyn<f64>>()?.to_vec().unwrap();
                Box::new(CsrMatrix::try_from_csr_data(shape[0], shape[1], indptr, indices, data).unwrap())
            },
            "float32" => {
                let data = $obj.getattr("data")?
                    .extract::<PyReadonlyArrayDyn<f32>>()?.to_vec().unwrap();
                Box::new(CsrMatrix::try_from_csr_data(shape[0], shape[1], indptr, indices, data).unwrap())
            },
            "int64" => {
                let data = $obj.getattr("data")?
                    .extract::<PyReadonlyArrayDyn<i64>>()?.to_vec().unwrap();
                Box::new(CsrMatrix::try_from_csr_data(shape[0], shape[1], indptr, indices, data).unwrap())
            },
            "int32" => {
                let data = $obj.getattr("data")?
                    .extract::<PyReadonlyArrayDyn<i32>>()?.to_vec().unwrap();
                Box::new(CsrMatrix::try_from_csr_data(shape[0], shape[1], indptr, indices, data).unwrap())
            },
            dtype => panic!("Converting csr type {} is not supported", dtype),
        })
    }};
}

macro_rules! to_rust_data_macro {
    ($py:expr, $obj:expr) => {
        if isinstance_of_arr($py, $obj)? {
            to_rust_arr_macro!($obj)
        } else if isinstance_of_csr($py, $obj)? {
            to_rust_csr_macro!($obj)
        } else {
            panic!("Cannot convert Python type \"{}\" to Rust data", $obj.get_type())
        }
    }
}

pub fn to_rust_data1<'py>(
    py: Python<'py>,
    obj: &'py PyAny,
) -> PyResult<Box<dyn DataIO>>
{
    if isinstance_of_arr(py, obj)? {
        to_rust_arr_macro!(obj)
    } else if isinstance_of_csr(py, obj)? {
        to_rust_csr_macro!(obj)
    } else if obj.is_instance_of::<pyo3::types::PyString>()? {
        Ok(Box::new(obj.extract::<String>()?))
    } else if obj.is_instance_of::<pyo3::types::PyBool>()? {
        Ok(Box::new(Scalar(obj.extract::<bool>()?)))
    } else if obj.is_instance_of::<pyo3::types::PyInt>()? {
        Ok(Box::new(Scalar(obj.extract::<i64>()?)))
    } else if obj.is_instance_of::<pyo3::types::PyFloat>()? {
        Ok(Box::new(Scalar(obj.extract::<f64>()?)))
    } else {
        panic!("Cannot convert Python type \"{}\" to Rust data", obj.get_type())
    }
}

pub fn to_rust_data2<'py>(
    py: Python<'py>,
    obj: &'py PyAny,
) -> PyResult<Box<dyn DataPartialIO>>
{
    to_rust_data_macro!(py, obj)
}
