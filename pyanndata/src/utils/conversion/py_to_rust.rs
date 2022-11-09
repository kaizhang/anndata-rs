use crate::utils::{
    instance::*,
    conversion::to_rust_df,
};

use pyo3::{
    prelude::*,
    PyResult, Python,
};
use numpy::PyReadonlyArrayDyn;
use nalgebra_sparse::csr::CsrMatrix;
use std::collections::HashMap;

use anndata_rs::data::{Mapping, Scalar, Data, MatrixData};

macro_rules! proc_py_numeric {
    ($obj:expr, $reader:expr, $fun:ident, $ty:tt) => {
        match $obj.getattr("dtype")?.getattr("name")?.extract::<&str>()? {
            "int8" => { let mat: $ty<i8> = $reader; $fun!(mat) },
            "int16" => { let mat: $ty<i16> = $reader; $fun!(mat) },
            "int32" => { let mat: $ty<i32> = $reader; $fun!(mat) },
            "int64" => { let mat: $ty<i64> = $reader; $fun!(mat) },
            "uint8" => { let mat: $ty<u8> = $reader; $fun!(mat) },
            "uint16" => { let mat: $ty<u16> = $reader; $fun!(mat) },
            "uint32" => { let mat: $ty<u32> = $reader; $fun!(mat) },
            "uint64" => { let mat: $ty<u64> = $reader; $fun!(mat) },
            "float32" => { let mat: $ty<f32> = $reader; $fun!(mat) },
            "float64" => { let mat: $ty<f64> = $reader; $fun!(mat) },
            "bool" => { let mat: $ty<bool> = $reader; $fun!(mat) },
            other => panic!("converting python type '{}' is not supported", other),
        }
    }
}

pub fn to_rust_data1<'py>(
    py: Python<'py>,
    obj: &'py PyAny,
) -> PyResult<Box<dyn Data>>
{
    macro_rules! _arr { ($x:expr) => { Ok(Box::new($x.to_owned_array())) }; }

    macro_rules! _csr {
        ($x:expr) => {{
            let shape: Vec<usize> = obj.getattr("shape")?.extract()?;
            let indices = obj.getattr("indices")?
                .extract::<PyReadonlyArrayDyn<i32>>()?.as_array().iter()
                .map(|x| (*x).try_into().unwrap()).collect();
            let indptr = obj.getattr("indptr")?
                .extract::<PyReadonlyArrayDyn<i32>>()?.as_array().iter()
                .map(|x| (*x).try_into().unwrap()).collect();
            Ok(Box::new(CsrMatrix::try_from_csr_data(
                shape[0], shape[1], indptr, indices, $x.to_vec().unwrap()
            ).unwrap()))
        }};
    }

    if isinstance_of_arr(py, obj)? {
        proc_py_numeric!(obj, obj.extract()?, _arr, PyReadonlyArrayDyn)
    } else if isinstance_of_csr(py, obj)? {
        proc_py_numeric!(obj, obj.getattr("data")?.extract()?,_csr, PyReadonlyArrayDyn)
    } else if obj.is_instance_of::<pyo3::types::PyString>()? {
        Ok(Box::new(obj.extract::<String>()?))
    } else if obj.is_instance_of::<pyo3::types::PyBool>()? {
        Ok(Box::new(Scalar(obj.extract::<bool>()?)))
    } else if obj.is_instance_of::<pyo3::types::PyInt>()? {
        Ok(Box::new(Scalar(obj.extract::<i64>()?)))
    } else if obj.is_instance_of::<pyo3::types::PyFloat>()? {
        Ok(Box::new(Scalar(obj.extract::<f64>()?)))
    } else if obj.is_instance_of::<pyo3::types::PyDict>()? {
        let data: HashMap<String, &'py PyAny> = obj.extract()?;
        let mapping = data.into_iter().map(|(k, v)| Ok((k, to_rust_data1(py, v)?)))
            .collect::<PyResult<HashMap<_, _>>>()?;
        Ok(Box::new(Mapping(mapping)))
    } else if isinstance_of_polars(py, obj)? {
        Ok(Box::new(to_rust_df(obj)?))
    } else if isinstance_of_pandas(py, obj)? {
        let obj_ = py.import("polars")?.call_method1("from_pandas", (obj, ))?;
        Ok(Box::new(to_rust_df(obj_)?))
    } else {
        panic!("Cannot convert Python type \"{}\" to Rust data", obj.get_type())
    }
}

pub fn to_rust_data2<'py>(
    py: Python<'py>,
    obj: &'py PyAny,
) -> PyResult<Box<dyn MatrixData>>
{
    macro_rules! _arr { ($x:expr) => { Ok(Box::new($x.to_owned_array())) }; }

    //TODO: support i64 types
    macro_rules! _csr {
        ($x:expr) => {{
            let shape: Vec<usize> = obj.getattr("shape")?.extract()?;
            let indices = extract_csr_indicies(obj.getattr("indices")?)?;
            let indptr = extract_csr_indicies(obj.getattr("indptr")?)?;
            Ok(Box::new(CsrMatrix::try_from_csr_data(
                shape[0], shape[1], indptr, indices, $x.to_vec().unwrap()
            ).unwrap()))
        }};
    }

    if isinstance_of_arr(py, obj)? {
        proc_py_numeric!(obj, obj.extract()?, _arr, PyReadonlyArrayDyn)
    } else if isinstance_of_csr(py, obj)? {
        proc_py_numeric!(obj, obj.getattr("data")?.extract()?, _csr, PyReadonlyArrayDyn)
    } else {
        panic!("Cannot convert Python type \"{}\" to Rust data", obj.get_type())
    }
}

fn extract_csr_indicies(indicies: &PyAny) -> PyResult<Vec<usize>> {
    let res = match indicies.getattr("dtype")?.getattr("name")?.extract::<&str>()? {
        "int32" => indicies.extract::<PyReadonlyArrayDyn<i32>>()?.as_array().iter()
            .map(|x| (*x).try_into().unwrap()).collect(),
        "int64" => indicies.extract::<PyReadonlyArrayDyn<i64>>()?.as_array().iter()
            .map(|x| (*x).try_into().unwrap()).collect(),
        other => panic!("CSR indicies type '{}' is not supported", other),
    };
    Ok(res)
}