use crate::utils::{instance::*, conversion::dataframe::to_rust_df};

use pyo3::{prelude::*, PyResult, Python};
use polars::frame::DataFrame;
use ndarray::ArrayD;
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

macro_rules! _box { ($x:expr) => { Ok(Box::new($x)) }; }

pub trait PyToRust<T> {
    fn into_rust<'py>(self, py: Python<'py>) -> PyResult<T>;
}

impl<T: numpy::Element> PyToRust<CsrMatrix<T>> for &PyAny {
    fn into_rust<'py>(self, _py: Python<'py>) -> PyResult<CsrMatrix<T>> {
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

        let shape: Vec<usize> = self.getattr("shape")?.extract()?;
        let indices = extract_csr_indicies(self.getattr("indices")?)?;
        let indptr = extract_csr_indicies(self.getattr("indptr")?)?;
        let data = self.getattr("data")?.extract::<PyReadonlyArrayDyn<T>>()?;
        let mat = CsrMatrix::try_from_csr_data(shape[0], shape[1], indptr, indices, data.to_vec().unwrap()).unwrap();
        Ok(mat)
    }
}

impl<T: numpy::Element> PyToRust<ArrayD<T>> for &PyAny {
    fn into_rust<'py>(self, _py: Python<'py>) -> PyResult<ArrayD<T>> {
        Ok(self.extract::<PyReadonlyArrayDyn<T>>()?.to_owned_array())
    }
}

impl PyToRust<DataFrame> for &PyAny {
    fn into_rust<'py>(self, py: Python<'py>) -> PyResult<DataFrame> {
        to_rust_df(py, self)
    }
}

impl PyToRust<Box<dyn Data>> for &PyAny {
    fn into_rust<'py>(self, py: Python<'py>) -> PyResult<Box<dyn Data>> {
        if isinstance_of_arr(py, self)? {
            proc_py_numeric!(self, self.into_rust(py)?, _box, ArrayD)
        } else if isinstance_of_csr(py, self)? {
            proc_py_numeric!(self, self.into_rust(py)?, _box, CsrMatrix)
        } else if self.is_instance_of::<pyo3::types::PyString>()? {
            Ok(Box::new(self.extract::<String>()?))
        } else if self.is_instance_of::<pyo3::types::PyBool>()? {
            Ok(Box::new(Scalar(self.extract::<bool>()?)))
        } else if self.is_instance_of::<pyo3::types::PyInt>()? {
            Ok(Box::new(Scalar(self.extract::<i64>()?)))
        } else if self.is_instance_of::<pyo3::types::PyFloat>()? {
            Ok(Box::new(Scalar(self.extract::<f64>()?)))
        } else if self.is_instance_of::<pyo3::types::PyDict>()? {
            let data: HashMap<String, &PyAny> = self.extract()?;
            let mapping = data.into_iter().map(|(k, v)| Ok((k, v.into_rust(py)?)))
                .collect::<PyResult<HashMap<_, _>>>()?;
            Ok(Box::new(Mapping(mapping)))
        } else if isinstance_of_polars(py, self)? {
            let df: DataFrame = self.into_rust(py)?;
            Ok(Box::new(df))
        } else if isinstance_of_pandas(py, self)? {
            let df: DataFrame = py.import("polars")?.call_method1("from_pandas", (self, ))?.into_rust(py)?;
            Ok(Box::new(df))
        } else {
            panic!("Cannot convert Python type \"{}\" to 'dyn Data'", self.get_type())
        }
    }
}

impl PyToRust<Box<dyn MatrixData>> for &PyAny {
    fn into_rust<'py>(self, py: Python<'py>) -> PyResult<Box<dyn MatrixData>> {
        if isinstance_of_arr(py, self)? {
            proc_py_numeric!(self, self.into_rust(py)?, _box, ArrayD)
        } else if isinstance_of_csr(py, self)? {
            proc_py_numeric!(self, self.into_rust(py)?, _box, CsrMatrix)
        } else {
            panic!("Cannot convert Python type \"{}\" to 'dyn MatrixData'", self.get_type())
        }
    }
}