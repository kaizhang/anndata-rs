mod instance;
mod slice;
mod array;

pub(crate) use instance::*;
use pyo3_polars::PyDataFrame;
pub use slice::{to_select_info, to_select_elem};

use std::{collections::HashMap, ops::Deref};
use pyo3::{prelude::*, types::PyDict};
use anndata::data::{Data, ArrayData, DynScalar, Mapping};

pub struct PyArrayData(ArrayData);

impl Deref for PyArrayData {
    type Target = ArrayData;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<ArrayData> for PyArrayData {
    fn from(value: ArrayData) -> Self {
        PyArrayData(value)
    }
}

impl Into<ArrayData> for PyArrayData {
    fn into(self) -> ArrayData {
        self.0
    }
}

impl<'py> FromPyObject<'py> for PyArrayData {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if isinstance_of_arr(ob)? {
            Ok(ArrayData::from(array::to_array(ob)?).into())
        } else if isinstance_of_csr(ob)? {
            if ob.getattr("has_canonical_format")?.extract()? {
                Ok(ArrayData::from(array::to_csr(ob)?).into())
            } else {
                Ok(ArrayData::from(array::to_csr_noncanonical(ob)?).into())
            }
        } else if isinstance_of_csc(ob)? {
            Ok(ArrayData::from(array::to_csc(ob)?).into())
        } else if isinstance_of_pandas(ob)? {
            let ob = ob.py().import("polars")?.call_method1("from_pandas", (ob, ))?;
            Ok(ArrayData::from(ob.extract::<PyDataFrame>()?.0).into())
        } else if isinstance_of_polars(ob)? {
            Ok(ArrayData::from(ob.extract::<PyDataFrame>()?.0).into())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                format!("Could not convert Python type {} to Rust data", ob.get_type())
            ))?
        }
    }
}

impl<'py> IntoPyObject<'py> for PyArrayData {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self.0 {
            ArrayData::CscMatrix(csc) => array::csc_to_py(csc, py),
            ArrayData::CsrMatrix(csr) => array::csr_to_py(csr, py),
            ArrayData::Array(arr) => array::arr_to_py(arr, py),
            ArrayData::CsrNonCanonical(csr) => array::csr_noncanonical_to_py(csr, py),
            ArrayData::DataFrame(df) => PyDataFrame(df).into_pyobject(py),
        }
    }
}

pub struct PyData(Data);

impl From<Data> for PyData {
    fn from(value: Data) -> Self {
        PyData(value)
    }
}

impl Into<Data> for PyData {
    fn into(self) -> Data {
        self.0
    }
}

impl<'py> FromPyObject<'py> for PyData {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let data = if let Ok(s) = to_scalar(ob) {
            PyData(Data::Scalar(s))
        } else if ob.is_instance_of::<pyo3::types::PyDict>() {
            let m = to_mapping(ob)?;
            PyData(Data::Mapping(m))
        } else {
            let arr: PyArrayData = ob.extract()?;
            PyData(Data::ArrayData(arr.0))
        };
        Ok(data)
    }
}

fn to_scalar(ob: &Bound<'_, PyAny>) -> PyResult<DynScalar> {
    if ob.is_instance_of::<pyo3::types::PyBool>() {
        ob.extract::<bool>().map(Into::into)
    } else if ob.is_instance_of::<pyo3::types::PyInt>() {
        ob.extract::<i64>().map(Into::into)
    } else if ob.is_instance_of::<pyo3::types::PyString>() {
        ob.extract::<String>().map(Into::into)
    } else if ob.is_instance_of::<pyo3::types::PyFloat>() {
        ob.extract::<f64>().map(Into::into)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Could not convert to Scalar"
        ))
    }
}

fn to_mapping(ob: &Bound<'_, PyAny>) -> PyResult<Mapping> {
    let data: HashMap<String, PyData> = ob.extract()?;
    let mapping = data.into_iter()
        .map(|(k, v)| (k, v.0))
        .collect::<HashMap<_, _>>().into();
    Ok(mapping)
}

impl<'py> IntoPyObject<'py> for PyData {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self.0 {
            Data::ArrayData(arr) => PyArrayData(arr).into_pyobject(py),
            Data::Mapping(m) => mapping_to_python(m, py),
            Data::Scalar(s) => scalar_to_py(s, py),
        }
    }
}

fn scalar_to_py<'py>(s: DynScalar, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    match s {
        DynScalar::I8(s) => Ok(s.into_pyobject(py)?.into_any()),
        DynScalar::I16(s) => Ok(s.into_pyobject(py)?.into_any()),
        DynScalar::I32(s) => Ok(s.into_pyobject(py)?.into_any()),
        DynScalar::I64(s) => Ok(s.into_pyobject(py)?.into_any()),
        DynScalar::U8(s) => Ok(s.into_pyobject(py)?.into_any()),
        DynScalar::U16(s) => Ok(s.into_pyobject(py)?.into_any()),
        DynScalar::U32(s) => Ok(s.into_pyobject(py)?.into_any()),
        DynScalar::U64(s) => Ok(s.into_pyobject(py)?.into_any()),
        DynScalar::F32(s) => Ok(s.into_pyobject(py)?.into_any()),
        DynScalar::F64(s) => Ok(s.into_pyobject(py)?.into_any()),
        DynScalar::Bool(s) => Ok(s.into_pyobject(py)?.to_owned().into_any()),
        DynScalar::String(s) => Ok(s.into_pyobject(py)?.into_any()),
    }
}

fn mapping_to_python<'py>(d: Mapping, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    let data: HashMap<String, Data> = d.into();
    data.into_iter().try_for_each(|(k, v)| {
        dict.set_item(k, PyData(v).into_pyobject(py)?)
    })?;
    Ok(dict.into_any())
}