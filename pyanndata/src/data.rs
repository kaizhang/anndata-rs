mod instance;
mod slice;
mod dataframe;
mod array;

pub use dataframe::{PyDataFrame, PySeries};
pub(crate) use instance::*;
pub(crate) use slice::{to_select_info, to_select_elem};

use polars::prelude::DataFrame;
use std::collections::HashMap;
use pyo3::{prelude::*, types::PyDict};
use anndata::data::{Data, ArrayData, DynArray, DynCsrMatrix, DynScalar, Mapping};

pub(crate) trait FromPython<'source>: Sized {
    fn from_python(ob: &'source PyAny) -> PyResult<Self>;
}

pub(crate) trait IntoPython {
    fn into_python(self, py: Python<'_>) -> PyResult<PyObject>;
}

pub struct PyArrayData(ArrayData);

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
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        let py = ob.py();
        if isinstance_of_arr(py, ob)? {
            Ok(ArrayData::from(DynArray::from_python(ob)?).into())
        } else if isinstance_of_csr(py, ob)? {
            Ok(ArrayData::from(DynCsrMatrix::from_python(ob)?).into())
        } else if isinstance_of_pandas(py, ob)? || isinstance_of_polars(py, ob)? {
            Ok(ArrayData::from(DataFrame::from(PyDataFrame::extract(ob)?)).into())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                format!("Could not convert Python type {} to Rust data", ob.get_type())
            ))?
        }
    }
}

impl IntoPy<PyObject> for PyArrayData {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self.0 {
            ArrayData::Array(arr) => arr.into_python(py).unwrap(),
            ArrayData::CsrMatrix(csr) => csr.into_python(py).unwrap(),
            ArrayData::DataFrame(df) => PyDataFrame::from(df).into_py(py),
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
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        let data = if let Ok(s) = DynScalar::from_python(ob) {
            PyData(Data::Scalar(s))
        } else if ob.is_instance_of::<pyo3::types::PyDict>()? {
            let m = Mapping::from_python(ob)?;
            PyData(Data::Mapping(m))
        } else {
            let arr: PyArrayData = ob.extract()?;
            PyData(Data::ArrayData(arr.0))
        };
        Ok(data)
    }
}

impl IntoPy<PyObject> for PyData {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self.0 {
            Data::ArrayData(arr) => PyArrayData(arr).into_py(py),
            Data::Scalar(s) => s.into_python(py).unwrap(),
            Data::Mapping(m) => m.into_python(py).unwrap(),
        }
    }
}

impl FromPython<'_> for DynScalar {
    fn from_python(ob: &PyAny) -> PyResult<Self> {
        if ob.is_instance_of::<pyo3::types::PyInt>()? {
            ob.extract::<i64>().map(Into::into)
        } else if ob.is_instance_of::<pyo3::types::PyBool>()? {
            ob.extract::<bool>().map(Into::into)
        } else if ob.is_instance_of::<pyo3::types::PyString>()? {
            ob.extract::<String>().map(Into::into)
        } else if ob.is_instance_of::<pyo3::types::PyFloat>()? {
            ob.extract::<f64>().map(Into::into)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Could not convert to Scalar"
            ))
        }
    }
}

impl FromPython<'_> for Mapping {
    fn from_python(ob: &PyAny) -> PyResult<Self> {
        let data: HashMap<String, PyData> = ob.extract()?;
        let mapping = data.into_iter()
            .map(|(k, v)| (k, v.0))
            .collect::<HashMap<_, _>>().into();
        Ok(mapping)
    }
}

impl IntoPython for DynScalar {
    fn into_python(self, py: Python<'_>) -> PyResult<PyObject> {
        match self {
            DynScalar::I8(s) => Ok(s.into_py(py)),
            DynScalar::I16(s) => Ok(s.into_py(py)),
            DynScalar::I32(s) => Ok(s.into_py(py)),
            DynScalar::I64(s) => Ok(s.into_py(py)),
            DynScalar::U8(s) => Ok(s.into_py(py)),
            DynScalar::U16(s) => Ok(s.into_py(py)),
            DynScalar::U32(s) => Ok(s.into_py(py)),
            DynScalar::U64(s) => Ok(s.into_py(py)),
            DynScalar::Usize(s) => Ok(s.into_py(py)),
            DynScalar::F32(s) => Ok(s.into_py(py)),
            DynScalar::F64(s) => Ok(s.into_py(py)),
            DynScalar::Bool(s) => Ok(s.into_py(py)),
            DynScalar::String(s) => Ok(s.into_py(py)),
        }
    }
}

impl IntoPython for Mapping {
    fn into_python(self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let data: HashMap<String, Data> = self.into();
        data.into_iter().try_for_each(|(k, v)| {
            dict.set_item(k, PyData(v).into_py(py))
        })?;
        Ok(dict.to_object(py))
    }
}