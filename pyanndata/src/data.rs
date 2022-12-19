mod instance;
mod slice;
mod dataframe;

use dataframe::{to_py_df, to_rust_df};
pub(crate) use instance::*;
use polars::prelude::DataFrame;
pub(crate) use slice::{to_select_info, to_select_elem};

use nalgebra_sparse::CsrMatrix;
use ndarray::ArrayD;
use numpy::{PyReadonlyArray2, PyReadonlyArrayDyn, IntoPyArray};
use pyo3::prelude::*;
use anndata::data::{Data, ArrayData, DynArray, DynCsrMatrix, SelectInfo, SelectInfoElem};

macro_rules! proc_py_numeric {
    ($dtype:expr, $data:expr, $ty_anno:tt) => {
        match $dtype {
            "int8" => {
                let x: $ty_anno<i8> = $data;
                x.into()
            }
            "int16" => {
                let x: $ty_anno<i16> = $data;
                x.into()
            }
            "int32" => {
                let x: $ty_anno<i32> = $data;
                x.into()
            }
            "int64" => {
                let x: $ty_anno<i64> = $data;
                x.into()
            }
            "uint8" => {
                let x: $ty_anno<u8> = $data;
                x.into()
            }
            "uint16" => {
                let x: $ty_anno<u16> = $data;
                x.into()
            }
            "uint32" => {
                let x: $ty_anno<u32> = $data;
                x.into()
            }
            "uint64" => {
                let x: $ty_anno<u64> = $data;
                x.into()
            }
            "float32" => {
                let x: $ty_anno<f32> = $data;
                x.into()
            }
            "float64" => {
                let x: $ty_anno<f64> = $data;
                x.into()
            }
            "bool" => {
                let x: $ty_anno<bool> = $data;
                x.into()
            }
            other => panic!("converting python type '{}' is not supported", other),
        }
    };
}

pub struct PyDataFrame(DataFrame);

impl From<DataFrame> for PyDataFrame {
    fn from(value: DataFrame) -> Self {
        PyDataFrame(value)
    }
}

impl Into<DataFrame> for PyDataFrame {
    fn into(self) -> DataFrame {
        self.0
    }
}

impl<'py> FromPyObject<'py> for PyDataFrame {
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        let py = ob.py();
        let df = if isinstance_of_pandas(py, ob)? {
            py.import("polars")?.call_method1("from_pandas", (ob, ))?
        } else if ob.is_instance_of::<pyo3::types::PyDict>()? {
            py.import("polars")?.call_method1("from_dict", (ob, ))?
        } else {
            ob
        };
        Ok(to_rust_df(ob.py(), df)?.into())
    }
}

impl IntoPy<PyObject> for PyDataFrame {
    fn into_py(self, py: Python<'_>) -> PyObject {
        to_py_df(py, self.0).unwrap()
    }
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
        } else {
            panic!("not an array")
        }
    }
}

impl IntoPy<PyObject> for PyArrayData {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self.0 {
            ArrayData::Array(arr) => arr.into_python(py).unwrap(),
            ArrayData::CsrMatrix(csr) => csr.into_python(py).unwrap(),
        }
    }
}

trait FromPython<'source>: Sized {
    fn from_python(ob: &'source PyAny) -> PyResult<Self>;
}

impl FromPython<'_> for DynArray {
    fn from_python(ob: &PyAny) -> PyResult<Self> {
        let ty = ob.getattr("dtype")?.getattr("name")?.extract::<&str>()?;
        let arr = proc_py_numeric!(ty, ob.extract::<PyReadonlyArrayDyn<_>>()?.to_owned_array(), ArrayD);
        Ok(arr)
    }
}

impl FromPython<'_> for DynCsrMatrix {
    fn from_python(ob: &PyAny) -> PyResult<Self> {
        fn extract_csr_indicies(indicies: &PyAny) -> PyResult<Vec<usize>> {
            let res = match indicies
                .getattr("dtype")?
                .getattr("name")?
                .extract::<&str>()?
            {
                "int32" => indicies
                    .extract::<PyReadonlyArrayDyn<i32>>()?
                    .as_array()
                    .iter()
                    .map(|x| (*x).try_into().unwrap())
                    .collect(),
                "int64" => indicies
                    .extract::<PyReadonlyArrayDyn<i64>>()?
                    .as_array()
                    .iter()
                    .map(|x| (*x).try_into().unwrap())
                    .collect(),
                other => panic!("CSR indicies type '{}' is not supported", other),
            };
            Ok(res)
        }

        let shape: Vec<usize> = ob.getattr("shape")?.extract()?;
        let indices = extract_csr_indicies(ob.getattr("indices")?)?;
        let indptr = extract_csr_indicies(ob.getattr("indptr")?)?;
        let ty = ob.getattr("data")?.getattr("dtype")?.getattr("name")?.extract::<&str>()?;

        let csr = proc_py_numeric!(
            ty,
            CsrMatrix::try_from_csr_data(
                shape[0],
                shape[1],
                indptr,
                indices,
                ob.getattr("data")?.extract::<PyReadonlyArrayDyn<_>>()?.to_vec().unwrap()
            ).unwrap(),
            CsrMatrix
        );
         Ok(csr)
    }
}

pub trait IntoPython {
    fn into_python<'py>(self, py: Python<'_>) -> PyResult<PyObject>;
}

impl IntoPython for DynArray {
    fn into_python<'py>(self, py: Python<'_>) -> PyResult<PyObject> {
        let res = match self {
            DynArray::I8(arr) => arr.into_pyarray(py).to_object(py),
            DynArray::I16(arr) => arr.into_pyarray(py).to_object(py),
            DynArray::I32(arr) => arr.into_pyarray(py).to_object(py),
            DynArray::I64(arr) => arr.into_pyarray(py).to_object(py),
            DynArray::U8(arr) => arr.into_pyarray(py).to_object(py),
            DynArray::U16(arr) => arr.into_pyarray(py).to_object(py),
            DynArray::U32(arr) => arr.into_pyarray(py).to_object(py),
            DynArray::U64(arr) => arr.into_pyarray(py).to_object(py),
            DynArray::Usize(arr) => arr.into_pyarray(py).to_object(py),
            DynArray::F32(arr) => arr.into_pyarray(py).to_object(py),
            DynArray::F64(arr) => arr.into_pyarray(py).to_object(py),
            DynArray::Bool(arr) => arr.into_pyarray(py).to_object(py),
            DynArray::String(_) => todo!(),
            DynArray::Categorical(_) => todo!(),
        };
        Ok(res)
    }
}

impl IntoPython for DynCsrMatrix {
    fn into_python<'py>(self, py: Python<'_>) -> PyResult<PyObject> {
        fn helper<T: numpy::Element>(csr: CsrMatrix<T>, py: Python<'_>) -> PyResult<PyObject> {
            let n = csr.nrows();
            let m = csr.ncols();
            let (intptr, indices, data) = csr.disassemble();
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

        match self {
            DynCsrMatrix::I8(csr) => helper(csr, py),
            DynCsrMatrix::I16(csr) => helper(csr, py),
            DynCsrMatrix::I32(csr) => helper(csr, py),
            DynCsrMatrix::I64(csr) => helper(csr, py),
            DynCsrMatrix::U8(csr) => helper(csr, py),
            DynCsrMatrix::U16(csr) => helper(csr, py),
            DynCsrMatrix::U32(csr) => helper(csr, py),
            DynCsrMatrix::U64(csr) => helper(csr, py),
            DynCsrMatrix::Usize(csr) => helper(csr, py),
            DynCsrMatrix::F32(csr) => helper(csr, py),
            DynCsrMatrix::F64(csr) => helper(csr, py),
            DynCsrMatrix::Bool(csr) => helper(csr, py),
            DynCsrMatrix::String(_) => todo!(),
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
        todo!()
    }
}

impl IntoPy<PyObject> for PyData {
    fn into_py(self, py: Python<'_>) -> PyObject {
        todo!()
    }
}

