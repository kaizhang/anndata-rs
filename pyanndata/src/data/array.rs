use crate::data::{FromPython, IntoPython};

use ndarray::ArrayD;
use nalgebra_sparse::{CsrMatrix, CscMatrix};
use pyo3::prelude::*;
use anndata::data::{DynArray, DynCsrMatrix, DynCscMatrix};
use numpy::{PyReadonlyArrayDyn, IntoPyArray};

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

impl FromPython<'_> for DynCscMatrix {
    fn from_python(ob: &PyAny) -> PyResult<Self> {
        fn extract_csc_indicies(indicies: &PyAny) -> PyResult<Vec<usize>> {
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
                other => panic!("CSC indicies type '{}' is not supported", other),
            };
            Ok(res)
        }

        let shape: Vec<usize> = ob.getattr("shape")?.extract()?;
        let indices = extract_csc_indicies(ob.getattr("indices")?)?;
        let indptr = extract_csc_indicies(ob.getattr("indptr")?)?;
        let ty = ob.getattr("data")?.getattr("dtype")?.getattr("name")?.extract::<&str>()?;

        let csc = proc_py_numeric!(
            ty,
            CscMatrix::try_from_csc_data(
                shape[0],
                shape[1],
                indptr,
                indices,
                ob.getattr("data")?.extract::<PyReadonlyArrayDyn<_>>()?.to_vec().unwrap()
            ).unwrap(),
            CscMatrix
        );
         Ok(csc)
    }
}

impl IntoPython for DynArray {
    fn into_python(self, py: Python<'_>) -> PyResult<PyObject> {
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
    fn into_python(self, py: Python<'_>) -> PyResult<PyObject> {
        fn helper<T: numpy::Element>(csr: CsrMatrix<T>, py: Python<'_>) -> PyResult<PyObject> {
            let n = csr.nrows();
            let m = csr.ncols();
            let (indptr, indices, data) = csr.disassemble();
            let scipy = PyModule::import(py, "scipy.sparse")?;
            Ok(scipy
                .getattr("csr_matrix")?
                .call1((
                    (
                        data.into_pyarray(py),
                        indices.into_pyarray(py),
                        indptr.into_pyarray(py),
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

impl IntoPython for DynCscMatrix {
    fn into_python(self, py: Python<'_>) -> PyResult<PyObject> {
        fn helper<T: numpy::Element>(csc: CscMatrix<T>, py: Python<'_>) -> PyResult<PyObject> {
            let n = csc.nrows();
            let m = csc.ncols();
            let (indptr, indices, data) = csc.disassemble();
            let scipy = PyModule::import(py, "scipy.sparse")?;
            Ok(scipy
                .getattr("csc_matrix")?
                .call1((
                    (
                        data.into_pyarray(py),
                        indices.into_pyarray(py),
                        indptr.into_pyarray(py),
                    ),
                    (n, m),
                ))?
                .to_object(py))
        }

        match self {
            DynCscMatrix::I8(csc) => helper(csc, py),
            DynCscMatrix::I16(csc) => helper(csc, py),
            DynCscMatrix::I32(csc) => helper(csc, py),
            DynCscMatrix::I64(csc) => helper(csc, py),
            DynCscMatrix::U8(csc) => helper(csc, py),
            DynCscMatrix::U16(csc) => helper(csc, py),
            DynCscMatrix::U32(csc) => helper(csc, py),
            DynCscMatrix::U64(csc) => helper(csc, py),
            DynCscMatrix::Usize(csc) => helper(csc, py),
            DynCscMatrix::F32(csc) => helper(csc, py),
            DynCscMatrix::F64(csc) => helper(csc, py),
            DynCscMatrix::Bool(csc) => helper(csc, py),
            DynCscMatrix::String(_) => todo!(),
        }
    }
}