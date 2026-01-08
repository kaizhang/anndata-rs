use crate::data::{isinstance_of_csc, isinstance_of_csr};

use anndata::data::{CsrNonCanonical, DynArray, DynCscMatrix, DynCsrMatrix, DynCsrNonCanonical};
use nalgebra_sparse::{CscMatrix, CsrMatrix};
use ndarray::ArrayD;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyTypeError, prelude::*};

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

pub(super) fn to_array(ob: &Bound<'_, PyAny>) -> PyResult<DynArray> {
    let py = ob.py();
    let dtype = ob.getattr("dtype")?.getattr("char")?;
    let dtype = dtype.extract::<&str>()?;
    let arr = if dtype == "U" || dtype == "S" {
        ob.getattr("astype")?
            .call1(("object",))?
            .extract::<PyReadonlyArrayDyn<PyObject>>()?
            .as_array()
            .map(|x| x.extract::<String>(py).unwrap())
            .into()
    } else if dtype == "O" {
        ob.extract::<PyReadonlyArrayDyn<PyObject>>()?
            .as_array()
            .map(|x| x.extract::<String>(py).unwrap())
            .into()
    } else {
        let ty = ob.getattr("dtype")?.getattr("name")?;
        let ty = ty.extract::<&str>()?;
        proc_py_numeric!(
            ty,
            ob.extract::<PyReadonlyArrayDyn<_>>()?.to_owned_array(),
            ArrayD
        )
    };
    Ok(arr)
}

fn extract_array_as_usize(arr: Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    arr.call_method1("astype", ("uintp",))?
        .extract::<Vec<usize>>()
}

pub(super) fn to_csr(ob: &Bound<'_, PyAny>) -> PyResult<DynCsrMatrix> {
    if !isinstance_of_csr(ob)? {
        return Err(PyTypeError::new_err("not a csr matrix"));
    }

    let shape: Vec<usize> = ob.getattr("shape")?.extract()?;
    let indices = extract_array_as_usize(ob.getattr("indices")?)?;
    let indptr = extract_array_as_usize(ob.getattr("indptr")?)?;
    let ty = ob.getattr("data")?.getattr("dtype")?.getattr("name")?;
    let ty = ty.extract::<&str>()?;

    let csr = proc_py_numeric!(
        ty,
        CsrMatrix::try_from_csr_data(
            shape[0],
            shape[1],
            indptr,
            indices,
            ob.getattr("data")?
                .extract::<PyReadonlyArrayDyn<_>>()?
                .to_vec()
                .unwrap()
        )
        .unwrap(),
        CsrMatrix
    );
    Ok(csr)
}

pub(super) fn to_csr_noncanonical(ob: &Bound<'_, PyAny>) -> PyResult<DynCsrNonCanonical> {
    if !isinstance_of_csr(ob)? {
        return Err(PyTypeError::new_err("not a csr matrix"));
    }

    let shape: Vec<usize> = ob.getattr("shape")?.extract()?;
    let indices = extract_array_as_usize(ob.getattr("indices")?)?;
    let indptr = extract_array_as_usize(ob.getattr("indptr")?)?;
    let ty = ob.getattr("data")?.getattr("dtype")?.getattr("name")?;
    let ty = ty.extract::<&str>()?;

    let csr = proc_py_numeric!(
        ty,
        CsrNonCanonical::from_csr_data(
            shape[0],
            shape[1],
            indptr,
            indices,
            ob.getattr("data")?
                .extract::<PyReadonlyArrayDyn<_>>()?
                .to_vec()
                .unwrap()
        ),
        CsrNonCanonical
    );
    Ok(csr)
}

pub(super) fn to_csc(ob: &Bound<'_, PyAny>) -> PyResult<DynCscMatrix> {
    if !isinstance_of_csc(ob)? {
        return Err(PyTypeError::new_err("not a csc matrix"));
    }

    let shape: Vec<usize> = ob.getattr("shape")?.extract()?;
    let indices = extract_array_as_usize(ob.getattr("indices")?)?;
    let indptr = extract_array_as_usize(ob.getattr("indptr")?)?;
    let ty = ob.getattr("data")?.getattr("dtype")?.getattr("name")?;
    let ty = ty.extract::<&str>()?;

    let csc = proc_py_numeric!(
        ty,
        CscMatrix::try_from_csc_data(
            shape[0],
            shape[1],
            indptr,
            indices,
            ob.getattr("data")?
                .extract::<PyReadonlyArrayDyn<_>>()?
                .to_vec()
                .unwrap()
        )
        .unwrap(),
        CscMatrix
    );
    Ok(csc)
}

pub(super) fn arr_to_py<'py>(arr: DynArray, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let res = match arr {
        DynArray::I8(arr) => arr.into_pyarray(py).into_any(),
        DynArray::I16(arr) => arr.into_pyarray(py).into_any(),
        DynArray::I32(arr) => arr.into_pyarray(py).into_any(),
        DynArray::I64(arr) => arr.into_pyarray(py).into_any(),
        DynArray::U8(arr) => arr.into_pyarray(py).into_any(),
        DynArray::U16(arr) => arr.into_pyarray(py).into_any(),
        DynArray::U32(arr) => arr.into_pyarray(py).into_any(),
        DynArray::U64(arr) => arr.into_pyarray(py).into_any(),
        DynArray::F32(arr) => arr.into_pyarray(py).into_any(),
        DynArray::F64(arr) => arr.into_pyarray(py).into_any(),
        DynArray::Bool(arr) => arr.into_pyarray(py).into_any(),
        DynArray::String(_) => todo!(),
    };
    Ok(res)
}

pub(super) fn csr_to_py<'py>(csr: DynCsrMatrix, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    fn helper<'py, T: numpy::Element>(
        csr: CsrMatrix<T>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let n = csr.nrows();
        let m = csr.ncols();
        let (indptr, indices, data) = csr.disassemble();
        let scipy = PyModule::import(py, "scipy.sparse")?;
        scipy.getattr("csr_matrix")?.call1((
            (
                data.into_pyarray(py),
                indices.into_pyarray(py),
                indptr.into_pyarray(py),
            ),
            (n, m),
        ))
    }

    match csr {
        DynCsrMatrix::I8(csr) => helper(csr, py),
        DynCsrMatrix::I16(csr) => helper(csr, py),
        DynCsrMatrix::I32(csr) => helper(csr, py),
        DynCsrMatrix::I64(csr) => helper(csr, py),
        DynCsrMatrix::U8(csr) => helper(csr, py),
        DynCsrMatrix::U16(csr) => helper(csr, py),
        DynCsrMatrix::U32(csr) => helper(csr, py),
        DynCsrMatrix::U64(csr) => helper(csr, py),
        DynCsrMatrix::F32(csr) => helper(csr, py),
        DynCsrMatrix::F64(csr) => helper(csr, py),
        DynCsrMatrix::Bool(csr) => helper(csr, py),
        DynCsrMatrix::String(_) => todo!(),
    }
}

pub(super) fn csr_noncanonical_to_py<'py>(csr: DynCsrNonCanonical, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    fn helper<'py, T: numpy::Element>(
        csr: CsrNonCanonical<T>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let n = csr.nrows();
        let m = csr.ncols();
        let (indptr, indices, data) = csr.disassemble();
        let scipy = PyModule::import(py, "scipy.sparse")?;
        scipy.getattr("csr_matrix")?.call1((
            (
                data.into_pyarray(py),
                indices.into_pyarray(py),
                indptr.into_pyarray(py),
            ),
            (n, m),
        ))
    }

    match csr {
        DynCsrNonCanonical::I8(csr) => helper(csr, py),
        DynCsrNonCanonical::I16(csr) => helper(csr, py),
        DynCsrNonCanonical::I32(csr) => helper(csr, py),
        DynCsrNonCanonical::I64(csr) => helper(csr, py),
        DynCsrNonCanonical::U8(csr) => helper(csr, py),
        DynCsrNonCanonical::U16(csr) => helper(csr, py),
        DynCsrNonCanonical::U32(csr) => helper(csr, py),
        DynCsrNonCanonical::U64(csr) => helper(csr, py),
        DynCsrNonCanonical::F32(csr) => helper(csr, py),
        DynCsrNonCanonical::F64(csr) => helper(csr, py),
        DynCsrNonCanonical::Bool(csr) => helper(csr, py),
        DynCsrNonCanonical::String(_) => todo!(),
    }
}

pub(super) fn csc_to_py<'py>(csc: DynCscMatrix, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    fn helper<'py, T: numpy::Element>(
        csc: CscMatrix<T>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let n = csc.nrows();
        let m = csc.ncols();
        let (indptr, indices, data) = csc.disassemble();
        let scipy = PyModule::import(py, "scipy.sparse")?;
        scipy.getattr("csc_matrix")?.call1((
            (
                data.into_pyarray(py),
                indices.into_pyarray(py),
                indptr.into_pyarray(py),
            ),
            (n, m),
        ))
    }

    match csc {
        DynCscMatrix::I8(csc) => helper(csc, py),
        DynCscMatrix::I16(csc) => helper(csc, py),
        DynCscMatrix::I32(csc) => helper(csc, py),
        DynCscMatrix::I64(csc) => helper(csc, py),
        DynCscMatrix::U8(csc) => helper(csc, py),
        DynCscMatrix::U16(csc) => helper(csc, py),
        DynCscMatrix::U32(csc) => helper(csc, py),
        DynCscMatrix::U64(csc) => helper(csc, py),
        DynCscMatrix::F32(csc) => helper(csc, py),
        DynCscMatrix::F64(csc) => helper(csc, py),
        DynCscMatrix::Bool(csc) => helper(csc, py),
        DynCscMatrix::String(_) => todo!(),
    }
}