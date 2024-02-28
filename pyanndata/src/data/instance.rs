use pyo3::{prelude::*, types::PyType, PyResult, Python};

pub fn isinstance_of_csr<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    obj.is_instance(
        py.import("scipy.sparse.csr")?
            .getattr("csr_matrix")?
            .downcast::<PyType>()
            .unwrap(),
    )
}

pub fn isinstance_of_csc<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    obj.is_instance(
        py.import("scipy.sparse.csc")?
            .getattr("csc_matrix")?
            .downcast::<PyType>()
            .unwrap(),
    )
}

pub fn isinstance_of_arr<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    obj.is_instance(
        py.import("numpy")?
            .getattr("ndarray")?
            .downcast::<PyType>()
            .unwrap(),
    )
}

pub fn isinstance_of_pyanndata<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    obj.is_instance(
        py.import("anndata")?
            .getattr("AnnData")?
            .downcast::<PyType>()
            .unwrap(),
    )
}

pub fn isinstance_of_pandas<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    obj.is_instance(
        py.import("pandas")?
            .getattr("DataFrame")?
            .downcast::<PyType>()
            .unwrap(),
    )
}

pub fn isinstance_of_polars<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    obj.is_instance(
        py.import("polars")?
            .getattr("DataFrame")?
            .downcast::<PyType>()
            .unwrap(),
    )
}

pub fn is_none_slice<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    Ok(
        obj.is_none() ||
        obj.is_ellipsis() ||
        (is_slice(obj) && obj.eq(py.eval("slice(None, None, None)", None, None)?)?)
    )
}

fn is_slice<'py>(obj: &'py PyAny) -> bool {
    obj.is_instance_of::<pyo3::types::PySlice>()
}