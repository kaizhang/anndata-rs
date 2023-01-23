use pyo3::{prelude::*, types::PyType, PyResult, Python};

pub fn isinstance_of_csr<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    obj.is_instance(
        py.import("scipy.sparse.csr")?
            .getattr("csr_matrix")?
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

pub fn is_list_of_bools<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    if obj.is_instance_of::<pyo3::types::PyList>()? {
        Ok(obj.extract::<Vec<PyObject>>()?.into_iter().all(|x| {
            x.as_ref(py)
                .is_instance_of::<pyo3::types::PyBool>()
                .unwrap()
        }))
    } else {
        Ok(false)
    }
}

pub fn is_list_of_ints<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    if obj.is_instance_of::<pyo3::types::PyList>()? {
        Ok(obj
            .extract::<Vec<PyObject>>()?
            .into_iter()
            .all(|x| x.as_ref(py).is_instance_of::<pyo3::types::PyInt>().unwrap()))
    } else {
        Ok(false)
    }
}

pub fn is_list_of_strings<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    if obj.is_instance_of::<pyo3::types::PyList>()? {
        Ok(obj.extract::<Vec<PyObject>>()?.into_iter().all(|x| {
            x.as_ref(py)
                .is_instance_of::<pyo3::types::PyString>()
                .unwrap()
        }))
    } else {
        Ok(false)
    }
}

pub fn is_none_slice<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    Ok(
        obj.is_none() ||
        is_ellipsis(py, obj)? ||
        (is_slice(obj)? && obj.eq(py.eval("slice(None, None, None)", None, None)?)?)
    )
}

fn is_slice<'py>(obj: &'py PyAny) -> PyResult<bool> {
    obj.is_instance_of::<pyo3::types::PySlice>()
}

fn is_ellipsis<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    obj.is_instance(py.eval("...", None, None)?.get_type())
}
