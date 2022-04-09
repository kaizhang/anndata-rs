use pyo3::{
    prelude::*,
    types::PyType,
    PyResult, Python,
};

pub fn isinstance_of_csr<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    obj.is_instance(
        py.import("scipy.sparse.csr")?.getattr("csr_matrix")?
        .downcast::<PyType>().unwrap()
    )
}

pub fn isinstance_of_arr<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    obj.is_instance(
        py.import("numpy")?.getattr("ndarray")?.downcast::<PyType>().unwrap()
    )
}

pub fn isinstance_of_pandas<'py>(py: Python<'py>, obj: &'py PyAny) -> PyResult<bool> {
    obj.is_instance(
        py.import("pandas")?.getattr("DataFrame")?.downcast::<PyType>().unwrap()
    )
}
