use pyanndata::{element, AnnData, AnnDataSet, read, read_mtx, read_dataset};

use pyo3::{
    prelude::*,
    pymodule, types::PyModule, PyResult, Python,
};

#[pymodule]
fn anndata_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AnnData>().unwrap();
    m.add_class::<AnnDataSet>().unwrap();
    m.add_class::<element::PyElem>().unwrap();
    m.add_class::<element::PyMatrixElem>().unwrap();
    m.add_class::<element::PyDataFrameElem>().unwrap();

    m.add_function(wrap_pyfunction!(read_mtx, m)?)?;
    m.add_function(wrap_pyfunction!(read, m)?)?;
    m.add_function(wrap_pyfunction!(read_dataset, m)?)?;
    Ok(())
}
