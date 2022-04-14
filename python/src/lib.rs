use pyanndata::*;

use pyo3::{
    prelude::*,
    pymodule, types::PyModule, PyResult, Python,
};

#[pymodule]
fn anndata_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AnnData>().unwrap();
    m.add_class::<AnnDataSet>().unwrap();

    m.add_function(wrap_pyfunction!(read, m)?)?;
    m.add_function(wrap_pyfunction!(read_mtx, m)?)?;
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(create_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(read_dataset, m)?)?;
    Ok(())
}
