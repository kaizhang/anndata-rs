pub mod element;
pub mod iterator;
pub(crate) mod utils;
mod anndata;

pub use crate::anndata::{AnnData, AnnDataSet, read_h5ad, read_mtx, read_dataset};

use pyo3::{
    prelude::*,
    pymodule, types::PyModule, PyResult, Python,
};

#[pymodule]
fn pyanndata(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AnnData>().unwrap();
    m.add_class::<element::PyElem>().unwrap();
    m.add_class::<element::PyMatrixElem>().unwrap();
    m.add_class::<element::PyDataFrameElem>().unwrap();
    m.add_class::<AnnDataSet>().unwrap();

    m.add_function(wrap_pyfunction!(read_mtx, m)?)?;
    m.add_function(wrap_pyfunction!(read_h5ad, m)?)?;
    m.add_function(wrap_pyfunction!(read_dataset, m)?)?;
    Ok(())
}