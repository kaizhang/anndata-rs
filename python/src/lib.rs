use pyanndata::*;

use pyo3::{prelude::*, pymodule, types::PyModule, PyResult};

#[pymodule]
fn anndata_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_class::<AnnData>().unwrap();
    m.add_class::<AnnDataSet>().unwrap();

    m.add_function(wrap_pyfunction!(read, m)?)?;
    m.add_function(wrap_pyfunction!(read_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(read_mtx, m)?)?;
    /*
    m.add_class::<StackedAnnData>().unwrap();
    m.add_class::<element::PyElemCollection>().unwrap();
    m.add_class::<element::PyAxisArrays>().unwrap();
    m.add_class::<element::PyElem>().unwrap();
    m.add_class::<element::PyMatrixElem>().unwrap();
    m.add_class::<element::PyDataFrameElem>().unwrap();
    m.add_class::<element::PyStackedMatrixElem>().unwrap();
    m.add_class::<element::PyStackedAxisArrays>().unwrap();
    m.add_class::<element::PyStackedDataFrame>().unwrap();

    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    */
    Ok(())
}
