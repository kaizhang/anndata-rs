mod backed;
mod memory;
mod dataset;

pub use backed::AnnData;
pub use memory::PyAnnData;
pub use dataset::AnnDataSet;

use pyo3::prelude::*;
use std::path::PathBuf;
use anyhow::Result;

/// Read `.h5ad`-formatted hdf5 file.
///
/// Parameters
/// ----------
///
/// filename: Path
///     File name of data file.
/// backed: Literal['r', 'r+'] | None
///     Default is `r+`.
///     If `'r'`, the file is opened in read-only mode.
///     If `'r+'`, the file is opened in read/write mode.
///     If `None`, the AnnData object is read into memory.
/// backend: Literal['hdf5'] | None
#[pyfunction(backed = "\"r+\"", backend = "None")]
#[pyo3(text_signature = "(filename, backed, backend /)")]
pub fn read<'py>(py: Python<'py>, filename: PathBuf, backed: Option<&str>, backend: Option<&str>) -> Result<PyObject> {
    let adata = match backed {
        Some(m) => AnnData::open(filename, m, backend)?.into_py(py),
        None => PyModule::import(py, "anndata")?
            .getattr("read_h5ad")?
            .call1((filename,))?
            .to_object(py),
    };
    Ok(adata)
}