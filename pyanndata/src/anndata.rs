mod backed;
pub mod memory;
mod dataset;

pub use backed::AnnData;
pub use memory::PyAnnData;
pub use dataset::AnnDataSet;

use anndata;
use anndata::Backend;
use anndata_hdf5::H5;
use pyo3::prelude::*;
use std::{path::PathBuf, collections::HashMap};
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

/// Read AnnDataSet object.
///
/// Read AnnDataSet from .h5ads file. If the file paths stored in AnnDataSet
/// object are relative paths, it will look for component .h5ad files in .h5ads file's parent directory.
///
/// Parameters
/// ----------
/// filename: Path
///     File name.
/// update_data_locations: Mapping[str, str]
///     If provided, locations of component anndata files will be updated.
/// mode: str
///     "r": Read-only mode; "r+": can modify annotation file but not component anndata files.
/// backend: Literal['hdf5'] | None
///
/// Returns
/// -------
/// AnnDataSet
#[pyfunction(update_data_locations = "None", mode = "\"r+\"")]
#[pyo3(text_signature = "(filename, update_data_locations, mode, backend, /)")]
pub fn read_dataset(
    filename: PathBuf,
    update_data_locations: Option<HashMap<String, String>>,
    mode: &str,
    backend: Option<&str>,
) -> Result<AnnDataSet> {
    match backend.unwrap_or(H5::NAME) {
        H5::NAME => {
            let file = match mode {
                "r" => H5::open(filename)?,
                "r+" => H5::open_rw(filename)?,
                _ => panic!("Unkown mode"),
            };
            Ok(anndata::AnnDataSet::<H5>::open(file, update_data_locations)?.into())
        },
        _ => todo!(),
    }
}