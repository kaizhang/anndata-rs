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
#[pyfunction]
#[pyo3(
    signature = (filename, backed="r+", backend=None),
    text_signature = "(filename, backed='r+', backend=None)",
)]
pub fn read<'py>(py: Python<'py>, filename: PathBuf, backed: Option<&str>, backend: Option<&str>) -> Result<PyObject> {
    let adata = match backed {
        Some(m) => AnnData::new_from(filename, m, backend).unwrap().into_py(py),
        None => PyModule::import_bound(py, "anndata")?
            .getattr("read_h5ad")?
            .call1((filename,))?
            .to_object(py),
    };
    Ok(adata)
}

/// Read Matrix Market file.
///
/// Parameters
/// ----------
///
/// mtx_file
///     File name of the input matrix market file.
/// obs_names
///     File that stores the observation names.
/// var_names
///     File that stores the variable names.
/// file
///     File name of the output ".h5ad" file.
/// backend
///     Backend to use for writing the output file.
/// sorted
///     If true, the input matrix is assumed to be sorted by rows.
///     Sorted input matrix can be read faster.
#[pyfunction]
#[pyo3(
    signature = (mtx_file, *, obs_names=None, var_names=None, file=None, backend=None, sorted=false),
    text_signature = "(mtx_file, *, obs_names=None, var_names=None, file=None, backend=None, sorted=False)",
)]
pub fn read_mtx(
    py: Python<'_>,
    mtx_file: PathBuf,
    obs_names: Option<PathBuf>,
    var_names: Option<PathBuf>,
    file: Option<PathBuf>,
    backend: Option<&str>,
    sorted: bool,
) -> Result<PyObject> {
    let mut reader = anndata::reader::MMReader::from_path(mtx_file)?;
    if let Some(obs_names) = obs_names {
        reader = reader.obs_names(obs_names)?;
    }
    if let Some(var_names) = var_names {
        reader = reader.var_names(var_names)?;
    }
    if sorted {
        reader = reader.is_sorted();
    }
    if let Some(file) =  file {
        match backend.unwrap_or(H5::NAME) {
            H5::NAME => {
                let adata = anndata::AnnData::<H5>::new(file)?;
                reader.finish(&adata)?;
                Ok(AnnData::from(adata).into_py(py))
            },
            backend => todo!("Backend {} is not supported", backend),
        }
    } else {
        let adata = PyAnnData::new(py)?;
        reader.finish(&adata)?;
        Ok(adata.to_object(py))
    }
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
/// adata_files_update: Mapping[str, Path] | Path | None
///     AnnDataSet internally stores links to component anndata files.
///     You can find this information in `.uns['AnnDataSet']`.
///     These links may be invalid if the anndata files are moved to a different location.
///     This parameter provides a way to update the locations of component anndata files.
///     The value of this parameter can be either a mapping from component anndata file names to their new locations,
///     or a directory containing component anndata files.
/// mode: str
///     "r": Read-only mode; "r+": can modify annotation file but not component anndata files.
/// backend: Literal['hdf5'] | None
///
/// Returns
/// -------
/// AnnDataSet
#[pyfunction]
#[pyo3(
    signature = (filename, *, adata_files_update=None, mode="r+", backend=None),
    text_signature = "(filename, *, adata_files_update=None, mode='r+', backend=None)",
)]
pub fn read_dataset(
    filename: PathBuf,
    adata_files_update: Option<LocationUpdate>,
    mode: &str,
    backend: Option<&str>,
) -> Result<AnnDataSet> {
    let adata_files_update = match adata_files_update {
        Some(LocationUpdate::Map(map)) => Some(Ok(map)),
        Some(LocationUpdate::Dir(dir)) => Some(Err(dir)),
        None => None,
    };
    match backend.unwrap_or(H5::NAME) {
        H5::NAME => {
            let file = match mode {
                "r" => H5::open(filename)?,
                "r+" => H5::open_rw(filename)?,
                _ => panic!("Unkown mode"),
            };
            Ok(anndata::AnnDataSet::<H5>::open(file, adata_files_update )?.into())
        },
        _ => todo!(),
    }
}

#[derive(FromPyObject)]
pub enum LocationUpdate {
    Map(HashMap<String, PathBuf>),
    Dir(PathBuf),
}