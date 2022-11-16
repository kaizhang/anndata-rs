use crate::{AnnData, AnnDataSet};

use anndata_rs::anndata;
use anyhow::Result;
use pyo3::{prelude::*, Python};
use std::{collections::HashMap, io::BufReader, path::PathBuf};

#[derive(FromPyObject)]
pub enum IOMode<'a> {
    Mode(&'a str),
    Backed(bool),
}

/// Read `.h5ad`-formatted hdf5 file.
///
/// Parameters
/// ----------
///
/// filename: Path
///     File name of data file.
/// backed: Literal['r', 'r+'] | bool | None
///     Default is `r+`.
///     If `'r'`, the file is opened in read-only mode.
///     If `'r+'` or `true`, the file is opened in read/write mode.
///     If `false` or `None`, the AnnData object is read into memory.
#[pyfunction(backed = "IOMode::Mode(\"r+\")")]
#[pyo3(text_signature = "(filename, backed, /)")]
pub fn read<'py>(filename: PathBuf, backed: Option<IOMode<'py>>) -> Result<PyObject> {
    Python::with_gil(|py| match backed {
        Some(IOMode::Mode("r")) => {
            let file = hdf5::File::open(filename)?;
            Ok(AnnData::wrap(anndata::AnnData::read(file)?).into_py(py))
        },
        Some(IOMode::Mode("r+")) | Some(IOMode::Backed(true)) => {
            let file = hdf5::File::open_rw(filename)?;
            Ok(AnnData::wrap(anndata::AnnData::read(file)?).into_py(py))
        }
        None | Some(IOMode::Backed(false)) => {
            Ok(PyModule::import(py, "anndata")?.getattr("read_h5ad")?.call1((filename,))?.to_object(py))
        }
        _ => panic!("Unkown mode"),
    })
}

/// Read Matrix Market file.
///
/// Parameters
/// ----------
///
/// mtx_file
///     File name of the input matrix market file.
/// file 
///     File name of the output ".h5ad" file.
/// sorted
///     Indicate whether the entries in the matrix market file have been
///     sorted by row and column indices. When the data is sorted, only
///     a small amount of (constant) memory will be used.
#[pyfunction(sorted = "false")]
#[pyo3(text_signature = "(mtx_file, file, sorted)")]
pub fn read_mtx<'py>(py: Python<'py>, mtx_file: PathBuf, file: PathBuf, sorted: bool) -> Result<AnnData> {
    let anndata = AnnData::new(py, file, None, None, None, None, None, None, None, None)?;
    let mut reader = BufReader::new(crate::utils::open_file(mtx_file));
    anndata.0.inner().import_matrix_market(&mut reader, sorted)?;
    Ok(anndata)
}

#[pyfunction(has_header = true, index_column = "None", delimiter = "b','")]
#[pyo3(text_signature = "(csv_file, file, has_header, index_column, delimiter, /)")]
pub fn read_csv(
    csv_file: PathBuf,
    file: PathBuf,
    has_header: bool,
    index_column: Option<usize>,
    delimiter: u8,
) -> Result<AnnData>
{
    let anndata = anndata::AnnData::new(file, 0, 0)?;
    anndata.import_csv(csv_file, has_header, index_column, delimiter)?;
    Ok(AnnData::wrap(anndata))
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
/// 
/// Returns
/// -------
/// AnnDataSet
#[pyfunction(update_data_locations="None", mode="\"r+\"")]
#[pyo3(text_signature = "(filename, update_data_locations, mode, /)")]
pub fn read_dataset(
    filename: PathBuf,
    update_data_locations: Option<HashMap<String, String>>,
    mode: &str,
) -> Result<AnnDataSet> {
    let file = match mode {
        "r" => hdf5::File::open(filename)?,
        "r+" => hdf5::File::open_rw(filename)?,
        _ => panic!("Unkown mode"),
    };
    let data = anndata::AnnDataSet::read(file, update_data_locations)?;
    Ok(AnnDataSet::wrap(data))
}