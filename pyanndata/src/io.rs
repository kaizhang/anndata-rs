use crate::{AnnData, AnnDataSet};

use anndata_rs::anndata;
use anyhow::{Result, bail};
use pyo3::{prelude::*, Python};
use std::{collections::HashMap, io::BufReader, path::PathBuf};
use rayon::iter::{ParallelIterator, IntoParallelIterator};

#[derive(FromPyObject)]
pub enum IOMode<'a> {
    Mode(&'a str),
    Backed(bool),
}

#[derive(FromPyObject)]
pub enum AnnDataFile {
    Path(PathBuf),
    Data(AnnData),
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
pub fn read_mtx<'py>(py: Python<'py>, mtx_file: &'py PyAny, file: &'py PyAny, sorted: bool) -> Result<AnnData> {
    let anndata = AnnData::new(py, file.str()?.to_str()?, None, None, None, None, None, None, None, None, None, None)?;
    let mut reader = BufReader::new(crate::utils::open_file(mtx_file.str()?.to_str()?));
    anndata.0.inner().import_matrix_market(&mut reader, sorted)?;
    Ok(anndata)
}

#[pyfunction(has_header = true, index_column = "None", delimiter = "b','")]
#[pyo3(text_signature = "(csv_file, file, has_header, index_column, delimiter, /)")]
pub fn read_csv(
    csv_file: &PyAny,
    file: &PyAny,
    has_header: bool,
    index_column: Option<usize>,
    delimiter: u8,
) -> Result<AnnData>
{
    let anndata = anndata::AnnData::new(file.str()?.to_str()?, 0, 0)?;
    anndata.import_csv(csv_file.str()?.to_str()?, has_header, index_column, delimiter)?;
    Ok(AnnData::wrap(anndata))
}

/// Read and stack vertically multiple `.h5ad`-formatted hdf5 files.
///
/// Parameters
/// ----------
/// adatas: list[(str, Path)] | list[(str, AnnData)]
///     List of key and file name (or backed AnnData object) pairs.
/// file: Path
///     File name of the output file containing the AnnDataSet object.
/// add_key: str
///     The column name in obs to store the keys
/// 
/// Returns
/// -------
/// AnnDataSet
#[pyfunction(add_key= "\"sample\"")]
#[pyo3(text_signature = "(adatas, file, add_key, /)")]
pub fn create_dataset(
    adatas: Vec<(String, AnnDataFile)>,
    file: PathBuf,
    add_key: &str,
) -> Result<AnnDataSet> {
    let anndatas = adatas.into_par_iter().map(|(key, data_file)| {
        let adata = match data_file {
            AnnDataFile::Data(data) => data.0.extract().unwrap(),
            AnnDataFile::Path(path) => {
                let file = hdf5::File::open(path)?;
                anndata::AnnData::read(file)?
            },
        };
        Ok((key, adata))
    }).collect::<Result<_>>()?;
    Ok(AnnDataSet::wrap(anndata::AnnDataSet::new(anndatas, file, add_key)?))
}

/// Read AnnDataSet object.
/// 
/// Read AnnDataSet from .h5ads file. If the file paths stored in AnnDataSet
/// object are relative paths, it will look for component .h5ad files in .h5ads file's parent directory.
///
/// Parameters
/// ----------
/// filename
///     File name.
/// update_data_locations
///     Mapping[str, str]: If provided, locations of component anndata files will be updated.
/// mode
///     "r": Read-only mode; "r+": can modify annotation file but not component anndata files.
/// no_check
///     If True, do not check the validility of the file, recommended if you know
///     the file is valid and want faster loading time.
/// 
/// Returns
/// -------
/// AnnDataSet
#[pyfunction(update_data_locations="None", mode="\"r+\"")]
#[pyo3(text_signature = "(filename, update_data_locations, mode, /)")]
pub fn read_dataset(
    filename: &PyAny,
    update_data_locations: Option<HashMap<String, String>>,
    mode: &str,
) -> Result<AnnDataSet> {
    let file = match mode {
        "r" => hdf5::File::open(filename.str()?.to_str()?)?,
        "r+" => hdf5::File::open_rw(filename.str()?.to_str()?)?,
        _ => panic!("Unkown mode"),
    };
    let data = anndata::AnnDataSet::read(file, update_data_locations)?;
    Ok(AnnDataSet::wrap(data))
}