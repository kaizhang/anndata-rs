use crate::element::*;
use crate::iterator::{PyChunkedMatrix, PyStackedChunkedMatrix};
use crate::utils::{
    conversion::{PyToRust, RustToPy},
    instance::*,
    to_indices,
};

use anndata::{Data, ArrayData, AnnDataOp};
use anndata::data::DataFrameIndex;
use anndata::container::{Slot, Inner};
use anndata_hdf5::H5;
use anyhow::Result;
use paste::paste;
use polars::frame::DataFrame;
use pyo3::{
    exceptions::{PyException, PyTypeError},
    prelude::*,
    types::IntoPyDict,
    PyResult, Python,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{collections::HashMap, ops::Deref, path::PathBuf};

impl AnnData {
    pub fn inner(&self) -> Inner<'_, anndata::AnnData> {
        self.0.inner()
    }

    pub fn wrap(anndata: anndata::AnnData) -> Self {
        AnnData(Slot::new(anndata))
    }

    fn normalize_index<'py>(
        &self,
        py: Python<'py>,
        indices: &'py PyAny,
        axis: u8,
    ) -> PyResult<Option<Vec<usize>>> {
        match indices
            .iter()?
            .map(|x| x.unwrap().extract())
            .collect::<PyResult<Vec<String>>>()
        {
            Ok(names) => {
                if axis == 0 {
                    Ok(Some(self.obs_ix(names)))
                } else {
                    Ok(Some(self.var_ix(names)))
                }
            }
            _ => {
                let length = if axis == 0 {
                    self.n_obs()
                } else {
                    self.n_vars()
                };
                Ok(to_indices(py, indices, length)?.0)
            }
        }
    }
}

/// Lazily concatenated AnnData objects.
#[pyclass]
#[repr(transparent)]
pub struct StackedAnnData(pub Slot<anndata::StackedAnnData>);

#[pymethods]
impl StackedAnnData {
    /// :class:`.PyStackedDataFrame`.
    #[getter(obs)]
    fn get_obs(&self) -> PyStackedDataFrame {
        PyStackedDataFrame(self.0.inner().obs.clone())
    }

    /// :class:`.PyStackedAxisArrays`.
    #[getter(obsm)]
    fn get_obsm(&self) -> PyStackedAxisArrays {
        PyStackedAxisArrays(self.0.inner().get_obsm().clone())
    }

    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl AnnDataSet {
    pub fn inner(&self) -> Inner<'_, anndata::AnnDataSet> {
        self.0.inner()
    }

    pub fn wrap(anndata: anndata::AnnDataSet) -> Self {
        AnnDataSet(Slot::new(anndata))
    }

    fn normalize_index<'py>(
        &self,
        py: Python<'py>,
        indices: &'py PyAny,
        axis: u8,
    ) -> PyResult<(Option<Vec<usize>>, bool)> {
        match indices
            .iter()?
            .map(|x| x.unwrap().extract())
            .collect::<PyResult<Vec<String>>>()
        {
            Ok(names) => {
                if axis == 0 {
                    Ok((Some(self.obs_ix(names)), false))
                } else {
                    Ok((Some(self.var_ix(names)), false))
                }
            }
            _ => {
                let length = if axis == 0 {
                    self.n_obs()
                } else {
                    self.n_vars()
                };
                to_indices(py, indices, length)
            }
        }
    }

    fn to_pyanndata<'py>(&self, py: Python<'py>, copy_X: bool) -> Result<PyAnnData<'py>> {
        let anndata = PyAnnData::from_anndata(py, self.inner().as_adata())?;
        if copy_X {
            anndata.set_x(self.inner().read_x()?)?;
        }
        Ok(anndata)
    }
}

#[derive(FromPyObject)]
pub enum AnnDataFile {
    Path(PathBuf),
    Data(AnnData),
}

#[pymethods]
impl AnnDataSet {
    #[pyo3(text_signature = "($self, names)")]
    pub fn var_ix(&self, names: Vec<String>) -> Vec<usize> {
        self.0.inner().var_ix(&names).unwrap()
    }

    #[pyo3(text_signature = "($self, names)")]
    pub fn obs_ix(&self, names: Vec<String>) -> Vec<usize> {
        self.0.inner().obs_ix(&names).unwrap()
    }

    /// View into the component AnnData objects.
    ///
    /// Returns
    /// -------
    /// StackedAnnData
    #[getter(adatas)]
    pub fn adatas(&self) -> StackedAnnData {
        StackedAnnData(self.0.inner().get_inner_adatas().clone())
    }

    /// Copy the AnnDataSet object to a new location.
    ///
    /// Copying AnnDataSet object will copy both the object itself and assocated
    /// AnnData objects.
    ///
    /// Parameters
    /// ----------
    /// dirname: Path
    ///     Name of the directory used to store the result.
    ///
    /// Returns
    /// -------
    /// AnnDataSet
    #[pyo3(text_signature = "($self, dirname)")]
    pub fn copy(&self, dirname: PathBuf) -> Result<Self> {
        let (adata, _) = self.0.inner().copy(None, None, dirname)?;
        Ok(AnnDataSet::wrap(adata))
    }

    /// Close the AnnDataSet object.
    #[pyo3(text_signature = "($self)")]
    pub fn close(&self) {
        if let Some(dataset) = self.0.extract() {
            dataset.close().unwrap();
        }
    }

    /// If the AnnDataSet object has been closed.
    ///
    /// Returns
    /// -------
    /// bool
    #[pyo3(text_signature = "($self)")]
    pub fn is_closed(&self) -> bool {
        self.0.inner().0.is_none()
    }

    /// Convert the AnnDataSet object to an AnnData object.
    ///
    /// Parameters
    /// ----------
    /// file: Optional[Path]
    ///     If provided, the resultant AnnData object will be backed. Default: None.
    /// copy_X: bool
    ///     Whether to copy the `.X` field. Default: True.
    ///
    /// Returns
    /// -------
    /// AnnData
    #[args(file = "None", copy_X = "true")]
    #[pyo3(text_signature = "($self, file, copy_X, /)")]
    pub fn to_adata<'py>(
        &self,
        py: Python<'py>,
        file: Option<PathBuf>,
        copy_X: bool,
    ) -> Result<PyObject> {
        if let Some(_) = file {
            unimplemented!("saving backed anndata");
        } else {
            Ok(self.to_pyanndata(py, copy_X)?.to_object(py))
        }
    }

    fn __repr__(&self) -> String {
        if self.is_closed() {
            "Closed AnnDataSet object".to_string()
        } else {
            format!("{}", self.0.inner().deref())
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}