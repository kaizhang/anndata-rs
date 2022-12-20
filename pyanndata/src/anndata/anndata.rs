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

#[pymethods]
impl AnnData {
    /// Return an iterator over the rows of the data matrix X.
    ///
    /// Parameters
    /// ----------
    /// chunk_size : int
    ///     Row size of a single chunk. Default: 500.
    ///
    /// Returns
    /// -------
    /// PyChunkedMatrix
    #[args(chunk_size = "500")]
    #[pyo3(text_signature = "($self, chunk_size, /)")]
    #[pyo3(name = "chunked_X")]
    pub fn chunked_x(&self, chunk_size: usize) -> PyChunkedMatrix {
        self.get_x().chunked(chunk_size)
    }



def_df_accessor!(AnnData, { obs, var });

def_arr_accessor!(
    AnnData,
    PyAxisArrays,
    Option<HashMap<String, &'py PyAny>>,
    { obsm, obsp, varm, varp }
);

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

    /// Names of observations.
    ///
    /// Returns
    /// -------
    /// list[str]
    #[getter]
    pub fn obs_names(&self) -> Vec<String> {
        self.0.inner().obs_names()
    }

    #[setter(obs_names)]
    pub fn set_obs_names(&self, names: &PyAny) -> PyResult<()> {
        let obs_names: PyResult<DataFrameIndex> = names
            .iter()?
            .map(|x| x.unwrap().extract::<String>())
            .collect();
        self.0.inner().set_obs_names(obs_names?).unwrap();
        Ok(())
    }

    #[pyo3(text_signature = "($self, names)")]
    pub fn obs_ix(&self, names: Vec<String>) -> Vec<usize> {
        self.0.inner().obs_ix(&names).unwrap()
    }

    /// Vertically concatenated data matrices of shape `n_obs` x `n_vars`.
    ///
    /// Returns
    /// -------
    /// PyStackedMatrixElem
    #[getter(X)]
    pub fn get_x(&self) -> Option<PyStackedMatrixElem> {
        self.0
            .inner()
            .get_inner_adatas()
            .lock()
            .as_ref()
            .map(|x| PyStackedMatrixElem(x.get_x().clone()))
    }

    /// Unstructured annotation (ordered dictionary).
    ///
    /// Returns
    /// -------
    /// PyElemCollection
    #[getter(uns)]
    pub fn get_uns(&self) -> PyElemCollection {
        PyElemCollection(self.0.inner().get_uns().clone())
    }

    #[setter(uns)]
    pub fn set_uns<'py>(
        &self,
        py: Python<'py>,
        uns: Option<HashMap<String, &'py PyAny>>,
    ) -> PyResult<()> {
        let data: PyResult<_> = uns
            .map(|mut x| x.drain().map(|(k, v)| Ok((k, v.into_rust(py)?))).collect())
            .transpose();
        self.0.inner().set_uns(data?).unwrap();
        Ok(())
    }

    /// Subsetting the AnnDataSet object.
    ///
    /// Note
    /// ----
    /// AnnDataSet will not move data across underlying AnnData objects. So the
    /// orders of rows in the resultant AnnDataSet object may not be consistent
    /// with the input `obs_indices`. This function will return a vector that can
    /// be used to reorder the `obs_indices` to match the final order of rows in
    /// the AnnDataSet.
    ///
    /// Parameters
    /// ----------
    /// obs_indices
    ///     obs indices
    /// var_indices
    ///     var indices
    /// out
    ///     Name of the directory used to store the new files. If provided,
    ///     the result will be saved to the directory and the original files
    ///     remains unchanged.
    ///
    /// Returns
    /// -------
    /// None | list[int] | Tuple[AnnDataSet, list[int]]
    ///     If `out` is not `None`, a new AnnDataSet object will be returned.
    ///     Otherwise, the operation will be performed inplace.
    ///     If the order of input `obs_indices` has been changed, it will
    ///     return the indices that would sort the `obs_indices` array.
    #[pyo3(text_signature = "($self, obs_indices, var_indices, out)")]
    pub fn subset<'py>(
        &self,
        py: Python<'py>,
        obs_indices: Option<&'py PyAny>,
        var_indices: Option<&'py PyAny>,
        out: Option<&str>,
    ) -> Result<PyObject> {
        let (i, obs_is_boolean) =
            obs_indices.map_or((None, false), |x| self.normalize_index(py, x, 0).unwrap());
        let j = var_indices.and_then(|x| self.normalize_index(py, x, 1).unwrap().0);
        match out {
            None => {
                let idx_order = self
                    .0
                    .inner()
                    .subset(i.as_ref().map(Vec::as_slice), j.as_ref().map(Vec::as_slice))?;
                let res = if obs_is_boolean { None } else { idx_order };
                Ok(res.to_object(py))
            }
            Some(dir) => {
                let (adata, idx_order) = self.0.inner().copy(
                    i.as_ref().map(Vec::as_slice),
                    j.as_ref().map(Vec::as_ref),
                    dir,
                )?;
                let ann = AnnDataSet::wrap(adata).into_py(py);
                let res = if obs_is_boolean {
                    (ann, None)
                } else {
                    (ann, idx_order)
                };
                Ok(res.to_object(py))
            }
        }
    }

    /// Return an iterator over the rows of the data matrix X.
    ///
    /// Parameters
    /// ----------
    /// chunk_size : int
    ///     Row size of a single chunk. Default: 500.
    #[args(chunk_size = "500")]
    #[pyo3(text_signature = "($self, chunk_size, /)")]
    #[pyo3(name = "chunked_X")]
    pub fn chunked_x(&self, chunk_size: usize) -> PyStackedChunkedMatrix {
        self.get_x().expect("X is empty").chunked(chunk_size)
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

    /// Whether the AnnDataSet object is backed. This is always true.
    ///
    /// Returns
    /// -------
    /// bool
    #[getter]
    pub fn isbacked(&self) -> bool {
        true
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