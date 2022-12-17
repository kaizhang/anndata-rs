use crate::container::{PyArrayElem, PyElem};
use crate::data::PyArrayData;

use anndata::data::DataFrameIndex;
use pyo3::prelude::*;
use anndata;
use anndata::{Backend, ArrayData, AnnDataOp};
use anndata::container::{Inner, Slot};
use anndata_hdf5::H5;
use std::collections::HashMap;
use std::ops::BitOrAssign;
use std::path::PathBuf;
use anyhow::{Result, bail};
use downcast_rs::{Downcast, impl_downcast};

/** An annotated data matrix.

    `AnnData` stores a data matrix `X` together with annotations of
    observations `obs` (`obsm`, `obsp`), variables `var` (`varm`, `varp`),
    and unstructured annotations `uns`.
    `AnnData` is stored as a HDF5 file. Opening/creating an AnnData object
    does not read data from the HDF5 file. Data is copied to memory only when
    individual element is requested (or when cache is turned on).

    Parameters
    ----------
    X
        A #observations × #variables data matrix. A view of the data is used if the
        data type matches, otherwise, a copy is made.
    obs
        Key-indexed one-dimensional observations annotation of length #observations.
    var
        Key-indexed one-dimensional variables annotation of length #variables.
    uns
        Key-indexed unstructured annotation.
    obsm
        Key-indexed multi-dimensional observations annotation of length #observations.
        If passing a :class:`~numpy.ndarray`, it needs to have a structured datatype.
    varm
        Key-indexed multi-dimensional variables annotation of length #variables.
        If passing a :class:`~numpy.ndarray`, it needs to have a structured datatype.
    filename
        Name of backing file.

    Note
    ----
    This is a backed AnnData. You can use :func:`~AnnData.to_memory` to convert
    it to an in-memory AnnData object. See
    `here <https://anndata.readthedocs.io/en/latest/index.html>`_
    for the documentation of in-memory AnnData objects.

    See Also
    --------
    read
    read_mtx
    read_csv
*/
#[pyclass]
#[pyo3(text_signature = "(*, filename, X, n_obs, n_vars, obs, var, obsm, varm, uns)")]
#[repr(transparent)]
#[derive(Clone)]
pub struct AnnData(Slot<Box<dyn AnnDataTrait>>);

impl AnnData {
    pub fn into_inner<B: Backend + 'static>(self) -> Option<anndata::AnnData<B>> {
        self.0.extract().map(|adata|
            *adata.into_any().downcast::<anndata::AnnData<B>>().expect("downcast failed")
        )
    }
}

impl<B: Backend + 'static> From<anndata::AnnData<B>> for AnnData {
    fn from(adata: anndata::AnnData<B>) -> Self {
        AnnData(Slot::new(Box::new(adata)))
    }
}

#[pymethods]
impl AnnData {
    #[new]
    #[args(
        "*",
        filename,
        X = "None",
        n_obs = "None",
        n_vars = "None",
        obs = "None",
        var = "None",
        obsm = "None",
        varm = "None",
        uns = "None",
        backend = "\"hdf5\"",
    )]
    pub fn new<'py>(
        py: Python<'py>,
        filename: PathBuf,
        X: Option<PyArrayData>,
        n_obs: Option<usize>,
        n_vars: Option<usize>,
        obs: Option<&'py PyAny>,
        var: Option<&'py PyAny>,
        obsm: Option<HashMap<String, &'py PyAny>>,
        varm: Option<HashMap<String, &'py PyAny>>,
        uns: Option<HashMap<String, &'py PyAny>>,
        backend: &str,
    ) -> Result<Self> {
        let adata: AnnData = match backend {
            "hdf5" => anndata::AnnData::<H5>::new(filename, n_obs.unwrap_or(0), n_vars.unwrap_or(0))?.into(),
            backend => bail!("Unknown backend: {}", backend),
        };

        if X.is_some() {
            adata.set_x(X)?;
        }
        /*
        if obs.is_some() {
            anndata.set_obs(py, obs)?;
        }
        if var.is_some() {
            anndata.set_var(py, var)?;
        }
        if obsm.is_some() {
            anndata.set_obsm(py, obsm)?;
        }
        if varm.is_some() {
            anndata.set_varm(py, varm)?;
        }
        if uns.is_some() {
            anndata.set_uns(py, uns)?;
        }
        */
        Ok(adata)
    }

    /// Shape of data matrix (`n_obs`, `n_vars`).
    ///
    /// Returns
    /// -------
    /// tuple[int, int]
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        self.0.inner().shape()
    }

    /// Number of observations.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub fn n_obs(&self) -> usize {
        self.shape().0
    }

    /// Number of variables/features.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub fn n_vars(&self) -> usize {
        self.shape().1
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
    pub fn set_obs_names(&self, names: &PyAny) -> Result<()> {
        self.0.inner().set_obs_names(names)
    }

    /// Names of variables.
    ///
    /// Returns
    /// -------
    /// list[str]
    #[getter]
    pub fn var_names(&self) -> Vec<String> {
        self.0.inner().var_names()
    }
    #[setter(var_names)]
    pub fn set_var_names(&self, names: &PyAny) -> Result<()> {
        self.0.inner().set_var_names(names)
    }

    /// Data matrix of shape n_obs × n_vars.
    ///
    /// Returns
    /// -------
    /// PyArrayElem
    #[getter(X)]
    pub fn get_x(&self) -> PyArrayElem {
        self.0.inner().get_x()
    }
    #[setter(X)]
    pub fn set_x(&self, data: Option<PyArrayData>) -> Result<()> {
        self.0.inner().set_x(data)
    }

    /// Filename of the backing .h5ad file.
    ///
    /// Returns
    /// -------
    /// Path
    #[getter]
    pub fn filename(&self) -> PathBuf {
        self.0.inner().filename()
    }

    /// Whether the AnnData object is backed. This is always true.
    ///
    /// Returns
    /// -------
    /// bool
    #[getter]
    pub fn isbacked(&self) -> bool {
        true
    }

    /// If the AnnData object has been closed.
    ///
    /// Returns
    /// -------
    /// bool
    #[pyo3(text_signature = "($self)")]
    pub fn is_closed(&self) -> bool {
        self.0.is_empty()
    }

    /// Close the AnnData object.
    #[pyo3(text_signature = "($self)")]
    pub fn close(&self) {
        self.0.drop();
    }
}

trait AnnDataTrait: Send + Downcast {
    fn shape(&self) -> (usize, usize);
    fn obs_names(&self) -> Vec<String>;
    fn set_obs_names(&self, names: &PyAny) -> Result<()>;
    fn var_names(&self) -> Vec<String>;
    fn set_var_names(&self, names: &PyAny) -> Result<()>;
    fn get_x(&self) -> PyArrayElem;
    fn set_x(&self, data: Option<PyArrayData>) -> Result<()>;
    fn filename(&self) -> PathBuf;
}
impl_downcast!(AnnDataTrait);

impl<B: Backend + 'static> AnnDataTrait for anndata::AnnData<B> {
    fn shape(&self) -> (usize, usize) {
        (self.n_obs(), self.n_vars())
    }

    fn obs_names(&self) -> Vec<String> {
        AnnDataOp::obs_names(self)
    }

    fn set_obs_names(&self, names: &PyAny) -> Result<()> {
      let obs_names: Result<DataFrameIndex> = names
            .iter()?
            .map(|x| Ok(x?.extract::<String>()?))
            .collect();
        AnnDataOp::set_obs_names(self, obs_names?)
    }

    fn var_names(&self) -> Vec<String> {
        AnnDataOp::var_names(self)
    }

    fn set_var_names(&self, names: &PyAny) -> Result<()> {
      let var_names: Result<DataFrameIndex> = names
            .iter()?
            .map(|x| Ok(x?.extract::<String>()?))
            .collect();
        AnnDataOp::set_var_names(self, var_names?)
    }

    fn get_x(&self) -> PyArrayElem {
        self.get_x().clone().into()
    }

    fn set_x(&self, data: Option<PyArrayData>) -> Result<()> {
        if let Some(d) = data {
            AnnDataOp::set_x::<ArrayData>(self, d.into())?;
        } else {
            self.del_x()?;
        }
        Ok(())
    }

    fn filename(&self) -> PathBuf {
        self.filename()
    }
}