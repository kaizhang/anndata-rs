use crate::container::{PyArrayElem, PyAxisArrays, PyDataFrameElem, PyElem, PyElemCollection};
use crate::data::{to_select_elem, to_select_info, PyArrayData, PyData, PyDataFrame};
use crate::anndata::PyAnnData;

use anndata;
use anndata::container::Slot;
use anndata::data::{DataFrameIndex, SelectInfoElem};
use anndata::{AnnDataOp, ArrayData, Backend, Data};
use anndata_hdf5::H5;
use anyhow::{bail, Result};
use downcast_rs::{impl_downcast, Downcast};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;

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
        self.0.extract().map(|adata| {
            *adata
                .into_any()
                .downcast::<anndata::AnnData<B>>()
                .expect("downcast failed")
        })
    }

    pub fn open(filename: PathBuf, mode: &str, backend: Option<&str>) -> Result<Self> {
        match backend.unwrap_or(H5::NAME) {
            H5::NAME => {
                let file = match mode {
                    "r" => H5::open(filename)?,
                    "r+" => H5::open_rw(filename)?,
                    _ => bail!("Unknown mode: {}", mode),
                };
                anndata::AnnData::<H5>::open(file).map(|adata| adata.into())
            }
            x => bail!("Unknown backend: {}", x),
        }
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
        backend = "H5::NAME"
    )]
    pub fn new(
        filename: PathBuf,
        X: Option<PyArrayData>,
        n_obs: Option<usize>,
        n_vars: Option<usize>,
        obs: Option<PyDataFrame>,
        var: Option<PyDataFrame>,
        obsm: Option<HashMap<String, PyArrayData>>,
        varm: Option<HashMap<String, PyArrayData>>,
        uns: Option<HashMap<String, PyData>>,
        backend: &str,
    ) -> Result<Self> {
        let adata: AnnData = match backend {
            H5::NAME => {
                anndata::AnnData::<H5>::new(filename, n_obs.unwrap_or(0), n_vars.unwrap_or(0))?
                    .into()
            }
            backend => bail!("Unknown backend: {}", backend),
        };

        if X.is_some() {
            adata.set_x(X)?;
        }
        if obs.is_some() {
            adata.set_obs(obs)?;
        }
        if var.is_some() {
            adata.set_var(var)?;
        }
        if obsm.is_some() {
            adata.set_obsm(obsm)?;
        }
        if varm.is_some() {
            adata.set_varm(varm)?;
        }
        if uns.is_some() {
            adata.set_uns(uns)?;
        }
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
    pub fn get_x(&self) -> Option<PyArrayElem> {
        self.0.inner().get_x()
    }
    #[setter(X)]
    pub fn set_x(&self, data: Option<PyArrayData>) -> Result<()> {
        self.0.inner().set_x(data)
    }

    /// Observation annotations.
    ///
    /// Returns
    /// -------
    /// PyDataFrameElem
    #[getter(obs)]
    fn get_obs(&self) -> Option<PyDataFrameElem> {
        self.0.inner().get_obs()
    }
    #[setter(obs)]
    fn set_obs(&self, obs: Option<PyDataFrame>) -> Result<()> {
        self.0.inner().set_obs(obs)
    }

    /// Variable annotations.
    ///
    /// Returns
    /// -------
    /// PyDataFrameElem
    #[getter(var)]
    fn get_var(&self) -> Option<PyDataFrameElem> {
        self.0.inner().get_var()
    }
    #[setter(var)]
    fn set_var(&self, var: Option<PyDataFrame>) -> Result<()> {
        self.0.inner().set_var(var)
    }

    /// Unstructured annotation (ordered dictionary).
    ///
    /// Returns
    /// -------
    /// PyElemCollection
    #[getter(uns)]
    pub fn get_uns(&self) -> Option<PyElemCollection> {
        self.0.inner().get_uns()
    }
    #[setter(uns)]
    pub fn set_uns(&self, uns: Option<HashMap<String, PyData>>) -> Result<()> {
        self.0.inner().set_uns(uns)
    }

    #[getter(obsm)]
    pub fn get_obsm(&self) -> Option<PyAxisArrays> {
        self.0.inner().get_obsm()
    }
    #[setter(obsm)]
    pub fn set_obsm(&self, obsm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.inner().set_obsm(obsm)
    }

    #[getter(obsp)]
    pub fn get_obsp(&self) -> Option<PyAxisArrays> {
        self.0.inner().get_obsp()
    }
    #[setter(obsp)]
    pub fn set_obsp(&self, obsp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.inner().set_obsp(obsp)
    }

    #[getter(varm)]
    pub fn get_varm(&self) -> Option<PyAxisArrays> {
        self.0.inner().get_varm()
    }
    #[setter(varm)]
    pub fn set_varm(&self, varm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.inner().set_varm(varm)
    }

    #[getter(varp)]
    pub fn get_varp(&self) -> Option<PyAxisArrays> {
        self.0.inner().get_varp()
    }
    #[setter(varp)]
    pub fn set_varp(&self, varp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.inner().set_varp(varp)
    }

    /// Subsetting the AnnData object.
    ///
    /// Parameters
    /// ----------
    /// obs_indices
    ///     obs indices
    /// var_indices
    ///     var indices
    /// out: Path | None
    ///     File name of the output `.h5ad` file. If provided, the result will be
    ///     saved to a new file and the original AnnData object remains unchanged.
    /// backend: str | None
    ///
    /// Returns
    /// -------
    /// Optional[AnnData]
    #[pyo3(text_signature = "($self, obs_indices, var_indices, out, backend)")]
    pub fn subset(
        &self,
        obs_indices: Option<&PyAny>,
        var_indices: Option<&PyAny>,
        out: Option<PathBuf>,
        backend: Option<&str>,
    ) -> Result<Option<AnnData>> {
        let i = obs_indices
            .map(|x| to_select_elem(x, self.n_obs()).unwrap())
            .unwrap_or(SelectInfoElem::full());
        let j = var_indices
            .map(|x| to_select_elem(x, self.n_vars()).unwrap())
            .unwrap_or(SelectInfoElem::full());
        self.0.inner().subset([i, j].as_slice(), out, backend)
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

    #[getter]
    pub fn backend(&self) -> String {
        self.0.inner().backend().to_string()
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

    /// Write .h5ad-formatted hdf5 file.
    ///
    /// Parameters
    /// ----------
    /// filename: Path
    ///     File name of the output `.h5ad` file.
    /// backend: str | None
    #[pyo3(text_signature = "($self, filename, backend)")]
    pub fn write(&self, filename: PathBuf, backend: Option<&str>) -> Result<()> {
        self.0.inner().write(filename, backend)
    }

    /// Copy the AnnData object.
    ///
    /// Parameters
    /// ----------
    /// filename
    ///     File name of the output `.h5ad` file.
    /// backend: str | None
    ///
    /// Returns
    /// -------
    /// AnnData
    #[pyo3(text_signature = "($self, filename, backend)")]
    fn copy(&self, filename: PathBuf, backend: Option<&str>) -> Result<Self> {
        self.0.inner().copy(filename, backend)
    }

    /// Return a new AnnData object with all backed arrays loaded into memory.
    ///
    /// Returns
    /// -------
    /// AnnData
    #[pyo3(text_signature = "($self)")]
    pub fn to_memory<'py>(&self, py: Python<'py>) -> Result<PyAnnData<'py>> {
        self.0.inner().to_memory(py)
    }

    fn __repr__(&self) -> String {
        if self.is_closed() {
            "Closed AnnData object".to_string()
        } else {
            self.0.inner().show()
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

trait AnnDataTrait: Send + Downcast {
    fn shape(&self) -> (usize, usize);
    fn obs_names(&self) -> Vec<String>;
    fn set_obs_names(&self, names: &PyAny) -> Result<()>;
    fn var_names(&self) -> Vec<String>;
    fn set_var_names(&self, names: &PyAny) -> Result<()>;

    fn get_x(&self) -> Option<PyArrayElem>;
    fn get_obs(&self) -> Option<PyDataFrameElem>;
    fn get_var(&self) -> Option<PyDataFrameElem>;
    fn get_uns(&self) -> Option<PyElemCollection>;
    fn get_obsm(&self) -> Option<PyAxisArrays>;
    fn get_obsp(&self) -> Option<PyAxisArrays>;
    fn get_varm(&self) -> Option<PyAxisArrays>;
    fn get_varp(&self) -> Option<PyAxisArrays>;

    fn set_x(&self, data: Option<PyArrayData>) -> Result<()>;
    fn set_obs(&self, obs: Option<PyDataFrame>) -> Result<()>;
    fn set_var(&self, var: Option<PyDataFrame>) -> Result<()>;
    fn set_uns(&self, uns: Option<HashMap<String, PyData>>) -> Result<()>;
    fn set_obsm(&self, obsm: Option<HashMap<String, PyArrayData>>) -> Result<()>;
    fn set_obsp(&self, obsp: Option<HashMap<String, PyArrayData>>) -> Result<()>;
    fn set_varm(&self, varm: Option<HashMap<String, PyArrayData>>) -> Result<()>;
    fn set_varp(&self, varp: Option<HashMap<String, PyArrayData>>) -> Result<()>;

    fn subset(
        &self,
        slice: &[SelectInfoElem],
        out: Option<PathBuf>,
        backend: Option<&str>,
    ) -> Result<Option<AnnData>>;

    fn write(&self, filename: PathBuf, backend: Option<&str>) -> Result<()>;
    fn copy(&self, filename: PathBuf, backend: Option<&str>) -> Result<AnnData>;
    fn to_memory<'py>(&self, py: Python<'py>) -> Result<PyAnnData<'py>>;

    fn filename(&self) -> PathBuf;
    fn backend(&self) -> &str;
    fn show(&self) -> String;
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
        let obs_names: Result<DataFrameIndex> =
            names.iter()?.map(|x| Ok(x?.extract::<String>()?)).collect();
        AnnDataOp::set_obs_names(self, obs_names?)
    }

    fn var_names(&self) -> Vec<String> {
        AnnDataOp::var_names(self)
    }

    fn set_var_names(&self, names: &PyAny) -> Result<()> {
        let var_names: Result<DataFrameIndex> =
            names.iter()?.map(|x| Ok(x?.extract::<String>()?)).collect();
        AnnDataOp::set_var_names(self, var_names?)
    }

    fn get_x(&self) -> Option<PyArrayElem> {
        let x = self.get_x();
        if x.is_empty() {
            None
        } else {
            Some(x.clone().into())
        }
    }
    fn get_obs(&self) -> Option<PyDataFrameElem> {
        let obs = self.get_obs();
        if obs.is_empty() {
            None
        } else {
            Some(obs.clone().into())
        }
    }
    fn get_var(&self) -> Option<PyDataFrameElem> {
        let var = self.get_var();
        if var.is_empty() {
            None
        } else {
            Some(var.clone().into())
        }
    }
    fn get_uns(&self) -> Option<PyElemCollection> {
        let uns = self.get_uns();
        if uns.is_empty() {
            None
        } else {
            Some(uns.clone().into())
        }
    }
    fn get_obsm(&self) -> Option<PyAxisArrays> {
        let obsm = self.get_obsm();
        if obsm.is_empty() {
            None
        } else {
            Some(obsm.clone().into())
        }
    }
    fn get_obsp(&self) -> Option<PyAxisArrays> {
        let obsp = self.get_obsp();
        if obsp.is_empty() {
            None
        } else {
            Some(obsp.clone().into())
        }
    }
    fn get_varm(&self) -> Option<PyAxisArrays> {
        let varm = self.get_varm();
        if varm.is_empty() {
            None
        } else {
            Some(varm.clone().into())
        }
    }
    fn get_varp(&self) -> Option<PyAxisArrays> {
        let varp = self.get_varp();
        if varp.is_empty() {
            None
        } else {
            Some(varp.clone().into())
        }
    }

    fn set_x(&self, data: Option<PyArrayData>) -> Result<()> {
        if let Some(d) = data {
            AnnDataOp::set_x::<ArrayData>(self, d.into())?;
        } else {
            self.del_x()?;
        }
        Ok(())
    }
    fn set_obs(&self, obs: Option<PyDataFrame>) -> Result<()> {
        if let Some(o) = obs {
            AnnDataOp::set_obs(self, o.into())?;
        } else {
            self.del_obs()?;
        }
        Ok(())
    }
    fn set_var(&self, var: Option<PyDataFrame>) -> Result<()> {
        if let Some(v) = var {
            AnnDataOp::set_var(self, v.into())?;
        } else {
            self.del_var()?;
        }
        Ok(())
    }
    fn set_uns(&self, uns: Option<HashMap<String, PyData>>) -> Result<()> {
        if let Some(u) = uns {
            AnnDataOp::set_uns(self, u.into_iter().map(|(k, v)| (k, v.into())))?;
        } else {
            self.del_uns()?;
        }
        Ok(())
    }
    fn set_obsm(&self, obsm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        if let Some(o) = obsm {
            AnnDataOp::set_obsm(self, o.into_iter().map(|(k, v)| (k, v.into())))?;
        } else {
            self.del_obsm()?;
        }
        Ok(())
    }
    fn set_obsp(&self, obsp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        if let Some(o) = obsp {
            AnnDataOp::set_obsp(self, o.into_iter().map(|(k, v)| (k, v.into())))?;
        } else {
            self.del_obsp()?;
        }
        Ok(())
    }
    fn set_varm(&self, varm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        if let Some(v) = varm {
            AnnDataOp::set_varm(self, v.into_iter().map(|(k, v)| (k, v.into())))?;
        } else {
            self.del_varm()?;
        }
        Ok(())
    }
    fn set_varp(&self, varp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        if let Some(v) = varp {
            AnnDataOp::set_varp(self, v.into_iter().map(|(k, v)| (k, v.into())))?;
        } else {
            self.del_varp()?;
        }
        Ok(())
    }

    fn subset(
        &self,
        slice: &[SelectInfoElem],
        out: Option<PathBuf>,
        backend: Option<&str>,
    ) -> Result<Option<AnnData>> {
        if let Some(out) = out {
            match backend.unwrap_or("hdf5") {
                "hdf5" => {
                    self.write_select::<H5, _, _>(slice, &out)?;
                    Ok(Some(AnnData::open(out, "r+", backend)?))
                }
                x => bail!("Unsupported backend: {}", x),
            }
        } else {
            self.subset(slice)?;
            Ok(None)
        }
    }

    fn write(&self, filename: PathBuf, backend: Option<&str>) -> Result<()> {
        match backend.unwrap_or(H5::NAME) {
            H5::NAME => self.write::<H5, _>(filename),
            x => bail!("Unsupported backend: {}", x),
        }
    }

    fn copy(&self, filename: PathBuf, backend: Option<&str>) -> Result<AnnData> {
        AnnDataTrait::write(self, filename.clone(), backend)?;
        AnnData::open(filename, "r+", backend)
    }

    fn to_memory<'py>(&self, py: Python<'py>) -> Result<PyAnnData<'py>> {
        Ok(PyAnnData::from_anndata(py, self)?)
    }

    fn filename(&self) -> PathBuf {
        self.filename()
    }

    fn backend(&self) -> &str {
        B::NAME
    }

    fn show(&self) -> String {
        format!("{}", self)
    }
}
