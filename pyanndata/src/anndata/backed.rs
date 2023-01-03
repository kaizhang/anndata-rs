use crate::container::{PyArrayElem, PyAxisArrays, PyDataFrameElem, PyElemCollection, PyChunkedArray};
use crate::data::{to_select_elem, PyArrayData, PyData, PyDataFrame};
use crate::anndata::PyAnnData;

use anndata;
use anndata::container::Slot;
use anndata::data::{DataFrameIndex, SelectInfoElem};
use anndata::{AnnDataOp, ArrayData, Backend};
use anndata_hdf5::H5;
use anyhow::{bail, Result};
use downcast_rs::{impl_downcast, Downcast};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::ops::Deref;

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
pub struct AnnData(Box<dyn AnnDataTrait>);

impl Clone for AnnData {
    fn clone(&self) -> Self {
        AnnData(self.0.clone_ref())
    }
}

impl AnnData {
    pub fn take_inner<B: Backend + 'static>(&self) -> Option<anndata::AnnData<B>> {
        self.0.downcast_ref::<Slot<anndata::AnnData<B>>>()
            .expect("downcast to anndata failed").extract()
    }

    pub fn inner_ref<B: Backend + 'static>(&self) -> anndata::container::Inner<'_, anndata::AnnData<B>> {
        self.0.downcast_ref::<Slot<anndata::AnnData<B>>>().expect("downcast to anndata failed").inner()
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

    fn select_obs(&self, ix: &PyAny) -> PyResult<SelectInfoElem> {
        let from_iter = ix.iter().and_then(|iter| 
            iter.map(|x| x.unwrap().extract::<String>()).collect::<PyResult<Vec<_>>>()
        ).and_then(|names| {
            let index = self.0.obs_names();
            names.into_iter().map(|name| index.get(&name).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown key: {}", name))
            })).collect::<PyResult<Vec<_>>>()
        });

        if let Ok(indices) = from_iter {
            Ok(indices.into())
        } else {
            let n = self.n_obs();
            to_select_elem(ix, n)
        }
    }

    fn select_var(&self, ix: &PyAny) -> PyResult<SelectInfoElem> {
        let from_iter = ix.iter().and_then(|iter| 
            iter.map(|x| x.unwrap().extract::<String>()).collect::<PyResult<Vec<_>>>()
        ).and_then(|names| {
            let index = self.0.var_names();
            names.into_iter().map(|name| index.get(&name).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown key: {}", name))
            })).collect::<PyResult<Vec<_>>>()
        });

        if let Ok(indices) = from_iter {
            Ok(indices.into())
        } else {
            let n = self.n_vars();
            to_select_elem(ix, n)
        }
    }
}

impl<B: Backend + 'static> From<anndata::AnnData<B>> for AnnData {
    fn from(adata: anndata::AnnData<B>) -> Self {
        AnnData(Box::new(Slot::new(adata)))
    }
}

#[pymethods]
impl AnnData {
    #[new]
    #[args(
        "*",
        filename,
        X = "None",
        obs = "None",
        var = "None",
        obsm = "None",
        varm = "None",
        uns = "None",
        backend = "None",
    )]
    pub fn new(
        filename: PathBuf,
        X: Option<PyArrayData>,
        obs: Option<PyDataFrame>,
        var: Option<PyDataFrame>,
        obsm: Option<HashMap<String, PyArrayData>>,
        varm: Option<HashMap<String, PyArrayData>>,
        uns: Option<HashMap<String, PyData>>,
        backend: Option<&str>,
    ) -> Result<Self> {
        let adata: AnnData = match backend.unwrap_or(H5::NAME) {
            H5::NAME => anndata::AnnData::<H5>::new(filename)?.into(),
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
        self.0.shape()
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
        self.0.obs_names().names
    }
    #[setter(obs_names)]
    pub fn set_obs_names(&self, names: &PyAny) -> Result<()> {
        self.0.set_obs_names(names)
    }

    #[pyo3(text_signature = "($self, names)")]
    fn obs_ix(&self, names: &PyAny) -> Result<Vec<usize>> { self.0.obs_ix(names) }

    /// Names of variables.
    ///
    /// Returns
    /// -------
    /// list[str]
    #[getter]
    pub fn var_names(&self) -> Vec<String> {
        self.0.var_names().names
    }
    #[setter(var_names)]
    pub fn set_var_names(&self, names: &PyAny) -> Result<()> {
        self.0.set_var_names(names)
    }

    #[pyo3(text_signature = "($self, names)")]
    fn var_ix(&self, names: &PyAny) -> Result<Vec<usize>> { self.0.var_ix(names) }

    /// Data matrix of shape n_obs × n_vars.
    ///
    /// Returns
    /// -------
    /// PyArrayElem
    #[getter(X)]
    pub fn get_x(&self) -> Option<PyArrayElem> {
        self.0.get_x()
    }
    #[setter(X)]
    pub fn set_x(&self, data: Option<PyArrayData>) -> Result<()> {
        self.0.set_x(data)
    }

    /// Observation annotations.
    ///
    /// Returns
    /// -------
    /// PyDataFrameElem
    #[getter(obs)]
    fn get_obs(&self) -> Option<PyDataFrameElem> {
        self.0.get_obs()
    }
    #[setter(obs)]
    fn set_obs(&self, obs: Option<PyDataFrame>) -> Result<()> {
        self.0.set_obs(obs)
    }

    /// Variable annotations.
    ///
    /// Returns
    /// -------
    /// PyDataFrameElem
    #[getter(var)]
    fn get_var(&self) -> Option<PyDataFrameElem> {
        self.0.get_var()
    }
    #[setter(var)]
    fn set_var(&self, var: Option<PyDataFrame>) -> Result<()> {
        self.0.set_var(var)
    }

    /// Unstructured annotation (ordered dictionary).
    ///
    /// Returns
    /// -------
    /// PyElemCollection
    #[getter(uns)]
    pub fn get_uns(&self) -> Option<PyElemCollection> {
        self.0.get_uns()
    }
    #[setter(uns)]
    pub fn set_uns(&self, uns: Option<HashMap<String, PyData>>) -> Result<()> {
        self.0.set_uns(uns)
    }

    #[getter(obsm)]
    pub fn get_obsm(&self) -> Option<PyAxisArrays> {
        self.0.get_obsm()
    }
    #[setter(obsm)]
    pub fn set_obsm(&self, obsm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.set_obsm(obsm)
    }

    #[getter(obsp)]
    pub fn get_obsp(&self) -> Option<PyAxisArrays> {
        self.0.get_obsp()
    }
    #[setter(obsp)]
    pub fn set_obsp(&self, obsp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.set_obsp(obsp)
    }

    #[getter(varm)]
    pub fn get_varm(&self) -> Option<PyAxisArrays> {
        self.0.get_varm()
    }
    #[setter(varm)]
    pub fn set_varm(&self, varm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.set_varm(varm)
    }

    #[getter(varp)]
    pub fn get_varp(&self) -> Option<PyAxisArrays> {
        self.0.get_varp()
    }
    #[setter(varp)]
    pub fn set_varp(&self, varp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.set_varp(varp)
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
            .map(|x| self.select_obs(x).unwrap())
            .unwrap_or(SelectInfoElem::full());
        let j = var_indices
            .map(|x| self.select_var(x).unwrap())
            .unwrap_or(SelectInfoElem::full());
        self.0.subset(&[i, j], out, backend)
    }

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
    pub fn chunked_x(&self, chunk_size: usize) -> PyChunkedArray {
        self.0.chunked_x(chunk_size)
    }

    /// Filename of the backing .h5ad file.
    ///
    /// Returns
    /// -------
    /// Path
    #[getter]
    pub fn filename(&self) -> PathBuf {
        self.0.filename()
    }

    #[getter]
    pub fn backend(&self) -> String {
        self.0.backend().to_string()
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
        self.0.is_closed()
    }

    /// Close the AnnData object.
    #[pyo3(text_signature = "($self)")]
    pub fn close(&self) -> Result<()> {
        self.0.close()
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
        self.0.write(filename, backend)
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
        self.0.copy(filename, backend)
    }

    /// Return a new AnnData object with all backed arrays loaded into memory.
    ///
    /// Returns
    /// -------
    /// AnnData
    #[pyo3(text_signature = "($self)")]
    pub fn to_memory<'py>(&self, py: Python<'py>) -> Result<PyAnnData<'py>> {
        self.0.to_memory(py)
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

trait AnnDataTrait: Send + Downcast {
    fn shape(&self) -> (usize, usize);
    fn obs_names(&self) -> DataFrameIndex;
    fn set_obs_names(&self, names: &PyAny) -> Result<()>;
    fn obs_ix(&self, index: &PyAny) -> Result<Vec<usize>>;
    fn var_names(&self) -> DataFrameIndex;
    fn set_var_names(&self, names: &PyAny) -> Result<()>;
    fn var_ix(&self, index: &PyAny) -> Result<Vec<usize>>;

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

    fn chunked_x(&self, chunk_size: usize) -> PyChunkedArray;

    fn write(&self, filename: PathBuf, backend: Option<&str>) -> Result<()>;
    fn copy(&self, filename: PathBuf, backend: Option<&str>) -> Result<AnnData>;
    fn to_memory<'py>(&self, py: Python<'py>) -> Result<PyAnnData<'py>>;

    fn filename(&self) -> PathBuf;
    fn backend(&self) -> &str;
    fn is_closed(&self) -> bool;
    fn show(&self) -> String;

    fn close(&self) -> Result<()>;
    fn clone_ref(&self) -> Box<dyn AnnDataTrait>;
}
impl_downcast!(AnnDataTrait);

impl<B: Backend> AnnDataTrait for Slot<anndata::AnnData<B>> {
    fn shape(&self) -> (usize, usize) {
        let inner = self.inner();
        (inner.n_obs(), inner.n_vars())
    }

    fn obs_names(&self) -> DataFrameIndex {
        self.inner().obs_names()
    }

    fn obs_ix(&self, index: &PyAny) -> Result<Vec<usize>> {
        self.inner().obs_ix(
            index.iter()?.map(|x| x.unwrap().extract::<&str>().unwrap())
        )
    }

    fn set_obs_names(&self, names: &PyAny) -> Result<()> {
        let obs_names: Result<DataFrameIndex> =
            names.iter()?.map(|x| Ok(x?.extract::<String>()?)).collect();
        self.inner().set_obs_names(obs_names?)
    }

    fn var_names(&self) -> DataFrameIndex {
        self.inner().var_names()
    }

    fn var_ix(&self, index: &PyAny) -> Result<Vec<usize>> {
        self.inner().var_ix(
            index.iter()?.map(|x| x.unwrap().extract::<&str>().unwrap())
        )
    }

    fn set_var_names(&self, names: &PyAny) -> Result<()> {
        let var_names: Result<DataFrameIndex> =
            names.iter()?.map(|x| Ok(x?.extract::<String>()?)).collect();
        self.inner().set_var_names(var_names?)
    }

    fn get_x(&self) -> Option<PyArrayElem> {
        let inner = self.inner();
        let x = inner.get_x();
        if x.is_empty() {
            None
        } else {
            Some(x.clone().into())
        }
    }
    fn get_obs(&self) -> Option<PyDataFrameElem> {
        let inner = self.inner();
        let obs = inner.get_obs();
        if obs.is_empty() {
            None
        } else {
            Some(obs.clone().into())
        }
    }
    fn get_var(&self) -> Option<PyDataFrameElem> {
        let inner = self.inner();
        let var = inner.get_var();
        if var.is_empty() {
            None
        } else {
            Some(var.clone().into())
        }
    }
    fn get_uns(&self) -> Option<PyElemCollection> {
        let inner = self.inner();
        let uns = inner.uns();
        if uns.is_empty() {
            None
        } else {
            Some(uns.clone().into())
        }
    }
    fn get_obsm(&self) -> Option<PyAxisArrays> {
        let inner = self.inner();
        let obsm = inner.obsm();
        if obsm.is_empty() {
            None
        } else {
            Some(obsm.clone().into())
        }
    }
    fn get_obsp(&self) -> Option<PyAxisArrays> {
        let inner = self.inner();
        let obsp = inner.obsp();
        if obsp.is_empty() {
            None
        } else {
            Some(obsp.clone().into())
        }
    }
    fn get_varm(&self) -> Option<PyAxisArrays> {
        let inner = self.inner();
        let varm = inner.varm();
        if varm.is_empty() {
            None
        } else {
            Some(varm.clone().into())
        }
    }
    fn get_varp(&self) -> Option<PyAxisArrays> {
        let inner = self.inner();
        let varp = inner.varp();
        if varp.is_empty() {
            None
        } else {
            Some(varp.clone().into())
        }
    }

    fn set_x(&self, data: Option<PyArrayData>) -> Result<()> {
        let inner = self.inner();
        if let Some(d) = data {
            inner.set_x::<ArrayData>(d.into())?;
        } else {
            inner.del_x()?;
        }
        Ok(())
    }
    fn set_obs(&self, obs: Option<PyDataFrame>) -> Result<()> {
        let inner = self.inner();
        if let Some(o) = obs {
            inner.set_obs(o.into())?;
        } else {
            inner.del_obs()?;
        }
        Ok(())
    }
    fn set_var(&self, var: Option<PyDataFrame>) -> Result<()> {
        let inner = self.inner();
        if let Some(v) = var {
            inner.set_var(v.into())?;
        } else {
            inner.del_var()?;
        }
        Ok(())
    }
    fn set_uns(&self, uns: Option<HashMap<String, PyData>>) -> Result<()> {
        let inner = self.inner();
        if let Some(u) = uns {
            inner.set_uns(u.into_iter().map(|(k, v)| (k, v.into())))?;
        } else {
            inner.del_uns()?;
        }
        Ok(())
    }
    fn set_obsm(&self, obsm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        let inner = self.inner();
        if let Some(o) = obsm {
            inner.set_obsm(o.into_iter().map(|(k, v)| (k, v.into())))?;
        } else {
            inner.del_obsm()?;
        }
        Ok(())
    }
    fn set_obsp(&self, obsp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        let inner = self.inner();
        if let Some(o) = obsp {
            inner.set_obsp(o.into_iter().map(|(k, v)| (k, v.into())))?;
        } else {
            inner.del_obsp()?;
        }
        Ok(())
    }
    fn set_varm(&self, varm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        let inner = self.inner();
        if let Some(v) = varm {
            inner.set_varm(v.into_iter().map(|(k, v)| (k, v.into())))?;
        } else {
            inner.del_varm()?;
        }
        Ok(())
    }
    fn set_varp(&self, varp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        let inner = self.inner();
        if let Some(v) = varp {
            inner.set_varp(v.into_iter().map(|(k, v)| (k, v.into())))?;
        } else {
            inner.del_varp()?;
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
            match backend.unwrap_or(H5::NAME) {
                H5::NAME => {
                    self.inner().write_select::<H5, _, _>(slice, &out)?;
                    Ok(Some(AnnData::open(out, "r+", backend)?))
                }
                x => bail!("Unsupported backend: {}", x),
            }
        } else {
            self.inner().subset(slice)?;
            Ok(None)
        }
    }

    fn chunked_x(&self, chunk_size: usize) -> PyChunkedArray {
        self.inner().get_x().chunked(chunk_size).into()
    }

    fn write(&self, filename: PathBuf, backend: Option<&str>) -> Result<()> {
        match backend.unwrap_or(H5::NAME) {
            H5::NAME => self.inner().write::<H5, _>(filename),
            x => bail!("Unsupported backend: {}", x),
        }
    }

    fn copy(&self, filename: PathBuf, backend: Option<&str>) -> Result<AnnData> {
        AnnDataTrait::write(self, filename.clone(), backend)?;
        AnnData::open(filename, "r+", backend)
    }

    fn to_memory<'py>(&self, py: Python<'py>) -> Result<PyAnnData<'py>> {
        Ok(PyAnnData::from_anndata(py, self.inner().deref())?)
    }

    fn filename(&self) -> PathBuf {
        self.inner().filename()
    }

    fn backend(&self) -> &str {
        B::NAME
    }

    fn is_closed(&self) -> bool {
        self.is_empty()
    }

    fn show(&self) -> String {
        if self.is_closed() {
            "Closed AnnData object".to_string()
        } else {
            format!("{}", self.inner().deref())
        }
    }

    fn close(&self) -> Result<()> {
        if let Some(inner) = self.extract() {
            inner.close()?;
        }
        Ok(())
    }

    fn clone_ref(&self) -> Box<dyn AnnDataTrait> {
        Box::new(self.clone())
    }
}
