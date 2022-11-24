use crate::element::*;
use crate::iterator::{PyChunkedMatrix, PyStackedChunkedMatrix};
use crate::utils::{
    conversion::{PyToRust, RustToPy},
    instance::*,
    to_indices,
};

use anndata_rs::{
    data::{Data, MatrixData},
    element::DataFrameIndex,
    AnnDataOp,
    {
        anndata,
        element::{Inner, Slot},
    },
};
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

macro_rules! def_df_accessor {
    ($name:ty, { $($field:ident),* }) => {
        paste! {
            #[pymethods]
            impl $name {
            $(
                /// One-dimensional annotation.
                ///
                /// Returns
                /// -------
                /// PyDataFrameElem
                #[getter($field)]
                fn [<get_ $field>](&self) -> Option<PyDataFrameElem> {
                    let item = self.0.inner().[<get_ $field>]().clone();
                    if item.is_empty() {
                        None
                    } else {
                        Some(PyDataFrameElem(item))
                    }
                }

                #[setter($field)]
                fn [<set_ $field>]<'py>(
                    &mut self,
                    py: Python<'py>,
                    df: Option<&'py PyAny>
                ) -> PyResult<()>
                {
                    let data = df.map(|x| {
                        let polars = py.import("polars")?;
                        let df_ = if isinstance_of_pandas(py, x)? {
                            polars.call_method1("from_pandas", (x, ))?
                        } else if x.is_instance_of::<pyo3::types::PyDict>()? {
                            polars.call_method1("from_dict", (x, ))?
                        } else {
                            x
                        };
                        df_.into_rust(py)
                    }).transpose()?;
                    self.0.inner().[<set_ $field>](data).unwrap();
                    Ok(())
                }
            )*
            }
        }
    }
}

macro_rules! def_arr_accessor {
    ($name:ty, $get_type:ty, $set_type:ty, { $($field:ident),* }) => {
        paste! {
            #[pymethods]
            impl $name {
            $(
                /// :class:`.PyAxisArrays`.
                #[getter($field)]
                fn [<get_ $field>](&self) -> $get_type {
                    $get_type(self.0.inner().[<get_ $field>]().clone())
                }

                #[setter($field)]
                fn [<set_ $field>]<'py>(
                    &self,
                    py: Python<'py>,
                    $field: $set_type
                ) -> PyResult<()>
                {
                    let data: PyResult<_> = $field.map(|mut x| x.drain().map(|(k, v)|
                        Ok((k, v.into_rust(py)?))
                    ).collect()).transpose();
                    self.0.inner().[<set_ $field>](data?).unwrap();
                    Ok(())
                }

            )*
            }
        }
    }
}

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
pub struct AnnData(pub(crate) Slot<anndata::AnnData>);

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
        uns = "None"
    )]
    pub fn new<'py>(
        py: Python<'py>,
        filename: PathBuf,
        X: Option<&'py PyAny>,
        n_obs: Option<usize>,
        n_vars: Option<usize>,
        obs: Option<&'py PyAny>,
        var: Option<&'py PyAny>,
        obsm: Option<HashMap<String, &'py PyAny>>,
        varm: Option<HashMap<String, &'py PyAny>>,
        uns: Option<HashMap<String, &'py PyAny>>,
    ) -> PyResult<Self> {
        let mut anndata = AnnData::wrap(
            anndata::AnnData::new(filename, n_obs.unwrap_or(0), n_vars.unwrap_or(0)).unwrap(),
        );

        if X.is_some() {
            anndata.set_x(py, X)?;
        }
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
        Ok(anndata)
    }

    /// Shape of data matrix (`n_obs`, `n_vars`).
    ///
    /// Returns
    /// -------
    /// tuple[int, int]
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        (self.n_obs(), self.n_vars())
    }

    /// Number of observations.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub fn n_obs(&self) -> usize {
        self.0.inner().n_obs()
    }

    #[setter(n_obs)]
    pub fn set_n_obs(&self, n: usize) {
        self.0.inner().set_n_obs(n)
    }

    /// Number of variables/features.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub fn n_vars(&self) -> usize {
        self.0.inner().n_vars()
    }

    #[setter(n_vars)]
    pub fn set_n_vars(&self, n: usize) {
        self.0.inner().set_n_vars(n)
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
    pub fn set_var_names(&self, names: &PyAny) -> PyResult<()> {
        let var_names: PyResult<DataFrameIndex> = names
            .iter()?
            .map(|x| x.unwrap().extract::<String>())
            .collect();
        self.0.inner().set_var_names(var_names?).unwrap();
        Ok(())
    }

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

    /// Data matrix of shape n_obs × n_vars.
    ///
    /// Returns
    /// -------
    /// PyMatrixElem
    #[getter(X)]
    pub fn get_x(&self) -> PyMatrixElem {
        PyMatrixElem(self.0.inner().get_x().clone())
    }

    #[setter(X)]
    pub fn set_x<'py>(&self, py: Python<'py>, data: Option<&'py PyAny>) -> PyResult<()> {
        if let Some(d) = data {
            if is_iterator(py, d)? {
                panic!("Setting X by an iterator is not implemented");
            } else {
                let d_: Box<dyn MatrixData> = d.into_rust(py)?;
                self.0.inner().set_x(Some(d_)).unwrap();
            }
        } else {
            self.0.inner().set_x::<Box<dyn MatrixData>>(None).unwrap();
        }
        Ok(())
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
        let uns_ = uns.map(|mut x| {
            x.drain()
                .map(|(k, v)| (k, v.into_rust(py).unwrap()))
                .collect()
        });
        self.0.inner().set_uns(uns_).unwrap();
        Ok(())
    }

    /// Subsetting the AnnData object.
    ///
    /// Parameters
    /// ----------
    /// obs_indices
    ///     obs indices
    /// var_indices
    ///     var indices
    /// out: Path
    ///     File name of the output `.h5ad` file. If provided, the result will be
    ///     saved to a new file and the original AnnData object remains unchanged.
    ///
    /// Returns
    /// -------
    /// Optional[AnnData]
    #[pyo3(text_signature = "($self, obs_indices, var_indices, out)")]
    pub fn subset<'py>(
        &self,
        py: Python<'py>,
        obs_indices: Option<&'py PyAny>,
        var_indices: Option<&'py PyAny>,
        out: Option<PathBuf>,
    ) -> PyResult<Option<AnnData>> {
        let i = obs_indices.and_then(|x| self.normalize_index(py, x, 0).unwrap());
        let j = var_indices.and_then(|x| self.normalize_index(py, x, 1).unwrap());
        Ok(match out {
            None => {
                self.0
                    .inner()
                    .subset(i.as_ref().map(Vec::as_slice), j.as_ref().map(Vec::as_slice))
                    .unwrap();
                None
            }
            Some(file) => Some(AnnData::wrap(
                self.0
                    .inner()
                    .copy(
                        i.as_ref().map(Vec::as_slice),
                        j.as_ref().map(Vec::as_slice),
                        file,
                    )
                    .unwrap(),
            )),
        })
    }

    /// Filename of the backing .h5ad file.
    ///
    /// Returns
    /// -------
    /// str
    #[getter]
    pub fn filename(&self) -> String {
        self.0.inner().filename()
    }

    /// Copy the AnnData object.
    ///
    /// Parameters
    /// ----------
    /// filename
    ///     File name of the output `.h5ad` file.
    ///
    /// Returns
    /// -------
    /// AnnData
    #[pyo3(text_signature = "($self, filename)")]
    pub fn copy(&self, filename: PathBuf) -> Self {
        AnnData::wrap(self.0.inner().copy(None, None, filename).unwrap())
    }

    /// Write .h5ad-formatted hdf5 file.
    ///
    /// Parameters
    /// ----------
    /// filename
    ///     File name of the output `.h5ad` file.
    #[pyo3(text_signature = "($self, filename)")]
    pub fn write(&self, filename: PathBuf) {
        self.0.inner().write(None, None, filename).unwrap();
    }

    /// Close the AnnData object.
    #[pyo3(text_signature = "($self)")]
    pub fn close(&self) {
        if let Some(anndata) = self.0.extract() {
            anndata.close().unwrap();
        }
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

    /// Return a new AnnData object with all backed arrays loaded into memory.
    ///
    /// Returns
    /// -------
    /// AnnData
    #[pyo3(text_signature = "($self)")]
    pub fn to_memory<'py>(&self, py: Python<'py>) -> Result<PyObject> {
        Ok(PyAnnData::from_anndata(py, self.inner().deref())?.to_object(py))
    }

    /// If the AnnData object has been closed.
    ///
    /// Returns
    /// -------
    /// bool
    #[pyo3(text_signature = "($self)")]
    pub fn is_closed(&self) -> bool {
        self.0.inner().0.is_none()
    }

    fn __repr__(&self) -> String {
        if self.is_closed() {
            "Closed AnnData object".to_string()
        } else {
            format!("{}", self.0.inner().deref())
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
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

/** Similar to `AnnData`, `AnnDataSet` contains annotations of
    observations `obs` (`obsm`, `obsp`), variables `var` (`varm`, `varp`),
    and unstructured annotations `uns`. Additionally it provides lazy access to
    concatenated component AnnData objects, including `X`, `obs`, `obsm`, `obsp`.

    Parameters
    ----------
    adatas: list[(str, Path)] | list[(str, AnnData)]
        List of key and file name (or backed AnnData object) pairs.
    filename: Path
        File name of the output file containing the AnnDataSet object.
    add_key: str
        The column name in obs to store the keys

    Note
    ----
    AnnDataSet does not copy underlying AnnData objects. It stores the references
    to individual anndata files. If you move the anndata files to a new location,
    remember to update the anndata file locations when opening an AnnDataSet object.

    See Also
    --------
    read_dataset
*/
#[pyclass]
#[pyo3(text_signature = "(*, adatas, filename, add_key /)")]
#[repr(transparent)]
#[derive(Clone)]
pub struct AnnDataSet(Slot<anndata::AnnDataSet>);

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
    #[new]
    #[args("*", adatas, filename, add_key = "\"sample\"")]
    pub fn new(
        adatas: Vec<(String, AnnDataFile)>,
        filename: PathBuf,
        add_key: &str,
    ) -> Result<Self> {
        let anndatas = adatas
            .into_par_iter()
            .map(|(key, data_file)| {
                let adata = match data_file {
                    AnnDataFile::Data(data) => data.0.extract().unwrap(),
                    AnnDataFile::Path(path) => {
                        let file = hdf5::File::open(path)?;
                        anndata::AnnData::read(file)?
                    }
                };
                Ok((key, adata))
            })
            .collect::<Result<_>>()?;
        Ok(AnnDataSet::wrap(anndata::AnnDataSet::new(
            anndatas, filename, add_key,
        )?))
    }

    /// Shape of data matrix (`n_obs`, `n_vars`).
    ///
    /// Returns
    /// -------
    /// tuple[int, int]
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        (self.n_obs(), self.n_vars())
    }

    /// Number of observations.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub fn n_obs(&self) -> usize {
        self.0.inner().n_obs()
    }

    /// Number of variables/features.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub fn n_vars(&self) -> usize {
        self.0.inner().n_vars()
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
    pub fn set_var_names(&self, names: &PyAny) -> PyResult<()> {
        let var_names: PyResult<DataFrameIndex> = names
            .iter()?
            .map(|x| x.unwrap().extract::<String>())
            .collect();
        self.0.inner().set_var_names(var_names?).unwrap();
        Ok(())
    }

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

def_df_accessor!(AnnDataSet, { obs, var });

def_arr_accessor!(
    AnnDataSet,
    PyAxisArrays,
    Option<HashMap<String, &'py PyAny>>,
    { obsm, obsp, varm, varp }
);

pub struct PyAnnData<'py>(&'py PyAny);

impl<'py> Deref for PyAnnData<'py> {
    type Target = PyAny;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'py> PyAnnData<'py> {
    pub fn new(py: Python<'py>) -> PyResult<Self> {
        PyModule::import(py, "anndata")?
            .call_method0("AnnData")?
            .extract()
    }

    pub fn from_anndata(py: Python<'py>, inner: &anndata::AnnData) -> Result<Self> {
        let adata = PyAnnData::new(py)?;
        {
            // Set X
            adata.set_n_obs(inner.n_obs())?;
            adata.set_n_vars(inner.n_vars())?;
            adata.set_x(inner.read_x()?)?;
        }
        {
            // Set obs and var
            adata.set_obs_names(inner.obs_names().into())?;
            adata.set_var_names(inner.var_names().into())?;
            adata.set_obs(Some(inner.read_obs()?))?;
            adata.set_var(Some(inner.read_var()?))?;
        }
        {
            // Set uns
            inner
                .uns_keys()
                .into_iter()
                .try_for_each(|k| adata.add_uns_item(&k, inner.read_uns_item(&k)?.unwrap()))?;
        }
        {
            // Set obsm
            inner
                .obsm_keys()
                .into_iter()
                .try_for_each(|k| adata.add_obsm_item(&k, inner.read_obsm_item(&k)?.unwrap()))?;
        }
        {
            // Set obsp
            inner
                .obsp_keys()
                .into_iter()
                .try_for_each(|k| adata.add_obsp_item(&k, inner.read_obsp_item(&k)?.unwrap()))?;
        }
        {
            // Set varm
            inner
                .varm_keys()
                .into_iter()
                .try_for_each(|k| adata.add_varm_item(&k, inner.read_varm_item(&k)?.unwrap()))?;
        }
        {
            // Set varp
            inner
                .varp_keys()
                .into_iter()
                .try_for_each(|k| adata.add_varp_item(&k, inner.read_varp_item(&k)?.unwrap()))?;
        }
        Ok(adata)
    }

    fn get_item<T>(&'py self, slot: &str, key: &str) -> Result<Option<T>>
    where
        &'py PyAny: PyToRust<T>,
    {
        let data = self
            .getattr(slot)?
            .call_method1("__getitem__", (key,))
            .ok()
            .map(|x| x.into_rust(self.py()));
        Ok(data.transpose()?)
    }

    fn set_item<T: RustToPy>(&'py self, slot: &str, key: &str, data: T) -> Result<()> {
        let py = self.py();
        let d = data.rust_into_py(py)?;
        let new_d = if isinstance_of_polars(py, d.as_ref(py))? {
            d.call_method0(py, "to_pandas")?
        } else {
            d
        };
        self.getattr(slot)?
            .call_method1("__setitem__", (key, new_d))?;
        Ok(())
    }

    fn get_keys(&self, slot: &str) -> PyResult<Vec<String>> {
        self.getattr(slot)?.call_method0("keys")?.extract()
    }

    pub(crate) fn set_n_obs(&self, n_obs: usize) -> PyResult<()> {
        let n = self.n_obs();
        if n == n_obs {
            Ok(())
        } else if n == 0 {
            self.0.setattr("_n_obs", n_obs)
        } else {
            Err(PyException::new_err("cannot set n_obs unless n_obs == 0"))
        }
    }

    pub(crate) fn set_n_vars(&self, n_vars: usize) -> PyResult<()> {
        let n = self.n_vars();
        if n == n_vars {
            Ok(())
        } else if n == 0 {
            self.0.setattr("_n_vars", n_vars)
        } else {
            Err(PyException::new_err("cannot set n_vars unless n_vars == 0"))
        }
    }
}

impl<'py> FromPyObject<'py> for PyAnnData<'py> {
    fn extract(obj: &'py PyAny) -> PyResult<Self> {
        Python::with_gil(|py| {
            if isinstance_of_pyanndata(py, obj)? {
                Ok(PyAnnData(obj))
            } else {
                Err(PyTypeError::new_err("Not a Python AnnData object"))
            }
        })
    }
}

impl<'py> ToPyObject for PyAnnData<'py> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.0.into_py(py)
    }
}

impl<'py> AnnDataOp for PyAnnData<'py> {
    fn read_x(&self) -> Result<Option<Box<dyn MatrixData>>> {
        let x = self.getattr("X")?;
        if x.is_none() {
            Ok(None)
        } else {
            Ok(Some(x.into_rust(self.py())?))
        }
    }

    fn set_x<D: MatrixData>(&self, data_: Option<D>) -> Result<()> {
        let py = self.py();
        if let Some(data) = data_ {
            self.set_n_obs(data.nrows())?;
            self.set_n_vars(data.ncols())?;
            self.setattr("X", data.into_dyn_matrix().rust_into_py(py)?)?;
        } else {
            self.setattr("X", py.None())?;
        }
        Ok(())
    }

    fn n_obs(&self) -> usize {
        self.0.getattr("n_obs").unwrap().extract().unwrap()
    }
    fn n_vars(&self) -> usize {
        self.0.getattr("n_vars").unwrap().extract().unwrap()
    }

    fn obs_names(&self) -> Vec<String> {
        self.0.getattr("obs_names").unwrap().extract().unwrap()
    }
    fn var_names(&self) -> Vec<String> {
        self.0.getattr("var_names").unwrap().extract().unwrap()
    }

    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()> {
        if self.getattr("obs")?.getattr("empty")?.is_true()? {
            let py = self.py();
            let df = py.import("pandas")?.call_method(
                "DataFrame",
                (),
                Some(&[("index", index.names)].into_py_dict(py)),
            )?;
            self.setattr("obs", df)?;
        } else {
            self.setattr("obs_names", index.names)?;
        }
        Ok(())
    }
    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> {
        if self.getattr("var")?.getattr("empty")?.is_true()? {
            let py = self.py();
            let df = py.import("pandas")?.call_method(
                "DataFrame",
                (),
                Some(&[("index", index.names)].into_py_dict(py)),
            )?;
            self.setattr("var", df)?;
        } else {
            self.setattr("var_names", index.names)?;
        }
        Ok(())
    }

    fn obs_ix(&self, _names: &[String]) -> Result<Vec<usize>> {
        todo!()
    }
    fn var_ix(&self, _names: &[String]) -> Result<Vec<usize>> {
        todo!()
    }

    fn read_obs(&self) -> Result<DataFrame> {
        let py = self.py();
        let df = py
            .import("polars")?
            .call_method1("from_pandas", (self.0.getattr("obs")?,))?;
        Ok(df.into_rust(py)?)
    }
    fn read_var(&self) -> Result<DataFrame> {
        let py = self.py();
        let df = py
            .import("polars")?
            .call_method1("from_pandas", (self.0.getattr("var")?,))?;
        Ok(df.into_rust(py)?)
    }

    fn set_obs(&self, obs_: Option<DataFrame>) -> Result<()> {
        match obs_ {
            None => {
                self.0.setattr("obs", None::<PyObject>)?;
            }
            Some(obs) => {
                let py = self.py();
                let index = self.getattr("obs")?.getattr("index")?;
                let df = if obs.is_empty() {
                    py.import("pandas")?
                        .call_method1("DataFrame", (py.None(), index))?
                        .into_py(py)
                } else {
                    obs.rust_into_py(py)?
                        .call_method0(py, "to_pandas")?
                        .call_method1(py, "set_index", (index,))?
                };
                self.setattr("obs", df)?;
            }
        }
        Ok(())
    }
    fn set_var(&self, var_: Option<DataFrame>) -> Result<()> {
        match var_ {
            None => {
                self.0.setattr("var", None::<PyObject>)?;
            }
            Some(var) => {
                let py = self.py();
                let index = self.getattr("var")?.getattr("index")?;
                let df = if var.is_empty() {
                    py.import("pandas")?
                        .call_method1("DataFrame", (py.None(), index))?
                        .into_py(py)
                } else {
                    var.rust_into_py(py)?
                        .call_method0(py, "to_pandas")?
                        .call_method1(py, "set_index", (index,))?
                };
                self.setattr("var", df)?;
            }
        }
        Ok(())
    }

    fn uns_keys(&self) -> Vec<String> {
        self.get_keys("uns").unwrap()
    }
    fn obsm_keys(&self) -> Vec<String> {
        self.get_keys("obsm").unwrap()
    }
    fn obsp_keys(&self) -> Vec<String> {
        self.get_keys("obsp").unwrap()
    }
    fn varm_keys(&self) -> Vec<String> {
        self.get_keys("varm").unwrap()
    }
    fn varp_keys(&self) -> Vec<String> {
        self.get_keys("varp").unwrap()
    }

    fn read_uns_item(&self, key: &str) -> Result<Option<Box<dyn Data>>> {
        self.get_item("uns", key)
    }
    fn read_obsm_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> {
        self.get_item("obsm", key)
    }
    fn read_obsp_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> {
        self.get_item("obsp", key)
    }
    fn read_varm_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> {
        self.get_item("varm", key)
    }
    fn read_varp_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> {
        self.get_item("varp", key)
    }

    fn add_uns_item<D: Data>(&self, key: &str, data: D) -> Result<()> {
        self.set_item("uns", key, data.into_dyn_data())
    }
    fn add_obsm_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()> {
        self.set_n_obs(data.nrows())?;
        self.set_item("obsm", key, data.into_dyn_matrix())
    }
    fn add_obsp_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()> {
        self.set_n_obs(data.nrows())?;
        self.set_item("obsp", key, data.into_dyn_matrix())
    }
    fn add_varm_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()> {
        self.set_n_vars(data.nrows())?;
        self.set_item("varm", key, data.into_dyn_matrix())
    }
    fn add_varp_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()> {
        self.set_n_vars(data.nrows())?;
        self.set_item("varp", key, data.into_dyn_matrix())
    }
}
