use crate::element::*;
use crate::iterator::{PyChunkedMatrix, PyStackedChunkedMatrix};
use crate::utils::{to_indices, instance::*, conversion::{RustToPy, PyToRust}};

use anndata_rs::{
    AnnDataOp, element::DataFrameIndex,
    {anndata, element::{Inner, Slot}}, data::{Data, MatrixData},
};
use polars::frame::DataFrame;
use anyhow::Result;
use pyo3::{prelude::*, PyResult, Python, types::{PyIterator, IntoPyDict}, exceptions::{PyException, PyTypeError}};
use std::{collections::HashMap, ops::Deref};
use paste::paste;

macro_rules! def_df_accessor {
    ($name:ty, { $($field:ident),* }) => {
        paste! {
            #[pymethods]
            impl $name {
            $(
                /// :class:`.PyDataFrameElem`.
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

/// An annotated data matrix. 
/// `AnnData` stores a data matrix `X` together with annotations of
/// observations `obs` (`obsm`, `obsp`), variables `var` (`varm`, `varp`),
/// and unstructured annotations `uns`.
/// `AnnData` is stored as a HDF5 file. Opening/creating an AnnData object 
/// does not read data from the HDF5 file. Data is copied to memory only when
/// individual element is requested (or when cache is turned on).
#[pyclass]
#[pyo3(text_signature =
    "(*, filename, X, n_obs, n_vars, obs, var, obsm, obsp, varm, varp, uns)"
)]
#[repr(transparent)]
#[derive(Clone)]
pub struct AnnData(Slot<anndata::AnnData>);

impl AnnData {
    pub fn inner(&self) -> Inner<'_, anndata::AnnData> { self.0.inner() }

    pub fn wrap(anndata: anndata::AnnData) -> Self {
        AnnData(Slot::new(anndata))
    }

    fn import_mtx(&self, filename: &str, sorted: bool) {
        if crate::utils::is_gzipped(filename) {
            let f = std::fs::File::open(filename).unwrap();
            let mut reader = std::io::BufReader::new(flate2::read::MultiGzDecoder::new(f));
            self.0.inner().import_matrix_market(&mut reader, sorted).unwrap();
        } else {
            let f = std::fs::File::open(filename).unwrap();
            let mut reader = std::io::BufReader::new(f);
            self.0.inner().import_matrix_market(&mut reader, sorted).unwrap();
        }
    }

    fn normalize_index<'py>(&self, py: Python<'py>, indices: &'py PyAny, axis: u8) -> PyResult<Option<Vec<usize>>> {
        match PyIterator::from_object(py, indices)?.map(|x| x.unwrap().extract()).collect::<PyResult<Vec<String>>>() {
            Ok(names) => if axis == 0 {
                Ok(Some(self.obs_ix(names)))
            } else {
                Ok(Some(self.var_ix(names)))
            },
            _ => {
                let length = if axis == 0 { self.n_obs() } else { self.n_vars() };
                to_indices(py, indices, length)
            },
        }
    }
}

#[pymethods]
impl AnnData {
    #[new]
    #[args("*", filename, X = "None", n_obs = "None", n_vars = "None", obs = "None", var = "None",
        obsm = "None", obsp = "None", varm = "None", varp = "None", uns = "None",
    )]
    fn new<'py>(
        py: Python<'py>,
        filename: &str,
        X: Option<&'py PyAny>,
        n_obs: Option<usize>,
        n_vars: Option<usize>,
        obs: Option<&'py PyAny>,
        var: Option<&'py PyAny>,
        obsm: Option<HashMap<String, &'py PyAny>>,
        obsp: Option<HashMap<String, &'py PyAny>>,
        varm: Option<HashMap<String, &'py PyAny>>,
        varp: Option<HashMap<String, &'py PyAny>>,
        uns: Option<HashMap<String, &'py PyAny>>,
    ) -> PyResult<Self> {
        let mut anndata = AnnData::wrap(anndata::AnnData::new(
            filename, n_obs.unwrap_or(0), n_vars.unwrap_or(0)
        ).unwrap());
        
        if X.is_some() { anndata.set_x(py, X)?; }
        if obs.is_some() { anndata.set_obs(py, obs)?; }
        if var.is_some() { anndata.set_var(py, var)?; }
        if obsm.is_some() { anndata.set_obsm(py, obsm)?; }
        if obsp.is_some() { anndata.set_obsp(py, obsp)?; }
        if varm.is_some() { anndata.set_varm(py, varm)?; }
        if varp.is_some() { anndata.set_varp(py, varp)?; }
        if uns.is_some() { anndata.set_uns(py, uns)?; }
        Ok(anndata)
    }

    /// Shape of data matrix (`n_obs`, `n_vars`).
    #[getter]
    fn shape(&self) -> (usize, usize) { (self.n_obs(), self.n_vars()) }

    /// Number of observations.
    #[getter]
    fn n_obs(&self) -> usize { self.0.inner().n_obs() }

    #[setter(n_obs)]
    fn set_n_obs(&self, n: usize) { self.0.inner().set_n_obs(n) }

    /// Number of variables/features.
    #[getter]
    fn n_vars(&self) -> usize { self.0.inner().n_vars() }

    #[setter(n_vars)]
    fn set_n_vars(&self, n: usize) { self.0.inner().set_n_vars(n) }

    /// Names of variables.
    #[getter]
    fn var_names(&self) -> Vec<String> { self.0.inner().var_names() }

    #[setter(var_names)]
    fn set_var_names<'py>(&self, py: Python<'py>, names: &'py PyAny) -> PyResult<()> {
        let var_names: PyResult<DataFrameIndex> = PyIterator::from_object(py, names)?
            .map(|x| x.unwrap().extract::<String>()).collect();
        self.0.inner().set_var_names(var_names?).unwrap();
        Ok(())
    }

    #[pyo3(text_signature = "($self, names)")]
    fn var_ix(&self, names: Vec<String>) -> Vec<usize> { self.0.inner().var_ix(&names).unwrap() }

    /// Names of observations.
    #[getter]
    fn obs_names(&self) -> Vec<String> { self.0.inner().obs_names() }

    #[setter(obs_names)]
    fn set_obs_names<'py>(&self, py: Python<'py>, names: &'py PyAny) -> PyResult<()> {
        let obs_names: PyResult<DataFrameIndex> = PyIterator::from_object(py, names)?
            .map(|x| x.unwrap().extract::<String>()).collect();
        self.0.inner().set_obs_names(obs_names?).unwrap();
        Ok(())
    }

    #[pyo3(text_signature = "($self, names)")]
    fn obs_ix(&self, names: Vec<String>) -> Vec<usize> { self.0.inner().obs_ix(&names).unwrap() }

    /// :class:`.PyMatrixElem` of shape `n_obs` x `n_vars`.
    #[getter(X)]
    fn get_x(&self) -> PyMatrixElem {
        PyMatrixElem(self.0.inner().get_x().clone())
    }

    #[setter(X)]
    fn set_x<'py>(&self, py: Python<'py>, data: Option<&'py PyAny>) -> PyResult<()> {
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

    /// :class:`.PyElemCollection`.
    #[getter(uns)]
    fn get_uns(&self) -> PyElemCollection {
        PyElemCollection(self.0.inner().get_uns().clone())
    }

    #[setter(uns)]
    fn set_uns<'py>(&self, py: Python<'py>, uns: Option<HashMap<String, &'py PyAny>>) -> PyResult<()> {
        let uns_ = uns.map(|mut x|
            x.drain().map(|(k, v)| (k, v.into_rust(py).unwrap())).collect()
        );
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
    /// out
    ///     File name of the output `.h5ad` file. If provided, the result will be
    ///     saved to a new file and the original AnnData object remains unchanged.
    /// 
    /// Returns
    /// -------
    /// Optional[AnnData]
    #[pyo3(text_signature = "($self, obs_indices, var_indices, out)")]
    fn subset<'py>(
        &self,
        py: Python<'py>,
        obs_indices: Option<&'py PyAny>,
        var_indices: Option<&'py PyAny>,
        out: Option<&str>,
    ) -> PyResult<Option<AnnData>> {
        let i = obs_indices.and_then(|x| self.normalize_index(py, x, 0).unwrap());
        let j = var_indices.and_then(|x| self.normalize_index(py, x, 1).unwrap());
        Ok(match out {
            None => {
                self.0.inner().subset(i.as_ref().map(Vec::as_slice), j.as_ref().map(Vec::as_slice)).unwrap();
                None
            },
            Some(file) => Some(AnnData::wrap(
                self.0.inner().copy(i.as_ref().map(Vec::as_slice), j.as_ref().map(Vec::as_slice), file).unwrap()
            )),
        })
    }
            
    /// Filename of the underlying .h5ad file.
    #[getter]
    fn filename(&self) -> String { self.0.inner().filename() }

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
    fn copy(&self, filename: &str) -> Self {
        AnnData::wrap(self.0.inner().copy(None, None, filename).unwrap())
    }

    /// Write .h5ad-formatted hdf5 file.
    /// 
    /// Parameters
    /// ----------
    /// filename
    ///     File name of the output `.h5ad` file. 
    #[pyo3(text_signature = "($self, filename)")]
    fn write(&self, filename: &str) {
        self.0.inner().write(None, None, filename).unwrap();
    }

    /// Close the AnnData object.
    #[pyo3(text_signature = "($self)")]
    fn close(&self) {
        if let Some(anndata) = self.0.extract() {
            anndata.close().unwrap();
        }
    }

    #[getter]
    fn isbacked(&self) -> bool { true }

    /// Return an iterator over the rows of the data matrix X.
    /// 
    /// Parameters
    /// ----------
    /// chunk_size : int
    ///     Row size of a single chunk.
    #[args(chunk_size = "500")]
    #[pyo3(text_signature = "($self, chunk_size, /)")]
    #[pyo3(name = "chunked_X")]
    fn chunked_x(&self, chunk_size: usize) -> PyChunkedMatrix  {
        self.get_x().chunked(chunk_size)
    }

    /// Return a new AnnData object with all backed arrays loaded into memory.
    #[pyo3(text_signature = "($self)")]
    fn to_memory<'py>(&self, py: Python<'py>) -> Result<PyObject> {
        let adata = PyAnnData::new(py)?;
        { // Set X
            adata.set_n_obs(self.n_obs())?;
            adata.set_n_vars(self.n_vars())?;
            adata.set_x(self.inner().read_x()?)?;
        }
        { // Set obs and var
            adata.set_obs_names(self.inner().obs_names().into())?;
            adata.set_var_names(self.inner().var_names().into())?;
            adata.set_obs(Some(self.inner().read_obs()?))?;
            adata.set_var(Some(self.inner().read_var()?))?;
        }
        { // Set uns
            self.inner().uns_keys().iter().try_for_each(|k|
                adata.add_uns_item(k, self.inner().read_uns_item(k)?.unwrap()) )?;
        }
        { // Set obsm
            self.inner().obsm_keys().iter().try_for_each(|k|
                adata.add_obsm_item(k, self.inner().read_obsm_item(k)?.unwrap()) )?;
        }
        { // Set obsp
            self.inner().obsp_keys().iter().try_for_each(|k|
                adata.add_obsp_item(k, self.inner().read_obsp_item(k)?.unwrap()) )?;
        }
        { // Set varm
            self.inner().varm_keys().iter().try_for_each(|k|
                adata.add_varm_item(k, self.inner().read_varm_item(k)?.unwrap()) )?;
        }
        { // Set varp
            self.inner().varp_keys().iter().try_for_each(|k|
                adata.add_varp_item(k, self.inner().read_varp_item(k)?.unwrap()) )?;
        }
        Ok(adata.to_object(py))
    }

    /// If the AnnData object has been closed.
    /// Returns
    /// -------
    /// bool
    #[pyo3(text_signature = "($self)")]
    fn is_closed(&self) -> bool { self.0.inner().0.is_none() }

    fn __repr__(&self) -> String {
        if self.is_closed() {
            "Closed AnnData object".to_string()
        } else {
            format!("{}", self.0.inner().deref())
        }
    }

    fn __str__(&self) -> String { self.__repr__() }
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

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}

/// Similar to `AnnData`, `AnnDataSet` contains annotations of
/// observations `obs` (`obsm`, `obsp`), variables `var` (`varm`, `varp`),
/// and unstructured annotations `uns`. Additionally it provides lazy access to 
/// concatenated component AnnData objects, including `X`, `obs`, `obsm`, `obsp`.
/// 
/// Notes
/// ------
/// AnnDataSet doesn't copy underlying AnnData objects. It stores the references
/// to individual anndata files. If you move the anndata files to a new location, 
/// remember to update the anndata file locations when opening an AnnDataSet object.
/// 
/// See Also
/// --------
/// create_dataset
/// read_dataset
#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct AnnDataSet(Slot<anndata::AnnDataSet>);

impl AnnDataSet {
    pub fn inner(&self) -> Inner<'_, anndata::AnnDataSet> { self.0.inner() }

    pub fn wrap(anndata: anndata::AnnDataSet) -> Self {
        AnnDataSet(Slot::new(anndata))
    }

    fn normalize_index<'py>(&self, py: Python<'py>, indices: &'py PyAny, axis: u8) -> PyResult<Option<Vec<usize>>> {
        match PyIterator::from_object(py, indices)?.map(|x| x.unwrap().extract()).collect::<PyResult<Vec<String>>>() {
            Ok(names) => if axis == 0 {
                Ok(Some(self.obs_ix(names)))
            } else {
                Ok(Some(self.var_ix(names)))
            },
            _ => {
                let length = if axis == 0 { self.n_obs() } else { self.n_vars() };
                to_indices(py, indices, length)
            },
        }
    }
}

#[pymethods]
impl AnnDataSet {
    #[new]
    fn new(adatas: Vec<(String, AnnData)>, filename: &str, add_key: &str) -> Self {
        let data = adatas.into_iter().map(|(k, v)| (k, v.0.extract().unwrap())).collect();
        AnnDataSet::wrap(anndata::AnnDataSet::new(data, filename, add_key).unwrap())
    }

    /// Shape of data matrix (`n_obs`, `n_vars`).
    #[getter]
    fn shape(&self) -> (usize, usize) { (self.n_obs(), self.n_vars()) }

    /// Number of observations.
    #[getter]
    fn n_obs(&self) -> usize { self.0.inner().n_obs() }

    /// Number of variables/features.
    #[getter]
    fn n_vars(&self) -> usize { self.0.inner().n_vars() }

    /// Names of variables.
    #[getter]
    fn var_names(&self) -> Vec<String> { self.0.inner().var_names() }

    #[pyo3(text_signature = "($self, names)")]
    fn var_ix(&self, names: Vec<String>) -> Vec<usize> {
        self.0.inner().var_ix(&names).unwrap()
    }

    /// Names of observations.
    #[getter]
    fn obs_names(&self) -> Vec<String> { self.0.inner().obs_names() }

    #[pyo3(text_signature = "($self, names)")]
    fn obs_ix(&self, names: Vec<String>) -> Vec<usize> {
        self.0.inner().obs_ix(&names).unwrap()
    }

    /// :class:`.PyStackedMatrixElem` of shape `n_obs` x `n_vars`.
    #[getter(X)]
    fn get_x(&self) -> PyStackedMatrixElem {
        PyStackedMatrixElem(self.0.inner().anndatas.inner().get_x().clone())
    }

    /// :class:`.PyElemCollection`.
    #[getter(uns)]
    fn get_uns(&self) -> PyElemCollection {
        PyElemCollection(self.0.inner().get_uns().clone())
    }

    #[setter(uns)]
    fn set_uns<'py>(
        &self,
        py: Python<'py>,
        uns: Option<HashMap<String, &'py PyAny>>,
    ) -> PyResult<()>
    {
        let data: PyResult<_> = uns.map(|mut x| x.drain().map(|(k, v)|
            Ok((k, v.into_rust(py)?))
        ).collect()).transpose();
        self.0.inner().set_uns(data?).unwrap();
        Ok(())
    }

    /// Subsetting the AnnDataSet object.
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
    /// Optional[AnnDataSet]
    #[pyo3(text_signature = "($self, obs_indices, var_indices, out)")]
    fn subset<'py>(
        &self,
        py: Python<'py>,
        obs_indices: Option<&'py PyAny>,
        var_indices: Option<&'py PyAny>,
        out: Option<&str>,
    ) -> PyResult<Option<AnnDataSet>> {
        let i = obs_indices.and_then(|x| self.normalize_index(py, x, 0).unwrap());
        let j = var_indices.and_then(|x| self.normalize_index(py, x, 1).unwrap());
        Ok(match out {
            None => {
                self.0.inner().subset(i.as_ref().map(Vec::as_slice), j.as_ref().map(Vec::as_slice)).unwrap();
                None
            },
            Some(dir) => Some(AnnDataSet::wrap(
                self.0.inner().copy(i.as_ref().map(Vec::as_slice), j.as_ref().map(Vec::as_ref), dir).unwrap()
            )),
        })
    }

    /// Return an iterator over the rows of the data matrix X.
    /// 
    /// Parameters
    /// ----------
    /// chunk_size : int
    ///     Row size of a single chunk.
    #[args(chunk_size = "500")]
    #[pyo3(text_signature = "($self, chunk_size, /)")]
    #[pyo3(name = "chunked_X")]
    fn chunked_x(&self, chunk_size: usize) -> PyStackedChunkedMatrix {
        self.get_x().chunked(chunk_size)
    }
 
    /// :class:`.StackedAnnData`.
    #[getter(adatas)]
    fn adatas(&self) -> StackedAnnData { StackedAnnData(self.0.inner().anndatas.clone()) }

    /// Copy the AnnDataSet object to a new location.
    /// 
    /// Copying AnnDataSet object will copy both the object itself and assocated
    /// AnnData objects.
    /// 
    /// Parameters
    /// ----------
    /// dirname
    ///     Name of the directory used to store the result.
    /// 
    /// Returns
    /// -------
    /// AnnDataSet
    #[pyo3(text_signature = "($self, dirname)")]
    fn copy(&self, dirname: &str) -> Self { AnnDataSet::wrap(self.0.inner().copy(None, None, dirname).unwrap()) }

    /// Close the AnnDataSet object.
    #[pyo3(text_signature = "($self)")]
    fn close(&self) {
        if let Some(dataset) = self.0.extract() { dataset.close().unwrap(); }
    }

    /// If the AnnData object has been closed.
    /// Returns
    /// -------
    /// bool
    #[pyo3(text_signature = "($self)")]
    fn is_closed(&self) -> bool { self.0.inner().0.is_none() }

    fn __repr__(&self) -> String {
        if self.is_closed() {
            "Closed AnnDataSet object".to_string()
        } else {
            format!("{}", self.0.inner().deref())
        }
    }

    fn __str__(&self) -> String { self.__repr__() }
}

def_df_accessor!(AnnDataSet, { obs, var });

def_arr_accessor!(
    AnnDataSet,
    PyAxisArrays,
    Option<HashMap<String, &'py PyAny>>,
    { obsm, obsp, varm, varp }
);

/// Read `.h5ad`-formatted hdf5 file.
///
/// Parameters
/// ----------
///
/// filename
///     File name of data file.
/// backed
///     If `'r'`, the file is opened in read-only mode.
///     If you want to modify the AnnData object, you need to choose `'r+'`.
#[pyfunction(backed = "\"r+\"")]
#[pyo3(text_signature = "(filename, backed)")]
pub fn read<'py>(py: Python<'py>, filename: &str, backed: Option<&str>) -> PyResult<PyObject> {
    match backed {
        Some("r") => {
            let file = hdf5::File::open(filename).unwrap();
            Ok(AnnData::wrap(anndata::AnnData::read(file).unwrap()).into_py(py))
        },
        Some("r+") => {
            let file = hdf5::File::open_rw(filename).unwrap();
            Ok(AnnData::wrap(anndata::AnnData::read(file).unwrap()).into_py(py))
        }
        None => {
            Ok(PyModule::import(py, "anndata")?.getattr("read_h5ad")?.call1((filename,))?.to_object(py))
        }
        _ => panic!("Unkown mode"),
    }
}

/// Read Matrix Market file.
///
/// Parameters
/// ----------
///
/// filename 
///     File name of matrix market file.
/// storage
///     File name of the output ".h5ad" file.
/// sorted
///     Indicate whether the entries in the matrix market file have been
///     sorted by row and column indices. When the data is sorted, only
///     a small amount of (constant) memory will be used.
#[pyfunction(sorted = "false")]
#[pyo3(text_signature = "(input, output, sorted)")]
pub fn read_mtx<'py>(py: Python<'py>, filename: &str, storage: &str, sorted: bool) -> PyResult<AnnData> {
    let anndata = AnnData::new(
        py, storage, None, None, None, None, None, None, None, None, None, None
    )?;
    anndata.import_mtx(filename, sorted);
    Ok(anndata)
}

#[pyfunction(has_header = true, index_column = "None", delimiter = "b','")]
#[pyo3(text_signature = "(filename, storage, has_header, index_column, delimiter)")]
pub fn read_csv(
    filename: &str,
    storage: &str,
    has_header: bool,
    index_column: Option<usize>,
    delimiter: u8,
) -> AnnData
{
    let anndata = anndata::AnnData::new(storage, 0, 0).unwrap();
    anndata.import_csv(filename, has_header, index_column, delimiter).unwrap();
    AnnData::wrap(anndata)
}


/// Read and stack vertically multiple `.h5ad`-formatted hdf5 files.
///
/// Parameters
/// ----------
/// files
///     List of key and file name pairs.
/// storage
///     File name of the output file containing the AnnDataSet object.
/// add_key
///     The column name in obs to store the keys
/// 
/// Returns
/// -------
/// AnnDataSet
#[pyfunction(add_key= "\"sample\"")]
#[pyo3(text_signature = "(files, storage)")]
pub fn create_dataset<'py>(
    py: Python<'py>,
    files: Vec<(String, &'py PyAny)>,
    storage: &str,
    add_key: &str
) -> PyResult<AnnDataSet> {
    let adatas = files.into_iter().map(|(key, obj)| {
        let data = if obj.is_instance_of::<pyo3::types::PyString>()? {
            read(py, obj.extract::<&str>()?, Some("r"))?.extract::<AnnData>(py)?
        } else if obj.is_instance_of::<AnnData>()? {
            obj.extract::<AnnData>()?
        } else {
            todo!()
        }.0.extract().unwrap();
        Ok((key, data))
    }).collect::<PyResult<_>>()?;
    Ok(AnnDataSet::wrap(anndata::AnnDataSet::new(adatas, storage, add_key).unwrap()))
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
#[pyfunction(anndatas = "None", mode = "\"r+\"", no_check = "false")]
#[pyo3(text_signature = "(filename, update_data_locations, mode, no_check)")]
pub fn read_dataset(
    filename: &str,
    update_data_locations: Option<HashMap<String, String>>,
    mode: &str,
    no_check: bool,
) -> AnnDataSet {
    let file = match mode {
        "r" => hdf5::File::open(filename).unwrap(),
        "r+" => hdf5::File::open_rw(filename).unwrap(),
        _ => panic!("Unkown mode"),
    };
    let data = anndata::AnnDataSet::read(file, update_data_locations, !no_check).unwrap();
    AnnDataSet::wrap(data)
}

pub struct PyAnnData<'py>(&'py PyAny);

impl<'py> Deref for PyAnnData<'py> {
    type Target = PyAny;

    fn deref(&self) -> &Self::Target { self.0 }
}

impl<'py> PyAnnData<'py> {
    pub fn new(py: Python<'py>) -> PyResult<Self> {
        PyModule::import(py, "anndata")?.call_method0("AnnData")?.extract()
    }

    fn get_item<T>(&'py self, slot: &str, key: &str) -> Result<Option<T>>
    where
        &'py PyAny: PyToRust<T>,
    {
        let data = self.getattr(slot)?.call_method1("__getitem__", (key,))
            .ok().map(|x| x.into_rust(self.py()));
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
        self.getattr(slot)?.call_method1("__setitem__", (key, new_d))?;
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
        Python::with_gil(|py| if isinstance_of_pyanndata(py, obj)? {
            Ok(PyAnnData(obj))
        } else {
            Err(PyTypeError::new_err("Not a Python AnnData object"))
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

    fn n_obs(&self) -> usize { self.0.getattr("n_obs").unwrap().extract().unwrap() }
    fn n_vars(&self) -> usize { self.0.getattr("n_vars").unwrap().extract().unwrap() }

    fn obs_names(&self) -> Vec<String> { self.0.getattr("obs_names").unwrap().extract().unwrap() }
    fn var_names(&self) -> Vec<String> { self.0.getattr("var_names").unwrap().extract().unwrap() }

    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()> {
        if self.getattr("obs")?.getattr("empty")?.is_true()? {
            let py = self.py();
            let df = py.import("pandas")?
                .call_method("DataFrame", (), Some(&[("index", index.names)].into_py_dict(py)))?;
            self.setattr("obs", df)?;
        } else {
            self.setattr("obs_names", index.names)?;
        }
        Ok(())
    }
    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> {
        if self.getattr("var")?.getattr("empty")?.is_true()? {
            let py = self.py();
            let df = py.import("pandas")?
                .call_method("DataFrame", (), Some(&[("index", index.names)].into_py_dict(py)))?;
            self.setattr("var", df)?;
        } else {
            self.setattr("var_names", index.names)?;
        }
        Ok(())
    }

    fn obs_ix(&self, _names: &[String]) -> Result<Vec<usize>> {todo!()}
    fn var_ix(&self, _names: &[String]) -> Result<Vec<usize>> {todo!()}

    fn read_obs(&self) -> Result<DataFrame> {
        let py = self.py();
        let df = py.import("polars")?.call_method1("from_pandas", (self.0.getattr("obs")?,))?;
        Ok(df.into_rust(py)?)
    }
    fn read_var(&self) -> Result<DataFrame> {
        let py = self.py();
        let df = py.import("polars")?.call_method1("from_pandas", (self.0.getattr("var")?,))?;
        Ok(df.into_rust(py)?)
    }

    fn set_obs(&self, obs_: Option<DataFrame>) -> Result<()> {
        match obs_ {
            None => { self.0.setattr("obs", None::<PyObject>)?; },
            Some(obs) => {
                let py = self.py();
                let df = obs.rust_into_py(py)?.call_method0(py, "to_pandas")?;
                let index = self.getattr("obs")?.getattr("index")?;
                if !index.getattr("empty")?.is_true()? {
                    self.setattr("obs", df.call_method1(py, "set_index", (index,))?)?;
                } else {
                    self.setattr("obs", df)?;
                }
            },
        }
        Ok(())
    }
    fn set_var(&self, var_: Option<DataFrame>) -> Result<()> {
        match var_ {
            None => { self.0.setattr("var", None::<PyObject>)?; },
            Some(var) => {
                let py = self.py();
                let df = var.rust_into_py(py)?.call_method0(py, "to_pandas")?;
                let index = self.getattr("var")?.getattr("index")?;
                if !index.getattr("empty")?.is_true()? {
                    self.setattr("var", df.call_method1(py, "set_index", (index,))?)?;
                } else {
                    self.setattr("var", df)?;
                }
            },
        }
        Ok(())
    }

    fn uns_keys(&self) -> Vec<String> { self.get_keys("uns").unwrap() }
    fn obsm_keys(&self) -> Vec<String> { self.get_keys("obsm").unwrap() }
    fn obsp_keys(&self) -> Vec<String> { self.get_keys("obsp").unwrap() }
    fn varm_keys(&self) -> Vec<String> { self.get_keys("varm").unwrap() }
    fn varp_keys(&self) -> Vec<String> { self.get_keys("varp").unwrap() }

    fn read_uns_item(&self, key: &str) -> Result<Option<Box<dyn Data>>> { self.get_item("uns", key) }
    fn read_obsm_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> { self.get_item("obsm", key) }
    fn read_obsp_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> { self.get_item("obsp", key) }
    fn read_varm_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> { self.get_item("varm", key) }
    fn read_varp_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> { self.get_item("varp", key) }

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