use crate::element::*;
use crate::utils::{
    to_indices, instance::*,
    conversion::{to_py_data1, to_py_data2, to_py_df, to_rust_df, to_rust_data1, to_rust_data2},
};

use anndata_rs::anndata::AnnDataOp;
use anndata_rs::element::DataFrameIndex;
use anndata_rs::iterator::RowIterator;
use anndata_rs::{anndata, element::Slot};
use anndata_rs::data::{Data, MatrixData};
use polars::frame::DataFrame;
use anyhow::Result;
use pyo3::{prelude::*, PyResult, Python, types::PyIterator, exceptions::PyException};
use std::collections::HashMap;
use paste::paste;
use std::ops::Deref;

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
                        to_rust_df(df_)
                    }).transpose()?;
                    self.0.inner().[<write_ $field>](data.as_ref()).unwrap();
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
                        Ok((k, to_rust_data2(py, v)?))
                    ).collect()).transpose();
                    self.0.inner().[<set_ $field>](data?.as_ref()).unwrap();
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
pub struct AnnData(pub Slot<anndata::AnnData>);

impl AnnData {
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
    #[args(
        "*",
        filename,
        X = "None",
        n_obs = "None",
        n_vars = "None",
        obs = "None",
        var = "None",
        obsm = "None",
        obsp = "None",
        varm = "None",
        varp = "None",
        uns = "None",
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
    fn var_ix(&self, names: Vec<String>) -> Vec<usize> {
        self.0.inner().var_ix(&names).unwrap()
    }

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
    fn obs_ix(&self, names: Vec<String>) -> Vec<usize> {
        self.0.inner().obs_ix(&names).unwrap()
    }

    /// :class:`.PyMatrixElem` of shape `n_obs` x `n_vars`.
    #[getter(X)]
    fn get_x(&self) -> PyMatrixElem {
        PyMatrixElem(self.0.inner().get_x().clone())
    }

    #[setter(X)]
    fn set_x<'py>(&self, py: Python<'py>, data: Option<&'py PyAny>) -> PyResult<()> {
        match data {
            None => self.0.inner().set_x::<Box<dyn MatrixData>>(None).unwrap(),
            Some(d) => if is_iterator(py, d)? {
                panic!("Setting X by an iterator is not implemented")
            } else {
                self.0.inner().set_x(Some(&to_rust_data2(py, d)?)).unwrap()
            },
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
            x.drain().map(|(k, v)| (k, to_rust_data1(py, v).unwrap())).collect()
        );
        self.0.inner().set_uns(uns_.as_ref()).unwrap();
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
        PyStackedAxisArrays(self.0.inner().obsm.clone())
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
pub struct AnnDataSet(pub Slot<anndata::AnnDataSet>);

impl AnnDataSet {
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
        let data = adatas.into_iter().map(|(k, v)| (k, v.0.inner().clone())).collect();
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
        PyStackedMatrixElem(self.0.inner().anndatas.inner().x.clone())
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
            Ok((k, to_rust_data1(py, v)?))
        ).collect()).transpose();
        self.0.inner().set_uns(data?.as_ref()).unwrap();
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
/// mode
///     If `'r'`, the file is opened in read-only mode.
///     If you want to modify the AnnData object, you need to choose `'r+'`.
#[pyfunction(mode = "\"r+\"")]
#[pyo3(text_signature = "(filename, mode)")]
pub fn read<'py>(py: Python<'py>, filename: &str, mode: Option<&str>) -> PyResult<PyObject> {
    match mode {
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
        }.0.inner().clone();
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

impl<'py> PyAnnData<'py> {
    pub fn new(py: Python<'py>) -> PyResult<Self> {
        PyModule::import(py, "anndata")?.getattr("AnnData")?.call0()?.extract()
    }

    fn set_n_obs(&self, n_obs: usize) -> PyResult<()> {
        if self.n_obs() == 0 {
            self.0.setattr("_n_obs", n_obs)
        } else {
            Err(PyException::new_err("cannot set n_obs unless n_obs == 0"))
        }
    }

    fn set_n_vars(&self, n_vars: usize) -> PyResult<()> {
        if self.n_vars() == 0 {
            self.0.setattr("_n_vars", n_vars)
        } else {
            Err(PyException::new_err("cannot set n_vars unless n_vars == 0"))
        }
    }
}

impl<'py> FromPyObject<'py> for PyAnnData<'py> {
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        Ok(PyAnnData(ob))
    }
}

impl<'py> ToPyObject for PyAnnData<'py> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.0.into_py(py)
    }
}

impl<'py> AnnDataOp for PyAnnData<'py> {
    type MatrixIter = PyObject;
    fn n_obs(&self) -> usize { self.0.getattr("n_obs").unwrap().extract().unwrap() }
    fn n_vars(&self) -> usize { self.0.getattr("n_vars").unwrap().extract().unwrap() }
    fn obs_names(&self) -> Vec<String> {todo!()}
    fn var_names(&self) -> Vec<String> {todo!()}

    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()> {todo!()}
    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> {todo!()}
    fn obs_ix(&self, names: &[String]) -> Result<Vec<usize>> {todo!()}
    fn var_ix(&self, names: &[String]) -> Result<Vec<usize>> {todo!()}

    fn read_obs(&self) -> Result<DataFrame> { todo!() }
    fn read_var(&self) -> Result<DataFrame> {todo!()}
    fn write_obs(&self, obs_: Option<&DataFrame>) -> Result<()> {
        match obs_ {
            None => { self.0.setattr("obs", None::<PyObject>)?; },
            Some(obs) => {
                self.set_n_obs(obs.height()).unwrap_or(());
                let key = obs.get_column_names()[0];
                let py = self.0.py();
                let df = to_py_df(obs.clone())?.call_method0(py, "to_pandas")?.call_method1(py, "set_index", (key, ))?;
                self.0.setattr("obs", df)?;
            },
        }
        Ok(())
    }
    fn write_var(&self, var_: Option<&DataFrame>) -> Result<()> {
        match var_ {
            None => { self.0.setattr("var", None::<PyObject>)?; },
            Some(var) => {
                self.set_n_vars(var.height()).unwrap_or(());
                let key = var.get_column_names()[0];
                let py = self.0.py();
                let df = to_py_df(var.clone())?.call_method0(py, "to_pandas")?.call_method1(py, "set_index", (key, ))?;
                self.0.setattr("var", df)?;
            },
        }
        Ok(())
    }

    fn read_uns_item(&self, key: &str) -> Result<Box<dyn Data>> {todo!()}
    fn write_uns_item<D: Data>(&self, key: &str, data: &D) -> Result<()> {
        // TODO: remove the Box.
        let data_: Box<dyn Data> = Box::new(dyn_clone::clone(data));
        self.0.getattr("uns")?.call_method1(
            "__setitem__",
            (key, to_py_data1(self.0.py(), data_)?),
        )?;
        Ok(())
    }

    fn read_x_iter(&self, chunk_size: usize) -> Self::MatrixIter {todo!()}
    fn write_x_from_row_iter<I>(&self, data: I) -> Result<()> where I: RowIterator {todo!()}

    fn write_obsm_item<D: MatrixData>(&self, key: &str, data: &D) -> Result<()> {
        self.set_n_obs(data.nrows()).unwrap_or(());
        // TODO: remove the Box.
        let data_: Box<dyn MatrixData> = Box::new(dyn_clone::clone(data));
        self.0.getattr("obsm")?.call_method1(
            "__setitem__",
            (key, to_py_data2(self.0.py(), data_)?),
        )?;
        Ok(())
    }
    fn write_obsm_item_from_row_iter<I>(&self, key: &str, data: I) -> Result<()> where I: RowIterator {todo!()}
}