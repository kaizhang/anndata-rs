use crate::element::*;
use crate::utils::{
    to_indices,
    conversion::{to_rust_df, to_rust_data1, to_rust_data2},
    instance::isinstance_of_pandas,
};

use anndata_rs::{
    anndata,
    element::Slot,
};
use pyo3::{
    prelude::*,
    PyResult, Python,
};
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
                fn [<get_ $field>](&self) -> PyDataFrameElem {
                    PyDataFrameElem(self.0.inner().[<get_ $field>]().clone()) 
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
                    self.0.inner().[<set_ $field>](data.as_ref()).unwrap();
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
    fn var_names(&self) -> Vec<String> {
        self.0.inner().var_names().unwrap()
    }

    /// Names of observations.
    #[getter]
    fn obs_names(&self) -> Vec<String> {
        self.0.inner().obs_names().unwrap()
    }

    /// :class:`.PyMatrixElem` of shape `n_obs` x `n_vars`.
    #[getter(X)]
    fn get_x(&self) -> PyMatrixElem {
        PyMatrixElem(self.0.inner().get_x().clone())
    }

    #[setter(X)]
    fn set_x<'py>(&self, py: Python<'py>, data: Option<&'py PyAny>) -> PyResult<()> {
        match data {
            None => self.0.inner().set_x(None).unwrap(),
            Some(d) => self.0.inner().set_x(Some(&to_rust_data2(py, d)?)).unwrap(),
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
        let n_obs = self.n_obs();
        let n_vars = self.n_vars();
        
        let i = obs_indices.map(|oidx| to_indices(py, oidx, n_obs)).transpose()?;
        let j = var_indices.map(|vidx| to_indices(py, vidx, n_vars)).transpose()?;
        Ok(match out {
            None => {
                self.0.inner().subset(
                    i.as_ref().map(Vec::as_slice),
                    j.as_ref().map(Vec::as_slice),
                );
                None
            },
            Some(file) => Some(AnnData::wrap(
                self.0.inner().copy_subset(
                    i.as_ref().map(Vec::as_slice),
                    j.as_ref().map(Vec::as_slice),
                    file,
                ).unwrap()
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
        AnnData::wrap(self.0.inner().copy(filename).unwrap())
    }

    /// Write .h5ad-formatted hdf5 file.
    /// 
    /// Parameters
    /// ----------
    /// filename
    ///     File name of the output `.h5ad` file. 
    #[pyo3(text_signature = "($self, filename)")]
    fn write(&self, filename: &str) {
        self.0.inner().write(filename).unwrap();
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
    fn var_names(&self) -> Vec<String> {
        self.0.inner().var_names().unwrap()
    }

    /// Names of observations.
    #[getter]
    fn obs_names(&self) -> Vec<String> {
        self.0.inner().obs_names().unwrap()
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
        let n_obs = self.n_obs();
        let n_vars = self.n_vars();
        
        let i = obs_indices.map(|oidx| to_indices(py, oidx, n_obs)).transpose()?;
        let j = var_indices.map(|vidx| to_indices(py, vidx, n_vars)).transpose()?;
        Ok(match out {
            None => {
                self.0.inner().subset(
                    i.as_ref().map(Vec::as_slice),
                    j.as_ref().map(Vec::as_slice),
                );
                None
            },
            Some(dir) => Some(AnnDataSet::wrap(
                self.0.inner().copy_subset(
                    i.as_ref().map(Vec::as_slice),
                    j.as_ref().map(Vec::as_slice),
                    dir,
                ).unwrap()
            )),
        })
    }
 
    /// :class:`.StackedAnnData`.
    #[getter(adatas)]
    fn adatas(&self) -> StackedAnnData {
        StackedAnnData(self.0.inner().anndatas.clone())
    }

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
    fn copy(&self, dirname: &str) -> Self {
        AnnDataSet::wrap(self.0.inner().copy(dirname).unwrap())
    }

    /// Close the AnnDataSet object.
    #[pyo3(text_signature = "($self)")]
    fn close(&self) {
        if let Some(dataset) = self.0.extract() {
            dataset.close().unwrap();
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
pub fn read(filename: &str, mode: &str) -> PyResult<AnnData> {
    let file = match mode {
        "r" => hdf5::File::open(filename).unwrap(),
        "r+" => hdf5::File::open_rw(filename).unwrap(),
        _ => panic!("Unkown mode"),
    };
    let anndata = anndata::AnnData::read(file).unwrap();
    Ok(AnnData::wrap(anndata))
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
    files: Vec<(String, &'py PyAny)>,
    storage: &str,
    add_key: &str
) -> PyResult<AnnDataSet> {
    let adatas = files.into_iter().map(|(key, obj)| {
        let data = if obj.is_instance_of::<pyo3::types::PyString>()? {
            read(obj.extract::<&str>()?, "r").unwrap()
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
/// Parameters
/// ----------
/// filename
///     File name.
/// update_data_locations
///     Mapping[str, str]: If provided, locations of component anndata files will be updated.
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