use crate::element::{
    PyElemCollection, PyAxisArrays,
    PyMatrixElem, PyDataFrameElem,
    PyStackedMatrixElem,
};

use crate::utils::{
    to_indices,
    conversion::{to_rust_df, to_rust_data1, to_rust_data2},
    instance::isinstance_of_pandas,
};

use anndata_rs::anndata;
use pyo3::{
    prelude::*,
    PyResult, Python,
};
use std::collections::HashMap;
use paste::paste;
use std::sync::{Arc, Mutex, MutexGuard};
use std::ops::Deref;
use std::ops::DerefMut;

macro_rules! def_df_accessor {
    ($name:ty, { $($field:ident),* }) => {
        paste! {
            #[pymethods]
            impl $name {
            $(
                #[getter($field)]
                fn [<get_ $field>](&self) -> Option<PyDataFrameElem> {
                    self.inner().[<get_ $field>]().0.as_ref().map(|x| 
                        PyDataFrameElem(x.clone()))
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
                    self.inner().[<set_ $field>](data.as_ref()).unwrap();
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
                #[getter($field)]
                fn [<get_ $field>](&self) -> Option<$get_type> {
                    self.inner().[<get_ $field>]().0.as_ref()
                        .map(|x| $get_type(x.clone()))
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
                    self.inner().[<set_ $field>](data?.as_ref()).unwrap();
                    Ok(())
                }

            )*
            }
        }
    }
}

/// Wrap the inner AnnData by Mutex so that we can close and
/// drop it in Python.
#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct AnnData(Arc<Mutex<Option<anndata::AnnData>>>);

pub struct InnerAnnData<'a>(MutexGuard<'a, Option<anndata::AnnData>>);

impl Deref for InnerAnnData<'_> {
    type Target = anndata::AnnData;

    fn deref(&self) -> &Self::Target {
        match &self.0.deref() {
            None => panic!("accessing a closed AnnData object"),
            Some(x) => x,
        }
    }
}

impl DerefMut for InnerAnnData<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self.0.deref_mut() {
            None => panic!("accessing a closed AnnData object"),
            Some(ref mut x) => x,
        }
    }
}

impl AnnData {
    pub fn wrap(anndata: anndata::AnnData) -> Self {
        AnnData(Arc::new(Mutex::new(Some(anndata))))
    }

    pub fn inner(&self) -> InnerAnnData<'_> {
        InnerAnnData(self.0.lock().unwrap())
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
       anndata.set_x(py, X)?;
       anndata.set_obs(py, obs)?;
       anndata.set_var(py, var)?;
       anndata.set_obsm(py, obsm)?;
       anndata.set_obsp(py, obsp)?;
       anndata.set_varm(py, varm)?;
       anndata.set_varp(py, varp)?;
       anndata.set_uns(py, uns)?;
       Ok(anndata)
    }

    #[getter]
    fn shape(&self) -> (usize, usize) { (self.n_obs(), self.n_vars()) }

    #[getter]
    fn n_obs(&self) -> usize { self.inner().n_obs() }

    #[getter]
    fn n_vars(&self) -> usize { self.inner().n_vars() }

    #[getter]
    fn var_names(&self) -> Option<Vec<String>> {
        self.inner().get_var().0.as_ref().map(|x|
            x.read().unwrap()[0].utf8().unwrap().into_iter()
            .map(|s| s.unwrap().to_string()).collect()
        )
    }

    #[getter]
    fn obs_names(&self) -> Option<Vec<String>> {
        self.inner().get_obs().0.as_ref().map(|x|
            x.read().unwrap()[0].utf8().unwrap().into_iter()
            .map(|s| s.unwrap().to_string()).collect()
        )
    }

    #[getter(X)]
    fn get_x(&self) -> Option<PyMatrixElem> {
        self.inner().get_x().0.as_ref().map(|x| PyMatrixElem(x.clone()))
    }

    #[setter(X)]
    fn set_x<'py>(&self, py: Python<'py>, data: Option<&'py PyAny>) -> PyResult<()> {
        match data {
            None => self.inner().set_x(None).unwrap(),
            Some(d) => self.inner().set_x(Some(&to_rust_data2(py, d)?)).unwrap(),
        }
        Ok(())
    }

    #[getter(uns)]
    fn get_uns(&self) -> Option<PyElemCollection> {
        self.inner().get_uns().0.as_ref().map(|x| PyElemCollection(x.clone()))
    }

    #[setter(uns)]
    fn set_uns<'py>(&self, py: Python<'py>, uns: Option<HashMap<String, &'py PyAny>>) -> PyResult<()> {
        let uns_ = uns.map(|mut x|
            x.drain().map(|(k, v)| (k, to_rust_data1(py, v).unwrap())).collect()
        );
        self.inner().set_uns(uns_.as_ref()).unwrap();
        Ok(())
    }

    fn subset<'py>(
        &self,
        py: Python<'py>,
        obs_indices: Option<&'py PyAny>,
        var_indices: Option<&'py PyAny>,
    ) -> PyResult<()> {
        let n_obs = self.n_obs();
        let n_vars = self.n_vars();
        match obs_indices {
            Some(oidx) => {
                let i = to_indices(py, oidx, n_obs)?;
                match var_indices {
                    Some(vidx) => {
                        let j = to_indices(py, vidx, n_vars)?;
                        self.inner().subset(i.as_slice(), j.as_slice());
                    },
                    None => self.inner().subset_obs(i.as_slice()),
                }
            },
            None => {
               if let Some(vidx) = var_indices {
                    let j = to_indices(py, vidx, n_vars)?;
                    self.inner().subset_var(j.as_slice());
               }
            },
        }
        Ok(())
    }
            
    #[getter]
    fn filename(&self) -> String { self.inner().filename() }

    fn write(&self, filename: &str) {
        self.inner().write(filename).unwrap();
    }

    fn close(&self) {
        let mut inner = self.0.lock().unwrap();
        if let Some(anndata) = std::mem::replace(inner.deref_mut(), None) {
            anndata.close().unwrap();
        }
    }

    fn is_closed(&self) -> bool { self.0.lock().unwrap().is_none() }

    fn import_mtx(&self, filename: &str, sorted: bool) {
        if crate::utils::is_gzipped(filename) {
            let f = std::fs::File::open(filename).unwrap();
            let mut reader = std::io::BufReader::new(flate2::read::MultiGzDecoder::new(f));
            self.inner().read_matrix_market(&mut reader, sorted).unwrap();
        } else {
            let f = std::fs::File::open(filename).unwrap();
            let mut reader = std::io::BufReader::new(f);
            self.inner().read_matrix_market(&mut reader, sorted).unwrap();
        }
    }

    fn __repr__(&self) -> String {
        if self.is_closed() {
            "Closed AnnData object".to_string()
        } else {
            format!("{}", self.inner().deref())
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

#[pyclass]
#[repr(transparent)]
pub struct AnnDataSet(Arc<Mutex<Option<anndata::AnnDataSet>>>);

pub struct InnerAnnDataSet<'a>(MutexGuard<'a, Option<anndata::AnnDataSet>>);

impl Deref for InnerAnnDataSet<'_> {
    type Target = anndata::AnnDataSet;

    fn deref(&self) -> &Self::Target {
        match &self.0.deref() {
            None => panic!("accessing a closed AnnDataSet object"),
            Some(x) => x,
        }
    }
}

impl DerefMut for InnerAnnDataSet<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self.0.deref_mut() {
            None => panic!("accessing a closed AnnDataSet object"),
            Some(ref mut x) => x,
        }
    }
}

impl AnnDataSet {
    pub fn wrap(anndata: anndata::AnnDataSet) -> Self {
        AnnDataSet(Arc::new(Mutex::new(Some(anndata))))
    }

    pub fn inner(&self) -> InnerAnnDataSet<'_> {
        InnerAnnDataSet(self.0.lock().unwrap())
    }
}

#[pymethods]
impl AnnDataSet {
    #[new]
    fn new(adatas: Vec<(String, AnnData)>, filename: &str, add_key: &str) -> Self {
        let data = adatas.into_iter().map(|(k, v)| (k, v.inner().clone())).collect();
        AnnDataSet::wrap(anndata::AnnDataSet::new(data, filename, add_key).unwrap())
    }

    #[getter]
    fn shape(&self) -> (usize, usize) { (self.n_obs(), self.n_vars()) }

    #[getter]
    fn n_obs(&self) -> usize { self.inner().n_obs() }

    #[getter]
    fn n_vars(&self) -> usize { self.inner().n_vars() }

    #[getter]
    fn var_names(&self) -> Option<Vec<String>> {
        self.inner().get_var().0.as_ref().map(|x|
            x.read().unwrap()[0].utf8().unwrap().into_iter()
            .map(|s| s.unwrap().to_string()).collect()
        )
    }

    #[getter]
    fn obs_names(&self) -> Option<Vec<String>> {
        self.inner().get_obs().0.as_ref().map(|x|
            x.read().unwrap()[0].utf8().unwrap().into_iter()
            .map(|s| s.unwrap().to_string()).collect()
        )
    }

    #[getter(X)]
    fn get_x(&self) -> PyStackedMatrixElem {
        PyStackedMatrixElem(self.inner().x.clone())
    }

    #[getter(uns)]
    fn get_uns(&self) -> Option<PyElemCollection> {
        self.inner().get_uns().0.as_ref().map(|x| PyElemCollection(x.clone()))
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
        self.inner().set_uns(data?.as_ref()).unwrap();
        Ok(())
    }

    fn close(&self) {
        let mut inner = self.0.lock().unwrap();
        if let Some(dataset) = std::mem::replace(inner.deref_mut(), None) {
            dataset.close().unwrap();
        }
    }

    fn is_closed(&self) -> bool { self.0.lock().unwrap().is_none() }

    fn __repr__(&self) -> String {
        if self.is_closed() {
            "Closed AnnDataSet object".to_string()
        } else {
            format!("{}", self.inner().deref())
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

/// Read and stack vertically multiple `.h5ad`-formatted hdf5 files.
///
/// Parameters
/// ----------
///
/// files
///     List of key and file name pairs.
/// storage
///     File name of the output file containing the AnnDataSet object.
/// add_key
///     The column name in obs to store the keys
#[pyfunction(add_key= "\"batch\"")]
#[pyo3(text_signature = "(files, storage)")]
pub fn create_dataset(files: Vec<(String, &str)>, storage: &str, add_key: &str) -> AnnDataSet {
    let adatas = files.into_iter().map(|(key, file)|
        ( key, read(file, "r").unwrap().inner().clone() )
    ).collect();
    AnnDataSet::wrap(anndata::AnnDataSet::new(adatas, storage, add_key).unwrap())
}


#[pyfunction(anndatas = "None", mode = "\"r+\"")]
#[pyo3(text_signature = "(filename, anndatas, mode)")]
pub fn read_dataset(
    filename: &str,
    anndatas: Option<HashMap<&str, &str>>,
    mode: &str,
) -> AnnDataSet {
    let file = match mode {
        "r" => hdf5::File::open(filename).unwrap(),
        "r+" => hdf5::File::open_rw(filename).unwrap(),
        _ => panic!("Unkown mode"),
    };
    AnnDataSet::wrap(anndata::AnnDataSet::read(file, anndatas).unwrap())
}