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

macro_rules! def_df_accessor {
    ($name:ty, { $($field:ident),* }) => {
        paste! {
            #[pymethods]
            impl $name {
            $(
                #[getter($field)]
                fn [<get_ $field>](&self) -> Option<PyDataFrameElem> {
                    self.0.[<get_ $field>]().lock().unwrap().as_ref().map(|x| 
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
                    self.0.[<set_ $field>](data.as_ref()).unwrap();
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
                fn [<get_ $field>](&self) -> $get_type {
                    $get_type(self.0.[<get_ $field>]().clone())
                }

                #[setter($field)]
                fn [<set_ $field>]<'py>(
                    &mut self,
                    py: Python<'py>,
                    mut $field: $set_type
                ) -> PyResult<()>
                {
                    let x: PyResult<_> = $field.drain().map(|(k, v)|
                        Ok((k, to_rust_data2(py, v)?))
                    ).collect();
                    self.0.[<set_ $field>](&x?).unwrap();
                    Ok(())
                }
            )*
            }
        }
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct AnnData(pub anndata::AnnData);

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
    ) -> PyResult<Self> {
        let mut anndata = AnnData(anndata::AnnData::new(
            filename, n_obs.unwrap_or(0), n_vars.unwrap_or(0)
        ).unwrap());
       anndata.set_x(py, X)?;
       anndata.set_obs(py, obs)?;
       anndata.set_var(py, var)?;
       if let Some(d) = obsm { anndata.set_obsm(py, d)?; }
       Ok(anndata)
    }

    #[getter]
    fn shape(&self) -> (usize, usize) { (self.n_obs(), self.n_vars()) }

    #[getter]
    fn n_obs(&self) -> usize { self.0.n_obs() }

    #[getter]
    fn n_vars(&self) -> usize { self.0.n_vars() }

    #[getter]
    fn var_names(&self) -> PyObject {
        todo!()
    }

    #[getter]
    fn obs_names(&self) -> PyObject {
        todo!()
    }

    #[getter(X)]
    fn get_x(&self) -> Option<PyMatrixElem> {
        self.0.x.lock().unwrap().as_ref().map(|x| PyMatrixElem(x.clone()))
    }

    #[setter(X)]
    fn set_x<'py>(&self, py: Python<'py>, data: Option<&'py PyAny>) -> PyResult<()> {
        match data {
            None => self.0.set_x(None).unwrap(),
            Some(d) => self.0.set_x(Some(&to_rust_data2(py, d)?)).unwrap(),
        }
        Ok(())
    }

    #[getter(uns)]
    fn get_uns(&self) -> PyElemCollection { PyElemCollection(self.0.get_uns().clone()) }

    #[setter(uns)]
    fn set_uns<'py>(&mut self, py: Python<'py>, mut uns: HashMap<String, &'py PyAny>) {
        let uns_ = uns.drain().map(|(k, v)| (k, to_rust_data1(py, v).unwrap())).collect();
        self.0.set_uns(&uns_).unwrap();
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
                        self.0.subset(i.as_slice(), j.as_slice());
                    },
                    None => self.0.subset_obs(i.as_slice()),
                }
            },
            None => {
               if let Some(vidx) = var_indices {
                    let j = to_indices(py, vidx, n_vars)?;
                    self.0.subset_var(j.as_slice());
               }
            },
        }
        Ok(())
    }
            
    #[getter]
    fn filename(&self) -> String { self.0.filename() }

    fn write(&self, filename: &str) {
        self.0.write(filename).unwrap();
    }

    fn import_mtx(&self, filename: &str, sorted: bool) {
        if crate::utils::is_gzipped(filename) {
            let f = std::fs::File::open(filename).unwrap();
            let mut reader = std::io::BufReader::new(flate2::read::MultiGzDecoder::new(f));
            self.0.read_matrix_market(&mut reader, sorted).unwrap();
        } else {
            let f = std::fs::File::open(filename).unwrap();
            let mut reader = std::io::BufReader::new(f);
            self.0.read_matrix_market(&mut reader, sorted).unwrap();
        }
    }

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}

def_df_accessor!(AnnData, { obs, var });

def_arr_accessor!(
    AnnData,
    PyAxisArrays,
    HashMap<String, &'py PyAny>,
    { obsm, obsp, varm, varp }
);

#[pyclass]
#[repr(transparent)]
pub struct AnnDataSet(pub anndata::AnnDataSet);

#[pymethods]
impl AnnDataSet {
    #[new]
    fn new(adatas: Vec<(String, AnnData)>, filename: &str) -> Self {
        let data = adatas.into_iter().map(|(k, v)| (k, v.0)).collect();
        AnnDataSet(anndata::AnnDataSet::new(data, filename).unwrap())
    }

    #[getter(X)]
    fn get_x(&self) -> PyStackedMatrixElem {
        PyStackedMatrixElem(self.0.x.clone())
    }

    #[getter]
    fn n_obs(&self) -> usize { self.0.n_obs() }

    #[getter]
    fn n_vars(&self) -> usize { self.0.n_vars() }

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}

def_df_accessor!(AnnDataSet, { obs, var });

def_arr_accessor!(
    AnnDataSet,
    PyAxisArrays,
    HashMap<String, &'py PyAny>,
    { obsm, obsp, varm, varp }
);

#[pyfunction]
pub fn read_dataset(files: Vec<(String, &str)>, storage: &str) -> AnnDataSet {
    let adatas = files.into_iter()
        .map(|(key, file)| (key, read_h5ad(file, "r").unwrap().0)).collect();
    AnnDataSet(anndata::AnnDataSet::new(adatas, storage).unwrap())
}

#[pyfunction(mode = "\"r+\"")]
pub fn read_h5ad(filename: &str, mode: &str) -> PyResult<AnnData> {
    let file = match mode {
        "r" => hdf5::File::open(filename).unwrap(),
        "r+" => hdf5::File::open_rw(filename).unwrap(),
        _ => panic!("Unkown mode"),
    };
    let anndata = anndata::AnnData::read(file).unwrap();
    Ok(AnnData(anndata))
}

#[pyfunction(sorted = "false")]
pub fn read_mtx<'py>(py: Python<'py>, input: &str, output: &str, sorted: bool) -> PyResult<AnnData> {
    let anndata = AnnData::new(py, output, None, None, None, None, None, None)?;
    anndata.import_mtx(input, sorted);
    Ok(anndata)
}