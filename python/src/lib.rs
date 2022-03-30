pub mod iterator;
pub mod utils;

use iterator::ChunkedMatrix;

use utils::conversion::{
    to_py_df, to_rust_df,
    to_rust_data1, to_rust_data2,
    to_py_data1, to_py_data2,
};

use pyo3::{
    prelude::*,
    pymodule, types::PyModule, PyResult, Python,
};
use std::collections::HashMap;

use anndata_rs::{
    base::AnnData,
    element::{Elem, MatrixElem, DataFrameElem},
};

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyAnnData(pub AnnData);

#[pymethods]
impl PyAnnData {
    #[new]
    fn new(filename: &str, n_obs: usize, n_vars: usize) -> Self {
        PyAnnData(AnnData::new(filename, n_obs, n_vars).unwrap())
    }

    #[getter]
    fn n_obs(&self) -> PyResult<usize> {
        Ok(self.0.n_obs)
    }

    #[getter]
    fn n_vars(&self) -> PyResult<usize> {
        Ok(self.0.n_vars)
    }

    fn set_x(&mut self, data: PyObject) -> PyResult<()> {
        Python::with_gil(|py| {
            self.0.set_x(&to_rust_data2(py, data.as_ref(py))?).unwrap();
            Ok(())
        })
    }

    fn get_x(&self) -> PyResult<Option<PyMatrixElem>> {
        Ok(self.0.x.clone().map(PyMatrixElem))
    }

    fn get_obs(&self) -> PyResult<Option<PyDataFrameElem>> {
        Ok(self.0.obs.clone().map(PyDataFrameElem))
    }

    fn set_obs(&mut self, df: PyObject) -> PyResult<()> {
        Python::with_gil(|py| {
            self.0.set_obs(&to_rust_df(df.as_ref(py))?).unwrap();
            Ok(())
        })
    }

    fn get_obsm(&self, key: &str) -> PyResult<PyMatrixElem> {
        Ok(PyMatrixElem(self.0.obsm.get(key).unwrap().clone()))
    }

    fn set_obsm(&mut self, mut obsm: HashMap<String, PyObject>) -> PyResult<()> {
        Python::with_gil(|py| {
            let obsm_ = obsm.drain().map(|(k, v)| (k, to_rust_data2(py, v.as_ref(py)).unwrap())).collect();
            self.0.set_obsm(&obsm_).unwrap();
            Ok(())
        })
    }
    
    fn list_obsm(&self) -> PyResult<Vec<String>> {
        Ok(self.0.obsm.keys().map(|x| x.to_string()).collect())
    }

    fn add_obsm(&mut self, key: &str, data: PyObject) -> PyResult<()> {
        Python::with_gil(|py| {
            self.0.add_obsm(key, &to_rust_data2(py, data.as_ref(py))?).unwrap();
            Ok(())
        })
    }

    fn get_var(&self) -> PyResult<Option<PyDataFrameElem>> {
        Ok(self.0.var.clone().map(PyDataFrameElem))
    }

    fn set_var(&mut self, df: PyObject) -> PyResult<()> {
        Python::with_gil(|py| {
            self.0.set_var(&to_rust_df(df.as_ref(py))?).unwrap();
            Ok(())
        })
    }

    fn get_varm(&self) -> PyResult<HashMap<String, PyMatrixElem>> {
        let varm = self.0.varm.iter()
            .map(|(k, x)| (k.clone(), PyMatrixElem(x.clone())))
            .collect();
        Ok(varm)
    }

    fn set_varm(&mut self, mut varm: HashMap<String, PyObject>) -> PyResult<()> {
        Python::with_gil(|py| {
            let varm_ = varm.drain().map(|(k, v)| (k, to_rust_data2(py, v.as_ref(py)).unwrap())).collect();
            self.0.set_varm(&varm_).unwrap();
            Ok(())
        })
    }
    
    fn list_varm(&self) -> PyResult<Vec<String>> {
        Ok(self.0.varm.keys().map(|x| x.to_string()).collect())
    }

    fn add_varm(&mut self, key: &str, data: PyObject) -> PyResult<()> {
        Python::with_gil(|py| {
            self.0.add_varm(key, &to_rust_data2(py, data.as_ref(py))?).unwrap();
            Ok(())
        })
    }

    fn get_uns(&self, key: &str) -> PyResult<PyElem> {
        Ok(PyElem(self.0.uns.get(key).unwrap().clone()))
    }

    fn set_uns(&mut self, mut uns: HashMap<String, PyObject>) -> PyResult<()> {
        Python::with_gil(|py| {
            let uns_ = uns.drain().map(|(k, v)| (k, to_rust_data1(py, v.as_ref(py)).unwrap())).collect();
            self.0.set_uns(&uns_).unwrap();
            Ok(())
        })
    }
    
    fn list_uns(&self) -> PyResult<Vec<String>> {
        Ok(self.0.uns.keys().map(|x| x.to_string()).collect())
    }

    fn add_uns(&mut self, key: &str, data: PyObject) -> PyResult<()> {
        Python::with_gil(|py| {
            self.0.add_uns(key, &to_rust_data1(py, data.as_ref(py))?).unwrap();
            Ok(())
        })
    }

    fn subset_rows(&mut self, idx: Vec<usize>) -> PyResult<()> {
        self.0.subset_obs(idx.as_slice());
        Ok(())
    }

    fn subset_cols(&mut self, idx: Vec<usize>) -> PyResult<()> {
        self.0.subset_var(idx.as_slice());
        Ok(())
    }

    fn subset(&mut self, ridx: Vec<usize>, cidx: Vec<usize>) -> PyResult<()> {
        self.0.subset(ridx.as_slice(), cidx.as_slice());
        Ok(())
    }

    fn write(&self, filename: &str) -> PyResult<()> {
        self.0.write(filename).unwrap();
        Ok(())
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyElem(Elem);

#[pymethods]
impl PyElem {
    fn get_data(&self) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| to_py_data1(py, self.0.0.read_dyn_elem()))
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyMatrixElem(MatrixElem);

#[pymethods]
impl PyMatrixElem {
    fn get_data(&self) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| to_py_data2(py, self.0.0.read_dyn_elem()))
    }

    fn chunked(&self, chunk_size: usize) -> ChunkedMatrix {
        ChunkedMatrix {
            elem: self.0.clone(),
            chunk_size,
            size: self.0.nrows(),
            current_index: 0,
        }
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyDataFrameElem(DataFrameElem);

#[pymethods]
impl PyDataFrameElem {
    fn get_data(&self) -> PyResult<PyObject> {
        to_py_df(self.0.0.read_elem())
    }
}

#[pyfunction]
fn read_anndata(filename: &str, mode: &str) -> PyResult<PyAnnData> {
    let file = match mode {
        "r" => hdf5::File::open(filename).unwrap(),
        "r+" => hdf5::File::open_rw(filename).unwrap(),
        _ => panic!("Unkown mode"),
    };
    let anndata = AnnData::read(file).unwrap();
    Ok(PyAnnData(anndata))
}

#[pymodule]
fn pyanndata(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyAnnData>().unwrap();
    m.add_class::<PyMatrixElem>().unwrap();

    m.add_function(wrap_pyfunction!(read_anndata, m)?)?;

    Ok(())
}