pub mod dataset;
pub mod iterator;
pub mod utils;

use iterator::PyChunkedMatrix;
use utils::conversion::{
    to_py_df, to_rust_df,
    to_rust_data1, to_rust_data2,
    to_py_data1, to_py_data2,
};

use anndata_rs::{
    base::AnnData,
    anndata_trait::DataType,
    element::{Elem, MatrixElem, MatrixElemOptional, DataFrameElem},
};
use pyo3::{
    prelude::*,
    exceptions::PyTypeError,
    PyResult, Python,
};
use std::collections::HashMap;
use rand::SeedableRng;
use rand::Rng;

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
    fn n_obs(&self) -> usize { self.0.n_obs() }

    #[getter]
    fn n_vars(&self) -> usize { self.0.n_vars() }

    fn set_x<'py>(&self, py: Python<'py>, data: &'py PyAny) -> PyResult<()> {
            self.0.set_x(&to_rust_data2(py, data)?).unwrap();
            Ok(())
    }

    fn get_x(&self) -> Option<PyMatrixElemOptional> {
        if self.0.x.is_empty() {
            None
        } else {
            Some(PyMatrixElemOptional(self.0.x.clone()))
        }
    }

    fn get_obs(&self) -> Option<PyDataFrameElem> {
        if self.0.obs.is_empty() {
            None
        } else {
            Some(PyDataFrameElem(self.0.obs.clone()))
        }
    }

    fn set_obs<'py>(&self, df: &'py PyAny) -> PyResult<()> {
        self.0.set_obs(&to_rust_df(df)?).unwrap();
        Ok(())
    }

    fn get_obsm(&self, key: &str) -> PyMatrixElem {
        PyMatrixElem(self.0.obsm.get(key).unwrap().clone())
    }

    fn set_obsm<'py>(&mut self, py: Python<'py>, mut obsm: HashMap<String, &'py PyAny>) {
        let obsm_ = obsm.drain().map(|(k, v)| (k, to_rust_data2(py, v).unwrap())).collect();
        self.0.set_obsm(&obsm_).unwrap();
    }
    
    fn list_obsm(&self) -> Vec<String> {
        self.0.obsm.keys().map(|x| x.to_string()).collect()
    }

    fn add_obsm<'py>(&mut self, py: Python<'py>, key: &str, data: &'py PyAny) -> PyResult<()> {
        self.0.add_obsm(key, &to_rust_data2(py, data)?).unwrap();
        Ok(())
    }

    fn get_obsp(&self, key: &str) -> PyMatrixElem {
        PyMatrixElem(self.0.obsp.get(key).unwrap().clone())
    }

    fn set_obsp<'py>(&mut self, py: Python<'py>, mut obsp: HashMap<String, &'py PyAny>) {
        let obsp_ = obsp.drain().map(|(k, v)| (k, to_rust_data2(py, v).unwrap())).collect();
        self.0.set_obsp(&obsp_).unwrap();
    }
    
    fn list_obsp(&self) -> Vec<String> {
        self.0.obsp.keys().map(|x| x.to_string()).collect()
    }

    fn add_obsp<'py>(&mut self, py: Python<'py>, key: &str, data: &'py PyAny) -> PyResult<()> {
        self.0.add_obsp(key, &to_rust_data2(py, data)?).unwrap();
        Ok(())
    }

    fn get_var(&self) -> Option<PyDataFrameElem> {
        if self.0.var.is_empty() {
            None
        } else {
            Some(PyDataFrameElem(self.0.var.clone()))
        }
    }

    fn set_var<'py>(&mut self, df: &'py PyAny) -> PyResult<()> {
        self.0.set_var(&to_rust_df(df)?).unwrap();
        Ok(())
    }

    fn get_varm(&self) -> HashMap<String, PyMatrixElem> {
        self.0.varm.iter().map(|(k, x)| (k.clone(), PyMatrixElem(x.clone())))
            .collect()
    }

    fn set_varm<'py>(&mut self, py: Python<'py>, mut varm: HashMap<String, &'py PyAny>) {
        let varm_ = varm.drain().map(|(k, v)| (k, to_rust_data2(py, v).unwrap())).collect();
        self.0.set_varm(&varm_).unwrap();
    }
    
    fn list_varm(&self) -> Vec<String> {
        self.0.varm.keys().map(|x| x.to_string()).collect()
    }

    fn add_varm<'py>(&mut self, py: Python<'py>, key: &str, data: &'py PyAny) -> PyResult<()> {
        self.0.add_varm(key, &to_rust_data2(py, data)?).unwrap();
        Ok(())
    }

    fn get_varp(&self) -> HashMap<String, PyMatrixElem> {
        self.0.varp.iter().map(|(k, x)| (k.clone(), PyMatrixElem(x.clone())))
            .collect()
    }

    fn set_varp<'py>(
        &mut self,
        py: Python<'py>,
        mut varp: HashMap<String, &'py PyAny>
    )
    {
        let varp_ = varp.drain().map(|(k, v)| (k, to_rust_data2(py, v).unwrap())).collect();
        self.0.set_varp(&varp_).unwrap();
    }
    
    fn list_varp(&self) -> Vec<String> {
        self.0.varp.keys().map(|x| x.to_string()).collect()
    }

    fn add_varp<'py>(&mut self, py: Python<'py>, key: &str, data: &'py PyAny) -> PyResult<()> {
        self.0.add_varp(key, &to_rust_data2(py, data)?).unwrap();
        Ok(())
    }

    fn get_uns(&self, key: &str) -> PyElem {
        PyElem(self.0.uns.get(key).unwrap().clone())
    }

    fn set_uns<'py>(
        &mut self,
        py: Python<'py>,
        mut uns: HashMap<String, &'py PyAny>,
    )
    {
        let uns_ = uns.drain().map(|(k, v)| (k, to_rust_data1(py, v).unwrap())).collect();
        self.0.set_uns(&uns_).unwrap();
    }
    
    fn list_uns(&self) -> Vec<String> {
        self.0.uns.keys().map(|x| x.to_string()).collect()
    }

    fn add_uns<'py>(&mut self, py: Python<'py>, key: &str, data: &'py PyAny) -> PyResult<()> {
        self.0.add_uns(key, &to_rust_data1(py, data)?).unwrap();
        Ok(())
    }

    fn subset_rows(&mut self, idx: Vec<usize>) {
        self.0.subset_obs(idx.as_slice());
    }

    fn subset_cols(&mut self, idx: Vec<usize>) {
        self.0.subset_var(idx.as_slice());
    }

    fn subset(&mut self, ridx: Vec<usize>, cidx: Vec<usize>) {
        self.0.subset(ridx.as_slice(), cidx.as_slice());
    }

    fn filename(&self) -> String { self.0.filename() }

    fn write(&self, filename: &str) {
        self.0.write(filename).unwrap();
    }

    fn read_mtx(&self, filename: &str, sorted: bool) {
        if utils::is_gzipped(filename) {
            let f = std::fs::File::open(filename).unwrap();
            let mut reader = std::io::BufReader::new(flate2::read::MultiGzDecoder::new(f));
            self.0.read_matrix_market(&mut reader, sorted).unwrap();
        } else {
            let f = std::fs::File::open(filename).unwrap();
            let mut reader = std::io::BufReader::new(f);
            self.0.read_matrix_market(&mut reader, sorted).unwrap();
        }
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyElem(Elem);

#[pymethods]
impl PyElem {
    fn enable_cache(&self) { self.0.enable_cache() }

    fn disable_cache(&self) { self.0.disable_cache() }

    fn is_scalar(&self) -> bool {
        match self.0.0.lock().unwrap().dtype {
            DataType::Scalar(_) => true,
            _ => false,
        }
    }

    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> PyResult<Py<PyAny>> {
        if subscript.eq(py.eval("...", None, None)?)? ||
            subscript.eq(py.eval("slice(None, None, None)", None, None)?)? {
            to_py_data1(py, self.0.read().unwrap())
        } else {
            Err(PyTypeError::new_err(
                "Please use '...' or ':' to retrieve value"
            ))
        }
    }

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}

#[pyclass]
#[repr(transparent)]
pub struct PyMatrixElem(MatrixElem);

#[pymethods]
impl PyMatrixElem {
    fn enable_cache(&mut self) { self.0.enable_cache() }

    fn disable_cache(&mut self) { self.0.disable_cache() }

    #[getter]
    fn shape(&self) -> (usize, usize) { (self.0.nrows(), self.0.ncols()) }

    // TODO: efficient partial data reading
    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> PyResult<Py<PyAny>> {
        if subscript.eq(py.eval("...", None, None)?)? ||
            subscript.eq(py.eval("slice(None, None, None)", None, None)?)? {
            to_py_data2(py, self.0.read().unwrap())
        } else {
            let data = to_py_data2(py, self.0.read().unwrap())?;
            data.call_method1(py, "__getitem__", (subscript,))
        }
    }
 
    #[args(
        replace = true,
        seed = 2022,
    )]
    fn chunk<'py>(
        &self,
        py: Python<'py>,
        size: usize,
        replace: bool,
        seed: u64,
    ) -> PyResult<PyObject> {
        let length = self.0.nrows();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let idx: Vec<usize> = if replace {
            std::iter::repeat_with(|| rng.gen_range(0..length)).take(size).collect()
        } else {
            rand::seq::index::sample(&mut rng, length, size).into_vec()
        };
        to_py_data2(py, self.0.0.lock().unwrap().read_rows(idx.as_slice()).unwrap())
    }

    fn chunked(&self, chunk_size: usize) -> PyChunkedMatrix {
        PyChunkedMatrix(self.0.chunked(chunk_size))
    }

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}

#[pyclass]
#[repr(transparent)]
pub struct PyMatrixElemOptional(MatrixElemOptional);

#[pymethods]
impl PyMatrixElemOptional {
    fn enable_cache(&mut self) { self.0.enable_cache() }

    fn disable_cache(&mut self) { self.0.disable_cache() }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.0.nrows().unwrap_or(0), self.0.ncols().unwrap_or(0))
    }

    // TODO: efficient partial data reading
    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> PyResult<Py<PyAny>> {
        if subscript.eq(py.eval("...", None, None)?)? ||
            subscript.eq(py.eval("slice(None, None, None)", None, None)?)? {
            to_py_data2(py, self.0.read().unwrap().unwrap())
        } else {
            let data = to_py_data2(py, self.0.read().unwrap().unwrap())?;
            data.call_method1(py, "__getitem__", (subscript,))
        }
    }
 
    #[args(
        replace = true,
        seed = 2022,
    )]
    fn chunk<'py>(
        &self,
        py: Python<'py>,
        size: usize,
        replace: bool,
        seed: u64,
    ) -> PyResult<PyObject> {
        let length = self.0.nrows().unwrap_or(0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let idx: Vec<usize> = if replace {
            std::iter::repeat_with(|| rng.gen_range(0..length)).take(size).collect()
        } else {
            rand::seq::index::sample(&mut rng, length, size).into_vec()
        };
        to_py_data2(py, self.0.0.lock().unwrap().as_ref().unwrap().read_rows(idx.as_slice()).unwrap())
    }

    fn chunked(&self, chunk_size: usize) -> PyChunkedMatrix {
        PyChunkedMatrix(self.0.chunked(chunk_size))
    }

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}




#[pyclass]
#[repr(transparent)]
pub struct PyDataFrameElem(DataFrameElem);

#[pymethods]
impl PyDataFrameElem {
    fn enable_cache(&mut self) { self.0.enable_cache() }

    fn disable_cache(&mut self) { self.0.disable_cache() }

    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> PyResult<Option<Py<PyAny>>> {
        if subscript.eq(py.eval("...", None, None)?)? ||
            subscript.eq(py.eval("slice(None, None, None)", None, None)?)? {
            match self.0.read() {
                None => Ok(None),
                Some(x) => Ok(Some(to_py_df(x.unwrap())?)),
            }
        } else {
            match self.0.read() {
                None => Ok(None),
                Some(x) => {
                    let data = to_py_df(x.unwrap())?;
                    Ok(Some(data.call_method1(py, "__getitem__", (subscript,))?))
                },
            }
        }
    }

    fn __setitem__<'py>(
        &self,
        py: Python<'py>,
        key: &'py PyAny,
        data: &'py PyAny,
    ) -> PyResult<()> {
        match self.0.read() {
            None => Err(PyTypeError::new_err(
                "Cannot set a empty dataframe"
            )),
            Some(value) => {
                let df = to_py_df(value.unwrap())?;
                df.call_method1(py, "__setitem__", (key, data))?;
                self.0.update(&to_rust_df(df.as_ref(py)).unwrap());
                Ok(())
            },
        }
    }
 
    fn __contains__(&self, key: &str) -> bool {
        self.0.read().map_or(false, |x| x.unwrap().find_idx_by_name(key).is_some())
    }

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}

#[pyfunction]
pub fn read_anndata(filename: &str, mode: &str) -> PyResult<PyAnnData> {
    let file = match mode {
        "r" => hdf5::File::open(filename).unwrap(),
        "r+" => hdf5::File::open_rw(filename).unwrap(),
        _ => panic!("Unkown mode"),
    };
    let anndata = AnnData::read(file).unwrap();
    Ok(PyAnnData(anndata))
}