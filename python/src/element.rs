use crate::iterator::{PyChunkedMatrix, PyStackedChunkedMatrix};
use crate::utils::conversion::{
    to_py_df, to_rust_df,
    to_py_data1, to_py_data2,
    to_rust_data1, to_rust_data2,
};

use anndata_rs::{
    anndata_trait::DataType,
    element::{
        ElemTrait, Stacked,
        Elem, MatrixElem, DataFrameElem,
        ElemCollection, AxisArrays,
    },
};
use pyo3::{
    prelude::*,
    exceptions::PyTypeError,
    PyResult, Python,
};
use rand::SeedableRng;
use rand::Rng;

#[pyclass]
#[repr(transparent)]
pub struct PyElem(pub(crate) Elem);

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
pub struct PyMatrixElem(pub(crate) MatrixElem);

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
pub struct PyDataFrameElem(pub(crate) DataFrameElem);

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


#[pyclass]
#[repr(transparent)]
pub struct PyElemCollection(pub(crate) ElemCollection);

#[pymethods]
impl PyElemCollection {
    pub fn keys(&self) -> Vec<String> {
        self.0.data.lock().unwrap().keys().map(|x| x.to_string()).collect()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Option<PyObject>> {
        match self.0.data.lock().unwrap().get(key) {
            None => Ok(None),
            Some(x) => Ok(Some(to_py_data1(py, x.read().unwrap())?)),
        }
    }

    fn __setitem__<'py>(&self, py: Python<'py>, key: &str, data: &'py PyAny) -> PyResult<()> {
        self.0.insert(key, &to_rust_data1(py, data)?).unwrap();
        Ok(())
    }

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}

#[pyclass]
#[repr(transparent)]
pub struct PyAxisArrays(pub(crate) AxisArrays);

#[pymethods]
impl PyAxisArrays {
    pub fn keys(&self) -> Vec<String> {
        self.0.data.lock().unwrap().keys().map(|x| x.to_string()).collect()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Option<PyObject>> {
        match self.0.data.lock().unwrap().get(key) {
            None => Ok(None),
            Some(x) => Ok(Some(to_py_data2(py, x.read().unwrap())?)),
        }
    }

    fn __setitem__<'py>(&self, py: Python<'py>, key: &str, data: &'py PyAny) -> PyResult<()> {
        self.0.insert(key, &to_rust_data2(py, data)?).unwrap();
        Ok(())
    }

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}

#[pyclass]
#[repr(transparent)]
pub struct PyStackedMatrixElem(pub(crate) Stacked<MatrixElem>);

#[pymethods]
impl PyStackedMatrixElem {
    fn enable_cache(&mut self) { self.0.enable_cache() }

    fn disable_cache(&mut self) { self.0.disable_cache() }

    fn chunked(&self, chunk_size: usize) -> PyStackedChunkedMatrix{
        PyStackedChunkedMatrix(self.0.chunked(chunk_size))
    }

    /*
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
    */

    fn get_rows<'py>(
        &self,
        py: Python<'py>,
        indices: &'py PyAny,
    ) -> PyResult<Py<PyAny>> {
        let idx = crate::utils::to_indices(py, indices, self.0.size)?;
        to_py_data2(py, self.0.read_rows(idx.as_slice()).unwrap())
    }
 
 
    //fn __repr__(&self) -> String { format!("{}", self.0) }

    //fn __str__(&self) -> String { self.__repr__() }
}

