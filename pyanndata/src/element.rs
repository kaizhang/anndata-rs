use crate::iterator::{PyChunkedMatrix, PyStackedChunkedMatrix};
use crate::utils::{
    to_indices, instance::is_none_slice,
    conversion::{
        to_py_df, to_rust_df, to_py_series, to_py_data1, to_py_data2,
        to_rust_data1, to_rust_data2,
    },
};
use anndata_rs::{
    anndata_trait::DataType,
    element::*,
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
        match self.0.dtype() {
            Some(DataType::Scalar(_)) => true,
            _ => false,
        }
    }

    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> PyResult<Py<PyAny>> {
        if is_none_slice(py, subscript)? {
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

/// An element that stores matrix objects such as arrays and sparse matrices.
/// Matrix elements support row and column indexing.
#[pyclass]
#[repr(transparent)]
pub struct PyMatrixElem(pub(crate) MatrixElem);

#[pymethods]
impl PyMatrixElem {
    /// Enable caching so that data will be stored in memory when the element
    /// is accessed the first time. Subsequent requests for the data will use
    /// the in-memory cache.
    #[pyo3(text_signature = "($self)")]
    fn enable_cache(&mut self) { self.0.inner().enable_cache() }

    /// Disable caching. In-memory cache will be cleared immediately.
    #[pyo3(text_signature = "($self)")]
    fn disable_cache(&mut self) { self.0.inner().disable_cache() }

    /// Shape of matrix.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        let guard = self.0.inner();
        (guard.nrows(), guard.ncols())
    }

    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> PyResult<Py<PyAny>> {
        if is_none_slice(py, subscript)? {
            to_py_data2(py, self.0.inner().read().unwrap())
        } else if subscript.is_instance_of::<pyo3::types::PyTuple>()? {
            let (r, c) = self.shape();
            let (ridx, cidx) = subscript.extract()?;
            if is_none_slice(py, ridx)? && is_none_slice(py, cidx)? {
                to_py_data2(py, self.0.inner().read().unwrap())
            } else if is_none_slice(py, ridx)? {
                to_py_data2(py, self.0.inner().read_columns(
                    to_indices(py, cidx, c)?.as_slice()
                ).unwrap())
            } else if is_none_slice(py, cidx)? {
                to_py_data2(py, self.0.inner().read_rows(
                    to_indices(py, ridx, r)?.as_slice()
                ).unwrap())
            } else {
                to_py_data2(py, self.0.inner().read_partial(
                    to_indices(py, ridx, r)?.as_slice(),
                    to_indices(py, cidx, c)?.as_slice(),
                ).unwrap())
            }
        } else {
            let data = to_py_data2(py, self.0.inner().read().unwrap())?;
            data.call_method1(py, "__getitem__", (subscript,))
        }
    }

    /// Return a chunk of the matrix with random indices.
    /// 
    /// Parameters
    /// ----------
    /// size
    ///     Number of rows of the returned random chunk.
    /// replace
    ///     True means random sampling of indices with replacement, False without replacement.
    /// seed
    ///     Random seed.
    /// 
    /// Returns
    /// -------
    /// A matrix
    #[args(
        replace = true,
        seed = 2022,
    )]
    #[pyo3(text_signature = "($self, size, replace, seed)")]
    fn chunk<'py>(
        &self,
        py: Python<'py>,
        size: usize,
        replace: bool,
        seed: u64,
    ) -> PyResult<PyObject> {
        let guard = self.0.inner();
        let length = guard.nrows();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let idx: Vec<usize> = if replace {
            std::iter::repeat_with(|| rng.gen_range(0..length)).take(size).collect()
        } else {
            rand::seq::index::sample(&mut rng, length, size).into_vec()
        };
        to_py_data2(py, guard.read_rows(idx.as_slice()).unwrap())
    }

    /// Return an iterator over the rows of the matrix.
    /// 
    /// Parameters
    /// ----------
    /// chunk_size
    ///     Number of rows of a single chunk.
    /// 
    /// Returns
    /// -------
    /// An iterator, of which the elements are matrices.
    #[pyo3(text_signature = "($self, chunk_size)")]
    fn chunked(&self, chunk_size: usize) -> PyChunkedMatrix {
        PyChunkedMatrix(self.0.chunked(chunk_size))
    }

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}

/// An element that stores dataframe objects.
#[pyclass]
#[repr(transparent)]
pub struct PyDataFrameElem(pub(crate) DataFrameElem);

#[pymethods]
impl PyDataFrameElem {
    /// Enable caching so that data will be stored in memory when the element
    /// is accessed the first time. Subsequent requests for the data will use
    /// the in-memory cache.
    #[pyo3(text_signature = "($self)")]
    fn enable_cache(&mut self) { self.0.enable_cache() }

    fn disable_cache(&mut self) { self.0.disable_cache() }

    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> PyResult<Py<PyAny>> {
        if subscript.eq(py.eval("...", None, None)?)? ||
            subscript.eq(py.eval("slice(None, None, None)", None, None)?)? {
                to_py_df(self.0.read().unwrap())
        } else {
            to_py_df(self.0.read().unwrap())?
                .call_method1(py, "__getitem__", (subscript,))?
                .call_method0(py, "to_numpy")
        }
    }

    fn __setitem__<'py>(
        &self,
        py: Python<'py>,
        key: &'py PyAny,
        data: &'py PyAny,
    ) -> PyResult<()> {
        let df = to_py_df(self.0.read().unwrap())?;
        let new_df = if key.is_instance_of::<pyo3::types::PyString>()? {
            let series = py.import("polars")?.call_method1("Series", (key, data))?;
            df.call_method1(py, "with_column", (series,))?
        } else {
            df.call_method1(py, "__setitem__", (key, data))?;
            df
        };
        self.0.update(&to_rust_df(new_df.as_ref(py)).unwrap());
        Ok(())
    }
 
    fn __contains__(&self, key: &str) -> bool {
        self.0.read().unwrap().find_idx_by_name(key).is_some()
    }

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}

/// Unstructured annotations (ordered dictionary).
#[pyclass]
#[repr(transparent)]
pub struct PyElemCollection(pub(crate) Slot<ElemCollection>);

#[pymethods]
impl PyElemCollection {
    pub fn keys(&self) -> Vec<String> {
        self.0.inner().keys().map(|x| x.to_string()).collect()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.0.inner().contains_key(key)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Option<PyObject>> {
        match self.0.inner().get_mut(key) {
            None => Ok(None),
            Some(x) => Ok(Some(to_py_data1(py, x.read().unwrap())?)),
        }
    }

    fn __setitem__<'py>(&self, py: Python<'py>, key: &str, data: &'py PyAny) -> PyResult<()> {
        self.0.inner().0.as_mut().unwrap().add_data(key, &to_rust_data1(py, data)?).unwrap();
        Ok(())
    }

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}

/// A mapping, in which each key is associated with an axisarray
/// (a two or higher-dimensional ndarray). 
/// It allows indexing and slicing along the associated axis.
/// 
/// Examples
/// --------
/// >>> data.obsm
/// AxisArrays (row) with keys: X_umap, insertion, X_spectral
/// >>> data.obsm['X_umap']
/// array([[13.279691  , -3.1859393 ],
///       [12.367847  , -1.9303571 ],
///       [11.376464  ,  0.36262953],
///       ...,
///       [12.1357565 , -2.777369  ],
///       [12.9115095 , -1.9225913 ],
///       [13.247231  , -4.200884  ]], dtype=float32)
/// >>> data.obsm.el('X_umap')
/// Array(Float(U4)) element, cache_enabled: no, cached: no
#[pyclass]
#[repr(transparent)]
pub struct PyAxisArrays(pub(crate) Slot<AxisArrays>);

#[pymethods]
impl PyAxisArrays {
    /// Return the keys.
    /// 
    /// Returns
    /// -------
    /// List[str]
    #[pyo3(text_signature = "($self)")]
    pub fn keys(&self) -> Vec<String> {
        self.0.inner().keys().map(|x| x.to_string()).collect()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.0.inner().contains_key(key)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Option<PyObject>> {
        match self.0.inner().get(key) {
            None => Ok(None),
            Some(x) => Ok(Some(to_py_data2(py, x.read().unwrap())?)),
        }
    }

    /// Provide a lazy access to the elements.
    /// 
    /// This function provides a lazy access to underlying elements. For example,
    /// calling `adata.obsm['elem']` will immediately read the data into memory,
    /// while using `adata.obsm.el('elem')` will return a :class:`.PyMatrixElem` object,
    /// which contains a reference to data stored in the disk.
    /// 
    /// /// Examples
    /// --------
    /// >>> data.obsm
    /// AxisArrays (row) with keys: X_umap, insertion, X_spectral
    /// >>> data.obsm['X_umap']
    /// array([[13.279691  , -3.1859393 ],
    ///       [12.367847  , -1.9303571 ],
    ///       [11.376464  ,  0.36262953],
    ///       ...,
    ///       [12.1357565 , -2.777369  ],
    ///       [12.9115095 , -1.9225913 ],
    ///       [13.247231  , -4.200884  ]], dtype=float32)
    /// >>> data.obsm.el('X_umap')
    /// Array(Float(U4)) element, cache_enabled: no, cached: no
    /// 
    /// Parameters
    /// ----------
    /// key
    ///     the name of the key.
    /// 
    /// Returns
    /// -------
    /// Optional[PyMatrixElem]
    #[pyo3(text_signature = "($self, key)")]
    fn el(&self, key: &str) -> PyResult<Option<PyMatrixElem>> {
        match self.0.inner().get(key) {
            None => Ok(None),
            Some(x) => Ok(Some(PyMatrixElem(x.clone()))),
        }
    }

    fn __setitem__<'py>(&self, py: Python<'py>, key: &str, data: &'py PyAny) -> PyResult<()> {
        self.0.inner().0.as_mut().unwrap().add_data(key, &to_rust_data2(py, data)?).unwrap();
        Ok(())
    }

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}

#[pyclass]
#[repr(transparent)]
pub struct PyStackedAxisArrays(pub(crate) StackedAxisArrays);

#[pymethods]
impl PyStackedAxisArrays {
    pub fn keys(&self) -> Vec<String> {
        self.0.data.keys().map(|x| x.to_string()).collect()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    fn __getitem__(&self, key: &str) -> PyResult<Option<PyStackedMatrixElem>> {
        match self.0.data.get(key) {
            None => Ok(None),
            Some(x) => Ok(Some(PyStackedMatrixElem(x.clone()))),
        }
    }

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}


/// Lazily concatenated matrix elements.
#[pyclass]
#[repr(transparent)]
pub struct PyStackedMatrixElem(pub(crate) Stacked<MatrixElem>);

#[pymethods]
impl PyStackedMatrixElem {
    /// Shape of matrix.
    #[getter]
    fn shape(&self) -> (usize, usize) { (self.0.nrows(), self.0.ncols()) }

    /// Enable caching so that data will be stored in memory when the element
    /// is accessed the first time. Subsequent requests for the data will use
    /// the in-memory cache.
    #[pyo3(text_signature = "($self)")]
    fn enable_cache(&mut self) { self.0.enable_cache() }

    fn disable_cache(&mut self) { self.0.disable_cache() }

    /// Return an iterator over the rows of the matrix.
    /// 
    /// Parameters
    /// ----------
    /// chunk_size
    ///     Number of rows of a single chunk.
    /// 
    /// Returns
    /// -------
    /// An iterator, of which the elements are matrices.
    #[pyo3(text_signature = "($self, chunk_size)")]
    fn chunked(&self, chunk_size: usize) -> PyStackedChunkedMatrix{
        PyStackedChunkedMatrix(self.0.chunked(chunk_size))
    }

    /// Return a chunk of the matrix with random indices.
    /// 
    /// Parameters
    /// ----------
    /// size
    ///     Number of rows of the returned random chunk.
    /// replace
    ///     True means random sampling of indices with replacement, False without replacement.
    /// seed
    ///     Random seed.
    /// 
    /// Returns
    /// -------
    /// A matrix
    #[args(
        replace = true,
        seed = 2022,
    )]
    #[pyo3(text_signature = "($self, size, replace, seed)")]
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
        to_py_data2(py, self.0.read_rows(idx.as_slice()).unwrap())
    }

    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> PyResult<Py<PyAny>> {
        if subscript.eq(py.eval("...", None, None)?)? ||
            subscript.eq(py.eval("slice(None, None, None)", None, None)?)? {
            to_py_data2(py, self.0.read().unwrap())
        } else {
            let idx = crate::utils::to_indices(py, subscript, self.0.nrows())?;
            to_py_data2(py, self.0.read_rows(idx.as_slice()).unwrap())
        }
    }

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}

#[pyclass]
#[repr(transparent)]
pub struct PyStackedDataFrame(pub(crate) StackedDataFrame);

#[pymethods]
impl PyStackedDataFrame {
    fn __getitem__<'py>(&self, key: &'py PyAny) -> PyResult<Py<PyAny>> {
        if key.is_instance_of::<pyo3::types::PyString>()? {
            to_py_series(self.0.column(key.extract()?).unwrap())
        } else {
            todo!()
        }
    }

    fn __repr__(&self) -> String { format!("{}", self.0) }

    fn __str__(&self) -> String { self.__repr__() }
}