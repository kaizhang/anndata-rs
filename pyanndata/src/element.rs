use crate::iterator::{PyChunkedMatrix, PyStackedChunkedMatrix};
use crate::utils::{
    conversion::{PyToRust, RustToPy},
    instance::is_none_slice,
    to_indices,
};
use anndata_rs::{data::DataType, element::*, Data, MatrixData};
use pyo3::{
    exceptions::{PyKeyError, PyTypeError},
    prelude::*,
    PyResult, Python,
};
use rand::Rng;
use rand::SeedableRng;

#[pyclass]
#[repr(transparent)]
pub struct PyElem(pub(crate) Elem);

#[pymethods]
impl PyElem {
    fn enable_cache(&self) {
        self.0.enable_cache()
    }

    fn disable_cache(&self) {
        self.0.disable_cache()
    }

    fn is_scalar(&self) -> bool {
        match self.0.dtype() {
            DataType::Scalar(_) => true,
            _ => false,
        }
    }

    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> PyResult<Py<PyAny>> {
        if is_none_slice(py, subscript)? {
            self.0.read().unwrap().rust_into_py(py)
        } else {
            Err(PyTypeError::new_err(
                "Please use '...' or ':' to retrieve value",
            ))
        }
    }

    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
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
    fn enable_cache(&mut self) {
        self.0.enable_cache()
    }

    /// Disable caching. In-memory cache will be cleared immediately.
    #[pyo3(text_signature = "($self)")]
    fn disable_cache(&mut self) {
        self.0.disable_cache()
    }

    /// Shape of matrix.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.0.nrows(), self.0.ncols())
    }

    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> PyResult<Py<PyAny>> {
        let ridx;
        let cidx;
        if is_none_slice(py, subscript)? {
            ridx = None;
            cidx = None;
        } else if subscript.is_instance_of::<pyo3::types::PyTuple>()? {
            let (r, c) = self.shape();
            let (i, j) = subscript.extract()?;
            ridx = to_indices(py, i, r)?.0;
            cidx = to_indices(py, j, c)?.0;
        } else {
            panic!("index type '{}' is not supported", subscript.get_type());
            //let data = to_py_data2(py, self.0.read(None, None).unwrap())?;
            //data.call_method1(py, "__getitem__", (subscript,))
        }
        self.0
            .read(
                ridx.as_ref().map(|x| x.as_slice()),
                cidx.as_ref().map(|x| x.as_slice()),
            )
            .unwrap()
            .rust_into_py(py)
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
    #[args(replace = true, seed = 2022)]
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
            std::iter::repeat_with(|| rng.gen_range(0..length))
                .take(size)
                .collect()
        } else {
            rand::seq::index::sample(&mut rng, length, size).into_vec()
        };
        self.0
            .read(Some(idx.as_slice()), None)
            .unwrap()
            .rust_into_py(py)
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
    pub fn chunked(&self, chunk_size: usize) -> PyChunkedMatrix {
        PyChunkedMatrix(self.0.chunked(chunk_size))
    }

    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// An element that stores dataframe objects.
#[pyclass]
#[repr(transparent)]
pub struct PyDataFrameElem(pub(crate) DataFrameElem);

#[pymethods]
impl PyDataFrameElem {
    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> PyResult<Py<PyAny>> {
        if is_none_slice(py, subscript)? {
            self.0.read().unwrap().rust_into_py(py)
        } else {
            self.0
                .read()
                .unwrap()
                .rust_into_py(py)?
                .call_method1(py, "__getitem__", (subscript,))
        }
    }

    //TODO: pandas dataframe should set index as well.
    fn __setitem__<'py>(&self, py: Python<'py>, key: &'py PyAny, data: &'py PyAny) -> PyResult<()> {
        let df = self.0.read().unwrap().rust_into_py(py)?;
        let new_df = if key.is_instance_of::<pyo3::types::PyString>()? {
            let series = py.import("polars")?.call_method1("Series", (key, data))?;
            df.call_method1(py, "with_column", (series,))?
        } else {
            df.call_method1(py, "__setitem__", (key, data))?;
            df
        };
        self.0
            .update(new_df.as_ref(py).into_rust(py).unwrap())
            .unwrap();
        Ok(())
    }

    fn __contains__(&self, key: &str) -> bool {
        self.0
            .get_column_names()
            .map(|x| x.contains(key))
            .unwrap_or(false)
    }

    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Unstructured annotations (ordered dictionary).
#[pyclass]
#[repr(transparent)]
pub struct PyElemCollection(pub(crate) ElemCollection);

#[pymethods]
impl PyElemCollection {
    pub fn keys(&self) -> Vec<String> {
        self.0.inner().keys().map(|x| x.to_string()).collect()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.0.inner().contains_key(key)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<PyObject> {
        match self.0.inner().get_mut(key) {
            None => Err(PyKeyError::new_err(key.to_owned())),
            Some(x) => x.read().unwrap().rust_into_py(py),
        }
    }

    fn __setitem__<'py>(&self, py: Python<'py>, key: &str, data: &'py PyAny) -> PyResult<()> {
        let d: Box<dyn Data> = data.into_rust(py)?;
        self.0.add_data(key, d).unwrap();
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
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
pub struct PyAxisArrays(pub(crate) AxisArrays);

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

    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<PyObject> {
        match self.0.inner().get(key) {
            None => Err(PyKeyError::new_err(key.to_owned())),
            Some(x) => x.read(None, None).unwrap().rust_into_py(py),
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
        let d: Box<dyn MatrixData> = data.into_rust(py)?;
        self.0.add_data(key, d).unwrap();
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
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

    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<PyObject> {
        match self.0.data.get(key) {
            None => Err(PyKeyError::new_err(key.to_owned())),
            Some(x) => x.read(None, None).unwrap().rust_into_py(py),
        }
    }

    fn el(&self, key: &str) -> PyResult<PyStackedMatrixElem> {
        match self.0.data.get(key) {
            None => Err(PyKeyError::new_err(key.to_owned())),
            Some(x) => Ok(PyStackedMatrixElem(x.clone())),
        }
    }

    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Lazily concatenated matrix elements.
#[pyclass]
#[repr(transparent)]
pub struct PyStackedMatrixElem(pub(crate) StackedMatrixElem);

#[pymethods]
impl PyStackedMatrixElem {
    /// Shape of matrix.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.0.nrows(), self.0.ncols())
    }

    /// Enable caching so that data will be stored in memory when the element
    /// is accessed the first time. Subsequent requests for the data will use
    /// the in-memory cache.
    #[pyo3(text_signature = "($self)")]
    fn enable_cache(&mut self) {
        self.0.enable_cache()
    }

    fn disable_cache(&mut self) {
        self.0.disable_cache()
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
    pub fn chunked(&self, chunk_size: usize) -> PyStackedChunkedMatrix {
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
    #[args(replace = true, seed = 2022)]
    #[pyo3(text_signature = "($self, size, replace, seed)")]
    pub fn chunk<'py>(
        &self,
        py: Python<'py>,
        size: usize,
        replace: bool,
        seed: u64,
    ) -> PyResult<PyObject> {
        let length = self.0.nrows();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let idx: Vec<usize> = if replace {
            std::iter::repeat_with(|| rng.gen_range(0..length))
                .take(size)
                .collect()
        } else {
            rand::seq::index::sample(&mut rng, length, size).into_vec()
        };
        self.0
            .read(Some(idx.as_slice()), None)
            .unwrap()
            .rust_into_py(py)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> PyResult<Py<PyAny>> {
        let ridx;
        let cidx;
        if is_none_slice(py, subscript)? {
            ridx = None;
            cidx = None;
        } else if subscript.is_instance_of::<pyo3::types::PyTuple>()? {
            let (r, c) = self.shape();
            let (i, j) = subscript.extract()?;
            ridx = to_indices(py, i, r)?.0;
            cidx = to_indices(py, j, c)?.0;
        } else {
            panic!("index type '{}' is not supported", subscript.get_type());
            //let data = to_py_data2(py, self.0.read(None, None).unwrap())?;
            //data.call_method1(py, "__getitem__", (subscript,))
        }
        self.0
            .read(
                ridx.as_ref().map(|x| x.as_slice()),
                cidx.as_ref().map(|x| x.as_slice()),
            )
            .unwrap()
            .rust_into_py(py)
    }

    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyStackedDataFrame(pub(crate) StackedDataFrame);

#[pymethods]
impl PyStackedDataFrame {
    fn __getitem__<'py>(&self, py: Python<'py>, key: &'py PyAny) -> PyResult<Py<PyAny>> {
        if is_none_slice(py, key)? {
            self.0.read().unwrap().rust_into_py(py)
        } else if key.is_instance_of::<pyo3::types::PyString>()? {
            self.0.column(key.extract()?).unwrap().rust_into_py(py)
        } else {
            todo!()
        }
    }

    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}
