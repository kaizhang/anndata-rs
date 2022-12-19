mod traits;

use crate::data::{PyData, PyArrayData, PyDataFrame};

use polars::prelude::DataFrame;
use pyo3::prelude::*;
use anndata::{Elem, Backend, ArrayData};
use traits::{ElemTrait, ArrayElemTrait, DataFrameElemTrait, AxisArrayTrait};
use anyhow::Result;

use self::traits::ElemCollectionTrait;


#[pyclass]
#[repr(transparent)]
pub struct PyElem(Box<dyn ElemTrait>);

impl<T: ElemTrait + 'static> From<T> for PyElem {
    fn from(elem: T) -> Self {
        Self(Box::new(elem))
    }
}

#[pymethods]
impl PyElem {
    fn enable_cache(&self) {
        self.0.enable_cache();
    }

    fn disable_cache(&self) {
        self.0.disable_cache();
    }

    fn is_scalar(&self) -> bool {
        self.0.is_scalar()
    }

    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> Result<PyData> {
        self.0.get(py, subscript)
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.0.show()
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyArrayElem(Box<dyn ArrayElemTrait>);

impl<T: ArrayElemTrait + 'static> From<T> for PyArrayElem {
    fn from(elem: T) -> Self {
        Self(Box::new(elem))
    }
}

#[pymethods]
impl PyArrayElem {
    fn enable_cache(&self) {
        self.0.enable_cache();
    }

    fn disable_cache(&self) {
        self.0.disable_cache();
    }

    fn is_scalar(&self) -> bool {
        self.0.is_scalar()
    }

    fn __getitem__<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> Result<PyArrayData> {
        self.0.get(py, subscript)
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.0.show()
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyDataFrameElem(Box<dyn DataFrameElemTrait>);

impl<T: DataFrameElemTrait + 'static> From<T> for PyDataFrameElem {
    fn from(elem: T) -> Self {
        Self(Box::new(elem))
    }
}

#[pymethods]
impl PyDataFrameElem {
    fn __getitem__<'py>(&self, subscript: &'py PyAny) -> Result<PyDataFrame> {
        Ok(self.0.get(subscript)?.into())
    }

    //TODO: pandas dataframe should set index as well.
    fn __setitem__<'py>(&self, py: Python<'py>, key: &'py PyAny, data: &'py PyAny) -> Result<()> {
        self.0.set(py, key, data)
    }

    fn __contains__(&self, key: &str) -> bool {
        self.0.contains(key)
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.0.show()
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyAxisArrays(Box<dyn AxisArrayTrait>);

impl<T: AxisArrayTrait + 'static> From<T> for PyAxisArrays {
    fn from(elem: T) -> Self {
        Self(Box::new(elem))
    }
}

#[pymethods]
impl PyAxisArrays {
    /// Return the keys.
    ///
    /// Returns
    /// -------
    /// List[str]
    #[pyo3(text_signature = "($self)")]
    pub fn keys(&self) -> Vec<String> {
        self.0.keys()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.0.contains(key)
    }

    fn __getitem__(&self, key: &str) -> Result<PyArrayData> {
        self.0.get(key)
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
    /// Optional[PyArrayElem]
    #[pyo3(text_signature = "($self, key)")]
    fn el(&self, key: &str) -> Result<PyArrayElem> {
        self.0.el(key)
    }

    fn __setitem__(&self, key: &str, data: PyArrayData) -> Result<()> {
        self.0.set(key, data)
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.0.show()
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyElemCollection(Box<dyn ElemCollectionTrait>);

impl<T: ElemCollectionTrait + 'static> From<T> for PyElemCollection {
    fn from(elem: T) -> Self {
        Self(Box::new(elem))
    }
}

#[pymethods]
impl PyElemCollection {
    pub fn keys(&self) -> Vec<String> {
        self.0.keys()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.0.contains(key)
    }

    fn __getitem__(&self, key: &str) -> Result<PyData> {
        self.0.get(key)
    }

    fn __setitem__(&self, key: &str, data: PyData) -> Result<()> {
        self.0.set(key, data)
    }

    fn __repr__(&self) -> String {
        self.0.show()
    }

    fn __str__(&self) -> String {
        self.0.show()
    }
}