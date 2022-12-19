use crate::data::{is_none_slice, to_select_info, PyArrayData, PyData};

use anndata::backend::DataType;
use anndata::{ArrayData, ArrayElem, Backend, Data, DataFrameElem, Elem, AxisArrays, ElemCollection};
use anyhow::{bail, Result, Context};
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::{PyArrayElem, PyElem};

/// Trait for `Elem` to abtract over different backends.
pub trait ElemTrait: Send {
    fn enable_cache(&self);
    fn disable_cache(&self);
    fn is_scalar(&self) -> bool;
    fn get<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> Result<PyData>;
    fn show(&self) -> String;
}

impl<B: Backend> ElemTrait for Elem<B> {
    fn enable_cache(&self) {
        self.lock().as_mut().map(|x| x.enable_cache());
    }

    fn disable_cache(&self) {
        self.lock().as_mut().map(|x| x.disable_cache());
    }

    fn is_scalar(&self) -> bool {
        match self.inner().dtype() {
            DataType::Scalar(_) => true,
            _ => false,
        }
    }

    fn get<'py>(&self, py: Python<'py>, slice: &'py PyAny) -> Result<PyData> {
        if is_none_slice(py, slice)? {
            Ok(self.inner().data::<Data>()?.into())
        } else {
            bail!("Please use None slice to retrieve data.")
        }
    }

    fn show(&self) -> String {
        format!("{}", self)
    }
}

pub trait ArrayElemTrait: Send {
    fn enable_cache(&self);
    fn disable_cache(&self);
    fn is_scalar(&self) -> bool;
    fn show(&self) -> String;
    fn get<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> Result<PyArrayData>;

    /// Shape of array.
    fn shape(&self) -> Vec<usize>;

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
    fn chunk<'py>(
        &self,
        py: Python<'py>,
        size: usize,
        replace: bool,
        seed: u64,
    ) -> PyResult<PyObject>;

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
    fn chunked(&self, chunk_size: usize) -> usize;
}

impl<B: Backend> ArrayElemTrait for ArrayElem<B> {
    fn enable_cache(&self) {
        self.lock().as_mut().map(|x| x.enable_cache());
    }

    fn disable_cache(&self) {
        self.lock().as_mut().map(|x| x.disable_cache());
    }

    fn is_scalar(&self) -> bool {
        match self.inner().dtype() {
            DataType::Scalar(_) => true,
            _ => false,
        }
    }

    fn get<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> Result<PyArrayData> {
        let slice = to_select_info(subscript, self.inner().shape())?;
        self.inner()
            .select::<ArrayData, _>(slice.as_ref())
            .map(|x| x.into())
    }

    fn show(&self) -> String {
        format!("{}", self)
    }

    fn shape(&self) -> Vec<usize> {
        self.inner().shape().as_ref().to_vec()
    }

    fn chunk<'py>(
        &self,
        py: Python<'py>,
        size: usize,
        replace: bool,
        seed: u64,
    ) -> PyResult<PyObject> {
        todo!()
    }

    fn chunked(&self, chunk_size: usize) -> usize {
        todo!()
    }
}

pub trait DataFrameElemTrait: Send {
    fn get<'py>(&self, subscript: &'py PyAny) -> Result<DataFrame>;
    fn set<'py>(&self, py: Python<'py>, key: &'py PyAny, data: &'py PyAny) -> Result<()>;
    fn contains(&self, key: &str) -> bool;
    fn show(&self) -> String;
}

impl<B: Backend> DataFrameElemTrait for DataFrameElem<B> {
    fn get<'py>(&self, subscript: &'py PyAny) -> Result<DataFrame> {
        let shape = [self.inner().width(), self.inner().height()].as_slice().into();
        let slice = to_select_info(subscript, &shape)?;
        self.inner().select(slice.as_ref())
    }

    fn set<'py>(&self, py: Python<'py>, key: &'py PyAny, data: &'py PyAny) -> Result<()> {
        todo!()
    }

    fn contains(&self, key: &str) -> bool {
        self.lock()
            .as_ref()
            .map(|x| x.get_column_names().contains(key))
            .unwrap_or(false)
    }

    fn show(&self) -> String {
        format!("{}", self)
    }
}

pub trait AxisArrayTrait: Send {
    fn keys(&self) -> Vec<String>;
    fn contains(&self, key: &str) -> bool;
    fn get(&self, key: &str) -> Result<PyArrayData>;
    fn el(&self, key: &str) -> Result<PyArrayElem>;
    fn set(&self, key: &str, data: PyArrayData) -> Result<()>;
    fn show(&self) -> String;
}

impl<B: Backend + 'static> AxisArrayTrait for AxisArrays<B> {
    fn keys(&self) -> Vec<String> {
        self.inner().keys().map(|x| x.to_string()).collect()
    }

    fn contains(&self, key: &str) -> bool {
        self.inner().contains_key(key)
    }

    fn get(&self, key: &str) -> Result<PyArrayData> {
        Ok(self.inner().get(key).context(format!("No such key: {}", key))?.inner().data::<ArrayData>()?.into())
    }

    fn el(&self, key: &str) -> Result<PyArrayElem> {
        Ok(self.inner().get(key).context(format!("No such key: {}", key))?.clone().into())
    }

    fn set(&self, key: &str, data: PyArrayData) -> Result<()> {
        self.inner().add_data::<ArrayData>(key, data.into())
    }

    fn show(&self) -> String {
        format!("{}", self)
    }
}

pub trait ElemCollectionTrait: Send {
    fn keys(&self) -> Vec<String>;
    fn contains(&self, key: &str) -> bool;
    fn get(&self, key: &str) -> Result<PyData>;
    fn el(&self, key: &str) -> Result<PyElem>;
    fn set(&self, key: &str, data: PyData) -> Result<()>;
    fn show(&self) -> String;
}

impl<B: Backend + 'static> ElemCollectionTrait for ElemCollection<B> {
    fn keys(&self) -> Vec<String> {
        self.inner().keys().map(|x| x.to_string()).collect()
    }

    fn contains(&self, key: &str) -> bool {
        self.inner().contains_key(key)
    }

    fn get(&self, key: &str) -> Result<PyData> {
        Ok(self.inner().get(key).context(format!("No such key: {}", key))?.inner().data::<Data>()?.into())
    }

    fn el(&self, key: &str) -> Result<PyElem> {
        Ok(self.inner().get(key).context(format!("No such key: {}", key))?.clone().into())
    }

    fn set(&self, key: &str, data: PyData) -> Result<()> {
        self.inner().add_data::<Data>(key, data.into())
    }

    fn show(&self) -> String {
        format!("{}", self)
    }
}

