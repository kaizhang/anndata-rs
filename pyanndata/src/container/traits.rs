use crate::data::{PyData, PyArrayData, is_none_slice, to_select_info};

use pyo3::prelude::*;
use anndata::{ArrayData, Data, Elem, Backend, ArrayElem};
use anndata::backend::DataType;
use anyhow::{Result, bail};

/// Trait for `Elem` to abtract over different backends.
pub trait ElemTrait: Send {
    fn enable_cache(&self);
    fn disable_cache(&self);
    fn is_scalar(&self) -> bool;
    fn data<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> Result<PyData>;
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

    fn data<'py>(&self, py: Python<'py>, slice: &'py PyAny) -> Result<PyData> {
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
    fn data<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> Result<PyArrayData>;

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

    fn data<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> Result<PyArrayData> {
        let slice = to_select_info(subscript, self.inner().shape())?;
        self.inner().select::<ArrayData, _>(slice.as_ref()).map(|x| x.into())
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

    fn chunked(&self, chunk_size: usize) -> usize {todo!()}
}