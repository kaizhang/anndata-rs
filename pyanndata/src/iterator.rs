use crate::utils::conversion::to_py_data2;
use crate::PyAnnData;

use anndata_rs::{MatrixData, AnnDataIterator, iterator::{ChunkedMatrix, StackedChunkedMatrix}};
use anndata_rs::iterator::RowIterator;
use hdf5::H5Type;
use anyhow::{bail, Result};
use pyo3::prelude::*;

#[pyclass]
pub struct PyChunkedMatrix(pub ChunkedMatrix);

#[pymethods]
impl PyChunkedMatrix {
    fn n_chunks(&self) -> usize { self.0.len() }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> { slf }

    fn __next__<'py>(mut slf: PyRefMut<Self>, py: Python<'py>) -> Option<PyObject> {
        slf.0.next().map(|data| to_py_data2(py, data).unwrap())
    }
}

#[pyclass]
pub struct PyStackedChunkedMatrix(pub StackedChunkedMatrix);

#[pymethods]
impl PyStackedChunkedMatrix {
    fn n_chunks(&self) -> usize { self.0.len() }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> { slf }

    fn __next__<'py>(mut slf: PyRefMut<Self>, py: Python<'py>) -> Option<PyObject> {
        slf.0.next().map(|data| to_py_data2(py, data).unwrap())
    }
}


impl<'py> AnnDataIterator for PyAnnData<'py> {
    type MatrixIter = StackedChunkedMatrix;

    fn read_x_iter(&self, chunk_size: usize) -> Self::MatrixIter {
        todo!()
    }
    fn set_x_from_row_iter<I, D>(&self, data: I) -> Result<()>
    where
        I: RowIterator<D>,
        D: H5Type + Copy + Send + Sync + std::cmp::PartialEq + std::fmt::Debug,
    {
        let csr: Box<dyn MatrixData> = Box::new(data.to_csr_matrix());
        self.set_n_obs(csr.nrows())?;
        self.set_n_vars(csr.ncols())?;
        self.setattr("X", to_py_data2(self.py(), csr)?)?;
        Ok(())
    }

    fn read_obsm_item_iter(&self, key: &str, chunk_size: usize) -> Result<Self::MatrixIter> {
        todo!()
    }

    fn add_obsm_item_from_row_iter<I, D>(&self, key: &str, data: I) -> Result<()>
    where
        I: RowIterator<D>,
        D: H5Type + Copy + Send + Sync + std::cmp::PartialEq + std::fmt::Debug,
    {
        let csr: Box<dyn MatrixData> = Box::new(data.to_csr_matrix());
        self.set_n_obs(csr.nrows())?;
        self.getattr("obsm")?.call_method1(
            "__setitem__",
            (key, to_py_data2(self.py(), csr)?),
        )?;
        Ok(())
    }
}