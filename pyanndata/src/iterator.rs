use crate::{PyAnnData, utils::conversion::{RustToPy, PyToRust}};

use anndata_rs::{MatrixData, AnnDataIterator, iterator::{ChunkedMatrix, StackedChunkedMatrix}};
use anndata_rs::iterator::RowIterator;
use hdf5::H5Type;
use anyhow::Result;
use pyo3::prelude::*;

#[pyclass]
pub struct PyChunkedMatrix(pub ChunkedMatrix);

#[pymethods]
impl PyChunkedMatrix {
    fn n_chunks(&self) -> usize { self.0.len() }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> { slf }

    fn __next__<'py>(mut slf: PyRefMut<Self>, py: Python<'py>) -> Option<(PyObject, usize, usize)> {
        slf.0.next().map(|(data, i, j)| (data.rust_into_py(py).unwrap(), i, j))
    }
}

#[pyclass]
pub struct PyStackedChunkedMatrix(pub StackedChunkedMatrix);

#[pymethods]
impl PyStackedChunkedMatrix {
    fn n_chunks(&self) -> usize { self.0.len() }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> { slf }

    fn __next__<'py>(mut slf: PyRefMut<Self>, py: Python<'py>) -> Option<(PyObject, usize, usize)> {
        slf.0.next().map(|(data, i, j)| (data.rust_into_py(py).unwrap(), i, j))
    }
}

pub struct PyMatrixIterator<'py> {
    matrix: &'py PyAny,
    chunk_size: usize,
    total_rows: usize,
    current_row: usize,
}

impl<'py> PyMatrixIterator<'py> {
    pub(crate) fn new(matrix: &'py PyAny, chunk_size: usize) -> PyResult<Self> {
        let total_rows = matrix.getattr("shape")?.downcast::<pyo3::types::PyTuple>()?.get_item(0)?.extract()?;
        Ok(Self { matrix, chunk_size, total_rows, current_row: 0 })
    }
}

impl<'py> Iterator for PyMatrixIterator<'py> {
    type Item = (Box<dyn MatrixData>, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.total_rows {
            None
        } else {
            let i = self.current_row;
            let j = std::cmp::min(self.total_rows, self.current_row + self.chunk_size);
            self.current_row = j;
            Python::with_gil(|py| -> PyResult<_> {
                let range = (pyo3::types::PySlice::new(py, i as isize, j as isize, 1), );
                let data = self.matrix.call_method1("__getitem__", (range,))?.into_rust(py)?;
                Ok(Some((data, i, j)))
            }).unwrap()
        }
    }
}

impl ExactSizeIterator for PyMatrixIterator<'_> {
    fn len(&self) -> usize {
        let n = self.total_rows / self.chunk_size;
        if self.total_rows % self.chunk_size == 0 { n } else { n + 1 }
    }
}

impl<'py> AnnDataIterator for PyAnnData<'py> {
    type MatrixIter<'a> = PyMatrixIterator<'a> where Self: 'a;

    fn read_x_iter<'a>(&'a self, chunk_size: usize) -> Self::MatrixIter<'a> {
        PyMatrixIterator::new(self.getattr("X").unwrap(), chunk_size).unwrap()
    }
    fn set_x_from_row_iter<I, D>(&self, data: I) -> Result<()>
    where
        I: RowIterator<D>,
        D: H5Type + Copy + Send + Sync + std::cmp::PartialEq + std::fmt::Debug,
    {
        let csr: Box<dyn MatrixData> = Box::new(data.to_csr_matrix());
        self.set_n_obs(csr.nrows())?;
        self.set_n_vars(csr.ncols())?;
        self.setattr("X", csr.rust_into_py(self.py())?)?;
        Ok(())
    }

    fn read_obsm_item_iter<'a>(&'a self, key: &str, chunk_size: usize) -> Result<Self::MatrixIter<'a>> {
        Ok(PyMatrixIterator::new(self.getattr("obsm")?.call_method1("__getitem__", (key,))?, chunk_size)?)
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
            (key, csr.rust_into_py(self.py())?),
        )?;
        Ok(())
    }
}