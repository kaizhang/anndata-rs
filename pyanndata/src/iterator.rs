use crate::{PyAnnData, utils::conversion::{to_py_data2, to_rust_data2}};

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

pub struct PyMatrixIterator<'py> {
    matrix: &'py PyAny,
    chunk_size: usize,
    size: usize,
    current_index: usize,
}

impl<'py> PyMatrixIterator<'py> {
    pub(crate) fn new(matrix: &'py PyAny, chunk_size: usize) -> PyResult<Self> {
        let size = matrix.getattr("shape")?.downcast::<pyo3::types::PyTuple>()?.get_item(0)?.extract()?;
        Ok(Self { matrix, chunk_size, size, current_index: 0 })
    }
}

impl<'py> Iterator for PyMatrixIterator<'py> {
    type Item = Box<dyn MatrixData>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.size {
            None
        } else {
            let i = self.current_index;
            let j = std::cmp::min(self.size, self.current_index + self.chunk_size);
            self.current_index = j;
            Python::with_gil(|py| -> PyResult<_> {
                let range = (pyo3::types::PySlice::new(py, i as isize, j as isize, 1), );
                let data = to_rust_data2(py, self.matrix.call_method1("__getitem__", (range,))?)?;
                Ok(Some(data))
            }).unwrap()
        }
    }
}

impl ExactSizeIterator for PyMatrixIterator<'_> {
    fn len(&self) -> usize {
        let n = self.size / self.chunk_size;
        if self.size % self.chunk_size == 0 { n } else { n + 1 }
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
        self.setattr("X", to_py_data2(self.py(), csr)?)?;
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
            (key, to_py_data2(self.py(), csr)?),
        )?;
        Ok(())
    }
}