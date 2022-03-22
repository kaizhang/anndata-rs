use crate::anndata_trait::data::{DataType, DataIO, DataContainer};

use ndarray::{Axis, ArrayD};
use hdf5::{H5Type, Result};
use nalgebra_sparse::csr::CsrMatrix;
use itertools::zip;
use hdf5::types::TypeDescriptor::*;
use polars::frame::DataFrame;

pub trait DataSubsetRow: DataIO {
    fn read_rows(
        container: &DataContainer,
        idx: &[usize],
    ) -> Self
    where Self: Sized,
    {
        let x: Self = DataIO::read(container).unwrap();
        x.get_rows(idx)
    }

    fn get_rows(&self, idx: &[usize]) -> Self where Self: Sized;
}

impl DataSubsetRow for DataFrame {
    fn get_rows(&self, idx: &[usize]) -> Self {
        self.take_iter(idx.iter().map(|i| *i)).unwrap()
    }
}

impl<T> DataSubsetRow for ArrayD<T>
where
    T: H5Type + Clone + Send + Sync,
{
    fn get_rows(&self, idx: &[usize]) -> Self { self.select(Axis(0), idx) }
}

impl<T> DataSubsetRow for CsrMatrix<T>
where
    T: H5Type + Copy + Send + Sync,
{
    fn get_rows(&self, idx: &[usize]) -> Self {
        create_csr_from_rows(idx.iter().map(|r| {
            let row = self.get_row(*r).unwrap();
            zip(row.col_indices(), row.values())
                .map(|(x, y)| (*x, *y)).collect()
        }),
        self.ncols()
        )
    }
}

pub trait DataSubsetCol: DataIO {
    fn read_columns(
        container: &DataContainer,
        idx: &[usize],
    ) -> Self
    where Self: Sized,
    {
        let x: Self = DataIO::read(container).unwrap();
        x.get_columns(idx)
    }

    fn get_columns(&self, idx: &[usize]) -> Self where Self: Sized;
}

impl DataSubsetCol for DataFrame {
    fn get_columns(&self, idx: &[usize]) -> Self {
        idx.iter().map(|i| self.select_at_idx(*i).unwrap().clone()).collect()
    }
}

impl<T> DataSubsetCol for ArrayD<T>
where
    T: H5Type + Clone + Send + Sync,
{
    fn get_columns(&self, idx: &[usize]) -> Self { self.select(Axis(1), idx) }
}

impl<T> DataSubsetCol for CsrMatrix<T>
where
    T: H5Type + Clone + Send + Sync,
{
    fn get_columns(&self, idx: &[usize]) -> Self {
        todo!()
    }
}

pub trait DataSubset2D: DataSubsetRow + DataSubsetCol {
    fn read_partial(
        container: &DataContainer,
        ridx: &[usize],
        cidx: &[usize],
    ) -> Self
    where Self: Sized,
    {
        let x: Self = DataSubsetRow::read_rows(container, ridx);
        x.get_columns(cidx)
    }

    fn subset(
        &self,
        ridx: &[usize],
        cidx: &[usize],
    ) -> Self
    where Self: Sized,
    {
        self.get_rows(ridx).get_columns(cidx)
    }
}

impl DataSubset2D for DataFrame {}
impl<T> DataSubset2D for CsrMatrix<T> where T: H5Type + Clone + Copy + Send + Sync, {}
impl<T> DataSubset2D for ArrayD<T> where T: H5Type + Clone + Send + Sync, {}

fn create_csr_from_rows<I, T>(iter: I, num_col: usize) -> CsrMatrix<T>
where
    I: Iterator<Item = Vec<(usize, T)>>,
    T: H5Type,
{
    let mut data: Vec<T> = Vec::new();
    let mut indices: Vec<usize> = Vec::new();
    let mut indptr: Vec<usize> = Vec::new();

    let n = iter.fold(0, |r_idx, row| {
        indptr.push(r_idx);
        let new_idx = r_idx + row.len();
        let (mut a, mut b) = row.into_iter().unzip();
        indices.append(&mut a);
        data.append(&mut b);
        new_idx
    });
    indptr.push(n);
    CsrMatrix::try_from_csr_data(indptr.len() - 1, num_col, indptr, indices, data).unwrap()
}

pub fn read_dyn_data_subset(
    container: &DataContainer,
    ridx: Option<&[usize]>,
    cidx: Option<&[usize]>,
) -> Result<Box<dyn DataSubset2D>> {
    match container.get_encoding_type()? {
        DataType::CsrMatrix(Integer(_)) => {
            let mat: CsrMatrix<i64> = read_data_subset(container, ridx, cidx);
            Ok(Box::new(mat))
        },
        DataType::CsrMatrix(Unsigned(_)) => {
            let mat: CsrMatrix<u64> = read_data_subset(container, ridx, cidx);
            Ok(Box::new(mat))
        },
        DataType::CsrMatrix(Float(_)) => {
            let mat: CsrMatrix<f64> = read_data_subset(container, ridx, cidx);
            Ok(Box::new(mat))
        },
        DataType::Array(Integer(_)) => {
            let mat: ArrayD<i64> = read_data_subset(container, ridx, cidx);
            Ok(Box::new(mat))
        },
        DataType::Array(Unsigned(_)) => {
            let mat: ArrayD<u64> = read_data_subset(container, ridx, cidx);
            Ok(Box::new(mat))
        },
        DataType::Array(Float(_)) => {
            let mat: ArrayD<f64> = read_data_subset(container, ridx, cidx);
            Ok(Box::new(mat))
        },
        DataType::DataFrame => {
            let df: DataFrame = read_data_subset(container, ridx, cidx);
            Ok(Box::new(df))
        },
        unknown => Err(hdf5::Error::Internal(
            format!("Not implemented: Dynamic reading of type {:?}", unknown)
        ))?,
    }
}

fn read_data_subset<T: DataSubset2D>(
    container: &DataContainer,
    ridx: Option<&[usize]>,
    cidx: Option<&[usize]>,
) -> T {
    match ridx {
        None => match cidx {
            None => DataIO::read(container).unwrap(),
            Some(j) => DataSubsetCol::read_columns(container, j),
        },
        Some(i) => match cidx {
            None => DataSubsetRow::read_rows(container, i),
            Some(j) => DataSubset2D::read_partial(container, i, j),
        },
    }
}