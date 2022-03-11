use crate::anndata_trait::data::{DataIO, DataContainer};

use ndarray::{Axis, Array2};
use hdf5::{H5Type};
use nalgebra_sparse::csr::CsrMatrix;
use itertools::zip;

pub trait DataSubsetRow: DataIO {
    fn read_rows(
        container: &Box<dyn DataContainer>,
        idx: &[usize],
    ) -> Self
    where Self: Sized,
    {
        let x: Self = DataIO::read(container).unwrap();
        x.get_rows(idx)
    }

    fn get_rows(&self, idx: &[usize]) -> Self where Self: Sized;
}

impl<T> DataSubsetRow for Array2<T>
where
    T: H5Type + Clone,
{
    fn get_rows(&self, idx: &[usize]) -> Self { self.select(Axis(0), idx) }
}

impl<T> DataSubsetRow for CsrMatrix<T>
where
    T: H5Type + Clone + Copy,
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
        container: &Box<dyn DataContainer>,
        idx: &[usize],
    ) -> Self
    where Self: Sized,
    {
        let x: Self = DataIO::read(container).unwrap();
        x.get_columns(idx)
    }

    fn get_columns(&self, idx: &[usize]) -> Self where Self: Sized;
}

impl<T> DataSubsetCol for Array2<T>
where
    T: H5Type + Clone,
{
    fn get_columns(&self, idx: &[usize]) -> Self { self.select(Axis(1), idx) }
}

impl<T> DataSubsetCol for CsrMatrix<T>
where
    T: H5Type + Clone,
{
    fn get_columns(&self, idx: &[usize]) -> Self {
        todo!()
    }
}

pub trait DataSubset2D: DataSubsetRow + DataSubsetCol {
    fn read_partial(
        container: &Box<dyn DataContainer>,
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

impl<T> DataSubset2D for CsrMatrix<T> where T: H5Type + Clone + Copy, {}
impl<T> DataSubset2D for Array2<T> where T: H5Type + Clone, {}

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