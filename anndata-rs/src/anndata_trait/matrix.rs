use crate::{
    anndata_trait::data::{DataContainer, ReadData, WriteData},
    utils::hdf5::read_str_attr,
};

use ndarray::{Axis, ArrayD};
use hdf5::{Result, Group, H5Type};
use nalgebra_sparse::{csr::CsrMatrix, SparseEntry};
use itertools::zip;
use polars::frame::DataFrame;

pub trait MatrixLike {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;

    fn get_rows(&self, idx: &[usize]) -> Self where Self: Sized;
    fn get_columns(&self, idx: &[usize]) -> Self where Self: Sized;

    fn subset(&self, ridx: &[usize], cidx: &[usize]) -> Self
    where Self: Sized,
    {
        self.get_rows(ridx).get_columns(cidx)
    }
}


impl MatrixLike for DataFrame {
    fn nrows(&self) -> usize { self.height() }

    fn ncols(&self) -> usize { self.height() }

    fn get_rows(&self, idx: &[usize]) -> Self {
        self.take_iter(idx.iter().map(|i| *i)).unwrap()
    }

    fn get_columns(&self, idx: &[usize]) -> Self { self.get_rows(idx) }
}


impl<T> MatrixLike for ArrayD<T>
where
    T: H5Type + Clone + Send + Sync,
{
    fn nrows(&self) -> usize { self.shape()[0] }

    fn ncols(&self) -> usize { self.shape()[1] }

    fn get_rows(&self, idx: &[usize]) -> Self { self.select(Axis(0), idx) }

    fn get_columns(&self, idx: &[usize]) -> Self { self.select(Axis(1), idx) }
}

impl<T> MatrixLike for CsrMatrix<T>
where
    T: H5Type + Copy + Send + Sync,
{
    fn nrows(&self) -> usize { self.nrows() }

    fn ncols(&self) -> usize { self.ncols() }

    fn get_rows(&self, idx: &[usize]) -> Self {
        create_csr_from_rows(
            idx.iter().map(|r| {
                let row = self.get_row(*r).unwrap();
                zip(row.col_indices(), row.values())
                    .map(|(x, y)| (*x, *y)).collect()
            }),
            self.ncols(),
        )
    }

    fn get_columns(&self, idx: &[usize]) -> Self {
        create_csr_from_rows(
            self.row_iter().map(|r| idx.iter().enumerate().filter_map(|(i, x)|
                match r.get_entry(*x).unwrap() {
                    SparseEntry::NonZero(v) => Some((i, *v)),
                    SparseEntry::Zero => None,
                }
            ).collect()),
            idx.len(),
        )
    }
}


pub trait MatrixIO: MatrixLike + ReadData + WriteData {
    fn get_nrows(container: &DataContainer) -> usize where Self: Sized;
    fn get_ncols(container: &DataContainer) -> usize where Self: Sized;

    fn read_rows(container: &DataContainer, idx: &[usize]) -> Self
    where Self: Sized,
    {
        Self::read(container).unwrap().get_rows(idx)
    }

    fn read_row_slice(container: &DataContainer, slice: std::ops::Range<usize>) -> Self
    where Self: Sized,
    {
        let idx: Vec<usize> = slice.collect();
        Self::read_rows(container, idx.as_slice())
    }

    fn read_columns(container: &DataContainer, idx: &[usize]) -> Self
    where Self: Sized,
    {
        Self::read(container).unwrap().get_columns(idx)
    }

    fn read_partial(container: &DataContainer, ridx: &[usize], cidx: &[usize]) -> Self
    where Self: Sized,
    {
        Self::read(container).unwrap().subset(ridx, cidx)
    }

    fn write_rows(
        &self,
        idx: &[usize],
        location: &Group,
        name: &str
    ) -> Result<DataContainer>
    where
        Self: Sized,
    {
        WriteData::write(&self.get_rows(idx), location, name)
    }

    fn write_columns(
        &self,
        idx: &[usize],
        location: &Group,
        name: &str
    ) -> Result<DataContainer>
    where
        Self: Sized,
    {
        WriteData::write(&self.get_columns(idx), location, name)
    }

    fn write_partial(
        &self,
        ridx: &[usize],
        cidx: &[usize],
        location: &Group,
        name: &str
    ) -> Result<DataContainer>
    where
        Self: Sized,
    {
        WriteData::write(&self.subset(ridx, cidx), location, name)
    }
}

impl MatrixIO for DataFrame {
    fn get_nrows(container: &DataContainer) -> usize {
        let group = container.get_group_ref().unwrap();
        let attr = read_str_attr(group, "_index").unwrap();
        group.dataset(attr.as_str()).unwrap().shape()[0]
    }

    fn get_ncols(container: &DataContainer) -> usize {
        Self::get_nrows(container)
    }
}


impl<T> MatrixIO for ArrayD<T>
where
    T: H5Type + Clone + Send + Sync,
{
    fn get_nrows(container: &DataContainer) -> usize {
        container.get_dataset_ref().unwrap().shape()[0]
    }

    fn get_ncols(container: &DataContainer) -> usize {
        container.get_dataset_ref().unwrap().shape()[1]
    }
}

impl<T> MatrixIO for CsrMatrix<T>
where
    T: H5Type + Copy + Send + Sync,
{
    fn get_nrows(container: &DataContainer) -> usize {
        container.get_group_ref().unwrap().attr("shape").unwrap()
            .read_1d().unwrap().to_vec()[0]
    }

    fn get_ncols(container: &DataContainer) -> usize {
        container.get_group_ref().unwrap().attr("shape").unwrap()
            .read_1d().unwrap().to_vec()[1]
    }

    fn read_row_slice(container: &DataContainer, slice: std::ops::Range<usize>) -> Self
    where Self: Sized + ReadData,
    {
        let group = container.get_group_ref().unwrap();
        let mut indptr: Vec<usize> = group.dataset("indptr").unwrap()
            .read_slice_1d(slice.start..slice.end+1).unwrap().to_vec();
        let lo = indptr[0];
        let hi = indptr[indptr.len() - 1];
        let data: Vec<T> = group.dataset("data").unwrap()
            .read_slice_1d(lo..hi).unwrap().to_vec();
        let indices: Vec<usize> = group.dataset("indices").unwrap()
            .read_slice_1d(lo..hi).unwrap().to_vec();
        indptr.iter_mut().for_each(|x| *x -= lo);
        CsrMatrix::try_from_csr_data(
            indptr.len() - 1,
            Self::get_ncols(container),
            indptr,
            indices,
            data
        ).unwrap()
    }
}

pub(crate) fn create_csr_from_rows<I, T>(iter: I, num_col: usize) -> CsrMatrix<T>
where
    I: Iterator<Item = Vec<(usize, T)>>,
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

#[cfg(test)]
mod dataframe_test {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_df() {
        let s1 = Series::new("Fruit", &["Apple", "Apple", "Pear"]);
        let s2 = Series::new("Color", &["Red", "Yellow", "Green"]);
        let df = DataFrame::new(vec![s1, s2]).unwrap();
        println!("{:?}", df);

        println!("{:?}", df.get_rows(&[]));
    }
}