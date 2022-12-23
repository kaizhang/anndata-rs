use crate::backend::{Backend, DataContainer, GroupOp, LocationOp, BackendData};
use crate::data::{
    ArrayData,
    array::utils::ExtendableDataset,
};

use anyhow::{bail, Result};
use ndarray::{Array, ArrayView1, ArrayD, Dimension};
use nalgebra_sparse::CsrMatrix;

pub trait ArrayChunk {
    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<Backend = B>;
}

impl ArrayChunk for ArrayData {
    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<Backend = B>,
    {todo!()}
}

impl<D, T> ArrayChunk for Array<D, T> {
    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<Backend = B>,
    {todo!()}
}

impl<T: BackendData> ArrayChunk for CsrMatrix<T> {
    fn write_by_chunk<B, G, I>(mut iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<Backend = B>,
    {
        let group = location.create_group(name)?;
        group.write_str_attr("encoding-type", "csr_matrix")?;
        group.write_str_attr("encoding-version", "0.2.0")?;
        group.write_str_attr("h5sparse_format", "csr")?;

        let mut data: ExtendableDataset<B, T> = ExtendableDataset::with_capacity(
            &group, "data", 100000.into(),
        )?;
        let mut indices: ExtendableDataset<B, i64> = ExtendableDataset::with_capacity(
            &group, "indices", 100000.into(),
        )?;
        let mut indptr: Vec<i64> = Vec::new();
        let mut num_rows = 0;
        let mut num_cols: Option<usize> = None;
        let mut nnz = 0;

        iter.try_for_each(|csr| {
            let c = csr.ncols();
            if num_cols.is_none() {
                num_cols = Some(c);
            }
            if num_cols.unwrap() == c {
                num_rows += csr.nrows();
                let (indptr_, indices_, data_) = csr.csr_data();
                indptr_[..indptr_.len() - 1]
                    .iter()
                    .for_each(|x| indptr.push(i64::try_from(*x).unwrap() + nnz));
                nnz += *indptr_.last().unwrap_or(&0) as i64;
                data.extend(ArrayView1::from_shape(data_.len(), data_)?)?;
                indices.extend(ArrayView1::from_shape(indices_.len(), indices_)?.mapv(|x| x as i64).view())
            } else {
                bail!("All matrices must have the same number of columns");
            }
        })?;

        indices.finish()?;
        data.finish()?;
        indptr.push(nnz);
        group.create_array_data("indptr", &indptr, Default::default())?;
        group.write_arr_attr("shape", &[num_rows, num_cols.unwrap_or(0)])?;
        Ok(DataContainer::Group(group))
    }
}

/*
impl<T: BackendData> ArrayChunk for Vec<Vec<(usize, T)>> {
    fn write_by_chunk<B, G, I>(mut iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<Backend = B>,
    {
        let group = location.create_group(name)?;
        group.write_str_attr("encoding-type", "csr_matrix")?;
        group.write_str_attr("encoding-version", "0.2.0")?;
        group.write_str_attr("h5sparse_format", "csr")?;

        let mut data: ExtendableDataset<B, T> = ExtendableDataset::with_capacity(
            &group, "data", 100000.into(),
        )?;
        let mut indices: ExtendableDataset<B, i64> = ExtendableDataset::with_capacity(
            &group, "indices", 100000.into(),
        )?;
        let mut indptr: Vec<i64> = Vec::new();
        let mut num_rows = 0;
        let mut num_cols: Option<usize> = None;
        let mut nnz = 0;

        let nnz: Result<i64> = self.iterator.try_fold(0, |accum, chunk| {
            data.extend(chunk.iter().flatten().map(|x| x.1))?;
            indices.extend(
                chunk
                    .iter()
                    .flatten()
                    .map(|x| -> i64 { x.0.try_into().unwrap() }),
            )?;
            Ok(chunk.iter().fold(accum, |accum_, vec| {
                indptr.push(accum_);
                accum_ + (vec.len() as i64)
            }))
        });
        indptr.push(nnz?);

        iter.try_for_each(|csr| {
            let c = csr.ncols();
            if num_cols.is_none() {
                num_cols = Some(c);
            }
            if num_cols.unwrap() == c {
                num_rows += csr.nrows();
                let (indptr_, indices_, data_) = csr.csr_data();
                indptr_[..indptr_.len() - 1]
                    .iter()
                    .for_each(|x| indptr.push(i64::try_from(*x).unwrap() + nnz));
                nnz += *indptr_.last().unwrap_or(&0) as i64;
                data.extend(ArrayView1::from_shape(data_.len(), data_)?)?;
                indices.extend(ArrayView1::from_shape(indices_.len(), indices_)?.mapv(|x| x as i64).view())
            } else {
                bail!("All matrices must have the same number of columns");
            }
        })?;

        indices.finish()?;
        data.finish()?;
        indptr.push(nnz);
        group.create_array_data("indptr", &indptr, Default::default())?;
        group.write_arr_attr("shape", &[num_rows, num_cols.unwrap_or(0)])?;
        Ok(DataContainer::Group(group))
    }
}
*/