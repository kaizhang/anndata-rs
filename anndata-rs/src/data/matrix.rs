use crate::{data::{DataContainer, ReadData, WriteData}, utils::hdf5::{read_str_attr, read_chunks_1d}};

use ndarray::{Axis, ArrayD};
use hdf5::{Result, Group, H5Type};
use nalgebra_sparse::csr::CsrMatrix;
use polars::frame::DataFrame;

pub trait MatrixOp {
    fn shape(&self) -> (usize, usize) { (self.nrows(), self.ncols()) }
    fn nrows(&self) -> usize { self.shape().0 }
    fn ncols(&self) -> usize { self.shape().1 }

    fn get_rows(&self, idx: &[usize]) -> Self where Self: Sized;
    fn get_columns(&self, idx: &[usize]) -> Self where Self: Sized;

    fn subset(&self, ridx: &[usize], cidx: &[usize]) -> Self
    where Self: Sized,
    {
        self.get_rows(ridx).get_columns(cidx)
    }
}


impl MatrixOp for DataFrame {
    fn nrows(&self) -> usize { self.height() }

    fn ncols(&self) -> usize { self.height() }

    fn get_rows(&self, idx: &[usize]) -> Self {
        self.take_iter(idx.iter().map(|i| *i)).unwrap()
    }

    fn get_columns(&self, idx: &[usize]) -> Self { self.get_rows(idx) }
}


impl<T> MatrixOp for ArrayD<T>
where
    T: H5Type + Clone + Send + Sync,
{
    fn shape(&self) -> (usize, usize) {
        let d = self.shape();
        (d[0], d[1])
    }

    fn get_rows(&self, idx: &[usize]) -> Self { self.select(Axis(0), idx) }

    fn get_columns(&self, idx: &[usize]) -> Self { self.select(Axis(1), idx) }
}

impl<T> MatrixOp for CsrMatrix<T>
where
    T: H5Type + Copy + Send + Sync + std::cmp::PartialEq + std::fmt::Debug
{
    fn nrows(&self) -> usize { self.nrows() }

    fn ncols(&self) -> usize { self.ncols() }

    fn get_rows(&self, idx: &[usize]) -> Self {
        create_csr_from_rows(
            idx.iter().map(|r| {
                let row = self.get_row(*r).unwrap();
                std::iter::zip(
                    row.col_indices().iter().map(|x| *x).collect::<Vec<_>>(),
                    row.values().iter().map(|x| *x).collect::<Vec<_>>(),
                )
            }),
            self.ncols(),
        )
    }

    fn get_columns(&self, idx: &[usize]) -> Self {
        let (nrow, ncol) = self.shape();
        let indptr = self.row_offsets();
        let indices = self.col_indices();

        // bincount(col_idxs)
        let mut col_offsets = vec![0; ncol];
        idx.iter().for_each(|i| col_offsets[*i] += 1);

        // Compute new indptr
        let mut new_indptr = vec![0; nrow+1];
        let mut new_nnz = 0;
        (0..nrow).for_each(|r| {
            (indptr[r]..indptr[r+1]).for_each(|i| new_nnz += col_offsets[indices[i]]);
            new_indptr[r+1] = new_nnz;
        });

        // cumsum in-place
        (1..ncol).for_each(|i| col_offsets[i] += col_offsets[i - 1]);

        let mut col_order: Vec<_> = (0..idx.len()).collect();
        col_order.sort_by_key(|&i| idx[i]);

        //  populates indices/data entries for selected columns.
        let mut new_indices = vec![0; new_nnz];
        let mut new_values: Vec<T> = Vec::with_capacity(new_nnz);
        let mut n = 0;
        std::iter::zip(indices, self.values()).for_each(|(&j, v)| {
            let offset = col_offsets[j];
            let prev_offset = if j == 0 { 0 } else { col_offsets[j-1] };
            if offset != prev_offset {
                (prev_offset..offset).for_each(|k| {
                    new_indices[n] = col_order[k];
                    unsafe { new_values.as_mut_ptr().add(n).write(*v); }
                    n += 1;
                });
            }
        });
        unsafe { new_values.set_len(new_nnz); }

        CsrMatrix::try_from_unsorted_csr_data(
            nrow, idx.len(), new_indptr, new_indices, new_values,
        ).unwrap()
    }
}

pub trait PartialIO: MatrixOp + ReadData + WriteData {
    fn get_shape(container: &DataContainer) -> (usize, usize) where Self: Sized {
        (Self::get_nrows(container), Self::get_ncols(container))
    }

    fn get_nrows(container: &DataContainer) -> usize where Self: Sized {
        Self::get_shape(container).0
    }

    fn get_ncols(container: &DataContainer) -> usize where Self: Sized {
        Self::get_shape(container).1
    }

    fn read_rows(container: &DataContainer, idx: &[usize]) -> Self
    where Self: Sized,
    {
        Self::read(container).unwrap().get_rows(idx)
    }

    fn read_row_slice(container: &DataContainer, slice: std::ops::Range<usize>) -> Result<Self>
    where Self: Sized,
    {
        let idx: Vec<usize> = slice.collect();
        Ok(Self::read_rows(container, idx.as_slice()))
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

    fn write_rows(&self, idx: &[usize], location: &Group, name: &str) -> Result<DataContainer>;
    fn write_columns(&self, idx: &[usize], location: &Group, name: &str) -> Result<DataContainer>;
    fn write_partial(&self, ridx: &[usize], cidx: &[usize], location: &Group, name: &str) -> Result<DataContainer>;
}


impl PartialIO for DataFrame {
    fn get_nrows(container: &DataContainer) -> usize {
        let group = container.get_group_ref().unwrap();
        let attr = read_str_attr(group, "_index").unwrap();
        group.dataset(attr.as_str()).unwrap().shape()[0]
    }

    fn get_ncols(container: &DataContainer) -> usize {
        Self::get_nrows(container)
    }

    fn write_rows(&self, idx: &[usize], location: &Group, name: &str) -> Result<DataContainer> {
        WriteData::write(&<Self as MatrixOp>::get_rows(self, idx), location, name)
    }

    fn write_columns(&self, idx: &[usize], location: &Group, name: &str) -> Result<DataContainer> {
        WriteData::write(&<Self as MatrixOp>::get_columns(self, idx), location, name)
    }

    fn write_partial(&self, ridx: &[usize], cidx: &[usize], location: &Group, name: &str) -> Result<DataContainer> {
        WriteData::write(&<Self as MatrixOp>::subset(self, ridx, cidx), location, name)
    }
}


impl<T> PartialIO for ArrayD<T>
where
    T: H5Type + Clone + Send + Sync,
{
    fn get_nrows(container: &DataContainer) -> usize {
        container.get_dataset_ref().unwrap().shape()[0]
    }

    fn get_ncols(container: &DataContainer) -> usize {
        container.get_dataset_ref().unwrap().shape()[1]
    }

    fn write_rows(&self, idx: &[usize], location: &Group, name: &str) -> Result<DataContainer> {
        WriteData::write(&self.get_rows(idx), location, name)
    }

    fn write_columns(&self, idx: &[usize], location: &Group, name: &str) -> Result<DataContainer> {
        WriteData::write(&self.get_columns(idx), location, name)
    }

    fn write_partial(&self, ridx: &[usize], cidx: &[usize], location: &Group, name: &str) -> Result<DataContainer> {
        WriteData::write(&self.subset(ridx, cidx), location, name)
    }
}

impl<T> PartialIO for CsrMatrix<T>
where
    T: H5Type + Copy + Send + Sync + std::cmp::PartialEq + std::fmt::Debug
{
    fn get_nrows(container: &DataContainer) -> usize {
        container.get_group_ref().unwrap().attr("shape").unwrap()
            .read_1d().unwrap().to_vec()[0]
    }

    fn get_ncols(container: &DataContainer) -> usize {
        container.get_group_ref().unwrap().attr("shape").unwrap()
            .read_1d().unwrap().to_vec()[1]
    }

    /* Slow
    fn read_rows(container: &DataContainer, idx: &[usize]) -> Self
    where Self: Sized,
    {
        let group = container.get_group_ref().unwrap();
        let indptr: Vec<usize> = group.dataset("indptr").unwrap()
            .read_1d().unwrap().to_vec();
        let data_ = group.dataset("data").unwrap();
        let indices_ = group.dataset("indices").unwrap();

        create_csr_from_rows(
            idx.iter().map(|i| {
                let lo = indptr[*i];
                let hi = indptr[*i+1];
                zip(
                    indices_.read_slice_1d(lo..hi).unwrap(),
                    data_.read_slice_1d(lo..hi).unwrap(),
                )
            }),
            Self::get_ncols(container),
        )
    }
    */

    fn read_row_slice(container: &DataContainer, slice: std::ops::Range<usize>) -> Result<Self>
    where Self: Sized + ReadData,
    {
        let group = container.get_group_ref()?;
        let mut indptr: Vec<usize> = group.dataset("indptr")?
            .read_slice_1d(slice.start..slice.end+1)?.to_vec();
        let lo = indptr[0];
        let hi = indptr[indptr.len() - 1];
        let data: Vec<T> = group.dataset("data")?.read_slice_1d(lo..hi)?.to_vec();
        let indices: Vec<usize> = group.dataset("indices")?.read_slice_1d(lo..hi)?.to_vec();
        indptr.iter_mut().for_each(|x| *x -= lo);
        Ok(CsrMatrix::try_from_csr_data(
            indptr.len() - 1,
            Self::get_ncols(container),
            indptr,
            indices,
            data
        ).unwrap())
    }

    fn read_columns(container: &DataContainer, idx: &[usize]) -> Self
    where Self: Sized,
    {
        let (nrow, ncol) = Self::get_shape(container);
        if idx.is_empty() {
            return CsrMatrix::try_from_csr_data(
                nrow, 0, vec![0; nrow + 1], Vec::new(), Vec::new()
            ).unwrap()
        }
        let group = container.get_group_ref().unwrap();
        let indptr: Vec<usize> = group.dataset("indptr").unwrap()
            .read_1d().unwrap().to_vec();
        let indices: Vec<usize> = group.dataset("indices").unwrap()
            .read_1d().unwrap().to_vec();

        // bincount(col_idxs)
        let mut col_offsets = vec![0; ncol];
        idx.iter().for_each(|i| col_offsets[*i] += 1);

        // Compute new indptr
        let mut new_indptr = vec![0; nrow+1];
        let mut new_nnz = 0;

        (0..nrow).for_each(|r| {
            (indptr[r]..indptr[r+1]).for_each(|i| new_nnz += col_offsets[indices[i]]);
            new_indptr[r+1] = new_nnz;
        });


        // cumsum in-place
        (1..ncol).for_each(|i| col_offsets[i] += col_offsets[i - 1]);

        let mut col_order: Vec<_> = (0..idx.len()).collect();
        col_order.sort_by_key(|&i| idx[i]);

        //  populates indices/data entries for selected columns.
        let mut new_indices = vec![0; new_nnz];
        let mut new_values: Vec<T> = Vec::with_capacity(new_nnz);
        let mut n = 0;
        let values = group.dataset("data").unwrap();
        indices.into_iter().zip(read_chunks_1d(&values).flatten()).for_each(|(j, v): (usize, T)| {
            let offset = col_offsets[j];
            let prev_offset = if j == 0 { 0 } else { col_offsets[j-1] };
            if offset != prev_offset {
                (prev_offset..offset).for_each(|k| {
                    new_indices[n] = col_order[k];
                    unsafe { new_values.as_mut_ptr().add(n).write(v); }
                    n += 1;
                });
            }
        });
        unsafe { new_values.set_len(new_nnz); }

        CsrMatrix::try_from_unsorted_csr_data(
            nrow, idx.len(), new_indptr, new_indices, new_values,
        ).unwrap()
    }

    fn write_rows(&self, idx: &[usize], location: &Group, name: &str) -> Result<DataContainer> {
        WriteData::write(&self.get_rows(idx), location, name)
    }

    fn write_columns(&self, idx: &[usize], location: &Group, name: &str) -> Result<DataContainer> {
        WriteData::write(&self.get_columns(idx), location, name)
    }

    fn write_partial(&self, ridx: &[usize], cidx: &[usize], location: &Group, name: &str) -> Result<DataContainer> {
        WriteData::write(&self.subset(ridx, cidx), location, name)
    }
}

#[inline]
pub(crate) fn create_csr_from_rows<I, T, R>(iter: I, num_col: usize) -> CsrMatrix<T>
where
    I: Iterator<Item = R>,
    R: IntoIterator<Item = (usize, T)>,
{
    let mut data: Vec<T> = Vec::new();
    let mut indices: Vec<usize> = Vec::new();
    let mut indptr: Vec<usize> = Vec::new();

    let n = iter.fold(0, |r_idx, row| {
        indptr.push(r_idx);
        let (mut a, mut b): (Vec<_>, Vec<_>) = row.into_iter().unzip();
        let new_index = r_idx + a.len();
        indices.append(&mut a);
        data.append(&mut b);
        new_index
    });
    indptr.push(n);
    CsrMatrix::try_from_csr_data(indptr.len() - 1, num_col, indptr, indices, data).unwrap()
}

#[cfg(test)]
mod matrix_tests {
    use super::*;
    use polars::prelude::*;
    use quickcheck_macros::quickcheck;
    use rand::Rng;
    use hdf5::*;
    use tempfile::tempdir;
    use std::path::PathBuf;
    use nalgebra_sparse::CooMatrix;

    pub fn with_tmp_dir<T, F: Fn(PathBuf) -> T>(func: F) -> T {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();
        func(path)
    }

    fn with_tmp_path<T, F: Fn(PathBuf) -> T>(func: F) -> T {
        with_tmp_dir(|dir| func(dir.join("foo.h5")))
    }

    fn with_tmp_file<T, F: Fn(File) -> T>(func: F) -> T {
        with_tmp_path(|path| {
            let file = File::create(&path).unwrap();
            func(file)
        })
    }

    #[test]
    fn test_df() {
        let s1 = Series::new("Fruit", &["Apple", "Apple", "Pear"]);
        let s2 = Series::new("Color", &["Red", "Yellow", "Green"]);
        let df = DataFrame::new(vec![s1, s2]).unwrap();
        println!("{:?}", df);

        println!("{:?}", df.get_rows(&[]));
    }

    #[test]
    fn test_csr() {
        let mut rng = rand::thread_rng();
        let n: usize = 1000;
        let m: usize = 1000;
        let mat: CsrMatrix<i64> = {
            let nnz: usize = 10000;
            let values: Vec<i64> = vec![1; nnz];
            let (row_indices, col_indices) = (0..nnz).map(|_| (rng.gen_range(0..n), rng.gen_range(0..m) )).unzip();
            (&CooMatrix::try_from_triplets(n, m, row_indices, col_indices, values).unwrap()).into()
        };

        let data = with_tmp_file(|file| {
            let g = file.create_group("foo").unwrap();
            mat.write(&g, "x").unwrap()
        });

        let ridx: Vec<usize> = (0..100).map(|_| rng.gen_range(0..n)).collect();
        assert_eq!(
            mat.get_rows(ridx.as_slice()),
            CsrMatrix::read_rows(&data, ridx.as_slice()),
        );

        let cidx: Vec<usize> = (0..100).map(|_| rng.gen_range(0..m)).collect();
        assert_eq!(
            mat.get_columns(cidx.as_slice()),
            CsrMatrix::read_columns(&data, cidx.as_slice()),
        );
    }


    /*
    #[quickcheck]
    fn double_reversal_is_identity(xs: Array2<i64>) -> bool {
        xs == xs
    }
    */
}