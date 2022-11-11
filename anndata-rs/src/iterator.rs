use crate::{
    anndata::{AnnData, AnnDataSet},
    data::{DataType, DataContainer, MatrixData, create_csr_from_rows},
    utils::hdf5::{ResizableVectorData, COMPRESSION, create_str_attr},
    element::{AxisArrays, MatrixElem, base::InnerMatrixElem},
};

use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{arr1, Array};
use hdf5::{Group, H5Type};
use anyhow::{Result, Context};
use itertools::Itertools;

pub trait RowIterator<T> {
    fn write(self, location: &Group, name: &str) -> Result<(DataContainer, usize)>;

    fn version(&self) -> &str;

    fn get_dtype(&self) -> DataType;

    fn ncols(&self) -> usize;

    fn update(self, container: &DataContainer) -> Result<(DataContainer, usize)>
    where Self: Sized,
    {
        let (file, name) = match container {
            DataContainer::H5Group(grp) => (grp.file()?, grp.name()),
            DataContainer::H5Dataset(data) => (data.file()?, data.name()),
        };
        let (path, obj) = name.as_str().rsplit_once("/")
            .unwrap_or(("", name.as_str()));
        if path.is_empty() {
            file.unlink(obj)?;
            self.write(&file, obj)
        } else {
            let g = file.group(path)?;
            g.unlink(obj)?;
            self.write(&g, obj)
        }
    }

    fn to_csr_matrix(self) -> CsrMatrix<T>;
}

pub struct CsrIterator<I> {
    pub iterator: I,
    pub num_cols: usize,
}

impl<I, D> RowIterator<D> for CsrIterator<I>
where
    I: Iterator<Item = Vec<Vec<(usize, D)>>>,
    D: H5Type + Copy,
{
    fn write(mut self, location: &Group, name: &str) -> Result<(DataContainer, usize)> {
        let group = location.create_group(name)?;
        create_str_attr(&group, "encoding-type", "csr_matrix")?;
        create_str_attr(&group, "encoding-version", self.version())?;
        create_str_attr(&group, "h5sparse_format", "csr")?;
        let chunk_size: usize = 50000;
        let data: ResizableVectorData<D> =
            ResizableVectorData::new(&group, "data", chunk_size)?;
        let mut indptr: Vec<usize> = Vec::new();

        if self.num_cols <= (i32::MAX as usize) {
            let indices: ResizableVectorData<i32> =
                ResizableVectorData::new(&group, "indices", chunk_size)?;
            let nnz: Result<usize> = self.iterator.try_fold(0, |accum, chunk| {
                data.extend(chunk.iter().flatten().map(|x| x.1))?;
                indices.extend(
                    chunk.iter().flatten().map(|x| -> i32 { x.0.try_into().unwrap() })
                )?;
                Ok(chunk.iter().fold(accum, |accum_, vec| {
                    indptr.push(accum_);
                    accum_ + vec.len()
                }))
            });
            indptr.push(nnz?);
        } else {
            let indices: ResizableVectorData<i64> =
                ResizableVectorData::new(&group, "indices", chunk_size)?;
            let nnz: Result<usize> = self.iterator.try_fold(0, |accum, chunk| {
                data.extend(chunk.iter().flatten().map(|x| x.1))?;
                indices.extend(
                    chunk.iter().flatten().map(|x| -> i64 { x.0.try_into().unwrap() })
                )?;
                Ok(chunk.iter().fold(accum, |accum_, vec| {
                    indptr.push(accum_);
                    accum_ + vec.len()
                }))
            });
            indptr.push(nnz?);
        }

        let num_rows = indptr.len() - 1;
        group.new_attr_builder()
            .with_data(&arr1(&[num_rows, self.num_cols]))
            .create("shape")?;

        let try_convert_indptr: Option<Vec<i32>> = indptr.iter()
            .map(|x| (*x).try_into().ok()).collect();
        match try_convert_indptr {
            Some(vec) => {
                group.new_dataset_builder().deflate(COMPRESSION)
                    .with_data(&Array::from_vec(vec)).create("indptr")?;
            },
            _ => {
                let vec: Vec<i64> = indptr.into_iter()
                    .map(|x| x.try_into().unwrap()).collect();
                group.new_dataset_builder().deflate(COMPRESSION)
                    .with_data(&Array::from_vec(vec)).create("indptr")?;
            },
        }
        Ok((DataContainer::H5Group(group), num_rows))
    }

    fn ncols(&self) -> usize { self.num_cols }
    fn get_dtype(&self) -> DataType { DataType::CsrMatrix(D::type_descriptor()) }
    fn version(&self) -> &str { "0.1.0" }

    fn to_csr_matrix(self) -> CsrMatrix<D> {
        create_csr_from_rows(self.iterator.flatten(), self.num_cols)
    }
}

pub struct IndexedCsrIterator<I> {
    pub iterator: I,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl<I, D> RowIterator<D> for IndexedCsrIterator<I>
where
    I: Iterator<Item = (usize, Vec<(usize, D)>)>,
    D: H5Type,
{
    fn write(self, location: &Group, name: &str) -> Result<(DataContainer, usize)> {
        let group = location.create_group(name)?;
        create_str_attr(&group, "encoding-type", "csr_matrix")?;
        create_str_attr(&group, "encoding-version", self.version())?;
        create_str_attr(&group, "h5sparse_format", "csr")?;
        let chunk_size: usize = 5000;
        let data: ResizableVectorData<D> =
            ResizableVectorData::new(&group, "data", chunk_size)?;
        let mut indptr: Vec<usize> = vec![0];
        let iter = self.iterator.scan(0, |state, (i, x)| {
            *state = *state + x.len();
            Some((i + 1, *state, x))
        });

        let mut cur_idx = 0;
        if self.num_cols <= (i32::MAX as usize) {
            let indices: ResizableVectorData<i32> =
                ResizableVectorData::new(&group, "indices", chunk_size)?;
            for chunk in &iter.chunks(chunk_size) {
                let (a, b): (Vec<i32>, Vec<D>) = chunk.map(|(i, x, vec)| {
                    assert!(i > cur_idx, "row index is not sorted");
                    let lst = *indptr.last().unwrap();
                    indptr.extend(std::iter::repeat(lst).take(i - cur_idx - 1).chain([x]));
                    cur_idx = i;
                    vec
                }).flatten().map(|(x, y)| -> (i32, D) {
                    (x.try_into().unwrap(), y)
                }).unzip();
                indices.extend(a.into_iter())?;
                data.extend(b.into_iter())?;
            }
        } else {
            let indices: ResizableVectorData<i64> =
                ResizableVectorData::new(&group, "indices", chunk_size)?;
            for chunk in &iter.chunks(chunk_size) {
                let (a, b): (Vec<i64>, Vec<D>) = chunk.map(|(i, x, vec)| {
                    assert!(i > cur_idx, "row index is not sorted");
                    let lst = *indptr.last().unwrap();
                    indptr.extend(std::iter::repeat(lst).take(i - cur_idx - 1).chain([x]));
                    cur_idx = i;
                    vec
                }).flatten().map(|(x, y)| -> (i64, D) {
                    (x.try_into().unwrap(), y)
                }).unzip();
                indices.extend(a.into_iter())?;
                data.extend(b.into_iter())?;
            }
        }

        let num_rows = self.num_rows;
        let lst = *indptr.last().unwrap();
        indptr.extend(std::iter::repeat(lst).take(num_rows + 1 - indptr.len()));

        group.new_attr_builder()
            .with_data(&arr1(&[num_rows, self.num_cols]))
            .create("shape")?;

        let try_convert_indptr: Option<Vec<i32>> = indptr.iter()
            .map(|x| (*x).try_into().ok()).collect();
        match try_convert_indptr {
            Some(vec) => {
                group.new_dataset_builder().deflate(COMPRESSION)
                    .with_data(&Array::from_vec(vec)).create("indptr")?;
            },
            _ => {
                let vec: Vec<i64> = indptr.into_iter()
                    .map(|x| x.try_into().unwrap()).collect();
                group.new_dataset_builder().deflate(COMPRESSION)
                    .with_data(&Array::from_vec(vec)).create("indptr")?;
            },
        }
        Ok((DataContainer::H5Group(group), num_rows))
    }

    fn ncols(&self) -> usize { self.num_cols }
    fn get_dtype(&self) -> DataType { DataType::CsrMatrix(D::type_descriptor()) }
    fn version(&self) -> &str { "0.1.0" }

    fn to_csr_matrix(self) -> CsrMatrix<D> {
        todo!()
        //create_csr_from_rows(self.iterator, self.num_cols)
    }
}

impl AxisArrays {
    pub fn insert_from_row_iter<I: RowIterator<D>, D>(&self, key: &str, data: I) -> Result<()> {
        let container = {
            let inner = self.inner();
            let mut size_guard = inner.size.lock();
            let mut n = *size_guard;

            if inner.contains_key(key) { inner.container.unlink(key)?; } 
            let (container, nrows) = data.write(&inner.container, key)?;

            if n == 0 { n = nrows; }
            assert!(
                n == nrows,
                "Number of observations mismatched, expecting {}, but found {}",
                n, nrows,
            );
            *size_guard = n;
            container
        };
 
        let elem = MatrixElem::try_from(container)?;
        self.inner().insert(key.to_string(), elem);
        Ok(())
    }
}

pub struct ChunkedMatrix {
    elem: MatrixElem,
    chunk_size: usize,
    total_rows: usize,
    current_row: usize,
}

impl ChunkedMatrix {
    pub(crate) fn new(elem: MatrixElem, chunk_size: usize) -> Self {
        let total_rows = elem.nrows();
        Self { elem, chunk_size, total_rows, current_row: 0 }
    }
}


impl Iterator for ChunkedMatrix {
    type Item = (Box<dyn MatrixData>, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.total_rows {
            None
        } else {
            let i = self.current_row;
            let j = std::cmp::min(self.total_rows, self.current_row + self.chunk_size);
            self.current_row = j;
            let data = self.elem.read_row_slice(i..j).unwrap();
            Some((data, i, j))
        }
    }
}

impl ExactSizeIterator for ChunkedMatrix {
    fn len(&self) -> usize {
        let n = self.total_rows / self.chunk_size;
        if self.total_rows % self.chunk_size == 0 { n } else { n + 1 }
    }
}

pub struct StackedChunkedMatrix {
    matrices: Vec<ChunkedMatrix>,
    current_row: usize,
    current_matrix: usize,
}

impl StackedChunkedMatrix {
    pub(crate) fn new<I: Iterator<Item = MatrixElem>>(elems: I, chunk_size: usize) -> Self {
        Self {
            matrices: elems.map(|x| ChunkedMatrix::new(x, chunk_size)).collect(),
            current_row: 0, current_matrix: 0,
        }
    }
}

impl Iterator for StackedChunkedMatrix {
    type Item = (Box<dyn MatrixData>, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(mat) = self.matrices.get_mut(self.current_matrix) {
            if let Some((data, start, stop)) = mat.next() {
                let new_start = self.current_row;
                let new_stop = new_start + stop - start;
                self.current_row = new_stop;
                Some((data, new_start, new_stop))
            } else {
                self.current_matrix += 1;
                self.next()
            }
        } else {
            None
        }
    }
}

impl ExactSizeIterator for StackedChunkedMatrix {
    fn len(&self) -> usize { self.matrices.iter().map(|x| x.len()).sum() }
}


pub trait AnnDataIterator {
    type MatrixIter<'a>: Iterator<Item =
        (Box<dyn MatrixData>, usize, usize)> + ExactSizeIterator where Self: 'a;

    fn read_x_iter<'a>(&'a self, chunk_size: usize) -> Self::MatrixIter<'a>;
    fn set_x_from_row_iter<I, D>(&self, data: I) -> Result<()>
    where
        I: RowIterator<D>,
        D: H5Type + Copy + Send + Sync + std::cmp::PartialEq + std::fmt::Debug;

    fn read_obsm_item_iter<'a>(&'a self, key: &str, chunk_size: usize) -> Result<Self::MatrixIter<'a>>;
    fn add_obsm_item_from_row_iter<I, D>(&self, key: &str, data: I) -> Result<()>
    where
        I: RowIterator<D>,
        D: H5Type + Copy + Send + Sync + std::cmp::PartialEq + std::fmt::Debug;
}

impl AnnDataIterator for AnnData {
    type MatrixIter<'a> = ChunkedMatrix;

    fn read_x_iter<'a>(&'a self, chunk_size: usize) -> Self::MatrixIter<'a> {
        self.get_x().chunked(chunk_size)
    }
    fn set_x_from_row_iter<I, D>(&self, data: I) -> Result<()>
    where
        I: RowIterator<D>,
        D: H5Type + Copy + Send + Sync + std::cmp::PartialEq + std::fmt::Debug,
    {
        self.set_n_vars(data.ncols());
        if !self.x.is_empty() { self.file.unlink("X")?; }
        let (container, nrows) = data.write(&self.file, "X")?;
        self.set_n_obs(nrows);
        self.x.insert(InnerMatrixElem::try_from(container)?);
        Ok(())
    }

    fn read_obsm_item_iter<'a>(&'a self, key: &str, chunk_size: usize) -> Result<Self::MatrixIter<'a>> {
        let res = self.get_obsm().inner().get(key)
            .context(format!("key '{}' not present in AxisArrays", key))?.chunked(chunk_size);
        Ok(res)
    }
    fn add_obsm_item_from_row_iter<I, D>(&self, key: &str, data: I) -> Result<()>
    where
        I: RowIterator<D>,
        D: H5Type + Copy + Send + Sync + std::cmp::PartialEq + std::fmt::Debug,
    {
        self.get_obsm().insert_from_row_iter(key, data)
    }
}

/// AnnDataIterator trait for AnnDataSet will iterate over the underlying AnnData objects.
impl AnnDataIterator for AnnDataSet {
    type MatrixIter<'a> = StackedChunkedMatrix;

    fn read_x_iter<'a>(&'a self, chunk_size: usize) -> Self::MatrixIter<'a> {
        self.get_x().chunked(chunk_size)
    }
    fn set_x_from_row_iter<I, D>(&self, _: I) -> Result<()>
    where
        I: RowIterator<D>,
        D: H5Type + Copy + Send + Sync + std::cmp::PartialEq + std::fmt::Debug,
    {
        todo!()
    }
    fn read_obsm_item_iter<'a>(&'a self, key: &str, chunk_size: usize) -> Result<Self::MatrixIter<'a>> {
        let res = self.get_inner_adatas().inner().get_obsm().data.get(key)
            .context(format!("key '{}' not present in StackedAxisArrays", key))?.chunked(chunk_size);
        Ok(res)
    }
    fn add_obsm_item_from_row_iter<I, D>(&self, _: &str, _: I) -> Result<()>
    where
        I: RowIterator<D>,
        D: H5Type + Copy + Send + Sync + std::cmp::PartialEq + std::fmt::Debug,
    {
        todo!()
    }
}