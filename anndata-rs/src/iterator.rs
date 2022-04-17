use crate::{
    anndata::AnnData,
    anndata_trait::{DataType, DataContainer, DataPartialIO},
    utils::hdf5::{ResizableVectorData, COMPRESSION, create_str_attr},
    element::{AxisArrays, ElemTrait, MatrixElem, RawMatrixElem},
};

use nalgebra_sparse::csr::{CsrMatrix, CsrRowIter};
use ndarray::{s, arr1, Array, Array1};
use hdf5::{Dataset, Group, H5Type};
use anyhow::Result;
use itertools::Itertools;

pub trait RowIterator {
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
}

pub struct CsrIterator<I> {
    pub iterator: I,
    pub num_cols: usize,
}

impl<I> CsrIterator<I> {
    pub fn to_csr_matrix<T>(self) -> CsrMatrix<T>
    where
        I: Iterator<Item = Vec<(usize, T)>>,
    {
        crate::anndata_trait::create_csr_from_rows(self.iterator, self.num_cols)
    }
}

impl<I, D> RowIterator for CsrIterator<I>
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
}

pub struct IndexedCsrIterator<I> {
    pub iterator: I,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl<I, D> RowIterator for IndexedCsrIterator<I>
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
}



impl AnnData {
    pub fn set_x_from_row_iter<I>(&self, data: I) -> Result<()>
    where
        I: RowIterator,
    {
        self.set_n_vars(data.ncols());
        let mut x_guard = self.get_x().inner();
        if x_guard.0.is_some() { self.file.unlink("X")?; }
        let (container, nrows) = data.write(&self.file, "X")?;
        self.set_n_obs(nrows);
        *x_guard.0 = Some(RawMatrixElem::new(container)?);
        Ok(())
    }
}

impl AxisArrays {
    pub fn insert_from_row_iter<I>(&mut self, key: &str, data: I) -> Result<()>
    where
        I: RowIterator,
    {
        let container = {
            let mut size_guard = self.size.lock();
            let mut n = *size_guard;

            if self.contains_key(key) { self.container.unlink(key)?; } 
            let (container, nrows) = data.write(&self.container, key)?;

            if n == 0 { n = nrows; }
            assert!(
                n == nrows,
                "Number of observations mismatched, expecting {}, but found {}",
                n, nrows,
            );
            *size_guard = n;
            container
        };
 
        let elem = MatrixElem::new_elem(container)?;
        self.insert(key.to_string(), elem);
        Ok(())
    }
}

pub trait IntoRowsIterator {
    type Rows;
    type IntoRowsIter: Iterator<Item = Self::Rows>;
    fn into_row_iter(self, chunk_size: usize) -> Self::IntoRowsIter;
}

impl<'a, T> IntoRowsIterator for &'a RawMatrixElem<CsrMatrix<T>>
where
    T: H5Type + Copy,
{
    type Rows = Vec<Vec<(usize, T)>>;
    type IntoRowsIter = CsrRowsIterator<'a, T>;
    fn into_row_iter(self, chunk_size: usize) -> Self::IntoRowsIter {
        match &self.inner.element {
            Some(csr) => CsrRowsIterator::Memory((csr.row_iter(), chunk_size)),
            None => { 
                let container = self.inner.container.get_group_ref().unwrap();
                let data = container.dataset("data").unwrap();
                let indices = container.dataset("indices").unwrap();
                let indptr: Vec<usize> = container.dataset("indptr").unwrap()
                    .read_1d().unwrap().to_vec();
                CsrRowsIterator::Disk((data, indices, indptr, 0, chunk_size))
            },
        }
    }
}

/*
impl RawMatrixElem<dyn DataPartialIO>
{
    pub fn into_csr_u32_iter<'a>(
        &'a self,
        chunk_size: usize
    ) -> impl Iterator<Item = Vec<Vec<(usize, u32)>>> + 'a
    {
        match self.inner.dtype {
            DataType::CsrMatrix(Integer(IntSize::U8)) =>
                self.downcast::<CsrMatrix<i64>>().into_row_iter(chunk_size).map(|x|
                    x.into_iter().map(|vec|
                        vec.into_iter().map(|(i, v)| (i, v.try_into().unwrap())).collect()
                    ).collect()
                ),
            _ => todo!()
        }
    }
}
*/

pub enum CsrRowsIterator<'a, T> {
    Memory((CsrRowIter<'a, T>, usize)),
    Disk((Dataset, Dataset, Vec<usize>, usize, usize)),
}

impl<'a, T> Iterator for CsrRowsIterator<'a, T>
where
    T: H5Type + Copy,
{
    type Item = Vec<Vec<(usize, T)>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CsrRowsIterator::Memory((iter, chunk_size)) => {
                let vec: Vec<_> = iter.take(*chunk_size).map(|r| r.col_indices().iter()
                    .zip(r.values()).map(|(i, v)| (*i, *v)).collect()
                ).collect();
                if vec.is_empty() {
                    None
                } else {
                    Some(vec)
                }
            },
            CsrRowsIterator::Disk((data_, indices_, indptr, current_row, chunk_size)) => {
                if *current_row >= indptr.len() - 1 {
                    None
                } else {
                    let i = *current_row;
                    let j = (i + *chunk_size).min(indptr.len() - 1);
                    let lo = indptr[i];
                    let hi = indptr[j];
                    let data: Array1<T> = data_.read_slice_1d(lo..hi).unwrap();
                    let indices: Array1<usize> = indices_.read_slice_1d(lo..hi).unwrap();
                    let result = (i..j).map(|idx| {
                        let a = indptr[idx] - lo;
                        let b = indptr[idx+1] - lo;
                        indices.slice(s![a..b]).into_iter().zip(
                            data.slice(s![a..b])
                        ).map(|(i, v)| (*i, *v)).collect()
                    }).collect();
                    *current_row = j;
                    Some(result)
                }
            },
        }
    }
}

pub struct ChunkedMatrix {
    pub(crate) elem: MatrixElem,
    pub(crate) chunk_size: usize,
    pub(crate) size: usize,
    pub(crate) current_index: usize,
}

impl Iterator for ChunkedMatrix {
    type Item = Box<dyn DataPartialIO>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.size {
            None
        } else {
            let i = self.current_index;
            let j = std::cmp::min(self.size, self.current_index + self.chunk_size);
            self.current_index = j;
            let data = self.elem.inner().read_dyn_row_slice(i..j).unwrap();
            Some(data)
        }
    }
}

pub struct StackedChunkedMatrix {
    pub(crate) matrices: Vec<ChunkedMatrix>,
    pub(crate) current_matrix_index: usize,
    pub(crate) n_mat: usize,
}

impl Iterator for StackedChunkedMatrix {
    type Item = Box<dyn DataPartialIO>;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.current_matrix_index;
        if i >= self.n_mat {
            None
        } else {
            match self.matrices[i].next() {
                None => { 
                    self.current_matrix_index += 1;
                    self.next()
                },
                r => r,
            }
        }
    }
}