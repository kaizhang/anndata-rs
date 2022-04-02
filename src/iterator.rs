use crate::utils::{ResizableVectorData, COMPRESSION, create_str_attr};
use crate::anndata_trait::{DataType, DataContainer};
use crate::base::AnnData;
use crate::element::{MatrixElem, RawMatrixElem};

use nalgebra_sparse::csr::{CsrMatrix, CsrRowIter};
use ndarray::{s, arr1, Array, Array1};
use hdf5::{Dataset, Group, H5Type, Result};
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

impl<I, D> RowIterator for CsrIterator<I>
where
    I: Iterator<Item = Vec<(usize, D)>>,
    D: H5Type,
{
    fn write(self, location: &Group, name: &str) -> Result<(DataContainer, usize)> {
        let group = location.create_group(name)?;
        create_str_attr(&group, "encoding-type", "csr_matrix")?;
        create_str_attr(&group, "encoding-version", self.version())?;
        create_str_attr(&group, "h5sparse_format", "csr")?;
        let data: ResizableVectorData<D> =
            ResizableVectorData::new(&group, "data", 10000)?;
        let mut indptr: Vec<usize> = vec![0];
        let iter = self.iterator.scan(0, |state, x| {
            *state = *state + x.len();
            Some((*state, x))
        });

        if self.num_cols <= (i32::MAX as usize) {
            let indices: ResizableVectorData<i32> =
                ResizableVectorData::new(&group, "indices", 10000)?;
            for chunk in &iter.chunks(10000) {
                let (a, b): (Vec<i32>, Vec<D>) = chunk.map(|(x, vec)| {
                    indptr.push(x);
                    vec
                }).flatten().map(|(x, y)| -> (i32, D) {(
                    x.try_into().expect(&format!("cannot convert '{}' to i32", x)),
                    y
                ) }).unzip();
                indices.extend(a.into_iter())?;
                data.extend(b.into_iter())?;
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
        } else {
            let indices: ResizableVectorData<i64> =
                ResizableVectorData::new(&group, "indices", 10000)?;
            for chunk in &iter.chunks(10000) {
                let (a, b): (Vec<i64>, Vec<D>) = chunk.map(|(x, vec)| {
                    indptr.push(x);
                    vec
                }).flatten().map(|(x, y)| -> (i64, D) {(
                    x.try_into().expect(&format!("cannot convert '{}' to i64", x)),
                    y
                ) }).unzip();
                indices.extend(a.into_iter())?;
                data.extend(b.into_iter())?;
            }

            let num_rows = indptr.len() - 1;
            group.new_attr_builder()
                .with_data(&arr1(&[num_rows, self.num_cols]))
                .create("shape")?;

            let vec: Vec<i64> = indptr.into_iter()
                .map(|x| x.try_into().unwrap()).collect();
            group.new_dataset_builder().deflate(COMPRESSION)
                .with_data(&Array::from_vec(vec)).create("indptr")?;
            Ok((DataContainer::H5Group(group), num_rows))
        }
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
        if self.n_vars() == 0 { self.set_n_vars(data.ncols()); }
        assert!(
            self.n_vars() == data.ncols(),
            "Number of variables mismatched, expecting {}, but found {}",
            self.n_vars(), data.ncols(),
        );

        if !self.x.is_empty() { self.file.unlink("X")?; }
        let (container, nrows) = data.write(&self.file, "X")?;
        if self.n_obs() == 0 { self.set_n_obs(nrows); }
        assert!(
            self.n_obs() == nrows,
            "Number of observations mismatched, expecting {}, but found {}",
            self.n_obs(), nrows,
        );
        self.x.insert(container)?;
        Ok(())
    }

    pub fn add_obsm_from_row_iter<I>(&mut self, key: &str, data: I) -> Result<()>
    where
        I: RowIterator,
    {
       let obsm = match self.file.group("obsm") {
            Ok(x) => x,
            _ => self.file.create_group("obsm").unwrap(),
        };
        if self.obsm.contains_key(key) { obsm.unlink(key)?; } 
        let (container, nrows) = data.write(&obsm, key)?;
        if self.n_obs() == 0 { self.set_n_obs(nrows); }

        assert!(
            self.n_obs() == nrows,
            "Number of observations mismatched, expecting {}, but found {}",
            self.n_obs(), nrows,
        );
 
        let elem = MatrixElem::new(container)?;
        self.obsm.insert(key.to_string(), elem);
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