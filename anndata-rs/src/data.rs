mod base;
mod matrix;

pub use base::*;
pub use matrix::*;

use crate::{proc_numeric_data, proc_numeric_data_ref, _box};

use anyhow::anyhow;
use ndarray::{ArrayD, Axis};
use itertools::Itertools;
use hdf5::{Result, Group};
use nalgebra_sparse::csr::CsrMatrix;
use polars::frame::DataFrame;
use downcast_rs::{Downcast, impl_downcast};
use std::{ops::Deref, fmt};

/// Super trait to deal with regular data IO.
pub trait Data: Send + Sync + Downcast + WriteData + ReadData {}
impl_downcast!(Data);

impl<T> Data for T where T: Send + Sync + WriteData + ReadData + 'static {}

impl fmt::Debug for Box<dyn Data> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "dyn Data: {}", self.get_dtype())
    }
}

impl Clone for Box<dyn Data> {
    fn clone(&self) -> Self { self.to_dyn_data() }
}

impl ReadData for Box<dyn Data> {
    fn read(container: &DataContainer) -> Result<Self> where Self: Sized {
        match container.get_encoding_type()? {
            DataType::String => Ok(Box::new(String::read(container)?)),
            DataType::DataFrame => Ok(Box::new(DataFrame::read(container)?)),
            DataType::Mapping => Ok(Box::new(Mapping::read(container)?)),
            DataType::Scalar(ty) => proc_numeric_data!(
                ty, ReadData::read(container)?, _box, Scalar
            ),
            DataType::Array(ty) => proc_numeric_data!(
                ty, ReadData::read(container)?, _box, ArrayD
            ),
            DataType::CsrMatrix(ty) => proc_numeric_data!(
                ty, ReadData::read(container)?, _box, CsrMatrix
            ),
            unknown => Err(hdf5::Error::Internal(
                format!("Not implemented: Dynamic reading of type '{:?}'", unknown)
            ))?,
        }
    }

    fn to_dyn_data(&self) -> Box<dyn Data> { self.deref().to_dyn_data() }
    fn into_dyn_data(self) -> Box<dyn Data> { self }
}

impl WriteData for Box<dyn Data> {
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer> {
        self.deref().write(location, name)
    }

    fn version(&self) -> &str { self.deref().version() }

    fn get_dtype(&self) -> DataType { self.deref().get_dtype() }

    fn dtype() -> DataType where Self: Sized {
        panic!("'.dtype()' cannot be called on a trait object, please use '.get_dtype()'")
    }
}

pub trait MatrixData: PartialIO + Data + Downcast {}
impl_downcast!(MatrixData);

impl<T> MatrixData for T where T: PartialIO + Data {}

impl fmt::Debug for Box<dyn MatrixData> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "dyn MatrixData: {}", self.get_dtype())
    }
}

impl Clone for Box<dyn MatrixData> {
    fn clone(&self) -> Self { self.to_dyn_matrix() }
}

impl ReadData for Box<dyn MatrixData> {
    fn read(container: &DataContainer) -> Result<Self> {
        read_dyn_data_subset(container, None, None)
    }
    fn to_dyn_data(&self) -> Box<dyn Data> { self.deref().to_dyn_data() }
    fn into_dyn_data(self) -> Box<dyn Data> { unimplemented!() }
}

impl WriteData for Box<dyn MatrixData> {
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer> {
        self.deref().write(location, name)
    }

    fn version(&self) -> &str { self.deref().version() }

    fn get_dtype(&self) -> DataType { self.deref().get_dtype() }

    fn dtype() -> DataType where Self: Sized {
        panic!("'.dtype()' cannot be called on a trait object, please use '.get_dtype()'")
    }
}

impl MatrixOp for Box<dyn MatrixData> {
    fn shape(&self) -> (usize, usize) { self.deref().shape() }
    fn nrows(&self) -> usize { self.deref().nrows() }
    fn ncols(&self) -> usize { self.deref().ncols() }
    fn get_rows(&self, idx: &[usize]) -> Self {
        macro_rules! _get { ($x:expr) => { Box::new($x.get_rows(idx)) }; }
        match self.get_dtype() {
            DataType::Array(ty) => proc_numeric_data_ref!(ty, self.downcast_ref().unwrap(), _get, ArrayD),
            DataType::CsrMatrix(ty) => proc_numeric_data_ref!(ty, self.downcast_ref().unwrap(), _get, CsrMatrix),
            unknown => panic!("Not implemented: Dynamic reading of type '{:?}'", unknown),
        }
    }
    fn get_columns(&self, idx: &[usize]) -> Self {
        macro_rules! _get { ($x:expr) => { Box::new($x.get_columns(idx)) }; }
        match self.get_dtype() {
            DataType::Array(ty) => proc_numeric_data_ref!(ty, self.downcast_ref().unwrap(), _get, ArrayD),
            DataType::CsrMatrix(ty) => proc_numeric_data_ref!(ty, self.downcast_ref().unwrap(), _get, CsrMatrix),
            unknown => panic!("Not implemented: Dynamic reading of type '{:?}'", unknown),
        }
    }
    fn subset(&self, ridx: &[usize], cidx: &[usize]) -> Self {
        macro_rules! _get { ($x:expr) => { Box::new($x.subset(ridx, cidx)) }; }
        match self.get_dtype() {
            DataType::Array(ty) => proc_numeric_data_ref!(ty, self.downcast_ref().unwrap(), _get, ArrayD),
            DataType::CsrMatrix(ty) => proc_numeric_data_ref!(ty, self.downcast_ref().unwrap(), _get, CsrMatrix),
            unknown => panic!("Not implemented: Dynamic reading of type '{:?}'", unknown),
        }
    }

    fn to_dyn_matrix(&self) -> Box<dyn MatrixData> { self.deref().to_dyn_matrix() }
    fn into_dyn_matrix(self) -> Box<dyn MatrixData> { self }
}

macro_rules! size_reader {
    ($container:expr, $ty:ident, $size:ident) => {
        match $container.get_encoding_type().unwrap() {
            DataType::CsrMatrix(_) => <CsrMatrix<i8> as $ty>::$size($container),
            DataType::Array(_) => <ArrayD<i8> as $ty>::$size($container),
            DataType::DataFrame => <DataFrame as $ty>::$size($container),
            unknown => panic!("Not implemented: Dynamic reading of type '{:?}'", unknown),
        }
    };
}

impl PartialIO for Box<dyn MatrixData> {
    fn get_nrows(container: &DataContainer) -> usize { size_reader!(container, PartialIO, get_nrows) }
    fn get_ncols(container: &DataContainer) -> usize { size_reader!(container, PartialIO, get_ncols) }

    fn read_rows(container: &DataContainer, idx: &[usize]) -> Self { read_dyn_data_subset(container, Some(idx), None).unwrap() }

    fn read_row_slice(container: &DataContainer, slice: std::ops::Range<usize>) -> Result<Self> {
        match container.get_encoding_type()? {
            DataType::Array(ty) => proc_numeric_data!(
                ty, PartialIO::read_row_slice(container, slice)?, _box, ArrayD
            ),
            DataType::CsrMatrix(ty) => proc_numeric_data!(
                ty, PartialIO::read_row_slice(container, slice)?, _box, CsrMatrix
            ),
            unknown => Err(hdf5::Error::Internal(
                format!("Not implemented: Dynamic reading of type '{:?}'", unknown)
            ))?,
        }
    }

    fn read_columns(container: &DataContainer, idx: &[usize]) -> Self {
        read_dyn_data_subset(container, None, Some(idx)).unwrap()
    }

    fn read_partial(container: &DataContainer, ridx: &[usize], cidx: &[usize]) -> Self {
        read_dyn_data_subset(container, Some(ridx), Some(cidx)).unwrap()
    }

    fn write_rows(&self, idx: &[usize], location: &Group, name: &str) -> Result<DataContainer> {
        self.deref().write_rows(idx, location, name)
    }

    fn write_columns( &self, idx: &[usize], location: &Group, name: &str) -> Result<DataContainer> {
        self.deref().write_columns(idx, location, name)
    }

    fn write_partial(&self, ridx: &[usize], cidx: &[usize], location: &Group, name: &str) -> Result<DataContainer> {
        self.deref().write_partial(ridx, cidx, location, name)
    }
}

fn read_dyn_data_subset(
    container: &DataContainer,
    ridx: Option<&[usize]>,
    cidx: Option<&[usize]>,
) -> Result<Box<dyn MatrixData>> {
    fn read_data_subset<T: MatrixData>(
        container: &DataContainer,
        ridx: Option<&[usize]>,
        cidx: Option<&[usize]>,
    ) -> T {
        match ridx {
            None => match cidx {
                None => ReadData::read(container).unwrap(),
                Some(j) => PartialIO::read_columns(container, j),
            },
            Some(i) => match cidx {
                None => PartialIO::read_rows(container, i),
                Some(j) => PartialIO::read_partial(container, i, j),
            },
        }
    }

    let dtype = container.get_encoding_type()?;
    match dtype {
        DataType::DataFrame => {
            let df: DataFrame = read_data_subset(container, ridx, cidx);
            Ok(Box::new(df))
        },
        DataType::Array(ty) => proc_numeric_data!(
            ty, read_data_subset(container, ridx, cidx), _box, ArrayD
        ),
        DataType::CsrMatrix(ty) => proc_numeric_data!(
            ty, read_data_subset(container, ridx, cidx), _box, CsrMatrix
        ),
        unknown => Err(hdf5::Error::Internal(
            format!("Not implemented: Dynamic reading of type '{:?}'", unknown)
        ))?,
    }
}

pub(crate) fn rstack_with_index(
    index: &[usize],
    mats: Vec<Box<dyn MatrixData>>
) -> Result<Box<dyn MatrixData>> {
    match mats[0].get_dtype() {
        DataType::Array(ty) => proc_numeric_data!(
            ty, rstack_arr_with_index(index, mats.into_iter().map(|x| x.downcast().map_err(|_|
                anyhow!("cannot downcast Array with type {}", ty)).unwrap()).collect()
            ), _box, ArrayD
        ),
        DataType::CsrMatrix(ty) => proc_numeric_data!(
            ty, rstack_csr_with_index(index, mats.into_iter().map(|x| x.downcast().map_err(|_| 
                anyhow!("cannot downcast Sparse Row Matrix with type {}", ty)).unwrap()).collect()
            ), _box, CsrMatrix
        ),
        x => panic!("type '{}' is not a supported matrix format", x),
    }
}

fn rstack_arr_with_index<T: Clone>(
    index: &[usize],
    mats: Vec<Box<ArrayD<T>>>,
) -> ArrayD<T> {
    let merged = mats.into_iter().reduce(|mut accum, other| {
        accum.as_mut().append(Axis(0), other.view()).unwrap();
        accum
    }).unwrap();
    let new_idx: Vec<_> = index.iter().enumerate().sorted_by_key(|x| *x.1)
        .map(|x| x.0).collect();
    merged.select(Axis(0), new_idx.as_slice())
}

fn rstack_csr_with_index<T: Clone>(
    index: &[usize],
    mats: Vec<Box<CsrMatrix<T>>>,
) -> CsrMatrix<T> {
    if !mats.iter().map(|x| x.ncols()).all_equal() {
        panic!("num cols mismatch");
    }
    let num_rows = mats.iter().map(|x| x.nrows()).sum();
    let num_cols = mats.iter().next().map(|x| x.ncols()).unwrap_or(0);
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_offsets = Vec::new();
    let nnz = mats.iter().map(|x| x.row_iter()).flatten()
        .zip(index).sorted_by_key(|x| *x.1).fold(0, |acc, x| {
            row_offsets.push(acc);
            values.extend_from_slice(x.0.values());
            col_indices.extend_from_slice(x.0.col_indices());
            acc + x.0.nnz()
        });
    row_offsets.push(nnz);
    CsrMatrix::try_from_csr_data(num_rows, num_cols, row_offsets, col_indices, values).unwrap()
}

pub(crate) fn rstack(mats: Vec<Box<dyn MatrixData>>) -> Result<Box<dyn MatrixData>> {
    match mats[0].get_dtype() {
        DataType::Array(ty) => proc_numeric_data!(
            ty, rstack_arr(mats.into_iter().map(|x| x.downcast().map_err(|_|
                anyhow!("cannot downcast Array with type {}", ty)).unwrap())), _box, ArrayD
        ),
        DataType::CsrMatrix(ty) => proc_numeric_data!(
            ty, rstack_csr(mats.into_iter().map(|x| x.downcast().map_err(|_|
                anyhow!("cannot downcast Sparse Row Matrix with type {}", ty)).unwrap())), _box, CsrMatrix
        ),
        x => panic!("type '{}' is not a supported matrix format", x),
    }
}

fn rstack_arr<I, T>(mats: I) -> ArrayD<T>
where
    I: Iterator<Item = Box<ArrayD<T>>>,
    T: Clone,
{
    *mats.reduce(|mut accum, other| {
        accum.as_mut().append(Axis(0), other.view()).unwrap();
        accum
    }).unwrap()
}

fn rstack_csr<I, T>(mats: I) -> CsrMatrix<T>
where
    I: Iterator<Item = Box<CsrMatrix<T>>>,
    T: Clone,
{

    let mut num_rows = 0;
    let mut num_cols = 0;

    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_offsets = Vec::new();
    let nnz = mats.fold(0, |acc, mat| {
        num_rows += mat.nrows();
        num_cols = mat.ncols();
        mat.row_iter().fold(acc, |cidx, row| {
            row_offsets.push(cidx);
            values.extend_from_slice(row.values());
            col_indices.extend_from_slice(row.col_indices());
            cidx + row.nnz()
        })
    });
    row_offsets.push(nnz);
    CsrMatrix::try_from_csr_data(num_rows, num_cols, row_offsets, col_indices, values).unwrap()
}