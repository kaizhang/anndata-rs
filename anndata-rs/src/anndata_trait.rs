mod data;
mod matrix;

use crate::{proc_numeric_data, _box};

pub use data::*;
pub use matrix::*;

use ndarray::{ArrayD, Axis};
use itertools::Itertools;
use hdf5::Result;
use nalgebra_sparse::csr::CsrMatrix;
use polars::frame::DataFrame;
use dyn_clone::DynClone;
use downcast_rs::Downcast;
use downcast_rs::impl_downcast;

pub trait DataIO: Send + Sync + DynClone + Downcast + WriteData + ReadData {}
impl_downcast!(DataIO);
dyn_clone::clone_trait_object!(DataIO);

impl<T> DataIO for T where T: Clone + Send + Sync + WriteData + ReadData + 'static {}

pub trait DataPartialIO: DataIO + MatrixIO {}
impl<T> DataPartialIO for T where T: DataIO + MatrixIO {}

pub fn read_dyn_data(container: &DataContainer) -> Result<Box<dyn DataIO>> {
    let dtype = container.get_encoding_type()?;
    match dtype {
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

pub fn read_dyn_data_subset(
    container: &DataContainer,
    ridx: Option<&[usize]>,
    cidx: Option<&[usize]>,
) -> Result<Box<dyn DataPartialIO>> {
    fn read_data_subset<T: DataPartialIO>(
        container: &DataContainer,
        ridx: Option<&[usize]>,
        cidx: Option<&[usize]>,
    ) -> T {
        match ridx {
            None => match cidx {
                None => ReadData::read(container).unwrap(),
                Some(j) => MatrixIO::read_columns(container, j),
            },
            Some(i) => match cidx {
                None => MatrixIO::read_rows(container, i),
                Some(j) => MatrixIO::read_partial(container, i, j),
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

pub fn read_dyn_row_slice(
    container: &DataContainer,
    slice: std::ops::Range<usize>,
) -> Result<Box<dyn DataPartialIO>> {
    let dtype = container.get_encoding_type()?;
    match dtype {
        DataType::Array(ty) => proc_numeric_data!(
            ty, MatrixIO::read_row_slice(container, slice), _box, ArrayD
        ),
        DataType::CsrMatrix(ty) => proc_numeric_data!(
            ty, MatrixIO::read_row_slice(container, slice), _box, CsrMatrix
        ),
        unknown => Err(hdf5::Error::Internal(
            format!("Not implemented: Dynamic reading of type '{:?}'", unknown)
        ))?,
    }
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

pub fn get_nrows(container: &DataContainer) -> usize {
    size_reader!(container, MatrixIO, get_nrows)
}
 
pub fn get_ncols(container: &DataContainer) -> usize {
    size_reader!(container, MatrixIO, get_ncols)
}

pub(crate) fn rstack_with_index(
    index: &[usize],
    mats: Vec<Box<dyn DataPartialIO>>
) -> Result<Box<dyn DataPartialIO>> {
    match mats[0].get_dtype() {
        DataType::Array(ty) => proc_numeric_data!(
            ty,
            rstack_arr_with_index(
                index,
                mats.into_iter().map(|x| x.into_any().downcast().unwrap()).collect(),
            ),
            _box,
            ArrayD
        ),
        DataType::CsrMatrix(ty) => proc_numeric_data!(
            ty,
            rstack_csr_with_index(
                index,
                mats.into_iter().map(|x| x.into_any().downcast().unwrap()).collect(),
            ),
            _box,
            CsrMatrix
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

pub(crate) fn rstack(mats: Vec<Box<dyn DataPartialIO>>) -> Result<Box<dyn DataPartialIO>> {
    match mats[0].get_dtype() {
        DataType::Array(ty) => proc_numeric_data!(
            ty,
            rstack_arr(
                mats.into_iter().map(|x| x.into_any().downcast().unwrap())
            ),
            _box,
            ArrayD
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

/*
fn rstack_csr<I, T>(mats: I, n_rows: usize, n_cols: usize) -> CsrMatrix<T>
where
    I: Iterator<Item = Box<CsrMatrix<T>>>,
    T: Clone,
{
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_offsets = Vec::new();
    let nnz = mats.map(|x| x.row_iter()).flatten().fold(0, |acc, x| {
        row_offsets.push(acc);
        values.extend_from_slice(x.values());
        col_indices.extend_from_slice(x.col_indices());
        acc + x.nnz()
    });
    row_offsets.push(nnz);
    CsrMatrix::try_from_csr_data(n_rows, n_cols, row_offsets, col_indices, values).unwrap()
}
*/