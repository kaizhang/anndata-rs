mod data;
mod matrix;

use crate::{proc_numeric_data, _box};

pub use data::*;
pub use matrix::*;

use ndarray::ArrayD;
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