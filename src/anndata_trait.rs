mod data;
mod subsetting;

pub use data::*;
pub use subsetting::*;

use ndarray::ArrayD;
use hdf5::Result;
use hdf5::types::TypeDescriptor::*;
use nalgebra_sparse::csr::CsrMatrix;
use polars::frame::DataFrame;
use dyn_clone::DynClone;
use downcast_rs::Downcast;
use downcast_rs::impl_downcast;

pub trait DataIO: Send + Sync + DynClone + Downcast + WriteData + ReadData {}
impl_downcast!(DataIO);
impl<T> DataIO for T where T: Clone + Send + Sync + WriteData + ReadData + 'static {}

pub trait DataPartialIO: DataIO + DataSubset2D + ReadPartial {}
impl<T> DataPartialIO for T where T: DataIO + DataSubset2D + ReadPartial {}

pub trait WritePartialData: DataSubset2D + WriteData {}
impl<T> WritePartialData for T where T: DataSubset2D + WriteData {}

macro_rules! dyn_data_reader {
    ($get_type:expr, $reader:expr) => {
        match $get_type {
            DataType::CsrMatrix(Integer(_)) => {
                let mat: CsrMatrix<i64> = $reader;
                Ok(Box::new(mat))
            },
            DataType::CsrMatrix(Unsigned(_)) => {
                let mat: CsrMatrix<u64> = $reader;
                Ok(Box::new(mat))
            },
            DataType::CsrMatrix(Float(_)) => {
                let mat: CsrMatrix<f64> = $reader;
                Ok(Box::new(mat))
            },
            DataType::Array(Integer(_)) => {
                let mat: ArrayD<i64> = $reader;
                Ok(Box::new(mat))
            },
            DataType::Array(Unsigned(_)) => {
                let mat: ArrayD<u64> = $reader;
                Ok(Box::new(mat))
            },
            DataType::Array(Float(_)) => {
                let mat: ArrayD<f64> = $reader;
                Ok(Box::new(mat))
            },
            DataType::DataFrame => {
                let df: DataFrame = $reader;
                Ok(Box::new(df))
            },
            unknown => Err(hdf5::Error::Internal(
                format!("Not implemented: Dynamic reading of type '{:?}'", unknown)
            ))?,
        }
    };
}

pub fn read_dyn_data(container: &DataContainer) -> Result<Box<dyn DataIO>> {
    dyn_data_reader!(container.get_encoding_type()?, ReadData::read(container)?)
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
                Some(j) => ReadCols::read_columns(container, j),
            },
            Some(i) => match cidx {
                None => ReadRows::read_rows(container, i),
                Some(j) => ReadPartial::read_partial(container, i, j),
            },
        }
    }

    dyn_data_reader!(container.get_encoding_type()?, read_data_subset(container, ridx, cidx))
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
    size_reader!(container, DataSubsetRow, container_nrows)
}
 
pub fn get_ncols(container: &DataContainer) -> usize {
    size_reader!(container, DataSubsetCol, container_ncols)
}