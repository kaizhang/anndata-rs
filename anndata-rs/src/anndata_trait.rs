mod data;
mod subsetting;

pub use data::*;
pub use subsetting::*;

use ndarray::ArrayD;
use hdf5::Result;
use hdf5::types::TypeDescriptor::*;
use hdf5::types::IntSize;
use hdf5::types::FloatSize;
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
            DataType::CsrMatrix(Integer(IntSize::U1)) => {
                let mat: CsrMatrix<i8> = $reader;
                Ok(Box::new(mat))
            },
            DataType::CsrMatrix(Integer(IntSize::U2)) => {
                let mat: CsrMatrix<i16> = $reader;
                Ok(Box::new(mat))
            },
            DataType::CsrMatrix(Integer(IntSize::U4)) => {
                let mat: CsrMatrix<i32> = $reader;
                Ok(Box::new(mat))
            },
            DataType::CsrMatrix(Integer(IntSize::U8)) => {
                let mat: CsrMatrix<i64> = $reader;
                Ok(Box::new(mat))
            },
            DataType::CsrMatrix(Unsigned(IntSize::U1)) => {
                let mat: CsrMatrix<u8> = $reader;
                Ok(Box::new(mat))
            },
            DataType::CsrMatrix(Unsigned(IntSize::U2)) => {
                let mat: CsrMatrix<u16> = $reader;
                Ok(Box::new(mat))
            },
            DataType::CsrMatrix(Unsigned(IntSize::U4)) => {
                let mat: CsrMatrix<u32> = $reader;
                Ok(Box::new(mat))
            },
            DataType::CsrMatrix(Unsigned(IntSize::U8)) => {
                let mat: CsrMatrix<u64> = $reader;
                Ok(Box::new(mat))
            },
            DataType::CsrMatrix(Float(FloatSize::U4)) => {
                let mat: CsrMatrix<f32> = $reader;
                Ok(Box::new(mat))
            },
            DataType::CsrMatrix(Float(FloatSize::U8)) => {
                let mat: CsrMatrix<f64> = $reader;
                Ok(Box::new(mat))
            },
            DataType::Array(Integer(IntSize::U1)) => {
                let mat: ArrayD<i8> = $reader;
                Ok(Box::new(mat))
            },
            DataType::Array(Integer(IntSize::U2)) => {
                let mat: ArrayD<i16> = $reader;
                Ok(Box::new(mat))
            },
            DataType::Array(Integer(IntSize::U4)) => {
                let mat: ArrayD<i32> = $reader;
                Ok(Box::new(mat))
            },
            DataType::Array(Integer(IntSize::U8)) => {
                let mat: ArrayD<i64> = $reader;
                Ok(Box::new(mat))
            },
            DataType::Array(Unsigned(IntSize::U1)) => {
                let mat: ArrayD<u8> = $reader;
                Ok(Box::new(mat))
            },
            DataType::Array(Unsigned(IntSize::U2)) => {
                let mat: ArrayD<u16> = $reader;
                Ok(Box::new(mat))
            },
            DataType::Array(Unsigned(IntSize::U4)) => {
                let mat: ArrayD<u32> = $reader;
                Ok(Box::new(mat))
            },
            DataType::Array(Unsigned(IntSize::U8)) => {
                let mat: ArrayD<u64> = $reader;
                Ok(Box::new(mat))
            },
            DataType::Array(Float(FloatSize::U4)) => {
                let mat: ArrayD<f32> = $reader;
                Ok(Box::new(mat))
            },
            DataType::Array(Float(FloatSize::U8)) => {
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
    let dtype = container.get_encoding_type()?;
    match dtype {
        DataType::Scalar(Integer(IntSize::U1)) => Ok(Box::new(Scalar::<i8>::read(container)?)),
        DataType::Scalar(Integer(IntSize::U2)) => Ok(Box::new(Scalar::<i16>::read(container)?)),
        DataType::Scalar(Integer(IntSize::U4)) => Ok(Box::new(Scalar::<i32>::read(container)?)),
        DataType::Scalar(Integer(IntSize::U8)) => Ok(Box::new(Scalar::<i64>::read(container)?)),

        DataType::Scalar(Unsigned(IntSize::U1)) => Ok(Box::new(Scalar::<u8>::read(container)?)),
        DataType::Scalar(Unsigned(IntSize::U2)) => Ok(Box::new(Scalar::<u16>::read(container)?)),
        DataType::Scalar(Unsigned(IntSize::U4)) => Ok(Box::new(Scalar::<u32>::read(container)?)),
        DataType::Scalar(Unsigned(IntSize::U8)) => Ok(Box::new(Scalar::<u64>::read(container)?)),

        DataType::Scalar(Float(FloatSize::U4)) => Ok(Box::new(Scalar::<f32>::read(container)?)),
        DataType::Scalar(Float(FloatSize::U8)) => Ok(Box::new(Scalar::<f64>::read(container)?)),

        DataType::Scalar(VarLenUnicode) => Ok(Box::new(String::read(container)?)),
        DataType::Scalar(Boolean) => Ok(Box::new(Scalar::<bool>::read(container)?)),

        _ => dyn_data_reader!(dtype, ReadData::read(container)?),
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

pub fn read_dyn_row_slice(
    container: &DataContainer,
    slice: std::ops::Range<usize>,
) -> Result<Box<dyn DataPartialIO>> {
    dyn_data_reader!(
        container.get_encoding_type()?,
        ReadRows::read_row_slice(container, slice)
    )
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