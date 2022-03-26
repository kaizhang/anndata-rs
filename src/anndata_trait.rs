mod data;
mod subsetting;

pub use data::*;
pub use subsetting::*;

use ndarray::ArrayD;
use hdf5::Result;
use hdf5::types::TypeDescriptor::*;
use nalgebra_sparse::csr::CsrMatrix;
use polars::frame::DataFrame;

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
    dyn_data_reader!(container.get_encoding_type()?, DataIO::read(container)?)
}

pub fn read_dyn_data_subset(
    container: &DataContainer,
    ridx: Option<&[usize]>,
    cidx: Option<&[usize]>,
) -> Result<Box<dyn DataSubset2D>> {
    fn read_data_subset<T: DataSubset2D>(
        container: &DataContainer,
        ridx: Option<&[usize]>,
        cidx: Option<&[usize]>,
    ) -> T {
        match ridx {
            None => match cidx {
                None => DataIO::read(container).unwrap(),
                Some(j) => DataSubsetCol::read_columns(container, j),
            },
            Some(i) => match cidx {
                None => DataSubsetRow::read_rows(container, i),
                Some(j) => DataSubset2D::read_partial(container, i, j),
            },
        }
    }

    dyn_data_reader!(container.get_encoding_type()?, read_data_subset(container, ridx, cidx))
}