mod array;
mod other;

pub use array::{Shape, SelectInfo, SelectInfoElem, HasShape, ArrayOp, WriteArrayData, ReadArrayData, DynArray, DynCsrMatrix, select_all, SLICE_FULL};
pub use other::{WriteData, ReadData, DynScalar, Mapping, DataFrameIndex};
use crate::backend::{Backend, GroupOp, DataContainer, DataType};

use polars::frame::DataFrame;
use anyhow::{bail, Result};
use ndarray::ArrayD;
use nalgebra_sparse::csr::CsrMatrix;

#[derive(Debug, Clone)]
pub enum ArrayData {
    Array(DynArray),
    CsrMatrix(DynCsrMatrix),
    //CscMatrix(DynCscMatrix),
}

impl<T: Clone + Into<ArrayData>> From<&T> for ArrayData {
    fn from(data: &T) -> Self {
        data.clone().into()
    }
}

impl From<DynArray> for ArrayData {
    fn from(data: DynArray) -> Self {
        ArrayData::Array(data)
    }
}

impl From<DynCsrMatrix> for ArrayData {
    fn from(data: DynCsrMatrix) -> Self {
        ArrayData::CsrMatrix(data)
    }
}

/// macro for implementing From trait for Data from a list of types
macro_rules! impl_into_matrix_data {
    ($($ty:ty),*) => {
        $(
            impl From<ArrayD<$ty>> for ArrayData {
                fn from(data: ArrayD<$ty>) -> Self {
                    ArrayData::Array(data.into())
                }
            }
            impl From<CsrMatrix<$ty>> for ArrayData {
                fn from(data: CsrMatrix<$ty>) -> Self {
                    ArrayData::CsrMatrix(data.into())
                }
            }
        )*
    };
}

impl_into_matrix_data!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool, String);

impl WriteData for ArrayData {
    fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
        match self {
            ArrayData::Array(data) => data.write(location, name),
            ArrayData::CsrMatrix(data) => data.write(location, name),
        }
    }
}

impl ReadData for ArrayData {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) => DynArray::read(container).map(ArrayData::Array),
            DataType::CsrMatrix(_) => DynCsrMatrix::read(container).map(ArrayData::CsrMatrix),
            DataType::CscMatrix(_) => todo!(),
            ty => bail!("Cannot read type '{:?}' as matrix data", ty),
        }
    }
}

impl HasShape for ArrayData {
    fn shape(&self) -> Shape {
        match self {
            ArrayData::Array(data) => data.shape(),
            ArrayData::CsrMatrix(data) => data.shape(),
        }
    }
}

impl ArrayOp for ArrayData {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        match self {
            ArrayData::Array(data) => data.get(index),
            ArrayData::CsrMatrix(data) => data.get(index),
        }
    }

    fn select<S, E>(&self, info: S) -> Self
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
    {
        match self {
            ArrayData::Array(data) => data.select(info).into(),
            ArrayData::CsrMatrix(data) => data.select(info).into(),
        }
    }
}

impl ReadArrayData for ArrayData {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) => DynArray::get_shape(container),
            DataType::CsrMatrix(_) => DynCsrMatrix::get_shape(container),
            ty => bail!("Cannot read shape information from type '{}'", ty),
        }
    }
}
impl WriteArrayData for ArrayData {}

#[derive(Debug, Clone)]
pub enum Data {
    ArrayData(ArrayData),
    Scalar(DynScalar),
    DataFrame(DataFrame),
    Mapping(Mapping),
}

/// Types that can be converted to Data
impl<T: Clone + Into<Data>> From<&T> for Data {
    fn from(data: &T) -> Self {
        data.clone().into()
    }
}

macro_rules! impl_into_data {
    ($from_type:ty, $to_type:ident) => {
        impl From<$from_type> for Data {
            fn from(data: $from_type) -> Self {
                Data::Scalar(DynScalar::$to_type(data))
            }
        }
        impl From<ArrayD<$from_type>> for Data {
            fn from(data: ArrayD<$from_type>) -> Self {
                Data::ArrayData(ArrayData::Array(DynArray::$to_type(data)))
            }
        }
        impl From<CsrMatrix<$from_type>> for Data {
            fn from(data: CsrMatrix<$from_type>) -> Self {
                Data::ArrayData(ArrayData::CsrMatrix(DynCsrMatrix::$to_type(data)))
            }
        }
    };
}

impl_into_data!(i8, I8);
impl_into_data!(i16, I16);
impl_into_data!(i32, I32);
impl_into_data!(i64, I64);
impl_into_data!(u8, U8);
impl_into_data!(u16, U16);
impl_into_data!(u32, U32);
impl_into_data!(u64, U64);
impl_into_data!(f32, F32);
impl_into_data!(f64, F64);
impl_into_data!(bool, Bool);
impl_into_data!(String, String);

macro_rules! impl_into_data2 {
    ($from_type:ty, $to_type:ident) => {
        impl From<$from_type> for Data {
            fn from(data: $from_type) -> Self {
                Data::$to_type(data)
            }
        }
    };
}

impl_into_data2!(DynScalar, Scalar);
impl_into_data2!(ArrayData, ArrayData);
impl_into_data2!(DataFrame, DataFrame);
impl_into_data2!(Mapping, Mapping);


/// Types that can be converted from Data
impl TryInto<i8> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<i8> {
        match self {
            Data::Scalar(DynScalar::I8(data)) => Ok(data),
            _ => bail!("Cannot convert data to i8"),
        }
    }
}

impl TryInto<i16> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<i16> {
        match self {
            Data::Scalar(DynScalar::I16(data)) => Ok(data),
            _ => bail!("Cannot convert data to i16"),
        }
    }
}

impl TryInto<i32> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<i32> {
        match self {
            Data::Scalar(DynScalar::I32(data)) => Ok(data),
            _ => bail!("Cannot convert data to i32"),
        }
    }
}

impl TryInto<i64> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<i64> {
        match self {
            Data::Scalar(DynScalar::I64(data)) => Ok(data),
            _ => bail!("Cannot convert data to i64"),
        }
    }
}

impl TryInto<u8> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<u8> {
        match self {
            Data::Scalar(DynScalar::U8(data)) => Ok(data),
            _ => bail!("Cannot convert data to u8"),
        }
    }
}

impl TryInto<u16> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<u16> {
        match self {
            Data::Scalar(DynScalar::U16(data)) => Ok(data),
            _ => bail!("Cannot convert data to u16"),
        }
    }
}

impl TryInto<u32> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<u32> {
        match self {
            Data::Scalar(DynScalar::U32(data)) => Ok(data),
            _ => bail!("Cannot convert data to u32"),
        }
    }
}

impl TryInto<u64> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<u64> {
        match self {
            Data::Scalar(DynScalar::U64(data)) => Ok(data),
            _ => bail!("Cannot convert data to u64"),
        }
    }
}

impl TryInto<f32> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<f32> {
        match self {
            Data::Scalar(DynScalar::F32(data)) => Ok(data),
            _ => bail!("Cannot convert data to f32"),
        }
    }
}

impl TryInto<f64> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<f64> {
        match self {
            Data::Scalar(DynScalar::F64(data)) => Ok(data),
            _ => bail!("Cannot convert data to f64"),
        }
    }
}

impl TryInto<ArrayD<i8>> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ArrayD<i8>> {
        match self {
            Data::ArrayData(ArrayData::Array(DynArray::I8(data))) => Ok(data),
            _ => bail!("Cannot convert data to ArrayD<i8>"),
        }
    }
}

impl TryInto<ArrayD<i16>> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ArrayD<i16>> {
        match self {
            Data::ArrayData(ArrayData::Array(DynArray::I16(data))) => Ok(data),
            _ => bail!("Cannot convert data to ArrayD<i16>"),
        }
    }
}

impl TryInto<ArrayD<i32>> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ArrayD<i32>> {
        match self {
            Data::ArrayData(ArrayData::Array(DynArray::I32(data))) => Ok(data),
            _ => bail!("Cannot convert data to ArrayD<i32>"),
        }
    }
}

impl TryInto<ArrayD<i64>> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ArrayD<i64>> {
        match self {
            Data::ArrayData(ArrayData::Array(DynArray::I64(data))) => Ok(data),
            _ => bail!("Cannot convert data to ArrayD<i64>"),
        }
    }
}

impl TryInto<ArrayD<u8>> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ArrayD<u8>> {
        match self {
            Data::ArrayData(ArrayData::Array(DynArray::U8(data))) => Ok(data),
            _ => bail!("Cannot convert data to ArrayD<u8>"),
        }
    }
}

impl TryInto<ArrayD<u16>> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ArrayD<u16>> {
        match self {
            Data::ArrayData(ArrayData::Array(DynArray::U16(data))) => Ok(data),
            _ => bail!("Cannot convert data to ArrayD<u16>"),
        }
    }
}

impl TryInto<ArrayD<u32>> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ArrayD<u32>> {
        match self {
            Data::ArrayData(ArrayData::Array(DynArray::U32(data))) => Ok(data),
            _ => bail!("Cannot convert data to ArrayD<u32>"),
        }
    }
}

impl TryInto<ArrayD<u64>> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ArrayD<u64>> {
        match self {
            Data::ArrayData(ArrayData::Array(DynArray::U64(data))) => Ok(data),
            _ => bail!("Cannot convert data to ArrayD<u64>"),
        }
    }
}

impl TryInto<ArrayD<f32>> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ArrayD<f32>> {
        match self {
            Data::ArrayData(ArrayData::Array(DynArray::F32(data))) => Ok(data),
            _ => bail!("Cannot convert data to ArrayD<f32>"),
        }
    }
}

impl TryInto<ArrayD<f64>> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ArrayD<f64>> {
        match self {
            Data::ArrayData(ArrayData::Array(DynArray::F64(data))) => Ok(data),
            _ => bail!("Cannot convert data to ArrayD<f64>"),
        }
    }
}

impl TryInto<ArrayD<bool>> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ArrayD<bool>> {
        match self {
            Data::ArrayData(ArrayData::Array(DynArray::Bool(data))) => Ok(data),
            _ => bail!("Cannot convert data to ArrayD<bool>"),
        }
    }
}

impl TryInto<ArrayD<String>> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ArrayD<String>> {
        match self {
            Data::ArrayData(ArrayData::Array(DynArray::String(data))) => Ok(data),
            _ => bail!("Cannot convert data to ArrayD<String>"),
        }
    }
}

impl TryInto<bool> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<bool> {
        match self {
            Data::Scalar(DynScalar::Bool(data)) => Ok(data),
            _ => bail!("Cannot convert data to bool"),
        }
    }
}

impl TryInto<String> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<String> {
        match self {
            Data::Scalar(DynScalar::String(data)) => Ok(data),
            _ => bail!("Cannot convert data to String"),
        }
    }
}

impl TryInto<DataFrame> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<DataFrame> {
        match self {
            Data::DataFrame(data) => Ok(data),
            _ => bail!("Cannot convert data to DataFrame"),
        }
    }
}

impl TryInto<Mapping> for Data {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Mapping> {
        match self {
            Data::Mapping(data) => Ok(data),
            _ => bail!("Cannot convert data to Mapping"),
        }
    }
}


impl WriteData for Data {
    fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
        match self {
            Data::ArrayData(data) => data.write(location, name),
            Data::Scalar(data) => data.write(location, name),
            Data::DataFrame(data) => data.write(location, name),
            Data::Mapping(data) => data.write(location, name),
        }
    }
}

impl ReadData for Data {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) => DynArray::read(container).map(|x| ArrayData::from(x).into()),
            DataType::CsrMatrix(_) => DynCsrMatrix::read(container).map(|x| ArrayData::from(x).into()),
            DataType::CscMatrix(_) => todo!(),
            DataType::Scalar(_) => DynScalar::read(container).map(|x| x.into()),
            DataType::DataFrame => DataFrame::read(container).map(|x| x.into()),
            DataType::Mapping => Mapping::read(container).map(|x| x.into()),
        }
    }
}