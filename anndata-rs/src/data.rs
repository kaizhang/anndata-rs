mod array;
mod other;
mod slice;

pub use array::{Shape, HasShape, ArrayOp, WriteArrayData, ReadArrayData, DynArray, DynCsrMatrix};
pub use other::{WriteData, ReadData, DynScalar, Mapping, DataFrameIndex};
pub use slice::{SelectInfo, SelectInfoElem, SLICE_FULL, BoundedSelectInfoElem, BoundedSelectInfo};

use crate::backend::*;

use polars::frame::DataFrame;
use anyhow::{bail, Result};
use ndarray::{Array, ArrayD, Dimension};
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
macro_rules! impl_into_array_data {
    ($($ty:ty),*) => {
        $(
            impl<D: Dimension> From<Array<$ty, D>> for ArrayData {
                fn from(data: Array<$ty, D>) -> Self {
                    ArrayData::Array(data.into_dyn().into())
                }
            }
            impl From<CsrMatrix<$ty>> for ArrayData {
                fn from(data: CsrMatrix<$ty>) -> Self {
                    ArrayData::CsrMatrix(data.into())
                }
            }

            impl<D: Dimension> TryFrom<ArrayData> for Array<$ty, D> {
                type Error = anyhow::Error;
                fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
                    match value {
                        ArrayData::Array(data) => data.try_into(),
                        _ => bail!("Cannot convert {:?} to $ty Array", value),
                    }
                }
            }
            impl TryFrom<ArrayData> for CsrMatrix<$ty> {
                type Error = anyhow::Error;
                fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
                    match value {
                        ArrayData::CsrMatrix(data) => data.try_into(),
                        _ => bail!("Cannot convert {:?} to $ty CsrMatrix", value),
                    }
                }
            }
        )*
    };
}

impl_into_array_data!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool, String);


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

    fn read_select<B, S, E>(container: &DataContainer<B>, info: S) -> Result<Self>
        where
            B: Backend,
            S: AsRef<[E]>,
            E: AsRef<SelectInfoElem>,
            Self: Sized {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) => DynArray::read_select(container, info).map(ArrayData::Array),
            DataType::CsrMatrix(_) => DynCsrMatrix::read_select(container, info).map(ArrayData::CsrMatrix),
            ty => bail!("Cannot read type '{:?}' as matrix data", ty),
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

macro_rules! impl_try_from_for_scalar {
    ($($from:ident, $to:ident), *) => {
        $(
            impl TryFrom<Data> for $to {
                type Error = anyhow::Error;
                fn try_from(data: Data) -> Result<Self> {
                    match data {
                        Data::Scalar(DynScalar::$from(data)) => Ok(data),
                        _ => bail!("Cannot convert data to $to"),
                    }
                }
            }

            impl<D: Dimension> TryFrom<Data> for Array<$to, D> {
                type Error = anyhow::Error;
                fn try_from(v: Data) -> Result<Self> {
                    match v {
                        Data::ArrayData(data) => data.try_into(),
                        _ => bail!("Cannot convert data to $to Array"),
                    }
                }
            }
        )*
    };
}

impl_try_from_for_scalar!(
    I8, i8,
    I16, i16,
    I32, i32,
    I64, i64,
    U8, u8,
    U16, u16,
    U32, u32,
    U64, u64,
    F32, f32,
    F64, f64,
    Bool, bool,
    String, String
);

impl TryFrom<Data> for DataFrame {
    type Error = anyhow::Error;

    fn try_from(v: Data) -> Result<Self> {
        match v {
            Data::DataFrame(data) => Ok(data),
            _ => bail!("Cannot convert data to DataFrame"),
        }
    }
}

impl TryFrom<Data> for Mapping {
    type Error = anyhow::Error;

    fn try_from(v: Data) -> Result<Self> {
        match v {
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