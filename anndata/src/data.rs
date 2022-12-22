pub mod array;
pub mod data_traits;
pub mod mapping;
pub mod scalar;

pub use array::*;
pub use data_traits::*;
pub use mapping::*;
pub use scalar::*;

use crate::backend::{Backend, DataContainer, DataType, GroupOp};

use ::ndarray::{Array, Dimension};
use anyhow::{bail, Ok, Result};
use nalgebra_sparse::csr::CsrMatrix;
use polars::frame::DataFrame;

#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    ArrayData(ArrayData),
    Scalar(DynScalar),
    Mapping(Mapping),
}

/// Types that can be converted to Data
impl<T: Clone + Into<Data>> From<&T> for Data {
    fn from(data: &T) -> Self {
        data.clone().into()
    }
}

impl From<DataFrame> for Data {
    fn from(data: DataFrame) -> Self {
        Data::ArrayData(ArrayData::DataFrame(data))
    }
}

macro_rules! impl_into_data {
    ($from_type:ty, $to_type:ident) => {
        impl From<$from_type> for Data {
            fn from(data: $from_type) -> Self {
                Data::Scalar(DynScalar::$to_type(data))
            }
        }
        impl<D: Dimension> From<Array<$from_type, D>> for Data {
            fn from(data: Array<$from_type, D>) -> Self {
                Data::ArrayData(ArrayData::Array(DynArray::$to_type(data.into_dyn())))
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
    I8, i8, I16, i16, I32, i32, I64, i64, U8, u8, U16, u16, U32, u32, U64, u64, F32, f32, F64, f64,
    Bool, bool, String, String
);

impl TryFrom<Data> for DataFrame {
    type Error = anyhow::Error;

    fn try_from(value: Data) -> Result<Self, Self::Error> {
        match value {
            Data::ArrayData(data) => data.try_into(),
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
    fn data_type(&self) -> DataType {
        match self {
            Data::ArrayData(data) => data.data_type(),
            Data::Scalar(data) => data.data_type(),
            Data::Mapping(data) => data.data_type(),
        }
    }
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        match self {
            Data::ArrayData(data) => data.write(location, name),
            Data::Scalar(data) => data.write(location, name),
            Data::Mapping(data) => data.write(location, name),
        }
    }
}

impl ReadData for Data {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) => {
                DynArray::read(container).map(|x| ArrayData::from(x).into())
            }
            DataType::CsrMatrix(_) => {
                DynCsrMatrix::read(container).map(|x| ArrayData::from(x).into())
            }
            DataType::CscMatrix(_) => todo!(),
            DataType::DataFrame => DataFrame::read(container).map(|x| ArrayData::from(x).into()),
            DataType::Scalar(_) => DynScalar::read(container).map(|x| x.into()),
            DataType::Mapping => Mapping::read(container).map(|x| x.into()),
        }
    }
}
