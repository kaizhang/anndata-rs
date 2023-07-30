mod ndarray;
pub mod slice;
pub mod dataframe;
mod utils;
mod sparse;
mod chunks;

pub use self::ndarray::{CategoricalArray, DynArray};
pub use slice::{BoundedSelectInfo, BoundedSelectInfoElem, SelectInfo, SelectInfoElem, Shape};
pub use sparse::{DynCsrMatrix, DynCscMatrix};
pub use dataframe::DataFrameIndex;
pub use utils::{concat_array_data, from_csr_rows};
pub use chunks::ArrayChunk;

use crate::backend::*;
use crate::data::{data_traits::*, scalar::DynScalar, DataType};

use polars::prelude::DataFrame;
use ::ndarray::{Array, RemoveAxis};
use anyhow::{bail, Result};
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::csc::CscMatrix;


#[derive(Debug, Clone, PartialEq)]
pub enum ArrayData {
    Array(DynArray),
    CsrMatrix(DynCsrMatrix),
    CscMatrix(DynCscMatrix),
    DataFrame(DataFrame),
}

impl<T: Clone + Into<ArrayData>> From<&T> for ArrayData {
    fn from(data: &T) -> Self {
        data.clone().into()
    }
}

impl From<DataFrame> for ArrayData {
    fn from(data: DataFrame) -> Self {
        ArrayData::DataFrame(data)
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
impl From<DynCscMatrix> for ArrayData {
    fn from(data: DynCscMatrix) -> Self {
        ArrayData::CscMatrix(data)
    }
}

impl TryFrom<ArrayData> for DynArray {
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        match value {
            ArrayData::Array(data) => Ok(data),
            _ => bail!("Cannot convert {:?} to DynArray", value),
        }
    }
}

impl TryFrom<ArrayData> for DynCsrMatrix {
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        match value {
            ArrayData::CsrMatrix(data) => Ok(data),
            _ => bail!("Cannot convert {:?} to DynCsrMatrix", value),
        }
    }
}


impl TryFrom<ArrayData> for DynCscMatrix {
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        match value {
            ArrayData::CscMatrix(data) => Ok(data),
            _ => bail!("Cannot convert {:?} to DynCscMatrix", value),
        }
    }
}

impl TryFrom<ArrayData> for DataFrame {
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        match value {
            ArrayData::DataFrame(data) => Ok(data),
            _ => bail!("Cannot convert {:?} to DataFrame", value),
        }
    }
}

/// macro for implementing From trait for Data from a list of types
macro_rules! impl_into_array_data {
    ($($ty:ty),*) => {
        $(
            impl<D: RemoveAxis> From<Array<$ty, D>> for ArrayData {
                fn from(data: Array<$ty, D>) -> Self {
                    ArrayData::Array(data.into_dyn().into())
                }
            }
            impl From<CsrMatrix<$ty>> for ArrayData {
                fn from(data: CsrMatrix<$ty>) -> Self {
                    ArrayData::CsrMatrix(data.into())
                }
            }
            impl From<CscMatrix<$ty>> for ArrayData {
                fn from(data: CscMatrix<$ty>) -> Self {
                    ArrayData::CscMatrix(data.into())
                }
            }
            impl<D: RemoveAxis> TryFrom<ArrayData> for Array<$ty, D> {
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
            impl TryFrom<ArrayData> for CscMatrix<$ty> {
                type Error = anyhow::Error;
                fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
                    match value {
                        ArrayData::CscMatrix(data) => data.try_into(),
                        _ => bail!("Cannot convert {:?} to $ty CsrMatrix", value),
                    }
                }
            }
        )*
    };
}

impl_into_array_data!(i8, i16, i32, i64, u8, u16, u32, u64, usize, f32, f64, bool, String);

impl WriteData for ArrayData {
    fn data_type(&self) -> DataType {
        match self {
            ArrayData::Array(data) => data.data_type(),
            ArrayData::CsrMatrix(data) => data.data_type(),
            ArrayData::CscMatrix(data) => data.data_type(),
            ArrayData::DataFrame(data) => data.data_type(),
        }
    }
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        match self {
            ArrayData::Array(data) => data.write(location, name),
            ArrayData::CsrMatrix(data) => data.write(location, name),
            ArrayData::CscMatrix(data) => data.write(location, name),
            ArrayData::DataFrame(data) => data.write(location, name),
        }
    }
}

impl ReadData for ArrayData {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) => {
                DynArray::read(container).map(ArrayData::Array)
            }
            DataType::CsrMatrix(_) => DynCsrMatrix::read(container).map(ArrayData::CsrMatrix),
            DataType::CscMatrix(_) => DynCscMatrix::read(container).map(ArrayData::CscMatrix),
            DataType::DataFrame => DataFrame::read(container).map(ArrayData::DataFrame),
            ty => bail!("Cannot read type '{:?}' as matrix data", ty),
        }
    }
}

impl HasShape for ArrayData {
    fn shape(&self) -> Shape {
        match self {
            ArrayData::Array(data) => data.shape(),
            ArrayData::CsrMatrix(data) => data.shape(),
            ArrayData::CscMatrix(data) => data.shape(),
            ArrayData::DataFrame(data) => HasShape::shape(data),
        }
    }
}

impl ArrayOp for ArrayData {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        match self {
            ArrayData::Array(data) => data.get(index),
            ArrayData::CsrMatrix(data) => data.get(index),
            ArrayData::CscMatrix(data) => data.get(index),
            ArrayData::DataFrame(data) => ArrayOp::get(data, index),
        }
    }

    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        match self {
            ArrayData::Array(data) => data.select(info).into(),
            ArrayData::CsrMatrix(data) => data.select(info).into(),
            ArrayData::CscMatrix(data) => data.select(info).into(),
            ArrayData::DataFrame(data) => ArrayOp::select(data,info).into(),
        }
    }

    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            ArrayData::Array(_) => DynArray::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            ArrayData::CsrMatrix(_) => DynCsrMatrix::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            ArrayData::CscMatrix(_) => DynCscMatrix::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            ArrayData::DataFrame(_) => <DataFrame as ArrayOp>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
        }
    }
}

impl ReadArrayData for ArrayData {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) => DynArray::get_shape(container),
            DataType::CsrMatrix(_) => DynCsrMatrix::get_shape(container),
            DataType::CscMatrix(_) => DynCscMatrix::get_shape(container),
            DataType::DataFrame => DataFrame::get_shape(container),
            ty => bail!("Cannot read shape information from type '{}'", ty),
        }
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) =>
                DynArray::read_select(container, info).map(ArrayData::Array),
            DataType::CsrMatrix(_) =>
                DynCsrMatrix::read_select(container, info).map(ArrayData::CsrMatrix),
            DataType::CscMatrix(_) =>
                DynCscMatrix::read_select(container, info).map(ArrayData::CscMatrix),
            DataType::DataFrame =>
                DataFrame::read_select(container, info).map(ArrayData::DataFrame),
            ty => bail!("Cannot read type '{:?}' as matrix data", ty),
        }
    }
}
impl WriteArrayData for ArrayData {}

impl WriteArrayData for &ArrayData {}
