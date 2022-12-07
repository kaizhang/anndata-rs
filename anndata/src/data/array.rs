//pub mod iterators;
mod ndarray;
pub mod slice;
mod sparse;

pub use self::ndarray::{CategoricalArray, DynArray};
pub use slice::{BoundedSelectInfo, BoundedSelectInfoElem, SelectInfo, SelectInfoElem, Shape};
pub use sparse::DynCsrMatrix;

use crate::backend::*;
use crate::data::{data_traits::*, scalar::DynScalar, DataType};

use ::ndarray::{Array, Dimension};
use anyhow::{bail, Result};
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

impl_into_array_data!(i8, i16, i32, i64, u8, u16, u32, u64, usize, f32, f64, bool, String);

impl WriteData for ArrayData {
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        match self {
            ArrayData::Array(data) => data.write(location, name),
            ArrayData::CsrMatrix(data) => data.write(location, name),
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
        Self: Sized,
    {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) => {
                DynArray::read_select(container, info).map(ArrayData::Array)
            }
            DataType::CsrMatrix(_) => {
                DynCsrMatrix::read_select(container, info).map(ArrayData::CsrMatrix)
            }
            ty => bail!("Cannot read type '{:?}' as matrix data", ty),
        }
    }
}
impl WriteArrayData for ArrayData {}
