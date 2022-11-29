mod array;
mod other;

pub use array::{Shape, SelectInfo, SelectInfoElem, ArrayOp, WriteArrayData, ReadArrayData, DynArray, DynCsrMatrix, select_all, SLICE_FULL};
pub use other::{WriteData, ReadData, DynScalar, Mapping, DataFrameIndex};
use crate::backend::{Backend, DataContainer, DataType};

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
    fn write<B: Backend>(&self, location: &B::Group, name: &str) -> Result<DataContainer<B>> {
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

impl ArrayOp for ArrayData {
    fn shape(&self) -> Shape {
        match self {
            ArrayData::Array(data) => data.shape(),
            ArrayData::CsrMatrix(data) => data.shape(),
        }
    }

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
    Matrix(ArrayData),
    Scalar(DynScalar),
    DataFrame(DataFrame),
    Mapping(Mapping),
}

impl<T: Into<ArrayData>> From<T> for Data {
    fn from(data: T) -> Self {
        Data::Matrix(data.into())
    }
}

impl From<DynScalar> for Data {
    fn from(data: DynScalar) -> Self {
        Data::Scalar(data)
    }
}

impl From<DataFrame> for Data {
    fn from(data: DataFrame) -> Self {
        Data::DataFrame(data)
    }
}

impl From<Mapping> for Data {
    fn from(data: Mapping) -> Self {
        Data::Mapping(data)
    }
}

macro_rules! impl_into_data {
    ($(($from_type:ty, $from_type_wrap:ident, $to_type:ident, $to_type_wrap:ident)),*) => {
        $(
            impl From<$from_type> for Data {
                fn from(data: $from_type) -> Self {
                    Data::$to_type_wrap($from_type_wrap::$to_type(data))
                }
            }
        )*
    };
}

impl_into_data!(
    (i8, DynScalar, I8, Scalar),
    (i16, DynScalar, I16, Scalar),
    (i32, DynScalar, I32, Scalar),
    (i64, DynScalar, I64, Scalar),
    (u8, DynScalar, U8, Scalar),
    (u16, DynScalar, U16, Scalar),
    (u32, DynScalar, U32, Scalar),
    (u64, DynScalar, U64, Scalar),
    (f32, DynScalar, F32, Scalar),
    (f64, DynScalar, F64, Scalar),
    (bool, DynScalar, Bool, Scalar),
    (String, DynScalar, String, Scalar)
);

impl WriteData for Data {
    fn write<B: Backend>(&self, location: &B::Group, name: &str) -> Result<DataContainer<B>> {
        match self {
            Data::Matrix(data) => data.write(location, name),
            Data::Scalar(data) => data.write(location, name),
            Data::DataFrame(data) => data.write(location, name),
            Data::Mapping(data) => data.write(location, name),
        }
    }
}

impl ReadData for Data {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) => DynArray::read(container).map(|x| x.into()),
            DataType::CsrMatrix(_) => DynCsrMatrix::read(container).map(|x| x.into()),
            DataType::CscMatrix(_) => todo!(),
            DataType::Scalar(_) => DynScalar::read(container).map(|x| x.into()),
            DataType::DataFrame => DataFrame::read(container).map(|x| x.into()),
            DataType::Mapping => Mapping::read(container).map(|x| x.into()),
        }
    }
}