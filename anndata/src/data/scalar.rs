use crate::backend::*;
use crate::data::data_traits::*;

use anyhow::{Result, bail};

#[derive(Debug, Clone, PartialEq)]
pub enum DynScalar {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    Usize(usize),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
}

/// macro to implement `From` trait for `DynScalar`
macro_rules! impl_from_dynscalar {
    ($($from:ident, $to:ident),*) => {
        $(
            impl From<$from> for DynScalar {
                fn from(val: $from) -> Self {
                    DynScalar::$to(val)
                }
            }

            impl ReadData for $from {
                fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
                    let dataset = container.as_dataset()?;
                    match dataset.dtype()? {
                        ScalarType::$to => Ok(dataset.read_scalar()?),
                        _ => bail!("Cannot read $from"),
                    }
                }
            }

            impl WriteData for $from {
                fn data_type(&self) -> DataType {
                    DataType::Scalar(ScalarType::$to)
                }
                fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
                    let dataset = location.create_scalar_data(name, self)?;
                    let container = DataContainer::Dataset(dataset);
                    let encoding_type = if $from::DTYPE == ScalarType::String {
                        "string"
                    } else {
                        "numeric-scalar"
                    };
                    container.write_str_attr("encoding-type", encoding_type)?;
                    container.write_str_attr("encoding-version", "0.2.0")?;
                    Ok(container)
                }
            }
        )*
    };
}

impl_from_dynscalar!(
    i8, I8,
    i16, I16,
    i32, I32,
    i64, I64,
    u8, U8,
    u16, U16,
    u32, U32,
    u64, U64,
    usize, Usize,
    f32, F32,
    f64, F64,
    bool, Bool,
    String, String
);

impl WriteData for DynScalar {
    fn data_type(&self) -> DataType {
        match self {
            DynScalar::I8(_) => DataType::Scalar(ScalarType::I8),
            DynScalar::I16(_) => DataType::Scalar(ScalarType::I16),
            DynScalar::I32(_) => DataType::Scalar(ScalarType::I32),
            DynScalar::I64(_) => DataType::Scalar(ScalarType::I64),
            DynScalar::U8(_) => DataType::Scalar(ScalarType::U8),
            DynScalar::U16(_) => DataType::Scalar(ScalarType::U16),
            DynScalar::U32(_) => DataType::Scalar(ScalarType::U32),
            DynScalar::U64(_) => DataType::Scalar(ScalarType::U64),
            DynScalar::Usize(_) => DataType::Scalar(ScalarType::Usize),
            DynScalar::F32(_) => DataType::Scalar(ScalarType::F32),
            DynScalar::F64(_) => DataType::Scalar(ScalarType::F64),
            DynScalar::Bool(_) => DataType::Scalar(ScalarType::Bool),
            DynScalar::String(_) => DataType::Scalar(ScalarType::String),
        }
    }

    fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
        match self {
            DynScalar::I8(data) => data.write(location, name),
            DynScalar::I16(data) => data.write(location, name),
            DynScalar::I32(data) => data.write(location, name),
            DynScalar::I64(data) => data.write(location, name),
            DynScalar::U8(data) => data.write(location, name),
            DynScalar::U16(data) => data.write(location, name),
            DynScalar::U32(data) => data.write(location, name),
            DynScalar::U64(data) => data.write(location, name),
            DynScalar::Usize(data) => data.write(location, name),
            DynScalar::F32(data) => data.write(location, name),
            DynScalar::F64(data) => data.write(location, name),
            DynScalar::Bool(data) => data.write(location, name),
            DynScalar::String(data) => data.write(location, name),
        }
    }
}

impl ReadData for DynScalar {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let dataset = container.as_dataset()?;
        match dataset.dtype()? {
            ScalarType::I8 => Ok(DynScalar::I8(dataset.read_scalar()?)),
            ScalarType::I16 => Ok(DynScalar::I16(dataset.read_scalar()?)),
            ScalarType::I32 => Ok(DynScalar::I32(dataset.read_scalar()?)),
            ScalarType::I64 => Ok(DynScalar::I64(dataset.read_scalar()?)),
            ScalarType::U8 => Ok(DynScalar::U8(dataset.read_scalar()?)),
            ScalarType::U16 => Ok(DynScalar::U16(dataset.read_scalar()?)),
            ScalarType::U32 => Ok(DynScalar::U32(dataset.read_scalar()?)),
            ScalarType::U64 => Ok(DynScalar::U64(dataset.read_scalar()?)),
            ScalarType::Usize => Ok(DynScalar::Usize(dataset.read_scalar()?)),
            ScalarType::F32 => Ok(DynScalar::F32(dataset.read_scalar()?)),
            ScalarType::F64 => Ok(DynScalar::F64(dataset.read_scalar()?)),
            ScalarType::Bool => Ok(DynScalar::Bool(dataset.read_scalar()?)),
            ScalarType::String => Ok(DynScalar::String(dataset.read_scalar()?)),
        }
    }
}