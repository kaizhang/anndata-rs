use crate::{
    backend::*,
    data::{
        data_traits::*,
        slice::{SelectInfoElem, Shape},
    },
};

use anyhow::{bail, ensure, Result};
use ndarray::{arr0, Array, ArrayD, ArrayView, CowArray, Dimension, IxDyn};
use paste::paste;
use polars::series::Series;

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
                fn write<B: Backend, G: GroupOp<B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
                    let dataset = location.new_scalar_dataset(name, self)?;
                    let mut container = DataContainer::Dataset(dataset);
                    let encoding_type = if $from::DTYPE == ScalarType::String {
                        "string"
                    } else {
                        "numeric-scalar"
                    };
                    container.new_str_attr("encoding-type", encoding_type)?;
                    container.new_str_attr("encoding-version", "0.2.0")?;
                    Ok(container)
                }
            }
        )*
    };
}

impl_from_dynscalar!(
    i8, I8, i16, I16, i32, I32, i64, I64, u8, U8, u16, U16, u32, U32, u64, U64, f32, F32, f64, F64,
    bool, Bool, String, String
);

impl WriteData for DynScalar {
    fn data_type(&self) -> DataType {
        macro_rules! dtype {
            ($variant:ident, $exp:expr) => {
                DataType::Scalar(ScalarType::$variant)
            };
        }
        crate::macros::dyn_map!(self, DynScalar, dtype)
    }

    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_map_fun!(self, DynScalar, write, location, name)
    }
}

impl ReadData for DynScalar {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let dataset = container.as_dataset()?;

        macro_rules! fun {
            ($variant:ident) => {
                DynScalar::I8(dataset.read_scalar()?)
            };
        }

        Ok(crate::macros::dyn_match!(dataset.dtype()?, ScalarType, fun))
    }
}

/// A dynamic-typed array.
#[derive(Debug, Clone, PartialEq)]
pub enum DynArray {
    I8(ArrayD<i8>),
    I16(ArrayD<i16>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    U8(ArrayD<u8>),
    U16(ArrayD<u16>),
    U32(ArrayD<u32>),
    U64(ArrayD<u64>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    Bool(ArrayD<bool>),
    String(ArrayD<String>),
}

macro_rules! impl_dynarray_into_array{
    ($($from_type:ident, $to_type:ident),*) => {
        $(
            paste! {
                /// Extract the concrete array type from the dynamic array.
                /// This function does not perform any conversion, it only checks if 
                /// the underlying data type is exactly the same as the target type.
                /// To perform conversion, use the `.try_into()` method.
                pub fn [<into_ $to_type:lower>]<D: Dimension>(self) -> Result<Array<$to_type, D>> {
                    match self {
                        DynArray::$from_type(x) => Ok(x.into_dimensionality()?),
                        v => bail!("Cannot convert {} to {}", v.data_type(), stringify!($to_type)),
                    }
                }
            }
        )*
    };
}

impl DynArray {
    pub fn ndim(&self) -> usize {
        crate::macros::dyn_map_fun!(self, DynArray, ndim)
    }

    pub fn shape(&self) -> Shape {
        crate::macros::dyn_map_fun!(self, DynArray, shape)
            .to_vec()
            .into()
    }

    pub fn len(&self) -> usize {
        crate::macros::dyn_map_fun!(self, DynArray, len)
    }

    impl_dynarray_into_array!(
        I8, i8, I16, i16, I32, i32, I64, i64, U8, u8, U16, u16, U32, u32, U64, u64, F32, f32, F64,
        f64, Bool, bool, String, String
    );
}

macro_rules! impl_to_dynarray{
    ($($from_type:ty, $to_type:ident),*) => {
        $(
            impl<D: Dimension> From<Array<$from_type, D>> for DynArray {
                fn from(data: Array<$from_type, D>) -> Self {
                    DynArray::$to_type(data.into_dyn())
                }
            }
        )*
    };
}

impl_to_dynarray!(
    i8, I8, i16, I16, i32, I32, i64, I64, u8, U8, u16, U16, u32, U32, u64, U64, f32, F32, f64, F64,
    bool, Bool, String, String
);

impl<D: Dimension> TryFrom<DynArray> for Array<i8, D> {
    type Error = anyhow::Error;
    fn try_from(v: DynArray) -> Result<Self> {
        match v {
            DynArray::I8(data) => Ok(data.into_dimensionality()?),
            DynArray::I16(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I32(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U8(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U16(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U32(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::Bool(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            _ => bail!("Cannot convert to i8 Array"),
        }
    }
}

impl<D: Dimension> TryFrom<DynArray> for Array<i16, D> {
    type Error = anyhow::Error;
    fn try_from(v: DynArray) -> Result<Self> {
        match v {
            DynArray::I16(data) => Ok(data.into_dimensionality()?),
            DynArray::I8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::I32(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U16(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U32(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::Bool(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            _ => bail!("Cannot convert to i16 Array"),
        }
    }
}

impl<D: Dimension> TryFrom<DynArray> for Array<i32, D> {
    type Error = anyhow::Error;
    fn try_from(v: DynArray) -> Result<Self> {
        match v {
            DynArray::I32(data) => Ok(data.into_dimensionality()?),
            DynArray::I8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::I16(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::I64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U16(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U32(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::Bool(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            _ => bail!("Cannot convert to i32 Array"),
        }
    }
}

impl<D: Dimension> TryFrom<DynArray> for Array<i64, D> {
    type Error = anyhow::Error;
    fn try_from(v: DynArray) -> Result<Self> {
        match v {
            DynArray::I64(data) => Ok(data.into_dimensionality()?),
            DynArray::I8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::I16(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::I32(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U16(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U32(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::Bool(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            _ => bail!("Cannot convert to i64 Array"),
        }
    }
}

impl<D: Dimension> TryFrom<DynArray> for Array<u8, D> {
    type Error = anyhow::Error;
    fn try_from(v: DynArray) -> Result<Self> {
        match v {
            DynArray::U8(data) => Ok(data.into_dimensionality()?),
            DynArray::I8(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I16(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I32(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U16(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U32(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::Bool(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            _ => bail!("Cannot convert to u8 Array"),
        }
    }
}

impl<D: Dimension> TryFrom<DynArray> for Array<u16, D> {
    type Error = anyhow::Error;
    fn try_from(v: DynArray) -> Result<Self> {
        match v {
            DynArray::U16(data) => Ok(data.into_dimensionality()?),
            DynArray::I8(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I16(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I32(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U32(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::Bool(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            _ => bail!("Cannot convert to u16 Array"),
        }
    }
}

impl<D: Dimension> TryFrom<DynArray> for Array<u32, D> {
    type Error = anyhow::Error;
    fn try_from(v: DynArray) -> Result<Self> {
        match v {
            DynArray::U32(data) => Ok(data.into_dimensionality()?),
            DynArray::I8(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I16(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I32(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U16(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::Bool(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            _ => bail!("Cannot convert to u32 Array"),
        }
    }
}

impl<D: Dimension> TryFrom<DynArray> for Array<u64, D> {
    type Error = anyhow::Error;
    fn try_from(v: DynArray) -> Result<Self> {
        match v {
            DynArray::U64(data) => Ok(data.into_dimensionality()?),
            DynArray::I8(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I16(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I32(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U16(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U32(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::Bool(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            _ => bail!("Cannot convert to u64 Array"),
        }
    }
}

impl<D: Dimension> TryFrom<DynArray> for Array<usize, D> {
    type Error = anyhow::Error;
    fn try_from(v: DynArray) -> Result<Self> {
        match v {
            DynArray::I8(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I16(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I32(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::I64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U16(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U32(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::U64(data) => Ok(data.mapv(|x| x.try_into().unwrap()).into_dimensionality()?),
            DynArray::Bool(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            _ => bail!("Cannot convert to usize Array"),
        }
    }
}

impl<D: Dimension> TryFrom<DynArray> for Array<f32, D> {
    type Error = anyhow::Error;
    fn try_from(v: DynArray) -> Result<Self> {
        match v {
            DynArray::F32(data) => Ok(data.into_dimensionality()?),
            DynArray::I8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::I16(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U16(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::Bool(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            _ => bail!("Cannot convert to f32 Array"),
        }
    }
}

impl<D: Dimension> TryFrom<DynArray> for Array<f64, D> {
    type Error = anyhow::Error;
    fn try_from(v: DynArray) -> Result<Self> {
        match v {
            DynArray::F64(data) => Ok(data.into_dimensionality()?),
            DynArray::I8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::I16(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::I32(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U8(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U16(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::U32(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::F32(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            DynArray::Bool(data) => Ok(data.mapv(|x| x.into()).into_dimensionality()?),
            _ => bail!("Cannot convert to f64 Array"),
        }
    }
}

impl<D: Dimension> TryFrom<DynArray> for Array<bool, D> {
    type Error = anyhow::Error;
    fn try_from(v: DynArray) -> Result<Self> {
        match v {
            DynArray::Bool(data) => Ok(data.into_dimensionality()?),
            _ => bail!("Cannot convert to bool Array"),
        }
    }
}

impl<D: Dimension> TryFrom<DynArray> for Array<String, D> {
    type Error = anyhow::Error;
    fn try_from(v: DynArray) -> Result<Self> {
        match v {
            DynArray::String(data) => Ok(data.into_dimensionality()?),
            DynArray::I8(data) => Ok(data.mapv(|x| x.to_string()).into_dimensionality()?),
            DynArray::I16(data) => Ok(data.mapv(|x| x.to_string()).into_dimensionality()?),
            DynArray::I32(data) => Ok(data.mapv(|x| x.to_string()).into_dimensionality()?),
            DynArray::I64(data) => Ok(data.mapv(|x| x.to_string()).into_dimensionality()?),
            DynArray::U8(data) => Ok(data.mapv(|x| x.to_string()).into_dimensionality()?),
            DynArray::U16(data) => Ok(data.mapv(|x| x.to_string()).into_dimensionality()?),
            DynArray::U32(data) => Ok(data.mapv(|x| x.to_string()).into_dimensionality()?),
            DynArray::U64(data) => Ok(data.mapv(|x| x.to_string()).into_dimensionality()?),
            DynArray::F32(data) => Ok(data.mapv(|x| x.to_string()).into_dimensionality()?),
            DynArray::F64(data) => Ok(data.mapv(|x| x.to_string()).into_dimensionality()?),
            DynArray::Bool(data) => Ok(data.mapv(|x| x.to_string()).into_dimensionality()?),
        }
    }
}

impl Into<Series> for DynArray {
    fn into(self) -> Series {
        match self {
            DynArray::I8(x) => x.iter().collect(),
            DynArray::I16(x) => x.iter().collect(),
            DynArray::I32(x) => x.iter().collect(),
            DynArray::I64(x) => x.iter().collect(),
            DynArray::U8(x) => x.iter().collect(),
            DynArray::U16(x) => x.iter().collect(),
            DynArray::U32(x) => x.iter().collect(),
            DynArray::U64(x) => x.iter().collect(),
            DynArray::F32(x) => x.iter().collect(),
            DynArray::F64(x) => x.iter().collect(),
            DynArray::Bool(x) => x.iter().collect(),
            DynArray::String(x) => x.iter().map(|x| x.as_str()).collect(),
        }
    }
}

impl WriteData for DynArray {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_map_fun!(self, DynArray, data_type)
    }

    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_map_fun!(self, Self, write, location, name)
    }
}

impl ReadData for DynArray {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        container.as_dataset()?.read_dyn_array()
    }
}

impl HasShape for DynArray {
    fn shape(&self) -> Shape {
        crate::macros::dyn_map_fun!(self, DynArray, shape)
            .to_vec()
            .into()
    }
}

impl ArrayOp for DynArray {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        macro_rules! fun {
            ($variant:ident, $exp:expr) => {
                $exp.get(index).map(|x| x.clone().into())
            };
        }

        crate::macros::dyn_map!(self, DynArray, fun)
    }

    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        macro_rules! fun {
            ($variant:ident, $exp:expr) => {
                ArrayOp::select($exp, info).into()
            };
        }
        crate::macros::dyn_map!(self, DynArray, fun)
    }

    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            DynArray::U8(_) => {
                ArrayD::vstack(iter.map(|x| x.into_u8().unwrap())).map(|x| x.into())
            }
            DynArray::U16(_) => {
                ArrayD::vstack(iter.map(|x| x.into_u16().unwrap())).map(|x| x.into())
            }
            DynArray::U32(_) => {
                ArrayD::vstack(iter.map(|x| x.into_u32().unwrap())).map(|x| x.into())
            }
            DynArray::U64(_) => {
                ArrayD::vstack(iter.map(|x| x.into_u64().unwrap())).map(|x| x.into())
            }
            DynArray::I8(_) => {
                ArrayD::vstack(iter.map(|x| x.into_i8().unwrap())).map(|x| x.into())
            }
            DynArray::I16(_) => {
                ArrayD::vstack(iter.map(|x| x.into_i16().unwrap())).map(|x| x.into())
            }
            DynArray::I32(_) => {
                ArrayD::vstack(iter.map(|x| x.into_i32().unwrap())).map(|x| x.into())
            }
            DynArray::I64(_) => {
                ArrayD::vstack(iter.map(|x| x.into_i64().unwrap())).map(|x| x.into())
            }
            DynArray::F32(_) => {
                ArrayD::vstack(iter.map(|x| x.into_f32().unwrap())).map(|x| x.into())
            }
            DynArray::F64(_) => {
                ArrayD::vstack(iter.map(|x| x.into_f64().unwrap())).map(|x| x.into())
            }
            DynArray::Bool(_) => {
                ArrayD::vstack(iter.map(|x| x.into_bool().unwrap())).map(|x| x.into())
            }
            DynArray::String(_) => {
                ArrayD::vstack(iter.map(|x| x.into_string().unwrap())).map(|x| x.into())
            }
        }
    }
}

impl WriteArrayData for DynArray {}
impl ReadArrayData for DynArray {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container.as_dataset()?.shape().into())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        container.as_dataset()?.read_dyn_array_slice(info)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DynCowArray<'a> {
    I8(CowArray<'a, i8, IxDyn>),
    I16(CowArray<'a, i16, IxDyn>),
    I32(CowArray<'a, i32, IxDyn>),
    I64(CowArray<'a, i64, IxDyn>),
    U8(CowArray<'a, u8, IxDyn>),
    U16(CowArray<'a, u16, IxDyn>),
    U32(CowArray<'a, u32, IxDyn>),
    U64(CowArray<'a, u64, IxDyn>),
    F32(CowArray<'a, f32, IxDyn>),
    F64(CowArray<'a, f64, IxDyn>),
    Bool(CowArray<'a, bool, IxDyn>),
    String(CowArray<'a, String, IxDyn>),
}

impl From<DynScalar> for DynCowArray<'_> {
    fn from(scalar: DynScalar) -> Self {
        macro_rules! fun {
            ($variant:ident, $exp:expr) => {
                DynCowArray::$variant(arr0($exp).into_dyn().into())
            };
        }
        crate::macros::dyn_map!(scalar, DynScalar, fun)
    }
}

impl DynCowArray<'_> {
    pub fn ndim(&self) -> usize {
        crate::macros::dyn_map_fun!(self, DynCowArray, ndim)
    }

    pub fn shape(&self) -> Shape {
        crate::macros::dyn_map_fun!(self, DynCowArray, shape)
            .to_vec()
            .into()
    }

    pub fn len(&self) -> usize {
        crate::macros::dyn_map_fun!(self, DynCowArray, len)
    }
}

macro_rules! impl_dyn_cowarray_convert {
    ($from_type:ty, $to_type:ident) => {
        impl<D: Dimension> From<Array<$from_type, D>> for DynCowArray<'_> {
            fn from(data: Array<$from_type, D>) -> Self {
                DynCowArray::$to_type(data.into_dyn().into())
            }
        }

        impl<'a, D: Dimension> From<ArrayView<'a, $from_type, D>> for DynCowArray<'a> {
            fn from(data: ArrayView<'a, $from_type, D>) -> Self {
                DynCowArray::$to_type(data.into_dyn().into())
            }
        }

        impl<'a, D: Dimension> From<CowArray<'a, $from_type, D>> for DynCowArray<'a> {
            fn from(data: CowArray<'a, $from_type, D>) -> Self {
                DynCowArray::$to_type(data.into_dyn())
            }
        }

        impl<D: Dimension> TryFrom<DynCowArray<'_>> for Array<$from_type, D> {
            type Error = anyhow::Error;
            fn try_from(v: DynCowArray) -> Result<Self, Self::Error> {
                match v {
                    DynCowArray::$to_type(data) => {
                        let arr: ArrayD<$from_type> = data.into_owned();
                        if let Some(n) = D::NDIM {
                            ensure!(
                                arr.ndim() == n,
                                format!("Dimension mismatch: {} (in) != {} (out)", arr.ndim(), n)
                            );
                        }
                        Ok(arr.into_dimensionality::<D>()?)
                    }
                    _ => bail!("Cannot convert to {} ArrayD", stringify!($from_type)),
                }
            }
        }
    };
}

impl_dyn_cowarray_convert!(i8, I8);
impl_dyn_cowarray_convert!(i16, I16);
impl_dyn_cowarray_convert!(i32, I32);
impl_dyn_cowarray_convert!(i64, I64);
impl_dyn_cowarray_convert!(u8, U8);
impl_dyn_cowarray_convert!(u16, U16);
impl_dyn_cowarray_convert!(u32, U32);
impl_dyn_cowarray_convert!(u64, U64);
impl_dyn_cowarray_convert!(f32, F32);
impl_dyn_cowarray_convert!(f64, F64);
impl_dyn_cowarray_convert!(bool, Bool);
impl_dyn_cowarray_convert!(String, String);