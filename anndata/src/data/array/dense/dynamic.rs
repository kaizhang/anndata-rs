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

            impl Readable for $from {
                fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
                    let dataset = container.as_dataset()?;
                    match dataset.dtype()? {
                        ScalarType::$to => Ok(dataset.read_scalar()?),
                        _ => bail!("Cannot read $from"),
                    }
                }
            }

            impl Element for $from {
                fn data_type(&self) -> DataType {
                    DataType::Scalar(ScalarType::$to)
                }

                fn metadata(&self) -> MetaData {
                    let encoding_type = if $from::DTYPE == ScalarType::String {
                        "string"
                    } else {
                        "numeric-scalar"
                    };
                    MetaData::new(encoding_type, "0.2.0", None)
                }
            }

            impl Writable for $from {
                fn write<B: Backend, G: GroupOp<B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
                    let dataset = location.new_scalar_dataset(name, self)?;
                    let mut container = DataContainer::Dataset(dataset);
                    self.metadata().save_metadata(&mut container)?;
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

impl Element for DynScalar {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_map_fun!(self, DynScalar, data_type)
    }

    fn metadata(&self) -> MetaData {
        crate::macros::dyn_map_fun!(self, DynScalar, metadata)
    }
}

impl Writable for DynScalar {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_map_fun!(self, DynScalar, write, location, name)
    }
}

impl Readable for DynScalar {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let dataset = container.as_dataset()?;

        macro_rules! fun {
            ($variant:ident) => {
                DynScalar::$variant(dataset.read_scalar()?)
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
    ($($variant:ident, $scalar_ty:ident),*) => {
        $(
            paste! {
                pub fn [<as_ $scalar_ty:lower>](&self) -> Result<&ArrayD<$scalar_ty>> {
                    match self {
                        DynArray::$variant(x) => Ok(x),
                        v => bail!("Cannot convert {} to {}", v.data_type(), stringify!($scalar_ty)),
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

macro_rules! impl_dynarray_traits{
    ($($scalar_ty:ty, $ident:ident),*) => {
        $(
            impl<D: Dimension> From<Array<$scalar_ty, D>> for DynArray {
                fn from(data: Array<$scalar_ty, D>) -> Self {
                    DynArray::$ident(data.into_dyn())
                }
            }

            impl<D: Dimension> TryFrom<DynArray> for Array<$scalar_ty, D> {
                type Error = anyhow::Error;
                fn try_from(arr: DynArray) -> Result<Self, Self::Error> {
                    match arr {
                        DynArray::$ident(x) => Ok(x.into_dimensionality::<D>()?),
                        v => bail!("Cannot convert {} to {}", v.data_type(), stringify!($scalar_ty)),
                    }
                }
            }
        )*
    };
}

impl_dynarray_traits!(
    i8, I8, i16, I16, i32, I32, i64, I64, u8, U8, u16, U16, u32, U32, u64, U64, f32, F32, f64, F64,
    bool, Bool, String, String
);

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

impl Element for DynArray {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_map_fun!(self, DynArray, data_type)
    }

    fn metadata(&self) -> MetaData {
        crate::macros::dyn_map_fun!(self, DynArray, metadata)
    }
}

impl Writable for DynArray {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_map_fun!(self, Self, write, location, name)
    }
}

impl Readable for DynArray {
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

impl Indexable for DynArray {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        macro_rules! fun {
            ($variant:ident, $exp:expr) => {
                $exp.get(index).map(|x| x.clone().into())
            };
        }
        crate::macros::dyn_map!(self, DynArray, fun)
    }
}

impl Selectable for DynArray {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        macro_rules! fun {
            ($variant:ident, $exp:expr) => {
                Selectable::select($exp, info).into()
            };
        }
        crate::macros::dyn_map!(self, DynArray, fun)
    }
}

impl Stackable for DynArray {
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            DynArray::U8(_) => {
                ArrayD::<u8>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::U16(_) => {
                ArrayD::<u16>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::U32(_) => {
                ArrayD::<u32>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::U64(_) => {
                ArrayD::<u64>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::I8(_) => {
                ArrayD::<i8>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::I16(_) => {
                ArrayD::<i16>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::I32(_) => {
                ArrayD::<i32>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::I64(_) => {
                ArrayD::<i64>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::F32(_) => {
                ArrayD::<f32>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::F64(_) => {
                ArrayD::<f64>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::Bool(_) => {
                ArrayD::<bool>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::String(_) => {
                ArrayD::<String>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
        }
    }
}

impl WritableArray for DynArray {}
impl ReadableArray for DynArray {
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

/// `ArrayConvert` trait for converting dynamic arrays to concrete arrays.
/// The `try_convert` method performs the conversion and returns the result.
pub trait ArrayConvert<T> {
    fn try_convert(self) -> Result<T>;
}

impl<D: Dimension> ArrayConvert<Array<i8, D>> for DynArray {
    fn try_convert(self) -> Result<Array<i8, D>> {
        match self {
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

impl<D: Dimension> ArrayConvert<Array<i16, D>> for DynArray {
    fn try_convert(self) -> Result<Array<i16, D>> {
        match self {
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

impl<D: Dimension> ArrayConvert<Array<i32, D>> for DynArray {
    fn try_convert(self) -> Result<Array<i32, D>> {
        match self {
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

impl<D: Dimension> ArrayConvert<Array<i64, D>> for DynArray {
    fn try_convert(self) -> Result<Array<i64, D>> {
        match self {
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

impl<D: Dimension> ArrayConvert<Array<u8, D>> for DynArray {
    fn try_convert(self) -> Result<Array<u8, D>> {
        match self {
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

impl<D: Dimension> ArrayConvert<Array<u16, D>> for DynArray {
    fn try_convert(self) -> Result<Array<u16, D>> {
        match self {
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

impl<D: Dimension> ArrayConvert<Array<u32, D>> for DynArray {
    fn try_convert(self) -> Result<Array<u32, D>> {
        match self {
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

impl<D: Dimension> ArrayConvert<Array<u64, D>> for DynArray {
    fn try_convert(self) -> Result<Array<u64, D>> {
        match self {
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

impl<D: Dimension> ArrayConvert<Array<usize, D>> for DynArray {
    fn try_convert(self) -> Result<Array<usize, D>> {
        match self {
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

impl<D: Dimension> ArrayConvert<Array<f32, D>> for DynArray {
    fn try_convert(self) -> Result<Array<f32, D>> {
        match self {
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

impl<D: Dimension> ArrayConvert<Array<f64, D>> for DynArray {
    fn try_convert(self) -> Result<Array<f64, D>> {
        match self {
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

impl<D: Dimension> ArrayConvert<Array<bool, D>> for DynArray {
    fn try_convert(self) -> Result<Array<bool, D>> {
        match self {
            DynArray::Bool(data) => Ok(data.into_dimensionality()?),
            _ => bail!("Cannot convert to bool Array"),
        }
    }
}

impl<D: Dimension> ArrayConvert<Array<String, D>> for DynArray {
    fn try_convert(self) -> Result<Array<String, D>> {
        match self {
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