use crate::backend::*;
use crate::data::ArrayConvert;
use crate::data::{
    data_traits::*,
    slice::{SelectInfoElem, Shape},
};

use anyhow::{bail, Result};
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use num::FromPrimitive;

#[derive(Debug, Clone, PartialEq)]
pub enum DynCsrMatrix {
    I8(CsrMatrix<i8>),
    I16(CsrMatrix<i16>),
    I32(CsrMatrix<i32>),
    I64(CsrMatrix<i64>),
    U8(CsrMatrix<u8>),
    U16(CsrMatrix<u16>),
    U32(CsrMatrix<u32>),
    U64(CsrMatrix<u64>),
    F32(CsrMatrix<f32>),
    F64(CsrMatrix<f64>),
    Bool(CsrMatrix<bool>),
    String(CsrMatrix<String>),
}

macro_rules! impl_dyncsr_traits {
    ($($scalar_ty:ty, $variant:ident),*) => {
        $(
            impl From<CsrMatrix<$scalar_ty>> for DynCsrMatrix {
                fn from(data: CsrMatrix<$scalar_ty>) -> Self {
                    DynCsrMatrix::$variant(data)
                }
            }
            impl TryFrom<DynCsrMatrix> for CsrMatrix<$scalar_ty> {
                type Error = anyhow::Error;
                fn try_from(data: DynCsrMatrix) -> Result<Self> {
                    match data {
                        DynCsrMatrix::$variant(data) => Ok(data),
                        _ => bail!(
                            "Cannot convert {} to {} CsrMatrix",
                            data.data_type(),
                            stringify!($scalar_ty)
                        ),
                    }
                }
            }
        )*
    };
}

impl_dyncsr_traits!(
    i8, I8, i16, I16, i32, I32, i64, I64, u8, U8, u16, U16, u32, U32, u64, U64, f32, F32, f64, F64,
    bool, Bool, String, String
);

impl Element for DynCsrMatrix {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_map_fun!(self, DynCsrMatrix, data_type)
    }

    fn metadata(&self) -> MetaData {
        crate::macros::dyn_map_fun!(self, DynCsrMatrix, metadata)
    }
}

impl Writable for DynCsrMatrix {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_map_fun!(self, DynCsrMatrix, write, location, name)
    }
}

impl Readable for DynCsrMatrix {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container {
            DataContainer::Group(group) => {
                macro_rules! fun {
                    ($variant:ident) => {
                        CsrMatrix::<$variant>::read(container).map(Into::into)
                    };
                }
                crate::macros::dyn_match!(group.open_dataset("data")?.dtype()?, ScalarType, fun)
            }
            _ => bail!("cannot read csr matrix from non-group container"),
        }
    }
}

impl HasShape for DynCsrMatrix {
    fn shape(&self) -> Shape {
        crate::macros::dyn_map_fun!(self, DynCsrMatrix, shape)
    }
}

impl Selectable for DynCsrMatrix {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        macro_rules! fun {
            ($variant:ident, $data:expr) => {
                $data.select(info).into()
            };
        }
        crate::macros::dyn_map!(self, DynCsrMatrix, fun)
    }
}

impl Stackable for DynCsrMatrix {
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            DynCsrMatrix::U8(_) => Ok(DynCsrMatrix::U8(CsrMatrix::<u8>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::U16(_) => Ok(DynCsrMatrix::U16(CsrMatrix::<u16>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::U32(_) => Ok(DynCsrMatrix::U32(CsrMatrix::<u32>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::U64(_) => Ok(DynCsrMatrix::U64(CsrMatrix::<u64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::I8(_) => Ok(DynCsrMatrix::I8(CsrMatrix::<i8>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::I16(_) => Ok(DynCsrMatrix::I16(CsrMatrix::<i16>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::I32(_) => Ok(DynCsrMatrix::I32(CsrMatrix::<i32>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::I64(_) => Ok(DynCsrMatrix::I64(CsrMatrix::<i64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::F32(_) => Ok(DynCsrMatrix::F32(CsrMatrix::<f32>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::F64(_) => Ok(DynCsrMatrix::F64(CsrMatrix::<f64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::Bool(_) => Ok(DynCsrMatrix::Bool(CsrMatrix::<bool>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrMatrix::String(_) => Ok(DynCsrMatrix::String(CsrMatrix::<String>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
        }
    }
}

impl WritableArray for DynCsrMatrix {}
impl ReadableArray for DynCsrMatrix {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into_iter()
            .collect())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        if let DataType::CsrMatrix(ty) = container.encoding_type()? {
            macro_rules! fun {
                ($variant:ident) => {
                    CsrMatrix::<$variant>::read_select(container, info)?.into()
                };
            }
            Ok(crate::macros::dyn_match!(ty, ScalarType, fun))
        } else {
            bail!("the container does not contain a csr matrix");
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DynCscMatrix {
    I8(CscMatrix<i8>),
    I16(CscMatrix<i16>),
    I32(CscMatrix<i32>),
    I64(CscMatrix<i64>),
    U8(CscMatrix<u8>),
    U16(CscMatrix<u16>),
    U32(CscMatrix<u32>),
    U64(CscMatrix<u64>),
    F32(CscMatrix<f32>),
    F64(CscMatrix<f64>),
    Bool(CscMatrix<bool>),
    String(CscMatrix<String>),
}

macro_rules! impl_dyncsc_traits {
    ($($from_type:ty, $to_type:ident),*) => {
        $(
            impl From<CscMatrix<$from_type>> for DynCscMatrix {
                fn from(data: CscMatrix<$from_type>) -> Self {
                    DynCscMatrix::$to_type(data)
                }
            }
            impl TryFrom<DynCscMatrix> for CscMatrix<$from_type> {
                type Error = anyhow::Error;
                fn try_from(data: DynCscMatrix) -> Result<Self> {
                    match data {
                        DynCscMatrix::$to_type(data) => Ok(data),
                        _ => bail!(
                            "Cannot convert {:?} to {} CscMatrix",
                            data.data_type(),
                            stringify!($from_type)
                        ),
                    }
                }
            }
        )*
    };
}

impl_dyncsc_traits!(
    i8, I8, i16, I16, i32, I32, i64, I64, u8, U8, u16, U16, u32, U32, u64, U64, f32, F32, f64, F64,
    bool, Bool, String, String
);

impl Element for DynCscMatrix {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_map_fun!(self, DynCscMatrix, data_type)
    }

    fn metadata(&self) -> MetaData {
        crate::macros::dyn_map_fun!(self, DynCscMatrix, metadata)
    }
}

impl Writable for DynCscMatrix {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_map_fun!(self, DynCscMatrix, write, location, name)
    }
}

impl Readable for DynCscMatrix {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container {
            DataContainer::Group(group) => {
                macro_rules! fun {
                    ($variant:ident) => {
                        CscMatrix::<$variant>::read(container).map(Into::into)
                    };
                }
                crate::macros::dyn_match!(group.open_dataset("data")?.dtype()?, ScalarType, fun)
            }
            _ => bail!("cannot read csc matrix from non-group container"),
        }
    }
}

impl HasShape for DynCscMatrix {
    fn shape(&self) -> Shape {
        crate::macros::dyn_map_fun!(self, DynCscMatrix, shape)
    }
}

impl Selectable for DynCscMatrix {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        macro_rules! select {
            ($variant:ident, $data:expr) => {
                $data.select(info).into()
            };
        }
        crate::macros::dyn_map!(self, DynCscMatrix, select)
    }
}

impl WritableArray for DynCscMatrix {}
impl ReadableArray for DynCscMatrix {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into_iter()
            .collect())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        if let DataType::CscMatrix(ty) = container.encoding_type()? {
            macro_rules! fun {
                ($variant:ident) => {
                    CscMatrix::<$variant>::read_select(container, info).map(Into::into)
                };
            }
            crate::macros::dyn_match!(ty, ScalarType, fun)
        } else {
            bail!("the container does not contain a csc matrix");
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// ArrayConvert implementations
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_arrayconvert {
    ($($ty:ident, $fun:expr),*) => {
        $(paste::paste! {

            impl ArrayConvert<$ty<u32>> for [<Dyn $ty>] {
                fn try_convert(self) -> Result<$ty<u32>> {
                    match self {
                        [<Dyn $ty>]::U32(data) => Ok(data),
                        [<Dyn $ty>]::I8(data) => $fun(data, |x| Ok(x.try_into()?)),
                        [<Dyn $ty>]::I16(data) => $fun(data, |x| Ok(x.try_into()?)),
                        [<Dyn $ty>]::I32(data) => $fun(data, |x| Ok(x.try_into()?)),
                        [<Dyn $ty>]::I64(data) => $fun(data, |x| Ok(x.try_into()?)),
                        [<Dyn $ty>]::U8(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::U16(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::U64(data) => $fun(data, |x| Ok(x.try_into()?)),
                        [<Dyn $ty>]::Bool(data) => $fun(data, |x| Ok(x.into())),
                        v => bail!("Cannot convert {} to {}<u32>", v.data_type(), stringify!($ty)),
                    }
                }
            }

            impl ArrayConvert<$ty<f32>> for [<Dyn $ty>] {
                fn try_convert(self) -> Result<$ty<f32>> {
                    match self {
                        [<Dyn $ty>]::F32(data) => Ok(data),
                        [<Dyn $ty>]::I8(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::I16(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::I32(data) => $fun(data, |x| Ok(f32::from_i32(x).unwrap())),
                        [<Dyn $ty>]::I64(data) => $fun(data, |x| Ok(f32::from_i64(x).unwrap())),
                        [<Dyn $ty>]::U8(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::U16(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::U32(data) => $fun(data, |x| Ok(f32::from_u32(x).unwrap())),
                        [<Dyn $ty>]::U64(data) => $fun(data, |x| Ok(f32::from_u64(x).unwrap())),
                        [<Dyn $ty>]::F64(data) => $fun(data, |x| Ok(f32::from_f64(x).unwrap())),
                        [<Dyn $ty>]::Bool(data) => $fun(data, |x| Ok(x.into())),
                        v => bail!("Cannot convert {} to {}<f32>", v.data_type(), stringify!($ty)),
                    }
                }
            }

            impl ArrayConvert<$ty<f64>> for [<Dyn $ty>] {
                fn try_convert(self) -> Result<$ty<f64>> {
                    match self {
                        [<Dyn $ty>]::F64(data) => Ok(data),
                        [<Dyn $ty>]::I8(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::I16(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::I32(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::I64(data) => $fun(data, |x| Ok(f64::from_i64(x).unwrap())),
                        [<Dyn $ty>]::U8(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::U16(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::U32(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::U64(data) => $fun(data, |x| Ok(f64::from_u64(x).unwrap())),
                        [<Dyn $ty>]::F32(data) => $fun(data, |x| Ok(x.into())),
                        [<Dyn $ty>]::Bool(data) => $fun(data, |x| Ok(x.into())),
                        v => bail!("Cannot convert {} to {}<f64>", v.data_type(), stringify!($ty)),
                    }
                }
            }

        })*
    };
}

impl_arrayconvert!(CsrMatrix, convert_csr_with, CscMatrix, convert_csc_with);

fn convert_csr_with<T, U, F>(csr: CsrMatrix<T>, f: F) -> Result<CsrMatrix<U>>
where
    F: Fn(T) -> Result<U>,
{
    let (pattern, values) = csr.into_pattern_and_values();
    let out = CsrMatrix::try_from_pattern_and_values(
        pattern,
        values.into_iter().map(|x| f(x)).collect::<Result<_, _>>()?,
    )
    .unwrap();
    Ok(out)
}

fn convert_csc_with<T, U, F>(csc: CscMatrix<T>, f: F) -> Result<CscMatrix<U>>
where
    F: Fn(T) -> Result<U>,
{
    let (pattern, values) = csc.into_pattern_and_values();
    let out = CscMatrix::try_from_pattern_and_values(
        pattern,
        values.into_iter().map(|x| f(x)).collect::<Result<_, _>>()?,
    )
    .unwrap();
    Ok(out)
}
