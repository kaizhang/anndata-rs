use crate::backend::*;
use crate::data::{
    array::DynScalar,
    data_traits::*,
    slice::{SelectInfoElem, Shape},
};

use anyhow::{bail, Context, Result};
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::Ix1;
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

macro_rules! impl_into_dyn_csr {
    ($from_type:ty, $to_type:ident) => {
        impl From<CsrMatrix<$from_type>> for DynCsrMatrix {
            fn from(data: CsrMatrix<$from_type>) -> Self {
                DynCsrMatrix::$to_type(data)
            }
        }
        impl TryFrom<DynCsrMatrix> for CsrMatrix<$from_type> {
            type Error = anyhow::Error;
            fn try_from(data: DynCsrMatrix) -> Result<Self> {
                match data {
                    DynCsrMatrix::$to_type(data) => Ok(data),
                    _ => bail!(
                        "Cannot convert {:?} to {} CsrMatrix",
                        data.data_type(),
                        stringify!($from_type)
                    ),
                }
            }
        }
    };
}

impl From<CsrMatrix<u32>> for DynCsrMatrix {
    fn from(data: CsrMatrix<u32>) -> Self {
        DynCsrMatrix::U32(data)
    }
}

impl TryFrom<DynCsrMatrix> for CsrMatrix<u32> {
    type Error = anyhow::Error;
    fn try_from(data: DynCsrMatrix) -> Result<Self> {
        match data {
            DynCsrMatrix::U32(data) => Ok(data),
            DynCsrMatrix::I8(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::I16(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::I32(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::I64(data) => Ok(from_i64_csr(data)?),
            DynCsrMatrix::U8(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::U16(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::U64(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::F32(_) => bail!("Cannot convert f32 to u32"),
            DynCsrMatrix::F64(_) => bail!("Cannot convert f64 to u32"),
            DynCsrMatrix::Bool(_) => bail!("Cannot convert bool to f64"),
            DynCsrMatrix::String(_) => bail!("Cannot convert string to f64"),
        }
    }
}

impl From<CsrMatrix<f64>> for DynCsrMatrix {
    fn from(data: CsrMatrix<f64>) -> Self {
        DynCsrMatrix::F64(data)
    }
}

impl TryFrom<DynCsrMatrix> for CsrMatrix<f64> {
    type Error = anyhow::Error;
    fn try_from(data: DynCsrMatrix) -> Result<Self> {
        match data {
            DynCsrMatrix::F64(data) => Ok(data),
            DynCsrMatrix::I8(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::I16(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::I32(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::I64(data) => Ok(from_i64_csr(data)?),
            DynCsrMatrix::U8(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::U16(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::U32(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::U64(_) => bail!("Cannot convert u64 to f64"),
            DynCsrMatrix::F32(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::Bool(_) => bail!("Cannot convert bool to f64"),
            DynCsrMatrix::String(_) => bail!("Cannot convert string to f64"),
        }
    }
}

impl_into_dyn_csr!(i8, I8);
impl_into_dyn_csr!(i16, I16);
impl_into_dyn_csr!(i32, I32);
impl_into_dyn_csr!(i64, I64);
impl_into_dyn_csr!(u8, U8);
impl_into_dyn_csr!(u16, U16);
impl_into_dyn_csr!(u64, U64);
impl_into_dyn_csr!(f32, F32);
impl_into_dyn_csr!(bool, Bool);
impl_into_dyn_csr!(String, String);

impl WriteData for DynCsrMatrix {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_map_fun!(self, DynCsrMatrix, data_type)
    }
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_map_fun!(self, DynCsrMatrix, write, location, name)
    }
}

impl ReadData for DynCsrMatrix {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container {
            DataContainer::Group(group) => {
                macro_rules! fun {
                    ($variant:ident) => {
                        CsrMatrix::read(container).map(DynCsrMatrix::$variant)
                    };
                }
                crate::macros::dyn_match!(group.open_dataset("data")?.dtype()?, ScalarType, fun)
            },
            _ => bail!("cannot read csr matrix from non-group container"),
        }
    }
}

impl HasShape for DynCsrMatrix {
    fn shape(&self) -> Shape {
        crate::macros::dyn_map_fun!(self, DynCsrMatrix, shape)
    }
}

impl ArrayOp for DynCsrMatrix {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        crate::macros::dyn_map_fun!(self, DynCsrMatrix, get, index)
    }

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

impl WriteArrayData for DynCsrMatrix {}
impl ReadArrayData for DynCsrMatrix {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_array_attr::<u64, Ix1>("shape")?
            .into_iter()
            .map(|x| x as usize)
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
                    CsrMatrix::read_select(container, info).map(DynCsrMatrix::$variant)
                };
            }
            crate::macros::dyn_match!(ty, ScalarType, fun)
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

macro_rules! impl_into_dyn_csc {
    ($from_type:ty, $to_type:ident) => {
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
    };
}

impl From<CscMatrix<u32>> for DynCscMatrix {
    fn from(data: CscMatrix<u32>) -> Self {
        DynCscMatrix::U32(data)
    }
}

impl TryFrom<DynCscMatrix> for CscMatrix<u32> {
    type Error = anyhow::Error;
    fn try_from(data: DynCscMatrix) -> Result<Self> {
        match data {
            DynCscMatrix::U32(data) => Ok(data),
            DynCscMatrix::I8(data) => Ok(cast_csc(data)?),
            DynCscMatrix::I16(data) => Ok(cast_csc(data)?),
            DynCscMatrix::I32(data) => Ok(cast_csc(data)?),
            DynCscMatrix::I64(data) => Ok(from_i64_csc(data)?),
            DynCscMatrix::U8(data) => Ok(cast_csc(data)?),
            DynCscMatrix::U16(data) => Ok(cast_csc(data)?),
            DynCscMatrix::U64(data) => Ok(cast_csc(data)?),
            DynCscMatrix::F32(_) => bail!("Cannot convert f32 to u32"),
            DynCscMatrix::F64(_) => bail!("Cannot convert f64 to u32"),
            DynCscMatrix::Bool(_) => bail!("Cannot convert bool to f64"),
            DynCscMatrix::String(_) => bail!("Cannot convert string to f64"),
        }
    }
}

impl From<CscMatrix<f64>> for DynCscMatrix {
    fn from(data: CscMatrix<f64>) -> Self {
        DynCscMatrix::F64(data)
    }
}

impl TryFrom<DynCscMatrix> for CscMatrix<f64> {
    type Error = anyhow::Error;
    fn try_from(data: DynCscMatrix) -> Result<Self> {
        match data {
            DynCscMatrix::F64(data) => Ok(data),
            DynCscMatrix::I8(data) => Ok(cast_csc(data)?),
            DynCscMatrix::I16(data) => Ok(cast_csc(data)?),
            DynCscMatrix::I32(data) => Ok(cast_csc(data)?),
            DynCscMatrix::I64(data) => Ok(from_i64_csc(data)?),
            DynCscMatrix::U8(data) => Ok(cast_csc(data)?),
            DynCscMatrix::U16(data) => Ok(cast_csc(data)?),
            DynCscMatrix::U32(data) => Ok(cast_csc(data)?),
            DynCscMatrix::U64(_) => bail!("Cannot convert u64 to f64"),
            DynCscMatrix::F32(data) => Ok(cast_csc(data)?),
            DynCscMatrix::Bool(_) => bail!("Cannot convert bool to f64"),
            DynCscMatrix::String(_) => bail!("Cannot convert string to f64"),
        }
    }
}

impl_into_dyn_csc!(i8, I8);
impl_into_dyn_csc!(i16, I16);
impl_into_dyn_csc!(i32, I32);
impl_into_dyn_csc!(i64, I64);
impl_into_dyn_csc!(u8, U8);
impl_into_dyn_csc!(u16, U16);
impl_into_dyn_csc!(u64, U64);
impl_into_dyn_csc!(f32, F32);
impl_into_dyn_csc!(bool, Bool);
impl_into_dyn_csc!(String, String);

impl WriteData for DynCscMatrix {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_map_fun!(self, DynCscMatrix, data_type)
    }

    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_map_fun!(self, DynCscMatrix, write, location, name)
    }
}

impl ReadData for DynCscMatrix {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container {
            DataContainer::Group(group) => {
                macro_rules! fun {
                    ($variant:ident) => {
                        CscMatrix::read(container).map(DynCscMatrix::$variant)
                    };
                }
                crate::macros::dyn_match!(group.open_dataset("data")?.dtype()?, ScalarType, fun)
            },
            _ => bail!("cannot read csc matrix from non-group container"),
        }
    }
}

impl HasShape for DynCscMatrix {
    fn shape(&self) -> Shape {
        crate::macros::dyn_map_fun!(self, DynCscMatrix, shape)
    }
}

impl ArrayOp for DynCscMatrix {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        crate::macros::dyn_map_fun!(self, DynCscMatrix, get, index)
    }

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

    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();

        macro_rules! fun {
            ($variant:ident, $data:expr) => {
                DynCscMatrix::$variant(CscMatrix::vstack(iter.map(|x| x.try_into().unwrap()))?)
            };
        }
        Ok(crate::macros::dyn_map!(iter.peek().unwrap(), DynCscMatrix, fun))
    }
}

impl WriteArrayData for DynCscMatrix {}
impl ReadArrayData for DynCscMatrix {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_array_attr::<u64, Ix1>("shape")?
            .into_iter()
            .map(|x| x as usize)
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
                    CscMatrix::read_select(container, info).map(DynCscMatrix::$variant)
                };
            }
            crate::macros::dyn_match!(ty, ScalarType, fun)
        } else {
            bail!("the container does not contain a csc matrix");
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

fn cast_csr<T, U>(csr: CsrMatrix<T>) -> Result<CsrMatrix<U>>
where
    T: TryInto<U>,
    <T as TryInto<U>>::Error: std::error::Error + Sync + Send + 'static,
{
    let (pattern, values) = csr.into_pattern_and_values();
    let out = CsrMatrix::try_from_pattern_and_values(
        pattern,
        values
            .into_iter()
            .map(|x| x.try_into())
            .collect::<Result<_, _>>()?,
    )
    .unwrap();
    Ok(out)
}

fn from_i64_csr<U: FromPrimitive>(csr: CsrMatrix<i64>) -> Result<CsrMatrix<U>> {
    let (pattern, values) = csr.into_pattern_and_values();
    let out = CsrMatrix::try_from_pattern_and_values(
        pattern,
        values
            .into_iter()
            .map(|x| U::from_i64(x).context("cannot convert from i64"))
            .collect::<Result<_>>()?,
    )
    .unwrap();
    Ok(out)
}

fn cast_csc<T, U>(csc: CscMatrix<T>) -> Result<CscMatrix<U>>
where
    T: TryInto<U>,
    <T as TryInto<U>>::Error: std::error::Error + Sync + Send + 'static,
{
    let (pattern, values) = csc.into_pattern_and_values();
    let out = CscMatrix::try_from_pattern_and_values(
        pattern,
        values
            .into_iter()
            .map(|x| x.try_into())
            .collect::<Result<_, _>>()?,
    )
    .unwrap();
    Ok(out)
}

fn from_i64_csc<U: FromPrimitive>(csc: CscMatrix<i64>) -> Result<CscMatrix<U>> {
    let (pattern, values) = csc.into_pattern_and_values();
    let out = CscMatrix::try_from_pattern_and_values(
        pattern,
        values
            .into_iter()
            .map(|x| U::from_i64(x).context("cannot convert from i64"))
            .collect::<Result<_>>()?,
    )
    .unwrap();
    Ok(out)
}
