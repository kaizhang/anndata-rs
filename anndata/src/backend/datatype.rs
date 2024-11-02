use crate::data::{DynArray, DynCowArray, DynScalar};

use anyhow::{bail, Result};
use core::fmt::{Display, Formatter, Debug};
use ndarray::{ArrayD, CowArray, IxDyn};
use serde::{Serialize, Deserialize};

/// All data types that can be stored in an AnnData object.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum DataType {
    Array(ScalarType),
    Categorical,
    CsrMatrix(ScalarType),
    CscMatrix(ScalarType),
    DataFrame,
    Scalar(ScalarType),
    Mapping,
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::Array(t) => write!(f, "Array({})", t),
            DataType::Categorical => write!(f, "Categorical"),
            DataType::CsrMatrix(t) => write!(f, "CsrMatrix({})", t),
            DataType::CscMatrix(t) => write!(f, "CscMatrix({})", t),
            DataType::DataFrame => write!(f, "DataFrame"),
            DataType::Scalar(t) => write!(f, "Scalar({})", t),
            DataType::Mapping => write!(f, "Mapping"),
        }
    }
}

/// All scalar types that are supported in an AnnData object.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ScalarType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Usize,
    F32,
    F64,
    Bool,
    String,
}

impl Display for ScalarType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarType::I8 => write!(f, "i8"),
            ScalarType::I16 => write!(f, "i16"),
            ScalarType::I32 => write!(f, "i32"),
            ScalarType::I64 => write!(f, "i64"),
            ScalarType::U8 => write!(f, "u8"),
            ScalarType::U16 => write!(f, "u16"),
            ScalarType::U32 => write!(f, "u32"),
            ScalarType::U64 => write!(f, "u64"),
            ScalarType::Usize => write!(f, "usize"),
            ScalarType::F32 => write!(f, "f32"),
            ScalarType::F64 => write!(f, "f64"),
            ScalarType::Bool => write!(f, "bool"),
            ScalarType::String => write!(f, "string"),
        }
    }
}

pub trait BackendData: Serialize + for<'a> Deserialize<'a> + Send + Sync + Clone + 'static {
    /// Type of the data
    const DTYPE: ScalarType;

    /// Convert to opaque representation.
    fn into_dyn(&self) -> DynScalar;

    /// Convert to opaque array representation.
    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a>;

    /// Convert from opaque representation.
    fn from_dyn(x: DynScalar) -> Result<Self>;

    /// Convert from opaque array representation.
    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>>;
}

impl BackendData for i8 {
    const DTYPE: ScalarType = ScalarType::I8;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::I8(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::I8(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::I8(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i8")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::I8(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i8 array")
        }
    }
}

impl BackendData for i16 {
    const DTYPE: ScalarType = ScalarType::I16;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::I16(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::I16(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::I16(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i16")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::I16(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i16 array")
        }
    }
}

impl BackendData for i32 {
    const DTYPE: ScalarType = ScalarType::I32;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::I32(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::I32(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::I32(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i32")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::I32(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i32 array")
        }
    }
}

impl BackendData for i64 {
    const DTYPE: ScalarType = ScalarType::I64;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::I64(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::I64(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::I64(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i64")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::I64(x) = x {
            Ok(x)
        } else {
            bail!("Expecting i64 array")
        }
    }
}

impl BackendData for u8 {
    const DTYPE: ScalarType = ScalarType::U8;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::U8(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::U8(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::U8(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u8")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::U8(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u8 array")
        }
    }
}

impl BackendData for u16 {
    const DTYPE: ScalarType = ScalarType::U16;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::U16(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::U16(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::U16(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u16")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::U16(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u16 array")
        }
    }
}

impl BackendData for u32 {
    const DTYPE: ScalarType = ScalarType::U32;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::U32(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::U32(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::U32(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u32")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::U32(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u32 array")
        }
    }
}

impl BackendData for u64 {
    const DTYPE: ScalarType = ScalarType::U64;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::U64(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::U64(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::U64(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u64")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::U64(x) = x {
            Ok(x)
        } else {
            bail!("Expecting u64 array")
        }
    }
}

impl BackendData for usize {
    const DTYPE: ScalarType = ScalarType::Usize;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::Usize(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::Usize(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::Usize(x) = x {
            Ok(x)
        } else {
            bail!("Expecting usize")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::Usize(x) = x {
            Ok(x)
        } else {
            bail!("Expecting usize array")
        }
    }
}

impl BackendData for f32 {
    const DTYPE: ScalarType = ScalarType::F32;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::F32(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::F32(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::F32(x) = x {
            Ok(x)
        } else {
            bail!("Expecting f32")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::F32(x) = x {
            Ok(x)
        } else {
            bail!("Expecting f32 array")
        }
    }
}

impl BackendData for f64 {
    const DTYPE: ScalarType = ScalarType::F64;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::F64(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::F64(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::F64(x) = x {
            Ok(x)
        } else {
            bail!("Expecting f64")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::F64(x) = x {
            Ok(x)
        } else {
            bail!("Expecting f64 array")
        }
    }
}

impl BackendData for String {
    const DTYPE: ScalarType = ScalarType::String;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::String(self.clone())
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::String(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::String(x) = x {
            Ok(x)
        } else {
            bail!("Expecting string")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::String(x) = x {
            Ok(x)
        } else {
            bail!("Expecting string array")
        }
    }
}

impl BackendData for bool {
    const DTYPE: ScalarType = ScalarType::Bool;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::Bool(*self)
    }

    fn into_dyn_arr<'a>(arr: CowArray<'a, Self, IxDyn>) -> DynCowArray<'a> {
        DynCowArray::Bool(arr)
    }

    fn from_dyn(x: DynScalar) -> Result<Self> {
        if let DynScalar::Bool(x) = x {
            Ok(x)
        } else {
            bail!("Expecting bool")
        }
    }

    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>> {
        if let DynArray::Bool(x) = x {
            Ok(x)
        } else {
            bail!("Expecting bool array")
        }
    }
}