use crate::data::DynScalar;

use anyhow::{bail, Result};
use ndarray::{Array, Array2, ArrayView};
use std::{ops::Deref, path::{Path, PathBuf}};
use core::fmt::{Display, Formatter};

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
            ScalarType::F32 => write!(f, "f32"),
            ScalarType::F64 => write!(f, "f64"),
            ScalarType::Bool => write!(f, "bool"),
            ScalarType::String => write!(f, "string"),
        }
    }
}   

/// A selection used for reading and writing to a Container.
pub enum Selection {
    All,
    Points(Array2<usize>),
}

pub enum DataContainer<B: Backend> {
    Group(B::Group),
    Dataset(B::Dataset),
}

impl<B: Backend> Deref for DataContainer<B> {
    type Target = B::Location;

    fn deref(&self) -> &Self::Target {
        match self {
            DataContainer::Group(group) => group,
            DataContainer::Dataset(dataset) => dataset,
        }
    }
}

impl<B: Backend> DataContainer<B> {
    pub fn open(group: &B::Group, name: &str) -> Result<Self> {
        if B::exists(group, name)? {
            B::open_dataset(group, name)
                .map(DataContainer::Dataset)
                .or(B::open_group(group, name).map(DataContainer::Group))
        } else {
            bail!("No group or dataset named '{}' in group", name);
        }
    }

    pub fn delete(container: DataContainer<B>) -> Result<()> {
        let (file, name) = match &container {
            DataContainer::Group(grp) => (B::file(grp)?, B::path(grp)?),
            DataContainer::Dataset(data) => (B::file(data)?, B::path(data)?),
        };
        let group = B::open_group(&file, name.parent().unwrap().to_str().unwrap())?;

        B::delete(&group, name.file_name().unwrap().to_str().unwrap())
    }

    pub fn encoding_type(&self) -> Result<DataType> {
        let enc = match self {
            DataContainer::Group(group) => {
                B::read_str_attr(&group, "encoding_type").unwrap_or("mapping".to_string())
            }
            DataContainer::Dataset(dataset) => {
                B::read_str_attr(&dataset, "encoding_type").unwrap_or("numeric-scalar".to_string())
            }
        };
        let ty = match enc.as_str() {
            "string" => DataType::Scalar(ScalarType::String),
            "numeric-scalar" => DataType::Scalar(B::dtype(self.as_dataset()?)?),
            "categorical" => DataType::Categorical,
            "string-array" => DataType::Array(ScalarType::String),
            "array" => DataType::Array(B::dtype(self.as_dataset()?)?),
            "csc_matrix" => todo!(),
            "csr_matrix" => {
                let ty = B::dtype(&B::open_dataset(self.as_group()?, "data")?)?;
                DataType::CsrMatrix(ty)
            },
            "dataframe" => DataType::DataFrame,
            "mapping" | "dict" => DataType::Mapping,
            ty => bail!("Unsupported type '{}'", ty),
        };
        Ok(ty)
    }

    pub fn as_group(&self) -> Result<&B::Group> {
        match self {
            Self::Group(x) => Ok(&x),
            _ => bail!("Expecting Group"),
        }
    }

    pub fn as_dataset(&self) -> Result<&B::Dataset> {
        match self {
            Self::Dataset(x) => Ok(&x),
            _ => bail!("Expecting Dataset"),
        }
    }
}

pub trait Backend {
    type File: Deref<Target = Self::Group>;

    /// Groups work like dictionaries.
    type Group: Deref<Target = Self::Location>;

    /// datasets contain arrays.
    type Dataset: Deref<Target = Self::Location>;

    type Location;

    fn create<P: AsRef<Path>>(path: P) -> Result<Self::File>;
    fn filename(file: &Self::File) -> PathBuf;
    fn close(file: Self::File) -> Result<()>;

    fn list(group: &Self::Group) -> Result<Vec<String>>;
    fn create_group(group: &Self::Group, name: &str) -> Result<Self::Group>;
    fn open_group(group: &Self::Group, name: &str) -> Result<Self::Group>;
    fn open_dataset(group: &Self::Group, name: &str) -> Result<Self::Dataset>;
    fn delete(group: &Self::Group, name: &str) -> Result<()>;
    fn exists(group: &Self::Group, name: &str) -> Result<bool>;

    fn dtype(dataset: &Self::Dataset) -> Result<ScalarType>;
    fn shape(dataset: &Self::Dataset) -> Result<Vec<usize>>;

    fn file(location: &Self::Location) -> Result<Self::File>;
    fn path(location: &Self::Location) -> Result<PathBuf>;

    fn write_str_attr(location: &Self::Location, name: &str, value: &str) -> Result<()>;
    fn write_str_arr_attr<'a, A, D>(location: &Self::Location, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, String, D>>;

    fn read_str_attr(location: &Self::Location, name: &str) -> Result<String>;
    fn read_str_arr_attr<D>(location: &Self::Location, name: &str) -> Result<Array<String, D>>;

    fn write_i8(group: &Self::Group, name: &str, data: &i8) -> Result<Self::Dataset>;
    fn write_i16(group: &Self::Group, name: &str, data: &i16) -> Result<Self::Dataset>;
    fn write_i32(group: &Self::Group, name: &str, data: &i32) -> Result<Self::Dataset>;
    fn write_i64(group: &Self::Group, name: &str, data: &i64) -> Result<Self::Dataset>;
    fn write_u8(group: &Self::Group, name: &str, data: &u8) -> Result<Self::Dataset>;
    fn write_u16(group: &Self::Group, name: &str, data: &u16) -> Result<Self::Dataset>;
    fn write_u32(group: &Self::Group, name: &str, data: &u32) -> Result<Self::Dataset>;
    fn write_u64(group: &Self::Group, name: &str, data: &u64) -> Result<Self::Dataset>;
    fn write_f32(group: &Self::Group, name: &str, data: &f32) -> Result<Self::Dataset>;
    fn write_f64(group: &Self::Group, name: &str, data: &f64) -> Result<Self::Dataset>;
    fn write_bool(group: &Self::Group, name: &str, data: &bool) -> Result<Self::Dataset>;
    fn write_string(group: &Self::Group, name: &str, data: &str) -> Result<Self::Dataset>;

    fn write_i8_arr<'a, A, S, D>(
        group: &Self::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<Self::Dataset>
    where
        A: Into<ArrayView<'a, i8, D>>,
        S: Into<Selection>;
    fn write_i16_arr<'a, A, S, D>(
        group: &Self::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<Self::Dataset>
    where
        A: Into<ArrayView<'a, i16, D>>,
        S: Into<Selection>;
    fn write_i32_arr<'a, A, S, D>(
        group: &Self::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<Self::Dataset>
    where
        A: Into<ArrayView<'a, i32, D>>,
        S: Into<Selection>;
    fn write_i64_arr<'a, A, S, D>(
        group: &Self::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<Self::Dataset>
    where
        A: Into<ArrayView<'a, i64, D>>,
        S: Into<Selection>;
    fn write_u8_arr<'a, A, S, D>(
        group: &Self::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<Self::Dataset>
    where
        A: Into<ArrayView<'a, u8, D>>,
        S: Into<Selection>;
    fn write_u16_arr<'a, A, S, D>(
        group: &Self::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<Self::Dataset>
    where
        A: Into<ArrayView<'a, u16, D>>,
        S: Into<Selection>;
    fn write_u32_arr<'a, A, S, D>(
        group: &Self::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<Self::Dataset>
    where
        A: Into<ArrayView<'a, u32, D>>,
        S: Into<Selection>;
    fn write_u64_arr<'a, A, S, D>(
        group: &Self::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<Self::Dataset>
    where
        A: Into<ArrayView<'a, u64, D>>,
        S: Into<Selection>;
    fn write_f32_arr<'a, A, S, D>(
        group: &Self::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<Self::Dataset>
    where
        A: Into<ArrayView<'a, f32, D>>,
        S: Into<Selection>;
    fn write_f64_arr<'a, A, S, D>(
        group: &Self::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<Self::Dataset>
    where
        A: Into<ArrayView<'a, f64, D>>,
        S: Into<Selection>;
    fn write_bool_arr<'a, A, S, D>(
        group: &Self::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<Self::Dataset>
    where
        A: Into<ArrayView<'a, bool, D>>,
        S: Into<Selection>;
    fn write_str_arr<'a, A, S, D>(
        group: &Self::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<Self::Dataset>
    where
        A: Into<ArrayView<'a, String, D>>,
        S: Into<Selection>;

    fn read_i8(dataset: &Self::Dataset) -> Result<i8>;
    fn read_i16(dataset: &Self::Dataset) -> Result<i16>;
    fn read_i32(dataset: &Self::Dataset) -> Result<i32>;
    fn read_i64(dataset: &Self::Dataset) -> Result<i64>;
    fn read_u8(dataset: &Self::Dataset) -> Result<u8>;
    fn read_u16(dataset: &Self::Dataset) -> Result<u16>;
    fn read_u32(dataset: &Self::Dataset) -> Result<u32>;
    fn read_u64(dataset: &Self::Dataset) -> Result<u64>;
    fn read_f32(dataset: &Self::Dataset) -> Result<f32>;
    fn read_f64(dataset: &Self::Dataset) -> Result<f64>;
    fn read_bool(dataset: &Self::Dataset) -> Result<bool>;
    fn read_string(dataset: &Self::Dataset) -> Result<String>;

    fn read_i8_arr<D, S>(dataset: &Self::Dataset, selection: S) -> Result<Array<i8, D>>
    where
        S: Into<Selection>;
    fn read_i16_arr<D, S>(dataset: &Self::Dataset, selection: S) -> Result<Array<i16, D>>
    where
        S: Into<Selection>;
    fn read_i32_arr<D, S>(dataset: &Self::Dataset, selection: S) -> Result<Array<i32, D>>
    where
        S: Into<Selection>;
    fn read_i64_arr<D, S>(dataset: &Self::Dataset, selection: S) -> Result<Array<i64, D>>
    where
        S: Into<Selection>;
    fn read_u8_arr<D, S>(dataset: &Self::Dataset, selection: S) -> Result<Array<u8, D>>
    where
        S: Into<Selection>;
    fn read_u16_arr<D, S>(dataset: &Self::Dataset, selection: S) -> Result<Array<u16, D>>
    where
        S: Into<Selection>;
    fn read_u32_arr<D, S>(dataset: &Self::Dataset, selection: S) -> Result<Array<u32, D>>
    where
        S: Into<Selection>;
    fn read_u64_arr<D, S>(dataset: &Self::Dataset, selection: S) -> Result<Array<u64, D>>
    where
        S: Into<Selection>;
    fn read_f32_arr<D, S>(dataset: &Self::Dataset, selection: S) -> Result<Array<f32, D>>
    where
        S: Into<Selection>;
    fn read_f64_arr<D, S>(dataset: &Self::Dataset, selection: S) -> Result<Array<f64, D>>
    where
        S: Into<Selection>;
    fn read_bool_arr<D, S>(dataset: &Self::Dataset, selection: S) -> Result<Array<bool, D>>
    where
        S: Into<Selection>;
    fn read_str_arr<D, S>(dataset: &Self::Dataset, selection: S) -> Result<Array<String, D>>
    where
        S: Into<Selection>;
}

pub trait BackendData: Send + Sync + Clone + 'static {
    const DTYPE: ScalarType;

    fn into_dyn(&self) -> DynScalar;

    fn write_data<B: Backend>(group: &B::Group, name: &str, data: &Self) -> Result<B::Dataset>;

    fn write_arr_data<'a, B: Backend, A, S, D>(
        group: &B::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<B::Dataset>
    where
        A: Into<ArrayView<'a, Self, D>>,
        S: Into<Selection>,
        Self: Sized + 'a;

    fn read_data<B: Backend>(dataset: &B::Dataset) -> Result<Self>
    where
        Self: Sized;

    fn read_arr_data<B: Backend, S, D>(
        dataset: &B::Dataset,
        selection: S,
    ) -> Result<Array<Self, D>>
    where
        S: Into<Selection>,
        Self: Sized;
}

impl BackendData for i8 {
    const DTYPE: ScalarType = ScalarType::I8;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::I8(*self)
    }

    fn write_data<B: Backend>(group: &B::Group, name: &str, data: &Self) -> Result<B::Dataset> {
        B::write_i8(group, name, data)
    }

    fn write_arr_data<'a, B: Backend, A, S, D>(
        group: &B::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<B::Dataset>
    where
        A: Into<ArrayView<'a, Self, D>>,
        S: Into<Selection>,
        Self: Sized + 'a,
    {
        B::write_i8_arr(group, name, data, selection)
    }

    fn read_data<B: Backend>(dataset: &B::Dataset) -> Result<Self>
    where
        Self: Sized,
    {
        B::read_i8(dataset)
    }

    fn read_arr_data<B: Backend, S, D>(dataset: &B::Dataset, selection: S) -> Result<Array<Self, D>>
    where
        S: Into<Selection>,
        Self: Sized,
    {
        B::read_i8_arr(dataset, selection)
    }
}

impl BackendData for i16 {
    const DTYPE: ScalarType = ScalarType::I16;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::I16(*self)
    }

    fn write_data<B: Backend>(group: &B::Group, name: &str, data: &Self) -> Result<B::Dataset> {
        B::write_i16(group, name, data)
    }

    fn write_arr_data<'a, B: Backend, A, S, D>(
        group: &B::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<B::Dataset>
    where
        A: Into<ArrayView<'a, Self, D>>,
        S: Into<Selection>,
        Self: Sized + 'a,
    {
        B::write_i16_arr(group, name, data, selection)
    }

    fn read_data<B: Backend>(dataset: &B::Dataset) -> Result<Self>
    where
        Self: Sized,
    {
        B::read_i16(dataset)
    }

    fn read_arr_data<B: Backend, S, D>(dataset: &B::Dataset, selection: S) -> Result<Array<Self, D>>
    where
        S: Into<Selection>,
        Self: Sized,
    {
        B::read_i16_arr(dataset, selection)
    }
}

impl BackendData for i32 {
    const DTYPE: ScalarType = ScalarType::I32;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::I32(*self)
    }

    fn write_data<B: Backend>(group: &B::Group, name: &str, data: &Self) -> Result<B::Dataset> {
        B::write_i32(group, name, data)
    }

    fn write_arr_data<'a, B: Backend, A, S, D>(
        group: &B::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<B::Dataset>
    where
        A: Into<ArrayView<'a, Self, D>>,
        S: Into<Selection>,
        Self: Sized + 'a,
    {
        B::write_i32_arr(group, name, data, selection)
    }

    fn read_data<B: Backend>(dataset: &B::Dataset) -> Result<Self>
    where
        Self: Sized,
    {
        B::read_i32(dataset)
    }

    fn read_arr_data<B: Backend, S, D>(dataset: &B::Dataset, selection: S) -> Result<Array<Self, D>>
    where
        S: Into<Selection>,
        Self: Sized,
    {
        B::read_i32_arr(dataset, selection)
    }
}

impl BackendData for i64 {
    const DTYPE: ScalarType = ScalarType::I64;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::I64(*self)
    }

    fn write_data<B: Backend>(group: &B::Group, name: &str, data: &Self) -> Result<B::Dataset> {
        B::write_i64(group, name, data)
    }

    fn write_arr_data<'a, B: Backend, A, S, D>(
        group: &B::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<B::Dataset>
    where
        A: Into<ArrayView<'a, Self, D>>,
        S: Into<Selection>,
        Self: Sized + 'a,
    {
        B::write_i64_arr(group, name, data, selection)
    }

    fn read_data<B: Backend>(dataset: &B::Dataset) -> Result<Self>
    where
        Self: Sized,
    {
        B::read_i64(dataset)
    }

    fn read_arr_data<B: Backend, S, D>(dataset: &B::Dataset, selection: S) -> Result<Array<Self, D>>
    where
        S: Into<Selection>,
        Self: Sized,
    {
        B::read_i64_arr(dataset, selection)
    }
}

impl BackendData for u8 {
    const DTYPE: ScalarType = ScalarType::U8;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::U8(*self)
    }

    fn write_data<B: Backend>(group: &B::Group, name: &str, data: &Self) -> Result<B::Dataset> {
        B::write_u8(group, name, data)
    }

    fn write_arr_data<'a, B: Backend, A, S, D>(
        group: &B::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<B::Dataset>
    where
        A: Into<ArrayView<'a, Self, D>>,
        S: Into<Selection>,
        Self: Sized + 'a,
    {
        B::write_u8_arr(group, name, data, selection)
    }

    fn read_data<B: Backend>(dataset: &B::Dataset) -> Result<Self>
    where
        Self: Sized,
    {
        B::read_u8(dataset)
    }

    fn read_arr_data<B: Backend, S, D>(dataset: &B::Dataset, selection: S) -> Result<Array<Self, D>>
    where
        S: Into<Selection>,
        Self: Sized,
    {
        B::read_u8_arr(dataset, selection)
    }
}

impl BackendData for u16 {
    const DTYPE: ScalarType = ScalarType::U16;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::U16(*self)
    }

    fn write_data<B: Backend>(group: &B::Group, name: &str, data: &Self) -> Result<B::Dataset> {
        B::write_u16(group, name, data)
    }

    fn write_arr_data<'a, B: Backend, A, S, D>(
        group: &B::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<B::Dataset>
    where
        A: Into<ArrayView<'a, Self, D>>,
        S: Into<Selection>,
        Self: Sized + 'a,
    {
        B::write_u16_arr(group, name, data, selection)
    }

    fn read_data<B: Backend>(dataset: &B::Dataset) -> Result<Self>
    where
        Self: Sized,
    {
        B::read_u16(dataset)
    }

    fn read_arr_data<B: Backend, S, D>(dataset: &B::Dataset, selection: S) -> Result<Array<Self, D>>
    where
        S: Into<Selection>,
        Self: Sized,
    {
        B::read_u16_arr(dataset, selection)
    }
}

impl BackendData for u32 {
    const DTYPE: ScalarType = ScalarType::U32;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::U32(*self)
    }

    fn write_data<B: Backend>(group: &B::Group, name: &str, data: &Self) -> Result<B::Dataset> {
        B::write_u32(group, name, data)
    }

    fn write_arr_data<'a, B: Backend, A, S, D>(
        group: &B::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<B::Dataset>
    where
        A: Into<ArrayView<'a, Self, D>>,
        S: Into<Selection>,
        Self: Sized + 'a,
    {
        B::write_u32_arr(group, name, data, selection)
    }

    fn read_data<B: Backend>(dataset: &B::Dataset) -> Result<Self>
    where
        Self: Sized,
    {
        B::read_u32(dataset)
    }

    fn read_arr_data<B: Backend, S, D>(dataset: &B::Dataset, selection: S) -> Result<Array<Self, D>>
    where
        S: Into<Selection>,
        Self: Sized,
    {
        B::read_u32_arr(dataset, selection)
    }
}

impl BackendData for u64 {
    const DTYPE: ScalarType = ScalarType::U64;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::U64(*self)
    }

    fn write_data<B: Backend>(group: &B::Group, name: &str, data: &Self) -> Result<B::Dataset> {
        B::write_u64(group, name, data)
    }

    fn write_arr_data<'a, B: Backend, A, S, D>(
        group: &B::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<B::Dataset>
    where
        A: Into<ArrayView<'a, Self, D>>,
        S: Into<Selection>,
        Self: Sized + 'a,
    {
        B::write_u64_arr(group, name, data, selection)
    }

    fn read_data<B: Backend>(dataset: &B::Dataset) -> Result<Self>
    where
        Self: Sized,
    {
        B::read_u64(dataset)
    }

    fn read_arr_data<B: Backend, S, D>(dataset: &B::Dataset, selection: S) -> Result<Array<Self, D>>
    where
        S: Into<Selection>,
        Self: Sized,
    {
        B::read_u64_arr(dataset, selection)
    }
}

impl BackendData for f32 {
    const DTYPE: ScalarType = ScalarType::F32;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::F32(*self)
    }

    fn write_data<B: Backend>(group: &B::Group, name: &str, data: &Self) -> Result<B::Dataset> {
        B::write_f32(group, name, data)
    }

    fn write_arr_data<'a, B: Backend, A, S, D>(
        group: &B::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<B::Dataset>
    where
        A: Into<ArrayView<'a, Self, D>>,
        S: Into<Selection>,
        Self: Sized + 'a,
    {
        B::write_f32_arr(group, name, data, selection)
    }

    fn read_data<B: Backend>(dataset: &B::Dataset) -> Result<Self>
    where
        Self: Sized,
    {
        B::read_f32(dataset)
    }

    fn read_arr_data<B: Backend, S, D>(dataset: &B::Dataset, selection: S) -> Result<Array<Self, D>>
    where
        S: Into<Selection>,
        Self: Sized,
    {
        B::read_f32_arr(dataset, selection)
    }
}

impl BackendData for f64 {
    const DTYPE: ScalarType = ScalarType::F64;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::F64(*self)
    }

    fn write_data<B: Backend>(group: &B::Group, name: &str, data: &Self) -> Result<B::Dataset> {
        B::write_f64(group, name, data)
    }

    fn write_arr_data<'a, B: Backend, A, S, D>(
        group: &B::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<B::Dataset>
    where
        A: Into<ArrayView<'a, Self, D>>,
        S: Into<Selection>,
        Self: Sized + 'a,
    {
        B::write_f64_arr(group, name, data, selection)
    }

    fn read_data<B: Backend>(dataset: &B::Dataset) -> Result<Self>
    where
        Self: Sized,
    {
        B::read_f64(dataset)
    }

    fn read_arr_data<B: Backend, S, D>(dataset: &B::Dataset, selection: S) -> Result<Array<Self, D>>
    where
        S: Into<Selection>,
        Self: Sized,
    {
        B::read_f64_arr(dataset, selection)
    }
}

impl BackendData for String {
    const DTYPE: ScalarType = ScalarType::String;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::String(self.clone())
    }

    fn write_data<B: Backend>(group: &B::Group, name: &str, data: &Self) -> Result<B::Dataset> {
        B::write_string(group, name, data)
    }

    fn write_arr_data<'a, B: Backend, A, S, D>(
        group: &B::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<B::Dataset>
    where
        A: Into<ArrayView<'a, Self, D>>,
        S: Into<Selection>,
        Self: Sized + 'a,
    {
        B::write_str_arr(group, name, data, selection)
    }

    fn read_data<B: Backend>(dataset: &B::Dataset) -> Result<Self>
    where
        Self: Sized,
    {
        B::read_string(dataset)
    }

    fn read_arr_data<B: Backend, S, D>(dataset: &B::Dataset, selection: S) -> Result<Array<Self, D>>
    where
        S: Into<Selection>,
        Self: Sized,
    {
        B::read_str_arr(dataset, selection)
    }
}

impl BackendData for bool {
    const DTYPE: ScalarType = ScalarType::Bool;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::Bool(*self)
    }

    fn write_data<B: Backend>(group: &B::Group, name: &str, data: &Self) -> Result<B::Dataset> {
        B::write_bool(group, name, data)
    }

    fn write_arr_data<'a, B: Backend, A, S, D>(
        group: &B::Group,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<B::Dataset>
    where
        A: Into<ArrayView<'a, Self, D>>,
        S: Into<Selection>,
        Self: Sized + 'a,
    {
        B::write_bool_arr(group, name, data, selection)
    }

    fn read_data<B: Backend>(dataset: &B::Dataset) -> Result<Self>
    where
        Self: Sized,
    {
        B::read_bool(dataset)
    }

    fn read_arr_data<B: Backend, S, D>(dataset: &B::Dataset, selection: S) -> Result<Array<Self, D>>
    where
        S: Into<Selection>,
        Self: Sized,
    {
        B::read_bool_arr(dataset, selection)
    }
}

pub fn iter_containers<B: Backend>(group: &B::Group) -> impl Iterator<Item = (String, DataContainer<B>)> + '_{
    B::list(group).unwrap().into_iter().map(|x| {
        let container = DataContainer::open(group, &x).unwrap();
        (x, container)
    })
}