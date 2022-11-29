//mod hdf5;

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

impl<B: Backend> LocationOp for DataContainer<B> {
    type Backend = B;

    fn file(&self) -> Result<B::File> {
        match self {
            DataContainer::Group(g) => g.file(),
            DataContainer::Dataset(d) => d.file(),
        }
    }
    fn path(&self) -> Result<PathBuf> {
        match self {
            DataContainer::Group(g) => g.path(),
            DataContainer::Dataset(d) => d.path(),
        }
    }

    fn write_str_attr(&self, name: &str, value: &str) -> Result<()> {
        match self {
            DataContainer::Group(g) => g.write_str_attr(name, value),
            DataContainer::Dataset(d) => d.write_str_attr(name, value),
        }
    }
    fn write_str_arr_attr<'a, A, D>(&self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, String, D>>
    {
        match self {
            DataContainer::Group(g) => g.write_str_arr_attr(name, value),
            DataContainer::Dataset(d) => d.write_str_arr_attr(name, value),
        }
    }

    fn read_str_attr(&self, name: &str) -> Result<String> {
        match self {
            DataContainer::Group(g) => g.read_str_attr(name),
            DataContainer::Dataset(d) => d.read_str_attr(name),
        }
    }
    fn read_str_arr_attr<D>(&self, name: &str) -> Result<Array<String, D>> {
        match self {
            DataContainer::Group(g) => g.read_str_arr_attr(name),
            DataContainer::Dataset(d) => d.read_str_arr_attr(name),
        }
    }
}

impl<B: Backend> DataContainer<B> {
    pub fn open(group: &B::Group, name: &str) -> Result<Self> {
        if group.exists(name)? {
            group.open_dataset(name)
                .map(DataContainer::Dataset)
                .or(group.open_group(name).map(DataContainer::Group))
        } else {
            bail!("No group or dataset named '{}' in group", name);
        }
    }

    pub fn delete(container: DataContainer<B>) -> Result<()> {
        let file = container.file()?;
        let name = container.path()?;
        let group = file.open_group(name.parent().unwrap().to_str().unwrap())?;

        group.delete(name.file_name().unwrap().to_str().unwrap())
    }

    pub fn encoding_type(&self) -> Result<DataType> {
        let enc = match self {
            DataContainer::Group(group) => {
                group.read_str_attr("encoding_type").unwrap_or("mapping".to_string())
            }
            DataContainer::Dataset(dataset) => {
                dataset.read_str_attr("encoding_type").unwrap_or("numeric-scalar".to_string())
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
                let ty = B::dtype(&self.as_group()?.open_dataset("data")?)?;
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

pub trait GroupOp {
    type Backend: Backend;

    fn list(&self) -> Result<Vec<String>>;
    fn create_group(&self, name: &str) -> Result<<<Self as GroupOp>::Backend as Backend>::Group>;
    fn open_group(&self, name: &str) -> Result<<<Self as GroupOp>::Backend as Backend>::Group>;
    fn open_dataset(&self, name: &str) -> Result<<<Self as GroupOp>::Backend as Backend>::Dataset>; 
    fn delete(&self, name: &str) -> Result<()>;
    fn exists(&self, name: &str) -> Result<bool>;

    fn write_scalar<D: BackendData>(&self, name: &str, data: &D) -> Result<<<Self as GroupOp>::Backend as Backend>::Dataset>;
    fn write_array<'a, A, S, D, Dim>(
        &self,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<<<Self as GroupOp>::Backend as Backend>::Dataset>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        S: Into<Selection>;
}

pub trait LocationOp {
    type Backend: Backend;

    fn file(&self) -> Result<<<Self as LocationOp>::Backend as Backend>::File>;
    fn path(&self) -> Result<PathBuf>;

    fn write_str_attr(&self, name: &str, value: &str) -> Result<()>;
    fn write_str_arr_attr<'a, A, D>(&self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, String, D>>;

    fn read_str_attr(&self, name: &str) -> Result<String>;
    fn read_str_arr_attr<D>(&self, name: &str) -> Result<Array<String, D>>;
}

pub trait Backend {
    type File: GroupOp<Backend = Self>;

    /// Groups work like dictionaries.
    type Group: GroupOp<Backend = Self> + LocationOp<Backend=Self>;

    /// datasets contain arrays.
    type Dataset: LocationOp<Backend=Self>;

    fn create<P: AsRef<Path>>(path: P) -> Result<Self::File>;
    fn filename(file: &Self::File) -> PathBuf;
    fn close(file: Self::File) -> Result<()>;

    fn dtype(dataset: &Self::Dataset) -> Result<ScalarType>;
    fn shape(dataset: &Self::Dataset) -> Result<Vec<usize>>;

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
    group.list().unwrap().into_iter().map(|x| {
        let container = DataContainer::open(group, &x).unwrap();
        (x, container)
    })
}