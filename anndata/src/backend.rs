use crate::data::{DynArray, DynScalar, SelectInfo, SelectInfoElem, Shape};

use anyhow::{bail, Result};
use core::fmt::{Display, Formatter, Debug};
use ndarray::{Array, ArrayD, ArrayView, Dimension};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct WriteConfig {
    pub compression: Option<u8>,
    pub block_size: Option<Shape>,
}

impl Default for WriteConfig {
    fn default() -> Self {
        Self {
            compression: Some(1),
            //compression: None,
            block_size: None,
        }
    }
}

pub trait Backend: 'static {
    const NAME: &'static str;

    /// File represents the root of the hierarchy.
    type File: FileOp<Backend = Self> + GroupOp<Backend = Self> + Send + Sync;

    /// Groups work like directories and can contain groups or datasets.
    type Group: GroupOp<Backend = Self> + LocationOp<Backend = Self> + Send + Sync;

    /// Datasets store multi-dimensional arrays.
    type Dataset: DatasetOp<Backend = Self> + LocationOp<Backend = Self> + Send + Sync;

    /// Create a new file at the given path.
    fn create<P: AsRef<Path>>(path: P) -> Result<Self::File>;

    /// Opens a file as read-only, file must exist.
    fn open<P: AsRef<Path>>(path: P) -> Result<Self::File>;

    /// Opens a file as read/write, file must exist.
    fn open_rw<P: AsRef<Path>>(path: P) -> Result<Self::File>;
}

pub trait FileOp {
    type Backend: Backend;

    /// Returns the file path.
    fn filename(&self) -> PathBuf;

    /// Close the file.
    fn close(self) -> Result<()>;
}

pub trait GroupOp {
    type Backend: Backend;

    /// List all groups and datasets in this group.
    fn list(&self) -> Result<Vec<String>>;

    /// Create a new group.
    fn create_group(&self, name: &str) -> Result<<Self::Backend as Backend>::Group>;

    /// Open an existing group.
    fn open_group(&self, name: &str) -> Result<<Self::Backend as Backend>::Group>;

    /// Create an empty dataset holding an array value.
    fn new_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<<Self::Backend as Backend>::Dataset>;
    fn open_dataset(&self, name: &str) -> Result<<Self::Backend as Backend>::Dataset>;

    /// Delete a group or dataset.
    fn delete(&self, name: &str) -> Result<()>;

    /// Check if a group or dataset exists.
    fn exists(&self, name: &str) -> Result<bool>;

    fn create_scalar_data<D: BackendData>(
        &self,
        name: &str,
        data: &D,
    ) -> Result<<Self::Backend as Backend>::Dataset>;

    fn create_array_data<'a, A, D, Dim>(
        &self,
        name: &str,
        arr: A,
        config: WriteConfig,
    ) -> Result<<Self::Backend as Backend>::Dataset>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension,
    {
        let arr_view = arr.into();
        let shape = arr_view.shape();
        let block_size = config.block_size.unwrap_or_else(|| if shape.len() == 1 {
            shape[0].min(10000).into()
        } else {
            shape.iter().map(|&x| x.min(100)).collect()
        });
        let compression = if arr_view.len() > 100 {
            config.compression
        } else {
            None
        };
        let new_config = WriteConfig {
            compression: compression,
            block_size: Some(block_size),
        };
        let dataset = self.new_dataset::<D>(name, &shape.into(), new_config)?;
        dataset.write_array(arr_view)?;
        Ok(dataset)
    }
}

pub trait LocationOp {
    type Backend: Backend;

    /// Returns the Root.
    fn file(&self) -> Result<<Self::Backend as Backend>::File>;

    /// Returns the path of the location relative to the file root.
    fn path(&self) -> PathBuf;

    /// Write a scalar attribute at a given location. This function should be able to
    /// overwrite existing attributes.
    fn write_scalar_attr<D: BackendData>(&self, name: &str, value: D) -> Result<()>;
    /// Write a array-like attribute at a given location.
    fn write_array_attr<'a, A, D, Dim>(&self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension;
    fn write_str_attr(&self, name: &str, value: &str) -> Result<()> {
        self.write_scalar_attr(name, value.to_string())
    }

    fn read_scalar_attr<T: BackendData>(&self, name: &str) -> Result<T>;
    fn read_array_attr<T: BackendData, D: Dimension>(&self, name: &str) -> Result<Array<T, D>>;
    fn read_str_attr(&self, name: &str) -> Result<String> {
        self.read_scalar_attr(name)
    }
}

pub trait DatasetOp {
    type Backend: Backend;

    fn dtype(&self) -> Result<ScalarType>;
    fn shape(&self) -> Shape;
    fn reshape(&self, shape: &Shape) -> Result<()>;

    fn read_scalar<T: BackendData>(&self) -> Result<T>;

    fn read_array<T: BackendData, D>(&self) -> Result<Array<T, D>>
    where
        D: Dimension,
    {
        self.read_array_slice(SelectInfo::all(self.shape().ndim()).as_ref())
    }

    fn read_array_slice<T: BackendData, S, D>(&self, selection: &[S]) -> Result<Array<T, D>>
    where
        S: AsRef<SelectInfoElem>,
        D: Dimension;

    fn write_array<'a, A, D, Dim>(
        &self,
        data: A,
    ) -> Result<()>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension,
    {
        let arr = data.into();
        let ndim = arr.ndim();
        self.write_array_slice(arr, SelectInfo::all(ndim).as_ref())
    }

    fn write_array_slice<'a, A, S, T, D>(
        &self,
        data: A,
        selection: &[S],
    ) -> Result<()>
    where
        A: Into<ArrayView<'a, T, D>>,
        T: BackendData,
        S: AsRef<SelectInfoElem>,
        D: Dimension;
}

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

pub enum DataContainer<B: Backend> {
    Group(B::Group),
    Dataset(B::Dataset),
}

impl<B: Backend> Debug for DataContainer<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            DataContainer::Group(g) => write!(f, "Group({:?})", g.path()),
            DataContainer::Dataset(d) => write!(f, "Dataset({:?})", d.path()),
        }
    }
}

impl<B: Backend> LocationOp for DataContainer<B> {
    type Backend = B;

    fn file(&self) -> Result<B::File> {
        match self {
            DataContainer::Group(g) => g.file(),
            DataContainer::Dataset(d) => d.file(),
        }
    }
    fn path(&self) -> PathBuf {
        match self {
            DataContainer::Group(g) => g.path(),
            DataContainer::Dataset(d) => d.path(),
        }
    }

    fn write_array_attr<'a, A, D, Dim>(&self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension,
    {
        match self {
            DataContainer::Group(g) => g.write_array_attr(name, value),
            DataContainer::Dataset(d) => d.write_array_attr(name, value),
        }
    }
    fn write_scalar_attr<D: BackendData>(&self, name: &str, value: D) -> Result<()> {
        match self {
            DataContainer::Group(g) => g.write_scalar_attr(name, value),
            DataContainer::Dataset(d) => d.write_scalar_attr(name, value),
        }
    }

    fn read_scalar_attr<T: BackendData>(&self, name: &str) -> Result<T> {
        match self {
            DataContainer::Group(g) => g.read_scalar_attr(name),
            DataContainer::Dataset(d) => d.read_scalar_attr(name),
        }
    }
    fn read_array_attr<T: BackendData, D: Dimension>(&self, name: &str) -> Result<Array<T, D>> {
        match self {
            DataContainer::Group(g) => g.read_array_attr(name),
            DataContainer::Dataset(d) => d.read_array_attr(name),
        }
    }
}

impl<B: Backend> DataContainer<B> {
    pub fn open<G: GroupOp<Backend = B>>(group: &G, name: &str) -> Result<Self> {
        if group.exists(name)? {
            group
                .open_dataset(name)
                .map(DataContainer::Dataset)
                .or(group.open_group(name).map(DataContainer::Group))
        } else {
            bail!("No group or dataset named '{}' in group", name);
        }
    }

    pub fn delete(container: DataContainer<B>) -> Result<()> {
        container.file()?.delete(&container.path().to_string_lossy())
    }

    pub fn encoding_type(&self) -> Result<DataType> {
        let enc = match self {
            DataContainer::Group(group) => group
                .read_str_attr("encoding-type")
                .unwrap_or("mapping".to_string()),
            DataContainer::Dataset(dataset) => dataset
                .read_str_attr("encoding-type")
                .unwrap_or("numeric-scalar".to_string()),
        };
        let ty = match enc.as_str() {
            "string" => DataType::Scalar(ScalarType::String),
            "numeric-scalar" => DataType::Scalar(self.as_dataset()?.dtype()?),
            "categorical" => DataType::Categorical,
            "string-array" => DataType::Array(ScalarType::String),
            "array" => DataType::Array(self.as_dataset()?.dtype()?),
            "csc_matrix" => todo!(),
            "csr_matrix" => {
                let ty = self.as_group()?.open_dataset("data")?.dtype()?;
                DataType::CsrMatrix(ty)
            }
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

pub trait BackendData: Send + Sync + Clone + 'static {
    const DTYPE: ScalarType;
    fn into_dyn(&self) -> DynScalar;
    fn into_dyn_arr<'a, D>(arr: ArrayView<'a, Self, D>) -> DynArrayView<'a, D>;
    fn from_dyn(x: DynScalar) -> Result<Self>;
    fn from_dyn_arr(x: DynArray) -> Result<ArrayD<Self>>;
}

impl BackendData for i8 {
    const DTYPE: ScalarType = ScalarType::I8;

    fn into_dyn(&self) -> DynScalar {
        DynScalar::I8(*self)
    }

    fn into_dyn_arr<'a, D>(arr: ArrayView<'a, Self, D>) -> DynArrayView<'a, D> {
        DynArrayView::I8(arr)
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

    fn into_dyn_arr<'a, D>(arr: ArrayView<'a, Self, D>) -> DynArrayView<'a, D> {
        DynArrayView::I16(arr)
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

    fn into_dyn_arr<'a, D>(arr: ArrayView<'a, Self, D>) -> DynArrayView<'a, D> {
        DynArrayView::I32(arr)
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

    fn into_dyn_arr<'a, D>(arr: ArrayView<'a, Self, D>) -> DynArrayView<'a, D> {
        DynArrayView::I64(arr)
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

    fn into_dyn_arr<'a, D>(arr: ArrayView<'a, Self, D>) -> DynArrayView<'a, D> {
        DynArrayView::U8(arr)
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

    fn into_dyn_arr<'a, D>(arr: ArrayView<'a, Self, D>) -> DynArrayView<'a, D> {
        DynArrayView::U16(arr)
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

    fn into_dyn_arr<'a, D>(arr: ArrayView<'a, Self, D>) -> DynArrayView<'a, D> {
        DynArrayView::U32(arr)
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

    fn into_dyn_arr<'a, D>(arr: ArrayView<'a, Self, D>) -> DynArrayView<'a, D> {
        DynArrayView::U64(arr)
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

    fn into_dyn_arr<'a, D>(arr: ArrayView<'a, Self, D>) -> DynArrayView<'a, D> {
        DynArrayView::Usize(arr)
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

    fn into_dyn_arr<'a, D>(arr: ArrayView<'a, Self, D>) -> DynArrayView<'a, D> {
        DynArrayView::F32(arr)
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

    fn into_dyn_arr<'a, D>(arr: ArrayView<'a, Self, D>) -> DynArrayView<'a, D> {
        DynArrayView::F64(arr)
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

    fn into_dyn_arr<'a, D>(arr: ArrayView<'a, Self, D>) -> DynArrayView<'a, D> {
        DynArrayView::String(arr)
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

    fn into_dyn_arr<'a, D>(arr: ArrayView<'a, Self, D>) -> DynArrayView<'a, D> {
        DynArrayView::Bool(arr)
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

pub enum DynArrayView<'a, D> {
    I8(ArrayView<'a, i8, D>),
    I16(ArrayView<'a, i16, D>),
    I32(ArrayView<'a, i32, D>),
    I64(ArrayView<'a, i64, D>),
    U8(ArrayView<'a, u8, D>),
    U16(ArrayView<'a, u16, D>),
    U32(ArrayView<'a, u32, D>),
    U64(ArrayView<'a, u64, D>),
    Usize(ArrayView<'a, usize, D>),
    F32(ArrayView<'a, f32, D>),
    F64(ArrayView<'a, f64, D>),
    String(ArrayView<'a, String, D>),
    Bool(ArrayView<'a, bool, D>),
}

pub fn iter_containers<B: Backend>(
    group: &B::Group,
) -> impl Iterator<Item = (String, DataContainer<B>)> + '_ {
    group.list().unwrap().into_iter().map(|x| {
        let container = DataContainer::open(group, &x).unwrap();
        (x, container)
    })
}