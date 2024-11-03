mod datatype;
use crate::data::{ArrayCast, DynArray, SelectInfo, SelectInfoElem, Shape};
pub use datatype::{BackendData, DataType, ScalarType};

use anyhow::{bail, Result};
use core::fmt::{Debug, Formatter};
use ndarray::{arr0, Array, ArrayView, CowArray, Dimension, Ix0, IxDyn};
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
    /// The name of the backend.
    const NAME: &'static str;

    /// Data store
    type Store: StoreOp<Self> + GroupOp<Self> + Send + Sync;

    /// Groups work like directories and can contain groups or datasets.
    type Group: GroupOp<Self> + AttributeOp<Self> + Send + Sync;

    /// Datasets store multi-dimensional arrays.
    type Dataset: DatasetOp<Self> + AttributeOp<Self> + Send + Sync;

    /// Create a new file at the given path.
    fn new<P: AsRef<Path>>(path: P) -> Result<Self::Store>;

    /// Opens a file as read-only, file must exist.
    fn open<P: AsRef<Path>>(path: P) -> Result<Self::Store>;

    /// Opens a file as read/write, file must exist.
    fn open_rw<P: AsRef<Path>>(path: P) -> Result<Self::Store>;
}

pub trait StoreOp<B: Backend + ?Sized> {
    /// Returns the file path.
    fn filename(&self) -> PathBuf;

    /// Close the file.
    fn close(self) -> Result<()>;
}

pub trait GroupOp<B: Backend + ?Sized> {
    /// List all groups and datasets in this group.
    fn list(&self) -> Result<Vec<String>>;

    /// Create a new group.
    fn new_group(&self, name: &str) -> Result<B::Group>;

    /// Open an existing group.
    fn open_group(&self, name: &str) -> Result<B::Group>;

    /// Create an empty dataset holding an array value.
    fn new_empty_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<B::Dataset>;
    fn open_dataset(&self, name: &str) -> Result<B::Dataset>;

    /// Delete a group or dataset.
    fn delete(&self, name: &str) -> Result<()>;

    /// Check if a group or dataset exists.
    fn exists(&self, name: &str) -> Result<bool>;

    fn new_array_dataset<'a, D, Dim>(
        &self,
        name: &str,
        arr: CowArray<'a, D, Dim>,
        config: WriteConfig,
    ) -> Result<B::Dataset>
    where
        D: BackendData,
        Dim: Dimension,
    {
        let shape = arr.shape();
        let block_size = config.block_size.unwrap_or_else(|| {
            if shape.len() == 1 {
                shape[0].min(10000).into()
            } else {
                shape.iter().map(|&x| x.min(100)).collect()
            }
        });
        let compression = if arr.len() > 100 {
            config.compression
        } else {
            None
        };
        let new_config = WriteConfig {
            compression: compression,
            block_size: Some(block_size),
        };
        let dataset = self.new_empty_dataset::<D>(name, &shape.into(), new_config)?;
        dataset.write_array(arr)?;
        Ok(dataset)
    }

    fn new_scalar_dataset<D: BackendData>(&self, name: &str, data: &D) -> Result<B::Dataset> {
        self.new_array_dataset(name, arr0(data.clone()).into(), WriteConfig::default())
    }
}

pub trait AttributeOp<B: Backend + ?Sized> {
    /// Returns the Root.
    fn store(&self) -> Result<B::Store>;

    /// Returns the path of the location relative to the file root.
    fn path(&self) -> PathBuf;

    /// Write a array-like attribute at a given location.
    fn new_array_attr<'a, A, D, Dim>(&mut self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension;

    /// Write a scalar attribute at a given location. This function should be able to
    /// overwrite existing attributes.
    fn new_scalar_attr<D: BackendData>(&mut self, name: &str, value: D) -> Result<()> {
        self.new_array_attr(name, &arr0(value.clone()))
    }

    fn new_str_attr(&mut self, name: &str, value: &str) -> Result<()> {
        self.new_scalar_attr(name, value.to_string())
    }

    fn get_array_attr<T: BackendData, D: Dimension>(&self, name: &str) -> Result<Array<T, D>>;

    fn get_scalar_attr<T: BackendData>(&self, name: &str) -> Result<T> {
        self.get_array_attr::<T, Ix0>(name).map(|x| x.into_scalar())
    }
    fn get_str_attr(&self, name: &str) -> Result<String> {
        self.get_scalar_attr(name)
    }
}

pub trait DatasetOp<B: Backend + ?Sized> {
    /// Required methods

    fn dtype(&self) -> Result<ScalarType>;
    fn shape(&self) -> Shape;
    fn reshape(&mut self, shape: &Shape) -> Result<()>;

    fn write_array_slice<S, T, D>(&self, arr: CowArray<'_, T, D>, selection: &[S]) -> Result<()>
    where
        T: BackendData,
        S: AsRef<SelectInfoElem>,
        D: Dimension;

    fn read_array_slice<T: BackendData, S, D>(&self, selection: &[S]) -> Result<Array<T, D>>
    where
        S: AsRef<SelectInfoElem>,
        D: Dimension;

    /// Optional methods

    fn read_dyn_array_slice<S>(&self, selection: &[S]) -> Result<DynArray>
    where
        S: AsRef<SelectInfoElem>
    {
        let arr = match self.dtype()? {
            ScalarType::I8 => self.read_array_slice::<i8, _, IxDyn>(selection)?.into(),
            ScalarType::I16 => self.read_array_slice::<i16, _, IxDyn>(selection)?.into(),
            ScalarType::I32 => self.read_array_slice::<i32, _, IxDyn>(selection)?.into(),
            ScalarType::I64 => self.read_array_slice::<i64, _, IxDyn>(selection)?.into(),
            ScalarType::U8 => self.read_array_slice::<u8, _, IxDyn>(selection)?.into(),
            ScalarType::U16 => self.read_array_slice::<u16, _, IxDyn>(selection)?.into(),
            ScalarType::U32 => self.read_array_slice::<u32, _, IxDyn>(selection)?.into(),
            ScalarType::U64 => self.read_array_slice::<u64, _, IxDyn>(selection)?.into(),
            ScalarType::F32 => self.read_array_slice::<f32, _, IxDyn>(selection)?.into(),
            ScalarType::F64 => self.read_array_slice::<f64, _, IxDyn>(selection)?.into(),
            ScalarType::Bool => self.read_array_slice::<bool, _, IxDyn>(selection)?.into(),
            ScalarType::String => self.read_array_slice::<String, _, IxDyn>(selection)?.into(),
        };
        Ok(arr)
    }

    fn read_array_slice_cast<T, D, S>(&self, selection: &[S]) -> Result<Array<T, D>>
    where
        DynArray: ArrayCast<T>,
        D: Dimension,
        S: AsRef<SelectInfoElem>
    {
        self.read_dyn_array_slice(selection)?.cast()
    }

    fn read_array<T: BackendData, D>(&self) -> Result<Array<T, D>>
    where
        D: Dimension,
    {
        self.read_array_slice(SelectInfo::full_slice(self.shape().ndim()).as_ref())
    }

    fn read_dyn_array(&self) -> Result<DynArray> {
        self.read_dyn_array_slice(SelectInfo::full_slice(self.shape().ndim()).as_ref())
    }

    fn read_array_cast<T, D>(&self) -> Result<Array<T, D>>
    where
        DynArray: ArrayCast<T>,
        D: Dimension,
    {
        ArrayCast::cast(self.read_dyn_array()?)
    }

    fn read_scalar<T: BackendData>(&self) -> Result<T> {
        self.read_array::<T, Ix0>().map(|x| x.into_scalar())
    }

    fn write_array<D, Dim>(&self, arr: CowArray<'_, D, Dim>) -> Result<()>
    where
        D: BackendData,
        Dim: Dimension,
    {
        let ndim = arr.ndim();
        self.write_array_slice(arr, SelectInfo::full_slice(ndim).as_ref())
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

impl<B: Backend> AttributeOp<B> for DataContainer<B> {
    fn store(&self) -> Result<B::Store> {
        match self {
            DataContainer::Group(g) => g.store(),
            DataContainer::Dataset(d) => d.store(),
        }
    }
    fn path(&self) -> PathBuf {
        match self {
            DataContainer::Group(g) => g.path(),
            DataContainer::Dataset(d) => d.path(),
        }
    }

    fn new_array_attr<'a, A, D, Dim>(&mut self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension,
    {
        match self {
            DataContainer::Group(g) => g.new_array_attr(name, value),
            DataContainer::Dataset(d) => d.new_array_attr(name, value),
        }
    }
    fn new_scalar_attr<D: BackendData>(&mut self, name: &str, value: D) -> Result<()> {
        match self {
            DataContainer::Group(g) => g.new_scalar_attr(name, value),
            DataContainer::Dataset(d) => d.new_scalar_attr(name, value),
        }
    }

    fn get_scalar_attr<T: BackendData>(&self, name: &str) -> Result<T> {
        match self {
            DataContainer::Group(g) => g.get_scalar_attr(name),
            DataContainer::Dataset(d) => d.get_scalar_attr(name),
        }
    }
    fn get_array_attr<T: BackendData, D: Dimension>(&self, name: &str) -> Result<Array<T, D>> {
        match self {
            DataContainer::Group(g) => g.get_array_attr(name),
            DataContainer::Dataset(d) => d.get_array_attr(name),
        }
    }
}

impl<B: Backend> DataContainer<B> {
    pub fn open<G: GroupOp<B>>(group: &G, name: &str) -> Result<Self> {
        if group.exists(name)? {
            match group.open_dataset(name) {
                Ok(gr) => Ok(DataContainer::Dataset(gr)),
                Err(e1) => {
                    group.open_group(name).map(DataContainer::Group).map_err(|e2|
                        e2.context(e1).context(format!(
                            "Error opening group or dataset named '{}' in group",
                            name
                        ))
                    )
                }
            }
        } else {
            bail!("No group or dataset named '{}' in group", name);
        }
    }

    pub fn delete(container: DataContainer<B>) -> Result<()> {
        container
            .store()?
            .delete(&container.path().to_string_lossy())
    }

    pub fn encoding_type(&self) -> Result<DataType> {
        let enc = match self {
            DataContainer::Group(group) => group
                .get_str_attr("encoding-type")
                .unwrap_or("mapping".to_string()),
            DataContainer::Dataset(dataset) => dataset
                .get_str_attr("encoding-type")
                .unwrap_or("numeric-scalar".to_string()),
        };
        let ty = match enc.as_str() {
            "string" => DataType::Scalar(ScalarType::String),
            "numeric-scalar" => DataType::Scalar(self.as_dataset()?.dtype()?),
            "categorical" => DataType::Categorical,
            "string-array" => DataType::Array(ScalarType::String),
            "array" => DataType::Array(self.as_dataset()?.dtype()?),
            "csc_matrix" => {
                let ty = self.as_group()?.open_dataset("data")?.dtype()?;
                DataType::CscMatrix(ty)
            }
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

pub fn iter_containers<B: Backend>(
    group: &B::Group,
) -> impl Iterator<Item = (String, DataContainer<B>)> + '_ {
    group.list().unwrap().into_iter().map(|x| {
        let container = DataContainer::open(group, &x).unwrap();
        (x, container)
    })
}
