use crate::{
    backend::{
        Backend, BackendData, DatasetOp, DynArrayView, FileOp, GroupOp, LocationOp, ScalarType,
        WriteConfig,
    },
    data::{ArrayOp, BoundedSelectInfo, DynArray, DynScalar, SelectInfoElem, Shape},
};

use anyhow::{bail, Result};
use n5::{
    filesystem::N5Filesystem, ndarray::N5NdarrayWriter, DataType, DatasetAttributes, N5Lister,
    N5Reader, N5Writer, ReadableDataBlock, ReflectedType, SliceDataBlock,
};
use ndarray::{Array, Array2, ArrayView, Dimension, IxDyn, IxDynImpl, SliceInfoElem};
use serde_json::value::Value;
use smallvec::smallvec;
use std::{
    fmt::write,
    ops::Deref,
    path::{Path, PathBuf},
};

/// The N5 backend.
/// N5 is a "Not HDF5" n-dimensional tensor file system storage format
/// created by the Saalfeld lab at Janelia Research Campus.
/// N5 always stores array data in column-major order.
pub struct N5;

pub struct Root {
    group: Group,
    filename: PathBuf,
}

impl Deref for Root {
    type Target = Group;

    fn deref(&self) -> &Self::Target {
        &self.group
    }
}

pub struct Group(Location);

impl Deref for Group {
    type Target = Location;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct Dataset {
    attributes: DatasetAttributes,
    loc: Location,
}

impl Deref for Dataset {
    type Target = Location;

    fn deref(&self) -> &Self::Target {
        &self.loc
    }
}

pub struct Location {
    path: PathBuf,
    root: N5Filesystem,
}

impl Location {
    fn child_path(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Backend for N5 {
    type File = Root;

    type Group = Group;

    /// datasets contain arrays.
    type Dataset = Dataset;

    fn create<P: AsRef<Path>>(path: P) -> Result<Self::File> {
        Ok(Root {
            filename: path.as_ref().to_path_buf(),
            group: Group(Location {
                path: PathBuf::new(),
                root: N5Filesystem::open_or_create(path)?,
            }),
        })
    }
}

impl FileOp for Root {
    type Backend = N5;

    /// Returns the file path.
    fn filename(&self) -> PathBuf {
        self.filename.clone()
    }

    /// Close the file.
    fn close(self) -> Result<()> {
        todo!()
    }
}

impl GroupOp for Group {
    type Backend = N5;

    /// List all groups and datasets in this group.
    fn list(&self) -> Result<Vec<String>> {
        Ok(self.root.list(&self.path.to_string_lossy())?)
    }

    /// Create a new group.
    fn create_group(&self, name: &str) -> Result<<Self::Backend as Backend>::Group> {
        let path = self.child_path(name);
        self.root.create_group(&path.to_string_lossy())?;
        Ok(Group(Location {
            path,
            root: self.root.clone(),
        }))
    }

    /// Open an existing group.
    fn open_group(&self, name: &str) -> Result<<Self::Backend as Backend>::Group> {
        let path = self.child_path(name);
        if self.root.exists(&path.to_string_lossy())? {
            Ok(Group(Location {
                path,
                root: self.root.clone(),
            }))
        } else {
            bail!("Group {} does not exist", name);
        }
    }

    /// Create an empty dataset holding an array value.
    fn new_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<<Self::Backend as Backend>::Dataset> {
        let dimensions = shape.as_ref().iter().map(|&d| d as u64).collect();
        let block_size = config
            .block_size
            .unwrap_or(vec![100; shape.ndim()].into())
            .as_ref()
            .iter()
            .map(|x| *x as u32)
            .collect();
        let data_type = get_data_type::<T>();

        let path = self.child_path(name);
        let attr = DatasetAttributes::new(dimensions, block_size, data_type, Default::default());
        self.root.create_dataset(&path.to_string_lossy(), &attr)?;

        Ok(Dataset {
            attributes: attr,
            loc: Location {
                path,
                root: self.root.clone(),
            },
        })
    }

    fn open_dataset(&self, name: &str) -> Result<<Self::Backend as Backend>::Dataset> {
        let path = self.child_path(name);
        let attributes = self.root.get_dataset_attributes(&path.to_string_lossy())?;
        if self.root.dataset_exists(&path.to_string_lossy())? {
            Ok(Dataset {
                attributes,
                loc: Location {
                    path,
                    root: self.root.clone(),
                },
            })
        } else {
            bail!("Dataset {} does not exist", name);
        }
    }

    /// Delete a group or dataset.
    fn delete(&self, name: &str) -> Result<()> {
        self.root.remove(&self.child_path(name).to_string_lossy())?;
        Ok(())
    }

    /// Check if a group or dataset exists.
    fn exists(&self, name: &str) -> Result<bool> {
        Ok(self.root.exists(&self.child_path(name).to_string_lossy())?)
    }

    fn create_scalar_data<D: BackendData>(
        &self,
        name: &str,
        data: &D,
    ) -> Result<<Self::Backend as Backend>::Dataset> {
        macro_rules! write_scalar {
            ($data:expr) => {{
                let path = self.child_path(name);
                let data_type = get_data_type::<D>();
                let data_size = $data.len();
                let attr = DatasetAttributes::new(
                    smallvec![data_size as u64],
                    smallvec![data_size as u32],
                    data_type,
                    Default::default(),
                );
                let block_in =
                    SliceDataBlock::new(smallvec![data_size as u32], smallvec![0], $data);
                self.root.create_dataset(&path.to_string_lossy(), &attr)?;
                self.root
                    .write_block(&path.to_string_lossy(), &attr, &block_in)?;
                anyhow::Ok(Dataset {
                    attributes: attr,
                    loc: Location {
                        path,
                        root: self.root.clone(),
                    },
                })
            }};
        }

        match data.into_dyn() {
            DynScalar::U8(x) => write_scalar!([x]),
            DynScalar::U16(x) => write_scalar!([x]),
            DynScalar::U32(x) => write_scalar!([x]),
            DynScalar::U64(x) => write_scalar!([x]),
            DynScalar::Usize(x) => {
                let dataset = write_scalar!([x as u64])?;
                dataset.write_str_attr("custom-data-type", "usize")?;
                Ok(dataset)
            }
            DynScalar::I8(x) => write_scalar!([x]),
            DynScalar::I16(x) => write_scalar!([x]),
            DynScalar::I32(x) => write_scalar!([x]),
            DynScalar::I64(x) => write_scalar!([x]),
            DynScalar::F32(x) => write_scalar!([x]),
            DynScalar::F64(x) => write_scalar!([x]),
            DynScalar::Bool(x) => {
                let dataset = write_scalar!([x as u8])?;
                dataset.write_str_attr("custom-data-type", "bool")?;
                Ok(dataset)
            }
            DynScalar::String(x) => {
                let dataset = write_scalar!(x.as_bytes())?;
                dataset.write_str_attr("custom-data-type", "string")?;
                Ok(dataset)
            }
        }
    }

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
        let block_size = config.block_size.unwrap_or_else(|| {
            if shape.len() == 1 {
                shape[0].min(10000).into()
            } else {
                shape.iter().map(|&x| x.min(100)).collect()
            }
        });
        let new_config = WriteConfig {
            compression: config.compression,
            block_size: Some(block_size),
        };
        let dataset = self.new_dataset::<D>(name, &shape.into(), new_config)?;
        dataset.write_array(arr_view)?;
        Ok(dataset)
    }
}

impl DatasetOp for Dataset {
    type Backend = N5;

    fn dtype(&self) -> Result<ScalarType> {
        match self
            .root
            .get_dataset_attributes(&self.path.to_string_lossy())?
            .get_data_type()
        {
            DataType::UINT8 => match self
                .read_str_attr("custom-data-type")
                .as_ref()
                .map(|x| x.as_str())
            {
                Ok("bool") => Ok(ScalarType::Bool),
                Ok("string") => Ok(ScalarType::String),
                _ => Ok(ScalarType::U8),
            },
            DataType::UINT16 => Ok(ScalarType::U16),
            DataType::UINT32 => Ok(ScalarType::U32),
            DataType::UINT64 => match self
                .read_str_attr("custom-data-type")
                .as_ref()
                .map(|x| x.as_str())
            {
                Ok("usize") => Ok(ScalarType::Usize),
                _ => Ok(ScalarType::U64),
            },
            DataType::INT8 => Ok(ScalarType::I8),
            DataType::INT16 => Ok(ScalarType::I16),
            DataType::INT32 => Ok(ScalarType::I32),
            DataType::INT64 => Ok(ScalarType::I64),
            DataType::FLOAT32 => Ok(ScalarType::F32),
            DataType::FLOAT64 => Ok(ScalarType::F64),
        }
    }
    fn shape(&self) -> Result<Shape> {
        todo!()
    }

    fn reshape(&self, shape: &Shape) -> Result<()> {
        todo!()
    }

    fn read_scalar<T: BackendData>(&self) -> Result<T> {
        fn read<T>(dataset: &Dataset) -> Result<Vec<T>>
        where
            T: ReflectedType,
            SliceDataBlock<T, Vec<T>>: ReadableDataBlock,
        {
            let path = dataset.path.to_string_lossy();
            let attr = &dataset.attributes;
            let data = dataset
                .root
                .read_block::<T>(&path, attr, smallvec![0])?
                .unwrap()
                .into_data();
            Ok(data)
        }

        let val = match T::DTYPE {
            ScalarType::U8 => read::<u8>(self)?[0].into_dyn(),
            ScalarType::U16 => read::<u16>(self)?[0].into_dyn(),
            ScalarType::U32 => read::<u32>(self)?[0].into_dyn(),
            ScalarType::U64 => read::<u64>(self)?[0].into_dyn(),
            ScalarType::Usize => (read::<u64>(self)?[0] as usize).into_dyn(),
            ScalarType::I8 => read::<i8>(self)?[0].into_dyn(),
            ScalarType::I16 => read::<i16>(self)?[0].into_dyn(),
            ScalarType::I32 => read::<i32>(self)?[0].into_dyn(),
            ScalarType::I64 => read::<i64>(self)?[0].into_dyn(),
            ScalarType::F32 => read::<f32>(self)?[0].into_dyn(),
            ScalarType::F64 => read::<f64>(self)?[0].into_dyn(),
            ScalarType::Bool => (read::<u8>(self)?[0] != 0).into_dyn(),
            ScalarType::String => {
                let data = read::<u8>(self)?;
                String::from_utf8(data).unwrap().into_dyn()
            }
        };
        BackendData::from_dyn(val)
    }

    fn read_array_slice<T: BackendData, S, E, D>(&self, selection: S) -> Result<Array<T, D>>
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
        D: Dimension,
    {
        todo!()
    }

    fn write_array_slice<'a, A, S, T, D, E>(&self, data: A, selection: S) -> Result<()>
    where
        A: Into<ArrayView<'a, T, D>>,
        T: BackendData,
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
        D: Dimension,
    {
        todo!()
    }
}

impl LocationOp for Location {
    type Backend = N5;

    fn file(&self) -> Result<<Self::Backend as Backend>::File> {
        todo!()
    }

    fn name(&self) -> PathBuf {
        self.path.file_name().unwrap().to_owned().into()
    }

    fn write_arr_attr<'a, A, D, Dim>(&self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension,
    {
        let path = self.child_path(name);
        match BackendData::into_dyn_arr(value.into()) {
            DynArrayView::U8(x) => {
                self.root
                    .set_attribute(&path.to_string_lossy(), name.to_string(), x.into_dyn())?;
            }
            _ => todo!(),
        }
        Ok(())
    }

    fn write_str_attr(&self, name: &str, value: &str) -> Result<()> {
        let val = Value::String(value.to_string());
        self.root
            .set_attribute(&self.path.to_string_lossy(), name.to_string(), val)?;
        Ok(())
    }

    fn read_str_attr(&self, name: &str) -> Result<String> {
        match self.root.get_attributes(&self.path.to_string_lossy())? {
            Value::Object(map) => match map.get(name) {
                Some(Value::String(s)) => Ok(s.clone()),
                _ => bail!("Attribute is not a string"),
            },
            _ => bail!("Attributes are not an object"),
        }
    }

    fn read_arr_attr<T: BackendData, D: Dimension>(&self, name: &str) -> Result<Array<T, D>> {
        todo!()
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Implementation through Deref
////////////////////////////////////////////////////////////////////////////////
impl GroupOp for Root {
    type Backend = N5;

    fn list(&self) -> Result<Vec<String>> {
        self.deref().list()
    }

    /// Create a new group.
    fn create_group(&self, name: &str) -> Result<<Self::Backend as Backend>::Group> {
        self.deref().create_group(name)
    }

    /// Open an existing group.
    fn open_group(&self, name: &str) -> Result<<Self::Backend as Backend>::Group> {
        self.deref().open_group(name)
    }

    /// Create an empty dataset holding an array value.
    fn new_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<<Self::Backend as Backend>::Dataset> {
        self.deref().new_dataset::<T>(name, shape, config)
    }

    fn open_dataset(&self, name: &str) -> Result<<Self::Backend as Backend>::Dataset> {
        self.deref().open_dataset(name)
    }

    /// Delete a group or dataset.
    fn delete(&self, name: &str) -> Result<()> {
        self.deref().delete(name)
    }

    /// Check if a group or dataset exists.
    fn exists(&self, name: &str) -> Result<bool> {
        self.deref().exists(name)
    }

    fn create_scalar_data<D: BackendData>(
        &self,
        name: &str,
        data: &D,
    ) -> Result<<Self::Backend as Backend>::Dataset> {
        self.deref().create_scalar_data(name, data)
    }

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
        self.deref().create_array_data(name, arr, config)
    }
}

impl LocationOp for Group {
    type Backend = N5;

    fn file(&self) -> Result<<Self::Backend as Backend>::File> {
        self.deref().file()
    }

    fn name(&self) -> PathBuf {
        self.deref().name()
    }

    fn write_arr_attr<'a, A, D, Dim>(&self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension,
    {
        self.deref().write_arr_attr(name, value)
    }

    fn write_str_attr(&self, name: &str, value: &str) -> Result<()> {
        self.deref().write_str_attr(name, value)
    }

    fn read_str_attr(&self, name: &str) -> Result<String> {
        self.deref().read_str_attr(name)
    }

    fn read_arr_attr<T: BackendData, D: Dimension>(&self, name: &str) -> Result<Array<T, D>> {
        self.deref().read_arr_attr(name)
    }
}

impl LocationOp for Dataset {
    type Backend = N5;

    fn file(&self) -> Result<<Self::Backend as Backend>::File> {
        self.deref().file()
    }

    fn name(&self) -> PathBuf {
        self.deref().name()
    }

    fn write_arr_attr<'a, A, D, Dim>(&self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension,
    {
        self.deref().write_arr_attr(name, value)
    }

    fn write_str_attr(&self, name: &str, value: &str) -> Result<()> {
        self.deref().write_str_attr(name, value)
    }

    fn read_str_attr(&self, name: &str) -> Result<String> {
        self.deref().read_str_attr(name)
    }

    fn read_arr_attr<T: BackendData, D: Dimension>(&self, name: &str) -> Result<Array<T, D>> {
        self.deref().read_arr_attr(name)
    }
}

fn write_array<'a, A, T, D, S, E>(container: &Dataset, data: A, selection: S) -> Result<()>
where
    A: Into<ArrayView<'a, T, D>>,
    T: BackendData,
    D: Dimension,
    S: AsRef<[E]>,
    E: AsRef<SelectInfoElem>,
{
    // TODO: check array dimension matches the selectioin

    macro_rules! impl_write {
        ($ty:ty, $data:expr, $fill:expr) => {{
            let path = container.path.to_string_lossy();
            let attr = container.attributes;
            if selection
                .as_ref()
                .into_iter()
                .all(|x| x.as_ref().is_slice())
            {
                let offsets = selection
                    .as_ref()
                    .into_iter()
                    .map(|x| match x.as_ref() {
                        SelectInfoElem::Slice(SliceInfoElem::Slice{ start, .. }) => *start as u64,
                        _ => unreachable!(),
                    })
                    .collect();
                container.root.write_ndarray::<$ty, _>(&path, &attr, offsets, $data, $fill)?;
                Ok(())
            } else {
                todo!()
            }
        }};
    }

    match BackendData::into_dyn_arr(data.into()) {
        DynArrayView::U16(x) => impl_write!(u16, x.into_dyn(), 0),
        _ => todo!(),

        /*
        DynArrayView::String(x) => {
            let data: Array<VarLenUnicode, D> = x.map(|x| x.parse().unwrap());
            write_array_impl(container, data.view(), selection)
        }
        */
    }
}

fn get_data_type<T: BackendData>() -> DataType {
    match T::DTYPE {
        ScalarType::I8 => DataType::INT8,
        ScalarType::I16 => DataType::INT16,
        ScalarType::I32 => DataType::INT32,
        ScalarType::I64 => DataType::INT64,
        ScalarType::U8 => DataType::UINT8,
        ScalarType::U16 => DataType::UINT16,
        ScalarType::U32 => DataType::UINT32,
        ScalarType::U64 => DataType::UINT64,
        ScalarType::Usize => DataType::UINT64,
        ScalarType::F32 => DataType::FLOAT32,
        ScalarType::F64 => DataType::FLOAT64,
        ScalarType::Bool => DataType::UINT8,
        ScalarType::String => DataType::UINT8,
    }
}

/// test module
#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_io<T: PartialEq + BackendData + std::fmt::Debug>(root: &Root, data: T) -> Result<()> {
        let root = N5::create("test.n5")?;
        let dataset = root.create_scalar_data("scalar", &data)?;
        assert_eq!(dataset.read_scalar::<T>()?, data);
        assert_eq!(dataset.dtype()?, T::DTYPE);
        Ok(())
    }

    fn scalar_attr_io(root: &Root, data: &str) {
        let root = N5::create("test.n5").unwrap();
        let dataset = root.create_scalar_data("scalar", &1u8).unwrap();
        dataset.write_str_attr("test", data).unwrap();
        assert_eq!(dataset.read_str_attr("test").unwrap().as_str(), data);
    }

    #[test]
    fn test_scalar() -> Result<()> {
        let root = N5::create("test.n5")?;

        scalar_io(&root, 10u8)?;
        scalar_io(&root, 10usize)?;
        scalar_io(&root, true)?;
        scalar_io(&root, "this is a test".to_string())?;
        Ok(())
    }

    #[test]
    fn test_scalar_attr() {
        let root = N5::create("test.n5").unwrap();

        scalar_attr_io(&root, "this is a test");
    }
}
