use anndata::{backend::*, data::{ArrayOp, BoundedSelectInfo, DynArray, DynScalar, SelectInfoElem, BoundedSelectInfoElem, Shape}};
use anndata::data::slice::BoundedSlice;

use zarrs::storage::{store::FilesystemStore, ListableStorageTraits, WritableStorageTraits};
use zarrs::group::Group;
use zarrs::array::{Element, data_type::DataType};
use std::{fs::File, str::FromStr, sync::Arc, vec};
use anyhow::{bail, Result};
use ndarray::{Array, Array2, ArrayD, ArrayView, Dimension, IxDyn, IxDynImpl, RemoveAxis, Slice, SliceInfoElem};
use std::{
    ops::{Index, Deref},
    path::{Path, PathBuf},
};

/// The Zarr backend.
pub struct Zarr;

pub struct ZarrStore(Arc<FilesystemStore>);

impl Deref for ZarrStore {
    type Target = FilesystemStore;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct ZarrGroup<T> {
    group: Group<T>,
    store: Arc<T>,
}

pub struct ZarrDataset<T> {
    dataset: zarrs::array::Array<T>,
    store: Arc<T>,
}

impl Backend for Zarr {
    const NAME: &'static str = "zarr";

    type Store = ZarrStore;

    type Group = ZarrGroup<FilesystemStore>;

    /// datasets contain arrays.
    type Dataset = ZarrDataset<FilesystemStore>;

    fn create<P: AsRef<Path>>(path: P) -> Result<Self::Store> {
        Ok(ZarrStore(Arc::new(FilesystemStore::new(path)?)))
    }

    /// Opens a file as read-only, file must exist.
    fn open<P: AsRef<Path>>(path: P) -> Result<Self::Store> {
        Ok(ZarrStore(Arc::new(FilesystemStore::new(path)?)))
    }

    /// Opens a file as read/write, file must exist.
    fn open_rw<P: AsRef<Path>>(path: P) -> Result<Self::Store> {
        Ok(ZarrStore(Arc::new(FilesystemStore::new(path)?)))
    }
}

impl StoreOp<Zarr> for ZarrStore {
    /// Returns the file path.
    fn filename(&self) -> PathBuf {
        self.key_to_fspath(&"".try_into().unwrap())
    }

    /// Close the file.
    fn close(self) -> Result<()> {
        todo!()
    }
}

impl GroupOp<Zarr> for ZarrStore {
    /// List all groups and datasets in this group.
    fn list(&self) -> Result<Vec<String>> {
        let result = self.list_dir(&"/".try_into().unwrap())?;
        Ok(result.keys().into_iter().map(|x| x.as_str().to_string()).chain(
            result.prefixes().into_iter().map(|x| x.as_str().to_string())
        ).collect())
    }

    /// Create a new group.
    fn create_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let path = if name.starts_with("/") {
            name.to_string()
        } else {
            format!("/{}", name)
        };
        let group = zarrs::group::GroupBuilder::new().build(self.0.clone(), &path)?;
        Ok(ZarrGroup {
            group,
            store: self.0.clone(),
        })
    }

    /// Open an existing group.
    fn open_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let group = zarrs::group::Group::new(self.0.clone(), name)?;
        Ok(ZarrGroup {
            group,
            store: self.0.clone(),
        })
    }

    /// Create an empty dataset holding an array value.
    fn new_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<<Zarr as Backend>::Dataset> {
        let shape = shape.as_ref();
        let sizes: Vec<u64> = match config.block_size {
            Some(s) => s.as_ref().into_iter().map(|x| (*x).max(1) as u64).collect(),
            _ => if shape.len() == 1 {
                vec![shape[0].min(10000).max(1) as u64]
            } else {
                shape.iter().map(|&x| x.min(100).max(1) as u64).collect()
            },
        };
        let chunk_size = zarrs::array::chunk_grid::ChunkGrid::new(
            zarrs::array::chunk_grid::regular::RegularChunkGrid::new(
                sizes.try_into().unwrap()
            )
        );
        
        let (datatype, fill) = match T::DTYPE {
            ScalarType::U8 => (DataType::UInt8, 0.into()),
            ScalarType::U16 => (DataType::UInt16, 0.into()),
            ScalarType::U32 => (DataType::UInt32, 0.into()),
            ScalarType::U64 => (DataType::UInt64, 0.into()),
            ScalarType::Usize => (DataType::UInt64, 0.into()),
            ScalarType::I8 => (DataType::Int8, 0.into()),
            ScalarType::I16 => (DataType::Int16, 0.into()),
            ScalarType::I32 => (DataType::Int32, 0.into()),
            ScalarType::I64 => (DataType::Int64, 0.into()),
            ScalarType::F32 => (DataType::Float32, zarrs::array::ZARR_NAN_F32.into()),
            ScalarType::F64 => (DataType::Float64, zarrs::array::ZARR_NAN_F64.into()),
            ScalarType::Bool => (DataType::Bool, false.into()),
            ScalarType::String => todo!(),
        };

        let array = zarrs::array::ArrayBuilder::new(
            shape.iter().map(|x| *x as u64).collect(),
            datatype,
            chunk_size,
            fill,
        ).build(self.0.clone(), name)?;
        array.store_metadata()?;
        Ok(ZarrDataset {
            dataset: array,
            store: self.0.clone(),
        })
    }

    fn open_dataset(&self, name: &str) -> Result<<Zarr as Backend>::Dataset> {
        let array = zarrs::array::Array::new(self.0.clone(), name)?;
        Ok(ZarrDataset {
            dataset: array,
            store: self.0.clone(),
        })
    }

    /// Delete a group or dataset.
    fn delete(&self, name: &str) -> Result<()> {
        self.0.erase(&name.try_into().unwrap())?;
        Ok(())
    }

    /// Check if a group or dataset exists.
    fn exists(&self, name: &str) -> Result<bool> {
        todo!()
    }
}

impl GroupOp<Zarr> for ZarrGroup<FilesystemStore> {
    fn list(&self) -> Result<Vec<String>> {
        let result = self.store.list_dir(
            &self.group.path().as_str().try_into().unwrap()
        )?;
        Ok(result.keys().into_iter().map(|x| x.as_str().to_string()).chain(
            result.prefixes().into_iter().map(|x| x.as_str().to_string())
        ).collect())
    }

    /// Create a new group.
    fn create_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let path = format!("{}/{}", self.group.path().as_str(), name);
        let group = zarrs::group::GroupBuilder::new().build(self.store.clone(), &path)?;
        Ok(ZarrGroup {
            group,
            store: self.store.clone(),
        })
    }

    /// Open an existing group.
    fn open_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let path = format!("{}/{}", self.group.path().as_str(), name);
        let group = zarrs::group::Group::new(self.store.clone(), &path)?;
        Ok(ZarrGroup {
            group,
            store: self.store.clone(),
        })
    }

    /// Create an empty dataset holding an array value.
    fn new_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<<Zarr as Backend>::Dataset> {
        let shape = shape.as_ref();
        let sizes: Vec<u64> = match config.block_size {
            Some(s) => s.as_ref().into_iter().map(|x| (*x).max(1) as u64).collect(),
            _ => if shape.len() == 1 {
                vec![shape[0].min(10000).max(1) as u64]
            } else {
                shape.iter().map(|&x| x.min(100).max(1) as u64).collect()
            },
        };
        let chunk_size = zarrs::array::chunk_grid::ChunkGrid::new(
            zarrs::array::chunk_grid::regular::RegularChunkGrid::new(
                sizes.try_into().unwrap()
            )
        );
        
        let (datatype, fill) = match T::DTYPE {
            ScalarType::U8 => (DataType::UInt8, 0u8.into()),
            ScalarType::U16 => (DataType::UInt16, 0u16.into()),
            ScalarType::U32 => (DataType::UInt32, 0u32.into()),
            ScalarType::U64 => (DataType::UInt64, 0u64.into()),
            ScalarType::Usize => (DataType::UInt64, 0u64.into()),
            ScalarType::I8 => (DataType::Int8, 0i8.into()),
            ScalarType::I16 => (DataType::Int16, 0i16.into()),
            ScalarType::I32 => (DataType::Int32, 0i32.into()),
            ScalarType::I64 => (DataType::Int64, 0i64.into()),
            ScalarType::F32 => (DataType::Float32, zarrs::array::ZARR_NAN_F32.into()),
            ScalarType::F64 => (DataType::Float64, zarrs::array::ZARR_NAN_F64.into()),
            ScalarType::Bool => (DataType::Bool, false.into()),
            ScalarType::String => todo!(),
        };

        let path = format!("{}/{}", self.group.path().as_str(), name);
        let array = zarrs::array::ArrayBuilder::new(
            shape.iter().map(|x| *x as u64).collect(),
            datatype,
            chunk_size,
            fill,
        ).build(self.store.clone(), &path)?;
        array.store_metadata()?;
        Ok(ZarrDataset {
            dataset: array,
            store: self.store.clone(),
        })
    }

    fn open_dataset(&self, name: &str) -> Result<<Zarr as Backend>::Dataset> {
        let path = format!("{}/{}", self.group.path().as_str(), name);
        let array = zarrs::array::Array::new(self.store.clone(), &path)?;
        Ok(ZarrDataset {
            dataset: array,
            store: self.store.clone(),
        })
    }

    /// Delete a group or dataset.
    fn delete(&self, name: &str) -> Result<()> {
        let path = format!("{}/{}", self.group.path().as_str(), name);
        self.store.erase(&path.as_str().try_into().unwrap())?;
        Ok(())
    }

    /// Check if a group or dataset exists.
    fn exists(&self, name: &str) -> Result<bool> {
        todo!()
    }
}

impl AttributeOp<Zarr> for ZarrGroup<FilesystemStore> {
    /// Returns the Root.
    fn store(&self) -> Result<<Zarr as Backend>::Store> {
        Ok(ZarrStore(self.store.clone()))
    }

    /// Returns the path of the location relative to the file root.
    fn path(&self) -> PathBuf {
        self.group.path().as_path().to_path_buf()
    }

    /// Write a array-like attribute at a given location.
    fn write_array_attr<'a, A, D, Dim>(&self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension,
    {
        todo!()
    }


    fn read_array_attr<T: BackendData, D: Dimension>(&self, name: &str) -> Result<Array<T, D>>
    {
        todo!()
    }
}


impl AttributeOp<Zarr> for ZarrDataset<FilesystemStore> {
    /// Returns the Root.
    fn store(&self) -> Result<<Zarr as Backend>::Store> {
        Ok(ZarrStore(self.store.clone()))
    }

    /// Returns the path of the location relative to the file root.
    fn path(&self) -> PathBuf {
        todo!()
    }

    /// Write a array-like attribute at a given location.
    fn write_array_attr<'a, A, D, Dim>(&self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension,
    {
        todo!()
    }


    fn read_array_attr<T: BackendData, D: Dimension>(&self, name: &str) -> Result<Array<T, D>>
    {
        todo!()
    }
}

impl DatasetOp<Zarr> for ZarrDataset<FilesystemStore> {
    fn dtype(&self) -> Result<ScalarType> {todo!()}
    fn shape(&self) -> Shape {todo!()}
    fn reshape(&self, shape: &Shape) -> Result<()> {todo!()}

    fn read_array_slice<T: BackendData, S, D>(&self, selection: &[S]) -> Result<Array<T, D>>
    where
        S: AsRef<SelectInfoElem>,
        D: Dimension,
    {
        todo!()
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
        D: Dimension,
    {
        fn write_array_impl<'a, T, D, S>(
            container: &ZarrDataset<T>,
            arr: ArrayView<'a, T, D>,
            selection: &[S],
        ) -> Result<()>
        where
            T: Element + WritableStorageTraits + 'static,
            D: Dimension,
            S: AsRef<SelectInfoElem>,
        {
            let subset_all = zarrs::array_subset::ArraySubset::new_with_shape(container.dataset.shape().to_vec());
            container.dataset.store_chunks_ndarray(&subset_all, select(arr.into(), selection))?;
            Ok(())
        }

        match BackendData::into_dyn_arr(data.into()) {
            DynArrayView::U8(x) => write_array_impl(self, x, selection),
            DynArrayView::U16(x) => write_array_impl(self, x, selection),
            DynArrayView::U32(x) => write_array_impl(self, x, selection),
            DynArrayView::U64(x) => write_array_impl(self, x, selection),
            DynArrayView::Usize(x) => write_array_impl(self, x, selection),
            DynArrayView::I8(x) => write_array_impl(self, x, selection),
            DynArrayView::I16(x) => write_array_impl(self, x, selection),
            DynArrayView::I32(x) => write_array_impl(self, x, selection),
            DynArrayView::I64(x) => write_array_impl(self, x, selection),
            DynArrayView::F32(x) => write_array_impl(self, x, selection),
            DynArrayView::F64(x) => write_array_impl(self, x, selection),
            DynArrayView::Bool(x) => write_array_impl(self, x, selection),
            DynArrayView::String(x) => {
                todo!()
            }
        }
    }
}

fn select<'a, S, T, D>(arr: ArrayView<'a, T, D>, info: &[S]) -> Array<T, D>
where
    S: AsRef<SelectInfoElem>,
    T: Clone,
    D: Dimension,
{
    let arr = arr.into_dyn();
    let slices = info.as_ref().into_iter().map(|x| match x.as_ref() {
        SelectInfoElem::Slice(slice) => Some(SliceInfoElem::from(slice.clone())),
        _ => None,
    }).collect::<Option<Vec<_>>>();
    if let Some(slices) = slices {
        arr.slice(slices.as_slice()).into_owned()
    } else {
        let shape = arr.shape();
        let select: Vec<_> = info.as_ref().into_iter().zip(shape)
            .map(|(x, n)| BoundedSelectInfoElem::new(x.as_ref(), *n)).collect();
        let new_shape = select.iter().map(|x| x.len()).collect::<Vec<_>>();
        ArrayD::from_shape_fn(new_shape, |idx| {
            let new_idx: Vec<_> = (0..idx.ndim()).into_iter().map(|i| select[i].index(idx[i])).collect();
            arr.index(new_idx.as_slice()).clone()
        })
    }.into_dimensionality::<D>().unwrap()
}

/// test module
#[cfg(test)]
mod tests {
    use super::*;
    use anndata::s;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use ndarray::{Array1, Axis, concatenate, Ix1};
    use std::path::PathBuf;
    use tempfile::tempdir;

    pub fn with_tmp_dir<T, F: FnMut(PathBuf) -> T>(mut func: F) -> T {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();
        func(path)
    }

    fn with_tmp_path<T, F: Fn(PathBuf) -> T>(func: F) -> T {
        with_tmp_dir(|dir| func(dir.join("temp")))
    }

    #[test]
    fn test_basic() -> Result<()> {
        with_tmp_path(|path| {
            let store = ZarrStore(Arc::new(FilesystemStore::new(&path)?));
            let group = store.create_group("group")?;
            let subgroup = group.create_group("subgroup")?;

            assert_eq!(subgroup.path(), PathBuf::from("/group/subgroup"));
            Ok(())
        })
    }

    #[test]
    fn test_write_empty() -> Result<()> {
        with_tmp_path(|path| {
            let store = ZarrStore(Arc::new(FilesystemStore::new(&path)?));
            let group = store.create_group("group")?;
            let config = WriteConfig {
                ..Default::default()
            };

            let empty = Array1::<u8>::from_vec(Vec::new());
            let dataset = group.create_array_data("test", &empty, config)?;
            assert_eq!(empty, dataset.read_array::<u8, Ix1>()?);
            Ok(())
        })
    }
}