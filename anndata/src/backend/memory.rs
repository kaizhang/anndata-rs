use crate::{
    backend::{
        Backend, BackendData, DatasetOp, DynArrayView, FileOp, GroupOp, LocationOp, ScalarType,
        SelectInfoElem,
    },
    data::{BoundedSelectInfo, DynArray, DynScalar, Shape, ArrayOp},
};

use anyhow::{bail, Result};
use ndarray::{Array, ArrayView, Dimension, SliceInfo};
use std::path::{Path, PathBuf};

/// In-memory backend.
pub struct Mem;

pub struct File;
pub struct Group;
pub struct Dataset;

impl Backend for Mem {
    type File = File;

    type Group = Group;

    type Dataset = Dataset;

    fn create<P: AsRef<Path>>(path: P) -> Result<Self::File> {
        Ok(File::create(path)?)
    }
}

impl FileOp for File {
    type Backend = H5;

    fn filename(&self) -> PathBuf {
        hdf5::Location::filename(&self).into()
    }

    fn close(self) -> Result<()> {
        Ok(self.close()?)
    }
}

impl LocationOp for Group {
    type Backend = H5;

    fn file(&self) -> Result<<Self::Backend as Backend>::File> {
        Ok(hdf5::Location::file(&self)?)
    }

    fn path(&self) -> PathBuf {
        hdf5::Location::name(&self).into()
    }

    fn write_arr_attr<'a, A, D, Dim>(&self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension,
    {
        write_arr_attr(self, name, value)
    }

    fn write_str_attr(&self, name: &str, value: &str) -> Result<()> {
        let value_: VarLenUnicode = value.parse().unwrap();
        let attr = self
            .attr(name)
            .or(self.new_attr::<VarLenUnicode>().create(name))?;
        attr.write_scalar(&value_)?;
        Ok(())
    }

    fn read_str_attr(&self, name: &str) -> Result<String> {
        let container = self.attr(name)?;
        match container.dtype()?.to_descriptor()? {
            TypeDescriptor::VarLenAscii => {
                let attr: VarLenAscii = container.read_scalar()?;
                Ok(attr.parse().unwrap())
            }
            TypeDescriptor::VarLenUnicode => {
                let attr: VarLenUnicode = container.read_scalar()?;
                Ok(attr.parse().unwrap())
            }
            ty => {
                panic!("Cannot read string from type '{}'", ty);
            }
        }
    }

    fn read_arr_attr<T: BackendData, D: Dimension>(&self, name: &str) -> Result<Array<T, D>> {
        read_arr_attr(self, name)
    }
}

impl LocationOp for Dataset {
    type Backend = H5;

    fn file(&self) -> Result<<Self::Backend as Backend>::File> {
        Ok(hdf5::Location::file(&self)?)
    }

    fn path(&self) -> PathBuf {
        hdf5::Location::name(&self).into()
    }

    fn write_arr_attr<'a, A, D, Dim>(&self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension,
    {
        write_arr_attr(self, name, value)
    }

    fn write_str_attr(&self, name: &str, value: &str) -> Result<()> {
        let value_: VarLenUnicode = value.parse().unwrap();
        let attr = self
            .attr(name)
            .or(self.new_attr::<VarLenUnicode>().create(name))?;
        attr.write_scalar(&value_)?;
        Ok(())
    }

    fn read_str_attr(&self, name: &str) -> Result<String> {
        let container = self.attr(name)?;
        match container.dtype()?.to_descriptor()? {
            TypeDescriptor::VarLenAscii => {
                let attr: VarLenAscii = container.read_scalar()?;
                Ok(attr.parse().unwrap())
            }
            TypeDescriptor::VarLenUnicode => {
                let attr: VarLenUnicode = container.read_scalar()?;
                Ok(attr.parse().unwrap())
            }
            ty => {
                panic!("Cannot read string from type '{}'", ty);
            }
        }
    }

    fn read_arr_attr<T: BackendData, D: Dimension>(&self, name: &str) -> Result<Array<T, D>> {
        read_arr_attr(self, name)
    }
}

impl GroupOp for Group {
    type Backend = H5;

    fn list(&self) -> Result<Vec<String>> {
        Ok(self.member_names()?)
    }

    fn create_group(&self, name: &str) -> Result<<Self::Backend as Backend>::Group> {
        Ok(hdf5::Group::create_group(self, name)?)
    }

    fn open_group(&self, name: &str) -> Result<<Self::Backend as Backend>::Group> {
        Ok(self.group(name)?)
    }

    fn open_dataset(&self, name: &str) -> Result<<Self::Backend as Backend>::Dataset> {
        Ok(self.dataset(name)?)
    }

    fn delete(&self, name: &str) -> Result<()> {
        Ok(self.unlink(name)?)
    }

    fn exists(&self, name: &str) -> Result<bool> {
        Ok(self.link_exists(name))
    }

    fn write_scalar<D: BackendData>(
        &self,
        name: &str,
        data: &D,
    ) -> Result<<Self::Backend as Backend>::Dataset> {
        write_scalar(self, name, data)
    }

    fn write_array<'a, A, D, Dim>(
        &self,
        name: &str,
        data: A,
    ) -> Result<<Self::Backend as Backend>::Dataset>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension,
    {
        write_array(self, name, data)
    }
}

impl GroupOp for File {
    type Backend = H5;

    fn list(&self) -> Result<Vec<String>> {
        Ok(self.member_names()?)
    }

    fn create_group(&self, name: &str) -> Result<<Self::Backend as Backend>::Group> {
        Ok(hdf5::Group::create_group(self, name)?)
    }

    fn open_group(&self, name: &str) -> Result<<Self::Backend as Backend>::Group> {
        Ok(self.group(name)?)
    }

    fn open_dataset(&self, name: &str) -> Result<<Self::Backend as Backend>::Dataset> {
        Ok(self.dataset(name)?)
    }

    fn delete(&self, name: &str) -> Result<()> {
        Ok(self.unlink(name)?)
    }

    fn exists(&self, name: &str) -> Result<bool> {
        Ok(self.link_exists(name))
    }

    fn write_scalar<D: BackendData>(
        &self,
        name: &str,
        data: &D,
    ) -> Result<<Self::Backend as Backend>::Dataset> {
        write_scalar(self, name, data)
    }

    fn write_array<'a, A, D, Dim>(
        &self,
        name: &str,
        data: A,
    ) -> Result<<Self::Backend as Backend>::Dataset>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension,
    {
        write_array(self, name, data)
    }
}

impl DatasetOp for Dataset {
    type Backend = H5;

    fn dtype(&self) -> Result<ScalarType> {
        let ty = match hdf5::Container::dtype(self)?.to_descriptor()? {
            TypeDescriptor::Unsigned(U1) => ScalarType::U8,
            TypeDescriptor::Unsigned(U2) => ScalarType::U16,
            TypeDescriptor::Unsigned(U4) => ScalarType::U32,
            TypeDescriptor::Unsigned(U8) => ScalarType::U64,
            TypeDescriptor::Integer(U1) => ScalarType::I8,
            TypeDescriptor::Integer(U2) => ScalarType::I16,
            TypeDescriptor::Integer(U4) => ScalarType::I32,
            TypeDescriptor::Integer(U8) => ScalarType::I64,
            TypeDescriptor::Float(FloatSize::U4) => ScalarType::F32,
            TypeDescriptor::Float(FloatSize::U8) => ScalarType::F64,
            TypeDescriptor::Boolean => ScalarType::Bool,
            TypeDescriptor::VarLenAscii => ScalarType::String,
            TypeDescriptor::VarLenUnicode => ScalarType::String,
            ty => bail!("Unsupported type: {:?}", ty),
        };
        Ok(ty)
    }

    fn shape(&self) -> Result<Shape> {
        Ok(hdf5::Container::shape(self).into())
    }

    fn read_scalar<T: BackendData>(&self) -> Result<T> {
        read_scalar(self)
    }

    fn read_array_slice<T: BackendData, S, E, D>(&self, selection: S) -> Result<Array<T, D>>
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
        D: Dimension,
    {
        read_array(self, selection)
    }
}

fn read_scalar<T: BackendData>(dataset: &Dataset) -> Result<T> {
    let val = match T::DTYPE {
        ScalarType::Bool => hdf5::Container::read_scalar::<bool>(dataset)?.into_dyn(),
        ScalarType::U8 => hdf5::Container::read_scalar::<u8>(dataset)?.into_dyn(),
        ScalarType::U16 => hdf5::Container::read_scalar::<u16>(dataset)?.into_dyn(),
        ScalarType::U32 => hdf5::Container::read_scalar::<u32>(dataset)?.into_dyn(),
        ScalarType::U64 => hdf5::Container::read_scalar::<u64>(dataset)?.into_dyn(),
        ScalarType::Usize => hdf5::Container::read_scalar::<usize>(dataset)?.into_dyn(),
        ScalarType::I8 => hdf5::Container::read_scalar::<i8>(dataset)?.into_dyn(),
        ScalarType::I16 => hdf5::Container::read_scalar::<i16>(dataset)?.into_dyn(),
        ScalarType::I32 => hdf5::Container::read_scalar::<i32>(dataset)?.into_dyn(),
        ScalarType::I64 => hdf5::Container::read_scalar::<i64>(dataset)?.into_dyn(),
        ScalarType::F32 => hdf5::Container::read_scalar::<f32>(dataset)?.into_dyn(),
        ScalarType::F64 => hdf5::Container::read_scalar::<f64>(dataset)?.into_dyn(),
        ScalarType::String => {
            let s = hdf5::Container::read_scalar::<VarLenUnicode>(dataset)?;
            s.to_string().into_dyn()
        }
    };
    BackendData::from_dyn(val)
}

fn read_arr_attr<T, D>(loc: &Location, name: &str) -> Result<Array<T, D>>
where
    T: BackendData,
    D: Dimension,
{
    let array: DynArray = match T::DTYPE {
        ScalarType::I8 => loc.attr(name)?.read::<i8, D>()?.into(),
        ScalarType::I16 => loc.attr(name)?.read::<i16, D>()?.into(),
        ScalarType::I32 => loc.attr(name)?.read::<i32, D>()?.into(),
        ScalarType::I64 => loc.attr(name)?.read::<i64, D>()?.into(),
        ScalarType::U8 => loc.attr(name)?.read::<u8, D>()?.into(),
        ScalarType::U16 => loc.attr(name)?.read::<u16, D>()?.into(),
        ScalarType::U32 => loc.attr(name)?.read::<u32, D>()?.into(),
        ScalarType::U64 => loc.attr(name)?.read::<u64, D>()?.into(),
        ScalarType::Usize => loc.attr(name)?.read::<usize, D>()?.into(),
        ScalarType::F32 => loc.attr(name)?.read::<f32, D>()?.into(),
        ScalarType::F64 => loc.attr(name)?.read::<f64, D>()?.into(),
        ScalarType::Bool => loc.attr(name)?.read::<bool, D>()?.into(),
        ScalarType::String => {
            let s = loc.attr(name)?.read::<VarLenUnicode, D>()?;
            s.map(|s| s.to_string()).into()
        }
    };
    Ok(BackendData::from_dyn_arr(array)?.into_dimensionality::<D>()?)
}

fn read_array<T, S, E, D>(dataset: &Dataset, selection: S) -> Result<Array<T, D>>
where
    T: BackendData,
    S: AsRef<[E]>,
    E: AsRef<SelectInfoElem>,
    D: Dimension,
{
    fn read_arr<T, S, E, D>(dataset: &Dataset, selection: S) -> Result<Array<T, D>>
    where
        T: H5Type + BackendData,
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
        D: Dimension,
    {
        if selection.as_ref().iter().any(|x| x.as_ref().is_index()) { // fancy indexing is too slow, just read all
            let arr = hdf5::Container::read::<T, D>(dataset)?;
            Ok(ArrayOp::select(&arr, selection))
        } else {
            let (select, shape) = into_selection(selection, dataset.shape()?);
            if matches!(select, Selection::Points(_)) {
                let slice_1d = hdf5::Container::read_slice_1d::<T, _>(dataset, select)?;
                Ok(slice_1d.into_shape(shape.as_ref())?.into_dimensionality::<D>()?)
            } else {
                Ok(hdf5::Container::read_slice::<T, _, D>(dataset, select)?)
            }
        }
    }

    let array: DynArray = match T::DTYPE {
        ScalarType::I8 => read_arr::<i8, _, _, D>(dataset, selection)?.into(),
        ScalarType::I16 => read_arr::<i16, _, _, D>(dataset, selection)?.into(),
        ScalarType::I32 => read_arr::<i32, _, _, D>(dataset, selection)?.into(),
        ScalarType::I64 => read_arr::<i64, _, _, D>(dataset, selection)?.into(),
        ScalarType::U8 => read_arr::<u8, _, _, D>(dataset, selection)?.into(),
        ScalarType::U16 => read_arr::<u16, _, _, D>(dataset, selection)?.into(),
        ScalarType::U32 => read_arr::<u32, _, _, D>(dataset, selection)?.into(),
        ScalarType::U64 => read_arr::<u64, _, _, D>(dataset, selection)?.into(),
        ScalarType::Usize => read_arr::<usize, _, _, D>(dataset, selection)?.into(),
        ScalarType::F32 => read_arr::<f32, _, _, D>(dataset, selection)?.into(),
        ScalarType::F64 => read_arr::<f64, _, _, D>(dataset, selection)?.into(),
        ScalarType::Bool => read_arr::<bool, _, _, D>(dataset, selection)?.into(),
        ScalarType::String => {
            if selection.as_ref().iter().any(|x| x.as_ref().is_index()) { // fancy indexing is too slow, just read all
                let arr = hdf5::Container::read::<VarLenUnicode, D>(dataset)?;
                let arr_ = arr.map(|s| s.to_string());
                let r: Result<_> = Ok(ArrayOp::select(&arr_, selection));
                r
            } else {
                let (select, shape) = into_selection(selection, dataset.shape()?);
                let arr: Result<_> = if matches!(select, Selection::Points(_)) {
                    let slice_1d = hdf5::Container::read_slice_1d::<VarLenUnicode, _>(dataset, select)?;
                    Ok(slice_1d.into_shape(shape.as_ref())?.into_dimensionality::<D>()?)
                } else {
                    Ok(hdf5::Container::read_slice::<VarLenUnicode, _, D>(dataset, select)?)
                };
                Ok(arr?.map(|s| s.to_string()))
            }?.into()
            /*
            let arr = read_arr::<VarLenUnicode, _, _, D>(dataset, selection)?;
            let arr = arr.map(|s| s.to_string());
            arr.into()
            */
        },
    };
    Ok(BackendData::from_dyn_arr(array)?.into_dimensionality::<D>()?)
}

fn write_scalar<D: BackendData>(group: &Group, name: &str, data: &D) -> Result<Dataset> {
    match data.into_dyn() {
        DynScalar::U8(x) => {
            let dataset = group.new_dataset::<u8>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::U16(x) => {
            let dataset = group.new_dataset::<u16>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::U32(x) => {
            let dataset = group.new_dataset::<u32>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::U64(x) => {
            let dataset = group.new_dataset::<u64>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::Usize(x) => {
            let dataset = group.new_dataset::<usize>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::I8(x) => {
            let dataset = group.new_dataset::<i8>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::I16(x) => {
            let dataset = group.new_dataset::<i16>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::I32(x) => {
            let dataset = group.new_dataset::<i32>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::I64(x) => {
            let dataset = group.new_dataset::<i64>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::F32(x) => {
            let dataset = group.new_dataset::<f32>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::F64(x) => {
            let dataset = group.new_dataset::<f64>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::Bool(x) => {
            let dataset = group.new_dataset::<bool>().create(name)?;
            dataset.write_scalar(&x)?;
            Ok(dataset)
        }
        DynScalar::String(x) => {
            let dataset = group.new_dataset::<VarLenUnicode>().create(name)?;
            dataset.write_scalar(&x.parse::<VarLenUnicode>().unwrap())?;
            Ok(dataset)
        }
    }
}

fn write_arr_attr<'a, A, D, Dim>(loc: &Location, name: &str, data: A) -> Result<()>
where
    A: Into<ArrayView<'a, D, Dim>>,
    D: BackendData,
    Dim: Dimension,
{
    match BackendData::into_dyn_arr(data.into()) {
        DynArrayView::U8(x) => loc.new_attr_builder().with_data(x).create(name)?,
        DynArrayView::U16(x) => loc.new_attr_builder().with_data(x).create(name)?,
        DynArrayView::U32(x) => loc.new_attr_builder().with_data(x).create(name)?,
        DynArrayView::U64(x) => loc.new_attr_builder().with_data(x).create(name)?,
        DynArrayView::Usize(x) => loc.new_attr_builder().with_data(x).create(name)?,
        DynArrayView::I8(x) => loc.new_attr_builder().with_data(x).create(name)?,
        DynArrayView::I16(x) => loc.new_attr_builder().with_data(x).create(name)?,
        DynArrayView::I32(x) => loc.new_attr_builder().with_data(x).create(name)?,
        DynArrayView::I64(x) => loc.new_attr_builder().with_data(x).create(name)?,
        DynArrayView::F32(x) => loc.new_attr_builder().with_data(x).create(name)?,
        DynArrayView::F64(x) => loc.new_attr_builder().with_data(x).create(name)?,
        DynArrayView::Bool(x) => loc.new_attr_builder().with_data(x).create(name)?,
        DynArrayView::String(x) => {
            let data: Array<VarLenUnicode, Dim> = x.map(|x| x.parse().unwrap());
            loc.new_attr_builder().with_data(data.view()).create(name)?
        }
    };
    Ok(())
}



fn write_array<'a, A, D, Dim>(group: &Group, name: &str, data: A) -> Result<Dataset>
where
    A: Into<ArrayView<'a, D, Dim>>,
    D: BackendData,
    Dim: Dimension,
{
    fn write_array_impl<'a, D, Dim>(
        group: &Group,
        name: &str,
        arr: ArrayView<'a, D, Dim>,
    ) -> Result<Dataset>
    where
        D: H5Type,
        Dim: Dimension,
    {
        let shape = arr.shape();
        let chunk_size = if shape.len() == 1 {
            vec![shape[0].min(100000)]
        } else {
            shape.iter().map(|&x| x.min(100)).collect()
        };
        let dataset = if arr.len() > 100 {
            group
                .new_dataset_builder()
                .deflate(3)
                .chunk(chunk_size)
                .with_data(arr)
                .create(name)?
        } else {
            group.new_dataset_builder().with_data(arr).create(name)?
        };
        Ok(dataset)
    }

    match BackendData::into_dyn_arr(data.into()) {
        DynArrayView::U8(x) => write_array_impl(group, name, x),
        DynArrayView::U16(x) => write_array_impl(group, name, x),
        DynArrayView::U32(x) => write_array_impl(group, name, x),
        DynArrayView::U64(x) => write_array_impl(group, name, x),
        DynArrayView::Usize(x) => write_array_impl(group, name, x),
        DynArrayView::I8(x) => write_array_impl(group, name, x),
        DynArrayView::I16(x) => write_array_impl(group, name, x),
        DynArrayView::I32(x) => write_array_impl(group, name, x),
        DynArrayView::I64(x) => write_array_impl(group, name, x),
        DynArrayView::F32(x) => write_array_impl(group, name, x),
        DynArrayView::F64(x) => write_array_impl(group, name, x),
        DynArrayView::Bool(x) => write_array_impl(group, name, x),
        DynArrayView::String(x) => {
            let data: Array<VarLenUnicode, Dim> = x.map(|x| x.parse().unwrap());
            write_array_impl(group, name, data.view())
        }
    }
}

fn into_selection<S, E>(selection: S, shape: Shape) -> (Selection, Shape)
where
    S: AsRef<[E]>,
    E: AsRef<SelectInfoElem>,
{
    if selection.as_ref().into_iter().all(|x| x.as_ref().is_full()) {
        (Selection::All, shape)
    } else {
        let bounded_selection = BoundedSelectInfo::new(&selection, &shape).unwrap();
        let out_shape = bounded_selection.out_shape();
        if let Some(idx) = bounded_selection.try_into_indices() {
            (Selection::from(idx), out_shape)
        } else {
            let slice: SliceInfo<_, _, _> = bounded_selection.try_into().unwrap();
            (Selection::try_from(slice).unwrap(), out_shape)
        }
    }
}