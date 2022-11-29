use crate::{
    backend::{
        Backend, BackendData, DatasetOp, FileOp, GroupOp, LocationOp, ScalarType, Selection, DynArrayView,
    },
    data::DynScalar,
};

use anyhow::{bail, Result};
use hdf5::{
    dataset::Dataset,
    types::{FloatSize, TypeDescriptor, VarLenAscii, VarLenUnicode},
    types::IntSize::*, 
    File, Group, H5Type, Location,
};
use ndarray::{Array, Array2, ArrayView, Dimension};
use polars::prelude::VarAggSeries;
use std::{
    fmt::write,
    ops::Deref,
    path::{Path, PathBuf},
    sync::Arc,
};

pub struct H5;

impl Backend for H5 {
    type File = File;

    type Group = Group;

    /// datasets contain arrays.
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

    fn write_str_attr(&self, name: &str, value: &str) -> Result<()> {
        let value_: VarLenUnicode = value.parse().unwrap();
        let attr = self
            .attr(name)
            .or(self.new_attr::<VarLenUnicode>().create(name))?;
        attr.write_scalar(&value_)?;
        Ok(())
    }

    fn write_str_arr_attr<'a, A, D>(&self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, String, D>>,
        D: ndarray::Dimension,
    {
        let array_view = value.into();
        let value_ = Array::<VarLenUnicode, D>::from_shape_vec(
            array_view.dim(),
            array_view.iter().map(|x| x.parse().unwrap()).collect(),
        )?;
        let attr = self
            .attr(name)
            .or(self.new_attr::<VarLenUnicode>().create(name))?;
        attr.write(&value_)?;
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

    fn read_str_arr_attr<D>(&self, name: &str) -> Result<Array<String, D>> {
        todo!()
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

    fn write_str_attr(&self, name: &str, value: &str) -> Result<()> {
        let value_: VarLenUnicode = value.parse().unwrap();
        let attr = self
            .attr(name)
            .or(self.new_attr::<VarLenUnicode>().create(name))?;
        attr.write_scalar(&value_)?;
        Ok(())
    }
    fn write_str_arr_attr<'a, A, D>(&self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, String, D>>,
        D: Dimension,
    {
        let array_view = value.into();
        let value_ = Array::<VarLenUnicode, D>::from_shape_vec(
            array_view.dim(),
            array_view.iter().map(|x| x.parse().unwrap()).collect(),
        )?;
        let attr = self
            .attr(name)
            .or(self.new_attr::<VarLenUnicode>().create(name))?;
        attr.write(&value_)?;
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

    fn read_str_arr_attr<D>(&self, name: &str) -> Result<Array<String, D>> {
        todo!()
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

    fn write_array<'a, A, S, D, Dim>(
        &self,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<<Self::Backend as Backend>::Dataset>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        S: Into<Selection>,
        Dim: Dimension,
    {
        write_array(self, name, data, selection)
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

    fn write_array<'a, A, S, D, Dim>(
        &self,
        name: &str,
        data: A,
        selection: S,
    ) -> Result<<Self::Backend as Backend>::Dataset>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        S: Into<Selection>,
        Dim: Dimension,
    {
        write_array(self, name, data, selection)
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

    fn shape(&self) -> Result<Vec<usize>> {
        Ok(hdf5::Container::shape(self))
    }

    fn read_scalar<T: BackendData>(&self) -> Result<T> {
        read_scalar(self)
    }

    fn read_array<T: BackendData, S, D>(&self, selection: S) -> Result<Array<T, D>>
    where
        S: Into<Selection>,
    {
        todo!()
    }
}

fn read_scalar<T: BackendData>(dataset: &Dataset) -> Result<T> {
    let val = match T::DTYPE {
        ScalarType::Bool => hdf5::Container::read_scalar::<bool>(dataset)?.into_dyn(),
        ScalarType::U8 => hdf5::Container::read_scalar::<u8>(dataset)?.into_dyn(),
        ScalarType::U16 => hdf5::Container::read_scalar::<u16>(dataset)?.into_dyn(),
        ScalarType::U32 => hdf5::Container::read_scalar::<u32>(dataset)?.into_dyn(),
        ScalarType::U64 => hdf5::Container::read_scalar::<u64>(dataset)?.into_dyn(),
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

fn write_array<'a, A, S, D, Dim>(
    group: &Group,
    name: &str,
    data: A,
    selection: S,
) -> Result<Dataset>
where
    A: Into<ArrayView<'a, D, Dim>>,
    D: BackendData,
    S: Into<Selection>,
    Dim: Dimension,
{
    fn write_array_impl<'a, S, D, Dim>(
        group: &Group,
        name: &str,
        arr: ArrayView<'a, D, Dim>,
        selection: S,
    ) -> Result<Dataset>
    where
        D: H5Type,
        S: Into<Selection>,
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
        DynArrayView::U8(x) => write_array_impl(group, name, x, selection),
        DynArrayView::U16(x) => write_array_impl(group, name, x, selection),
        DynArrayView::U32(x) => write_array_impl(group, name, x, selection),
        DynArrayView::U64(x) => write_array_impl(group, name, x, selection),
        DynArrayView::I8(x) => write_array_impl(group, name, x, selection),
        DynArrayView::I16(x) => write_array_impl(group, name, x, selection),
        DynArrayView::I32(x) => write_array_impl(group, name, x, selection),
        DynArrayView::I64(x) => write_array_impl(group, name, x, selection),
        DynArrayView::F32(x) => write_array_impl(group, name, x, selection),
        DynArrayView::F64(x) => write_array_impl(group, name, x, selection),
        DynArrayView::Bool(x) => write_array_impl(group, name, x, selection),
        DynArrayView::String(x) => {
            let data = Array::<VarLenUnicode, Dim>::from_shape_vec(
                x.dim(),
                x.iter().map(|x| x.parse().unwrap()).collect(),
            )?;
            write_array_impl(group, name, data.view(), selection)
        }
    }
}
