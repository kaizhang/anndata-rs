use anndata::{
    backend::{
        Backend, BackendData, DatasetOp, DynArrayView, FileOp, GroupOp, LocationOp, ScalarType,
        WriteConfig,
    },
    data::{ArrayOp, BoundedSelectInfo, DynArray, DynScalar, SelectInfoElem, Shape},
};

use anyhow::{bail, Result, Ok};
use hdf5::{
    dataset::Dataset,
    types::IntSize::*,
    types::{FloatSize, TypeDescriptor, VarLenAscii, VarLenUnicode},
    File, Group, H5Type, Location, Selection,
};
use ndarray::{Array, ArrayView, Dimension, SliceInfo};
use std::ops::Deref;
use std::path::{Path, PathBuf};

///////////////////////////////////////////////////////////////////////////////
/// Type definitions
///////////////////////////////////////////////////////////////////////////////

pub struct H5;

pub struct H5File(File);

impl Deref for H5File {
    type Target = File;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct H5Group(Group);

impl Deref for H5Group {
    type Target = Group;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct H5Dataset(Dataset);

impl Deref for H5Dataset {
    type Target = Dataset;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

///////////////////////////////////////////////////////////////////////////////
/// Backend implementation
///////////////////////////////////////////////////////////////////////////////

impl Backend for H5 {
    const NAME: &'static str = "hdf5";

    type File = H5File;

    type Group = H5Group;

    /// datasets contain arrays.
    type Dataset = H5Dataset;

    fn create<P: AsRef<Path>>(path: P) -> Result<Self::File> {
        Ok(H5File(File::create(path)?))
    }

    /// Opens a file as read-only, file must exist.
    fn open<P: AsRef<Path>>(path: P) -> Result<Self::File> {
        Ok(File::open(path).map(H5File)?)
    }

    /// Opens a file as read/write, file must exist.
    fn open_rw<P: AsRef<Path>>(path: P) -> Result<Self::File> {
        Ok(File::open_rw(path).map(H5File)?)
    }
}

impl FileOp for H5File {
    type Backend = H5;

    fn filename(&self) -> PathBuf {
        hdf5::Location::filename(&self).into()
    }

    fn close(self) -> Result<()> {
        Ok(self.0.close()?)
    }
}

// Generic GroupOp implementations

fn list(group: &Group) -> Result<Vec<String>> {
    Ok(group.member_names()?)
}

fn create_group(group: &Group, name: &str) -> Result<H5Group> {
    Ok(H5Group(group.create_group(name)?))
}

fn open_group(group: &Group, name: &str) -> Result<H5Group> {
    Ok(H5Group(group.group(name)?))
}

fn new_dataset<T: BackendData>(
    group: &Group,
    name: &str,
    shape: &Shape,
    config: WriteConfig,
) -> Result<H5Dataset> {
    let mut builder = match T::DTYPE {
        ScalarType::U8 => group.new_dataset::<u8>(),
        ScalarType::U16 => group.new_dataset::<u16>(),
        ScalarType::U32 => group.new_dataset::<u32>(),
        ScalarType::U64 => group.new_dataset::<u64>(),
        ScalarType::Usize => group.new_dataset::<usize>(),
        ScalarType::I8 => group.new_dataset::<i8>(),
        ScalarType::I16 => group.new_dataset::<i16>(),
        ScalarType::I32 => group.new_dataset::<i32>(),
        ScalarType::I64 => group.new_dataset::<i64>(),
        ScalarType::F32 => group.new_dataset::<f32>(),  
        ScalarType::F64 => group.new_dataset::<f64>(),
        ScalarType::Bool => group.new_dataset::<bool>(),
        ScalarType::String => group.new_dataset::<VarLenUnicode>(),
    };

    builder = if let Some(compression) = config.compression {
        //builder.deflate(compression)
        builder.blosc_blosclz(compression, hdf5::filters::BloscShuffle::Byte)
    } else {
        builder
    };

    builder = if let Some(s) = config.block_size {
        if s.as_ref().iter().all(|&x| x > 0) {
            builder.chunk(s.as_ref())
        } else {
            builder
        }
    } else {
        builder
    };

    let s: hdf5::Extents = hdf5::SimpleExtents::resizable(shape.as_ref()).into();
    let dataset = builder.shape(s).create(name)?;
    Ok(H5Dataset(dataset))
}

fn open_dataset(group: &Group, name: &str) -> Result<H5Dataset> {
    Ok(H5Dataset(group.dataset(name)?))
}

fn delete(group: &Group, name: &str) -> Result<()> {
    Ok(group.unlink(name)?)
}

fn exists(group: &Group, name: &str) -> Result<bool> {
    Ok(group.link_exists(name))
}

fn create_scalar_data<D: BackendData>(group: &Group, name: &str, data: &D) -> Result<H5Dataset> {
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
    .map(H5Dataset)
}

impl DatasetOp for H5Dataset {
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

    fn shape(&self) -> Shape {
        hdf5::Container::shape(self).into()
    }

    fn reshape(&self, shape: &Shape) -> Result<()> {
        Ok(Dataset::resize(self, shape.as_ref())?)
    }

    fn read_scalar<T: BackendData>(&self) -> Result<T> {
        let val = match T::DTYPE {
            ScalarType::Bool => self.deref().read_scalar::<bool>()?.into_dyn(),
            ScalarType::U8 => self.deref().read_scalar::<u8>()?.into_dyn(),
            ScalarType::U16 => self.deref().read_scalar::<u16>()?.into_dyn(),
            ScalarType::U32 => self.deref().read_scalar::<u32>()?.into_dyn(),
            ScalarType::U64 => self.deref().read_scalar::<u64>()?.into_dyn(),
            ScalarType::Usize => self.deref().read_scalar::<usize>()?.into_dyn(),
            ScalarType::I8 => self.deref().read_scalar::<i8>()?.into_dyn(),
            ScalarType::I16 => self.deref().read_scalar::<i16>()?.into_dyn(),
            ScalarType::I32 => self.deref().read_scalar::<i32>()?.into_dyn(),
            ScalarType::I64 => self.deref().read_scalar::<i64>()?.into_dyn(),
            ScalarType::F32 => self.deref().read_scalar::<f32>()?.into_dyn(),
            ScalarType::F64 => self.deref().read_scalar::<f64>()?.into_dyn(),
            ScalarType::String => {
                let s = self.deref().read_scalar::<VarLenUnicode>()?;
                s.to_string().into_dyn()
            }
        };
        BackendData::from_dyn(val)
    }

    fn read_array_slice<T, S, D>(&self, selection: &[S]) -> Result<Array<T, D>>
    where
        T: BackendData,
        S: AsRef<SelectInfoElem>,
        D: Dimension,
    {
        fn read_arr<T, S, D>(dataset: &H5Dataset, selection: &[S]) -> Result<Array<T, D>>
        where
            T: H5Type + BackendData,
            S: AsRef<SelectInfoElem>,
            D: Dimension,
        {
            if selection.iter().any(|x| x.as_ref().is_index()) {
                // fancy indexing is too slow, just read all
                let arr = dataset.deref().read::<T, D>()?;
                Ok(ArrayOp::select(&arr, selection))
            } else {
                let (select, shape) = into_selection(selection, dataset.shape());
                if matches!(select, Selection::Points(_)) {
                    let slice_1d = hdf5::Container::read_slice_1d::<T, _>(dataset, select)?;
                    Ok(slice_1d
                        .into_shape(shape.as_ref())?
                        .into_dimensionality::<D>()?)
                } else {
                    Ok(hdf5::Container::read_slice::<T, _, D>(dataset, select)?)
                }
            }
        }

        let array: DynArray = match T::DTYPE {
            ScalarType::I8 => read_arr::<i8, _, D>(self, selection)?.into(),
            ScalarType::I16 => read_arr::<i16, _, D>(self, selection)?.into(),
            ScalarType::I32 => read_arr::<i32, _, D>(self, selection)?.into(),
            ScalarType::I64 => read_arr::<i64, _, D>(self, selection)?.into(),
            ScalarType::U8 => read_arr::<u8, _, D>(self, selection)?.into(),
            ScalarType::U16 => read_arr::<u16, _, D>(self, selection)?.into(),
            ScalarType::U32 => read_arr::<u32, _, D>(self, selection)?.into(),
            ScalarType::U64 => read_arr::<u64, _, D>(self, selection)?.into(),
            ScalarType::Usize => read_arr::<usize, _, D>(self, selection)?.into(),
            ScalarType::F32 => read_arr::<f32, _, D>(self, selection)?.into(),
            ScalarType::F64 => read_arr::<f64, _, D>(self, selection)?.into(),
            ScalarType::Bool => read_arr::<bool, _, D>(self, selection)?.into(),
            ScalarType::String => {
                if selection.as_ref().iter().any(|x| x.as_ref().is_index()) {
                    // fancy indexing is too slow, just read all
                    let arr = self.deref().read::<VarLenUnicode, D>()?;
                    let arr_ = arr.map(|s| s.to_string());
                    let r: Result<_> = Ok(ArrayOp::select(&arr_, selection));
                    r
                } else {
                    let (select, shape) = into_selection(selection, self.shape());
                    let arr: Result<_> = if matches!(select, Selection::Points(_)) {
                        let slice_1d = self.deref().read_slice_1d::<VarLenUnicode, _>(select)?;
                        Ok(slice_1d
                            .into_shape(shape.as_ref())?
                            .into_dimensionality::<D>()?)
                    } else {
                        Ok(self.deref().read_slice::<VarLenUnicode, _, D>(select)?)
                    };
                    Ok(arr?.map(|s| s.to_string()))
                }?
                .into()
                /*
                let arr = read_arr::<VarLenUnicode, _, _, D>(dataset, selection)?;
                let arr = arr.map(|s| s.to_string());
                arr.into()
                */
            }
        };
        Ok(BackendData::from_dyn_arr(array)?.into_dimensionality::<D>()?)
    }

    fn write_array_slice<'a, A, S, T, D>(&self, data: A, selection: &[S]) -> Result<()>
    where
        A: Into<ArrayView<'a, T, D>>,
        T: BackendData,
        S: AsRef<SelectInfoElem>,
        D: Dimension,
    {
        fn write_array_impl<'a, T, D, S>(
            container: &H5Dataset,
            arr: ArrayView<'a, T, D>,
            selection: &[S],
        ) -> Result<()>
        where
            T: H5Type + Clone,
            D: Dimension,
            S: AsRef<SelectInfoElem>,
        {
            let (select, _) = into_selection(selection, container.shape());
            container.deref().write_slice(&arr.as_standard_layout(), select)?;
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
                let data: Array<VarLenUnicode, D> = x.map(|x| x.parse().unwrap());
                write_array_impl(self, data.view(), selection)
            }
        }
    }
}

// Generic `LocationOp` functions

fn file(loc: &Location) -> Result<H5File> {
    Ok(H5File(hdf5::Location::file(loc)?))
}

fn path(loc: &Location) -> PathBuf {
    hdf5::Location::name(loc).into()
}

fn write_array_attr<'a, A, D, Dim>(loc: &Location, name: &str, value: A) -> Result<()>
where
    A: Into<ArrayView<'a, D, Dim>>,
    D: BackendData,
    Dim: Dimension,
{
    del_attr(loc, name);
    match BackendData::into_dyn_arr(value.into()) {
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

fn write_str_attr(loc: &Location, name: &str, value: &str) -> Result<()> {
    let value_: VarLenUnicode = value.parse().unwrap();
    let attr = loc
        .attr(name)
        .or(loc.new_attr::<VarLenUnicode>().create(name))?;
    attr.write_scalar(&value_)?;
    Ok(())
}

fn write_scalar_attr<D: BackendData>(loc: &Location, name: &str, value: D) -> Result<()> {
    del_attr(loc, name);
    match value.into_dyn() {
        DynScalar::U8(x) => loc.new_attr::<u8>().create(name)?.write_scalar(&x)?,
        DynScalar::U16(x) => loc.new_attr::<u16>().create(name)?.write_scalar(&x)?,
        DynScalar::U32(x) => loc.new_attr::<u32>().create(name)?.write_scalar(&x)?,
        DynScalar::U64(x) => loc.new_attr::<u64>().create(name)?.write_scalar(&x)?,
        DynScalar::Usize(x) => loc.new_attr::<usize>().create(name)?.write_scalar(&x)?,
        DynScalar::I8(x) => loc.new_attr::<i8>().create(name)?.write_scalar(&x)?,
        DynScalar::I16(x) => loc.new_attr::<i16>().create(name)?.write_scalar(&x)?,
        DynScalar::I32(x) => loc.new_attr::<i32>().create(name)?.write_scalar(&x)?,
        DynScalar::I64(x) => loc.new_attr::<i64>().create(name)?.write_scalar(&x)?,
        DynScalar::F32(x) => loc.new_attr::<f32>().create(name)?.write_scalar(&x)?,
        DynScalar::F64(x) => loc.new_attr::<f64>().create(name)?.write_scalar(&x)?,
        DynScalar::Bool(x) => loc.new_attr::<bool>().create(name)?.write_scalar(&x)?,
        DynScalar::String(x) => {
            let value_: VarLenUnicode = x.parse().unwrap();
            loc.new_attr::<VarLenUnicode>().create(name)?.write_scalar(&value_)?
        },
    };
    Ok(())
}

fn read_str_attr(loc: &Location, name: &str) -> Result<String> {
    let container = loc.attr(name)?;
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

fn read_array_attr<T: BackendData, D: Dimension>(loc: &Location, name: &str) -> Result<Array<T, D>> {
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

////////////////////////////////////////////////////////////////////////////////
/// Derived implementations
////////////////////////////////////////////////////////////////////////////////

impl GroupOp for H5File {
    type Backend = H5;

    fn list(&self) -> Result<Vec<String>> {
        list(self)
    }

    fn create_group(&self, name: &str) -> Result<<Self::Backend as Backend>::Group> {
        create_group(self, name)
    }

    fn open_group(&self, name: &str) -> Result<<Self::Backend as Backend>::Group> {
        open_group(self, name)
    }

    fn new_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<<Self::Backend as Backend>::Dataset> {
        new_dataset::<T>(self, name, shape, config)
    }

    fn open_dataset(&self, name: &str) -> Result<<Self::Backend as Backend>::Dataset> {
        open_dataset(self, name)
    }

    fn delete(&self, name: &str) -> Result<()> {
        delete(self, name)
    }

    fn exists(&self, name: &str) -> Result<bool> {
        exists(self, name)
    }

    fn create_scalar_data<D: BackendData>(
        &self,
        name: &str,
        data: &D,
    ) -> Result<<Self::Backend as Backend>::Dataset> {
        create_scalar_data(self, name, data)
    }
}

impl GroupOp for H5Group {
    type Backend = H5;

    fn list(&self) -> Result<Vec<String>> {
        list(self)
    }

    fn create_group(&self, name: &str) -> Result<<Self::Backend as Backend>::Group> {
        create_group(self, name)
    }

    fn open_group(&self, name: &str) -> Result<<Self::Backend as Backend>::Group> {
        open_group(self, name)
    }

    fn new_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<<Self::Backend as Backend>::Dataset> {
        new_dataset::<T>(self, name, shape, config)
    }

    fn open_dataset(&self, name: &str) -> Result<<Self::Backend as Backend>::Dataset> {
        open_dataset(self, name)
    }

    fn delete(&self, name: &str) -> Result<()> {
        delete(self, name)
    }

    fn exists(&self, name: &str) -> Result<bool> {
        exists(self, name)
    }

    fn create_scalar_data<D: BackendData>(
        &self,
        name: &str,
        data: &D,
    ) -> Result<<Self::Backend as Backend>::Dataset> {
        create_scalar_data(self, name, data)
    }
}

impl LocationOp for H5Group {
    type Backend = H5;

    fn file(&self) -> Result<<Self::Backend as Backend>::File> {
        file(self)
    }

    fn path(&self) -> PathBuf {
        path(self)
    }

    fn write_array_attr<'a, A, D, Dim>(&self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension,
    {
        write_array_attr(self, name, value)
    }

    fn write_scalar_attr<D: BackendData>(&self, name: &str, value: D) -> Result<()> {
        write_scalar_attr(self, name, value)
    }

    fn write_str_attr(&self, name: &str, value: &str) -> Result<()> {
        write_str_attr(self, name, value)
    }

    fn read_str_attr(&self, name: &str) -> Result<String> {
        read_str_attr(self, name)
    }

    fn read_array_attr<T: BackendData, D: Dimension>(&self, name: &str) -> Result<Array<T, D>> {
        read_array_attr(self, name)
    }
}

impl LocationOp for H5Dataset {
    type Backend = H5;

    fn file(&self) -> Result<<Self::Backend as Backend>::File> {
        file(self)
    }

    fn path(&self) -> PathBuf {
        path(self)
    }

    fn write_array_attr<'a, A, D, Dim>(&self, name: &str, value: A) -> Result<()>
    where
        A: Into<ArrayView<'a, D, Dim>>,
        D: BackendData,
        Dim: Dimension,
    {
        write_array_attr(self, name, value)
    }

    fn write_scalar_attr<D: BackendData>(&self, name: &str, value: D) -> Result<()> {
        write_scalar_attr(self, name, value)
    }

    fn write_str_attr(&self, name: &str, value: &str) -> Result<()> {
        write_str_attr(self, name, value)
    }

    fn read_str_attr(&self, name: &str) -> Result<String> {
        read_str_attr(self, name)
    }

    fn read_array_attr<T: BackendData, D: Dimension>(&self, name: &str) -> Result<Array<T, D>> {
        read_array_attr(self, name)
    }
}

///////////////////////////////////////////////////////////////////////////////
/// Auxiliary functions
///////////////////////////////////////////////////////////////////////////////

fn into_selection<S, E>(selection: S, shape: Shape) -> (Selection, Shape)
where
    S: AsRef<[E]>,
    E: AsRef<SelectInfoElem>,
{
    if selection.as_ref().into_iter().all(|x| x.as_ref().is_full()) {
        (Selection::All, shape)
    } else {
        let bounded_selection = BoundedSelectInfo::new(&selection, &shape);
        let out_shape = bounded_selection.out_shape();
        if let Some(idx) = bounded_selection.try_into_indices() {
            (Selection::from(idx), out_shape)
        } else {
            let slice: SliceInfo<_, _, _> = bounded_selection.try_into().unwrap();
            (Selection::try_from(slice).unwrap(), out_shape)
        }
    }
}

fn del_attr(loc: &Location, name: &str) {
    unsafe {
        let c_name = std::ffi::CString::new(name).unwrap().into_raw();
        if hdf5_sys::h5a::H5Aexists(loc.id(), c_name) != 0 {
            hdf5_sys::h5a::H5Adelete(loc.id(), c_name);
        }
    }
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
        with_tmp_dir(|dir| func(dir.join("temp.h5")))
    }

    #[test]
    fn test_basic() -> Result<()> {
        with_tmp_path(|path| {
            let file = H5::create(path.clone())?;
            let group = file.create_group("group")?;
            let subgroup = group.create_group("subgroup")?;

            assert_eq!(subgroup.path(), PathBuf::from("/group/subgroup"));
            Ok(())
        })
    }

    #[test]
    fn test_write_empty() -> Result<()> {
        with_tmp_path(|path| {
            let file = H5::create(&path)?;
            let group = file.create_group("group")?;
            let config = WriteConfig {
                ..Default::default()
            };

            let empty = Array1::<u8>::from_vec(Vec::new());
            let dataset = group.create_array_data("test", &empty, config)?;
            assert_eq!(empty, dataset.read_array::<u8, Ix1>()?);
            Ok(())
        })
    }

    #[test]
    fn test_write_slice() -> Result<()> {
        with_tmp_path(|path| -> Result<()> {
            let file = H5::create(&path)?;
            let config = WriteConfig {
                ..Default::default()
            };

            let dataset = file.new_dataset::<i32>("test", &[20, 50].as_slice().into(), config)?;
            let arr = Array::random((20, 50), Uniform::new(0, 100));

            // Repeatitive writes
            dataset.write_array_slice(&arr, s![.., ..].as_ref())?;
            dataset.write_array_slice(&arr, s![.., ..].as_ref())?;

            // Out-of-bounds writes should fail
            assert!(dataset.write_array_slice(&arr, s![20..40, ..].as_ref()).is_err());

            // Reshape and write
            dataset.reshape(&[40, 50].as_slice().into())?;
            dataset.write_array_slice(&arr, s![20..40, ..].as_ref())?;

            // Read back is OK
            let merged = concatenate(Axis(0), &[arr.view(), arr.view()])?;
            assert_eq!(merged, dataset.read_array::<i32, _>()?);

            // Shrinking is OK
            dataset.reshape(&[20, 50].as_slice().into())?;
            assert_eq!(arr, dataset.read_array::<i32, _>()?);

            Ok(())
        })
    }
}

