use crate::data::array::{CategoricalArray, DynArray};
use crate::data::Data;

use crate::backend::{Backend, GroupOp, LocationOp, DatasetOp, BackendData, DataContainer, ScalarType};
use anyhow::{bail, Result, Ok};
use ndarray::Array1;
use polars::{
    datatypes::CategoricalChunkedBuilder, datatypes::DataType, frame::DataFrame,
    prelude::IntoSeries, series::Series,
};
use std::collections::HashMap;

pub trait WriteData {
    fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>>;
    fn overwrite<B: Backend>(&self, container: DataContainer<B>) -> Result<DataContainer<B>> {
        let file = container.file()?;
        let path = container.path();
        let group = file.open_group(path.parent().unwrap().to_str().unwrap())?;
        let name = path.file_name().unwrap().to_str().unwrap();
        group.delete(name)?;
        self.write(&group, name)
    }
}

pub trait ReadData {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self>
    where
        Self: Sized;
}

#[derive(Debug, Clone, PartialEq)]
pub enum DynScalar {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    Usize(usize),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
}

/// macro to implement `From` trait for `DynScalar`
macro_rules! impl_from_dynscalar {
    ($($from:ty, $to:ident),*) => {
        $(
            impl From<$from> for DynScalar {
                fn from(val: $from) -> Self {
                    DynScalar::$to(val)
                }
            }
            impl ReadData for $from {
                fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
                    let dataset = container.as_dataset()?;
                    match dataset.dtype()? {
                        ScalarType::$to => Ok(dataset.read_scalar()?),
                        _ => bail!("Cannot read $from"),
                    }
                }
            }
        )*
    };
}

impl_from_dynscalar!(
    i8, I8,
    i16, I16,
    i32, I32,
    i64, I64,
    u8, U8,
    u16, U16,
    u32, U32,
    u64, U64,
    usize, Usize,
    f32, F32,
    f64, F64,
    bool, Bool,
    String, String
);

impl<T: BackendData> WriteData for T {
    fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
        let dataset = location.write_scalar(name, self)?;
        let container = DataContainer::Dataset(dataset);
        let encoding_type = if T::DTYPE == ScalarType::String {
            "string"
        } else {
            "numeric-scalar"
        };
        container.write_str_attr("encoding-type", encoding_type)?;
        container.write_str_attr("encoding-version", "0.2.0")?;
        Ok(container)
    }
}

impl WriteData for DynScalar {
    fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
        match self {
            DynScalar::I8(data) => data.write(location, name),
            DynScalar::I16(data) => data.write(location, name),
            DynScalar::I32(data) => data.write(location, name),
            DynScalar::I64(data) => data.write(location, name),
            DynScalar::U8(data) => data.write(location, name),
            DynScalar::U16(data) => data.write(location, name),
            DynScalar::U32(data) => data.write(location, name),
            DynScalar::U64(data) => data.write(location, name),
            DynScalar::Usize(data) => data.write(location, name),
            DynScalar::F32(data) => data.write(location, name),
            DynScalar::F64(data) => data.write(location, name),
            DynScalar::Bool(data) => data.write(location, name),
            DynScalar::String(data) => data.write(location, name),
        }
    }
}

impl ReadData for DynScalar {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let dataset = container.as_dataset()?;
        match dataset.dtype()? {
            ScalarType::I8 => Ok(DynScalar::I8(dataset.read_scalar()?)),
            ScalarType::I16 => Ok(DynScalar::I16(dataset.read_scalar()?)),
            ScalarType::I32 => Ok(DynScalar::I32(dataset.read_scalar()?)),
            ScalarType::I64 => Ok(DynScalar::I64(dataset.read_scalar()?)),
            ScalarType::U8 => Ok(DynScalar::U8(dataset.read_scalar()?)),
            ScalarType::U16 => Ok(DynScalar::U16(dataset.read_scalar()?)),
            ScalarType::U32 => Ok(DynScalar::U32(dataset.read_scalar()?)),
            ScalarType::U64 => Ok(DynScalar::U64(dataset.read_scalar()?)),
            ScalarType::Usize => Ok(DynScalar::Usize(dataset.read_scalar()?)),
            ScalarType::F32 => Ok(DynScalar::F32(dataset.read_scalar()?)),
            ScalarType::F64 => Ok(DynScalar::F64(dataset.read_scalar()?)),
            ScalarType::Bool => Ok(DynScalar::Bool(dataset.read_scalar()?)),
            ScalarType::String => Ok(DynScalar::String(dataset.read_scalar()?)),
        }
    }
}

impl WriteData for DataFrame {
    fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
        let group = if location.exists(name)? {
            location.open_group(name)?
        } else {
            location.create_group(name)?
        };
        group.write_str_attr("encoding-type", "dataframe")?;
        group.write_str_attr("encoding-version", "0.2.0")?;

        let columns: Array1<String> = self
            .get_column_names()
            .into_iter()
            .map(|x| x.to_owned())
            .collect();
        group.write_arr_attr("column-order", &columns)?;
        self.iter()
            .try_for_each(|x| write_series::<B>(x, &group, x.name()).map(|_| ()))?;

        // Create an index as python anndata package enforce it.
        //DataFrameIndex::from(self.height()).write(&container)?;
        Ok(DataContainer::Group(group))
    }

    fn overwrite<B: Backend>(&self, container: DataContainer<B>) -> Result<DataContainer<B>> {
        let index_name = container.read_str_attr("_index")?;
        for obj in container.as_group()?.list()? {
            if obj != index_name {
                container.as_group()?.delete(&obj)?;
            }
        }

        let columns: Array1<String> = self
            .get_column_names()
            .into_iter()
            .map(|x| x.to_owned())
            .collect();
        container.write_arr_attr("column-order", &columns)?;
        self.iter()
            .try_for_each(|x| write_series::<B>(x, container.as_group()?, x.name()).map(|_| ()))?;

        Ok(container)
    }
}

impl ReadData for DataFrame {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let columns: Array1<String> = container.read_arr_attr("column-order")?;
        columns
            .into_iter()
            .map(|x| {
                let name = x.as_str();
                let mut series = container.as_group()?.open_dataset(name)
                    .map(DataContainer::Dataset)
                    .and_then(|x| read_series::<B>(&x))?;
                series.rename(name);
                Ok(series)
            })
            .collect()
    }
}

fn write_series<B: Backend>(
    data: &Series,
    group: &B::Group,
    name: &str,
) -> Result<DataContainer<B>> {
    let array: DynArray = match data.dtype() {
        DataType::UInt8 => data
            .u8()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .into_dyn()
            .into(),
        DataType::UInt16 => data
            .u16()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .into_dyn()
            .into(),
        DataType::UInt32 => data
            .u32()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .into_dyn()
            .into(),
        DataType::UInt64 => data
            .u64()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .into_dyn()
            .into(),
        DataType::Int8 => data
            .i8()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .into_dyn()
            .into(),
        DataType::Int16 => data
            .i16()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .into_dyn()
            .into(),
        DataType::Int32 => data
            .i32()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .into_dyn()
            .into(),
        DataType::Int64 => data
            .i64()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .into_dyn()
            .into(),
        DataType::Float32 => data
            .f32()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .into_dyn()
            .into(),
        DataType::Float64 => data
            .f64()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .into_dyn()
            .into(),
        DataType::Boolean => data
            .bool()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .into_dyn()
            .into(),
        DataType::Utf8 => data
            .utf8()?
            .into_iter()
            .map(|x| x.unwrap().to_string())
            .collect::<Array1<_>>()
            .into_dyn()
            .into(),
        DataType::Categorical(_) => data
            .categorical()?
            .iter_str()
            .map(|x| x.unwrap())
            .collect::<CategoricalArray>()
            .into(),
        other => bail!("Unsupported series data type: {:?}", other),
    };
    array.write(group, name)
}

fn read_series<B: Backend>(container: &DataContainer<B>) -> Result<Series> {
    match DynArray::read(container)? {
        DynArray::I8(x) => Ok(x.iter().collect::<Series>()),
        DynArray::I16(x) => Ok(x.iter().collect::<Series>()),
        DynArray::I32(x) => Ok(x.iter().collect::<Series>()),
        DynArray::I64(x) => Ok(x.iter().collect::<Series>()),
        DynArray::U8(x) => Ok(x.iter().collect::<Series>()),
        DynArray::U16(x) => Ok(x.iter().collect::<Series>()),
        DynArray::U32(x) => Ok(x.iter().collect::<Series>()),
        DynArray::U64(x) => Ok(x.iter().collect::<Series>()),
        DynArray::Usize(x) => Ok(x.iter().map(|x| *x as u64).collect::<Series>()),
        DynArray::F32(x) => Ok(x.iter().collect::<Series>()),
        DynArray::F64(x) => Ok(x.iter().collect::<Series>()),
        DynArray::Bool(x) => Ok(x.iter().collect::<Series>()),
        DynArray::String(x) => Ok(x.iter().map(|x| x.as_str()).collect::<Series>()),
        DynArray::Categorical(arr) => {
            let mut builder = CategoricalChunkedBuilder::new("", arr.codes.len());
            builder.drain_iter(
                arr.codes
                    .into_iter()
                    .map(|i| Some(arr.categories[i as usize].as_str())),
            );
            Ok(builder.finish().into_series())
        }
    }
}

pub struct DataFrameIndex {
    pub index_name: String,
    pub names: Vec<String>,
    pub index_map: HashMap<String, usize>,
}

impl DataFrameIndex {
    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn get(&self, k: &String) -> Option<usize> {
        self.index_map.get(k).map(|x| *x)
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

impl WriteData for DataFrameIndex {
    fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
        let group = if location.exists(name)? {
            location.open_group(name)?
        } else {
            location.create_group(name)?
        };
        group.write_str_attr("_index", &self.index_name)?;
        let data: Array1<String> = self.names.iter().map(|x| x.clone()).collect();
        location.write_array(&self.index_name, &data, )?;
        Ok(DataContainer::Group(group))
    }
    fn overwrite<B: Backend>(&self, container: DataContainer<B>) -> Result<DataContainer<B>> {
        let index_name = container.read_str_attr("_index")?;
        container.as_group()?.delete(&index_name)?;
        container.write_str_attr("_index", &self.index_name)?;

        let data: Array1<String> = self.names.iter().map(|x| x.clone()).collect();
        container.as_group()?.write_array(&self.index_name, &data)?;
        Ok(container)
    }
}

impl ReadData for DataFrameIndex {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let index_name = container.read_str_attr("_index")?;
        let dataset = container.as_group()?.open_dataset(&index_name)?;
        let data = dataset.read_array()?;
        let mut index: DataFrameIndex = data.to_vec().into();
        index.index_name = index_name;
        Ok(index)
    }
}

impl From<Vec<String>> for DataFrameIndex {
    fn from(names: Vec<String>) -> Self {
        let index_map = names
            .clone()
            .into_iter()
            .enumerate()
            .map(|(a, b)| (b, a))
            .collect();
        Self {
            index_name: "index".to_owned(),
            names,
            index_map,
        }
    }
}

impl From<usize> for DataFrameIndex {
    fn from(size: usize) -> Self {
        (0..size).map(|x| x.to_string()).collect()
    }
}

impl FromIterator<String> for DataFrameIndex {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = String>,
    {
        let names: Vec<_> = iter.into_iter().collect();
        let index_map = names
            .clone()
            .into_iter()
            .enumerate()
            .map(|(a, b)| (b, a))
            .collect();
        Self {
            index_name: "index".to_owned(),
            names,
            index_map,
        }
    }
}

impl<'a> FromIterator<&'a str> for DataFrameIndex {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a str>,
    {
        let names: Vec<_> = iter.into_iter().map(|x| x.to_owned()).collect();
        let index_map = names
            .clone()
            .into_iter()
            .enumerate()
            .map(|(a, b)| (b, a))
            .collect();
        Self {
            index_name: "index".to_owned(),
            names,
            index_map,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mapping(HashMap<String, Data>);

impl WriteData for Mapping {
    fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
        let group = location.create_group(name)?;
        self.0
            .iter()
            .try_for_each(|(k, v)| v.write(&group, k).map(|_| ()))?;
        Ok(DataContainer::Group(group))
    }
}

impl ReadData for Mapping {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        todo!()
    }
}