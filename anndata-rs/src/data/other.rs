use crate::data::array::{CategoricalArray, DynArray};
use crate::data::Data;

use crate::backend::{Backend, BackendData, DataContainer, ScalarType, Selection};
use anyhow::{bail, Result, Ok};
use ndarray::Array1;
use polars::{
    datatypes::CategoricalChunkedBuilder, datatypes::DataType, frame::DataFrame,
    prelude::IntoSeries, series::Series,
};
use std::collections::HashMap;

pub trait WriteData {
    fn write<B: Backend>(&self, location: &B::Group, name: &str) -> Result<DataContainer<B>>;
    fn overwrite<B: Backend>(&self, container: DataContainer<B>) -> Result<DataContainer<B>> {
        let (file, path) = match &container {
            DataContainer::Group(grp) => (B::file(grp)?, B::path(grp)?),
            DataContainer::Dataset(data) => (B::file(data)?, B::path(data)?),
        };
        let group = B::open_group(&file, path.parent().unwrap().to_str().unwrap())?;
        let name = path.file_name().unwrap().to_str().unwrap();
        B::delete(&group, name)?;
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
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
}

impl From<i8> for DynScalar {
    fn from(v: i8) -> Self {
        DynScalar::I8(v)
    }
}
impl From<i16> for DynScalar {
    fn from(v: i16) -> Self {
        DynScalar::I16(v)
    }
}
impl From<i32> for DynScalar {
    fn from(v: i32) -> Self {
        DynScalar::I32(v)
    }
}
impl From<i64> for DynScalar {
    fn from(v: i64) -> Self {
        DynScalar::I64(v)
    }
}
impl From<u8> for DynScalar {
    fn from(v: u8) -> Self {
        DynScalar::U8(v)
    }
}
impl From<u16> for DynScalar {
    fn from(v: u16) -> Self {
        DynScalar::U16(v)
    }
}
impl From<u32> for DynScalar {
    fn from(v: u32) -> Self {
        DynScalar::U32(v)
    }
}
impl From<u64> for DynScalar {
    fn from(v: u64) -> Self {
        DynScalar::U64(v)
    }
}
impl From<f32> for DynScalar {
    fn from(v: f32) -> Self {
        DynScalar::F32(v)
    }
}
impl From<f64> for DynScalar {
    fn from(v: f64) -> Self {
        DynScalar::F64(v)
    }
}
impl From<bool> for DynScalar {
    fn from(v: bool) -> Self {
        DynScalar::Bool(v)
    }
}
impl From<String> for DynScalar {
    fn from(v: String) -> Self {
        DynScalar::String(v)
    }
}

impl<T: BackendData> WriteData for T {
    fn write<B: Backend>(&self, location: &B::Group, name: &str) -> Result<DataContainer<B>> {
        let dataset = T::write_data::<B>(location, name, self)?;
        let container = DataContainer::Dataset(dataset);
        let encoding_type = if T::DTYPE == ScalarType::String {
            "string"
        } else {
            "numeric_scalar"
        };
        B::write_str_attr(&container, "encoding-type", encoding_type)?;
        B::write_str_attr(&container, "encoding-version", "0.2.0")?;
        Ok(container)
    }
}

impl WriteData for DynScalar {
    fn write<B: Backend>(&self, location: &B::Group, name: &str) -> Result<DataContainer<B>> {
        match self {
            DynScalar::I8(data) => data.write(location, name),
            DynScalar::I16(data) => data.write(location, name),
            DynScalar::I32(data) => data.write(location, name),
            DynScalar::I64(data) => data.write(location, name),
            DynScalar::U8(data) => data.write(location, name),
            DynScalar::U16(data) => data.write(location, name),
            DynScalar::U32(data) => data.write(location, name),
            DynScalar::U64(data) => data.write(location, name),
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
        match B::dtype(dataset)? {
            ScalarType::I8 => Ok(DynScalar::I8(i8::read_data::<B>(dataset)?)),
            ScalarType::I16 => Ok(DynScalar::I16(i16::read_data::<B>(dataset)?)),
            ScalarType::I32 => Ok(DynScalar::I32(i32::read_data::<B>(dataset)?)),
            ScalarType::I64 => Ok(DynScalar::I64(i64::read_data::<B>(dataset)?)),
            ScalarType::U8 => Ok(DynScalar::U8(u8::read_data::<B>(dataset)?)),
            ScalarType::U16 => Ok(DynScalar::U16(u16::read_data::<B>(dataset)?)),
            ScalarType::U32 => Ok(DynScalar::U32(u32::read_data::<B>(dataset)?)),
            ScalarType::U64 => Ok(DynScalar::U64(u64::read_data::<B>(dataset)?)),
            ScalarType::F32 => Ok(DynScalar::F32(f32::read_data::<B>(dataset)?)),
            ScalarType::F64 => Ok(DynScalar::F64(f64::read_data::<B>(dataset)?)),
            ScalarType::Bool => Ok(DynScalar::Bool(bool::read_data::<B>(dataset)?)),
            ScalarType::String => Ok(DynScalar::String(String::read_data::<B>(dataset)?)),
        }
    }
}

impl WriteData for DataFrame {
    fn write<B: Backend>(&self, location: &B::Group, name: &str) -> Result<DataContainer<B>> {
        let group = if B::exists(location, name)? {
            B::open_group(location, name)?
        } else {
            B::create_group(location, name)?
        };
        B::write_str_attr(&group, "encoding-type", "dataframe")?;
        B::write_str_attr(&group, "encoding-version", "0.2.0")?;

        let columns: Array1<String> = self
            .get_column_names()
            .into_iter()
            .map(|x| x.to_owned())
            .collect();
        B::write_str_arr_attr(&group, "column-order", &columns)?;
        self.iter()
            .try_for_each(|x| write_series::<B>(x, &group, x.name()).map(|_| ()))?;

        // Create an index as python anndata package enforce it.
        //DataFrameIndex::from(self.height()).write(&container)?;
        Ok(DataContainer::Group(group))
    }

    fn overwrite<B: Backend>(&self, container: DataContainer<B>) -> Result<DataContainer<B>> {
        let index_name = B::read_str_attr(&container, "_index")?;
        for obj in B::list(container.as_group()?)? {
            if obj != index_name {
                B::delete(container.as_group()?, &obj)?;
            }
        }

        let columns: Array1<String> = self
            .get_column_names()
            .into_iter()
            .map(|x| x.to_owned())
            .collect();
        B::write_str_arr_attr(&container, "column-order", &columns)?;
        self.iter()
            .try_for_each(|x| write_series::<B>(x, container.as_group()?, x.name()).map(|_| ()))?;

        Ok(container)
    }
}

impl ReadData for DataFrame {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let columns: Array1<String> = B::read_str_arr_attr(container, "column-order")?;
        columns
            .into_iter()
            .map(|x| {
                let name = x.as_str();
                let mut series = B::open_dataset(container.as_group()?, name)
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
        DynArray::ArrayI8(x) => Ok(x.iter().collect::<Series>()),
        DynArray::ArrayI16(x) => Ok(x.iter().collect::<Series>()),
        DynArray::ArrayI32(x) => Ok(x.iter().collect::<Series>()),
        DynArray::ArrayI64(x) => Ok(x.iter().collect::<Series>()),
        DynArray::ArrayU8(x) => Ok(x.iter().collect::<Series>()),
        DynArray::ArrayU16(x) => Ok(x.iter().collect::<Series>()),
        DynArray::ArrayU32(x) => Ok(x.iter().collect::<Series>()),
        DynArray::ArrayU64(x) => Ok(x.iter().collect::<Series>()),
        DynArray::ArrayF32(x) => Ok(x.iter().collect::<Series>()),
        DynArray::ArrayF64(x) => Ok(x.iter().collect::<Series>()),
        DynArray::ArrayBool(x) => Ok(x.iter().collect::<Series>()),
        DynArray::ArrayString(x) => Ok(x.iter().map(|x| x.as_str()).collect::<Series>()),
        DynArray::ArrayCategorical(arr) => {
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
    fn write<B: Backend>(&self, location: &B::Group, name: &str) -> Result<DataContainer<B>> {
        let group = if B::exists(location, name)? {
            B::open_group(location, name)?
        } else {
            B::create_group(location, name)?
        };
        B::write_str_attr(&group, "_index", &self.index_name)?;
        let data: Array1<String> = self.names.iter().map(|x| x.clone()).collect();
        let dataset = String::write_arr_data::<B, _, _, _>(location, &self.index_name, &data, Selection::All)?;
        Ok(DataContainer::Group(group))
    }
    fn overwrite<B: Backend>(&self, container: DataContainer<B>) -> Result<DataContainer<B>> {
        let index_name = B::read_str_attr(&container, "_index")?;
        B::delete(container.as_group()?, &index_name)?;
        B::write_str_attr(&container, "_index", &self.index_name)?;

        let data: Array1<String> = self.names.iter().map(|x| x.clone()).collect();
        let dataset = String::write_arr_data::<B, _, _, _>(container.as_group()?, &self.index_name, &data, Selection::All)?;
        Ok(container)
    }
}

impl ReadData for DataFrameIndex {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let index_name = B::read_str_attr(&container, "_index")?;
        let dataset = B::open_dataset(container.as_group()?, &index_name)?;
        let data = String::read_arr_data::<B, _, _>(&dataset, Selection::All)?;
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
    fn write<B: Backend>(&self, location: &B::Group, name: &str) -> Result<DataContainer<B>> {
        let group = B::create_group(location, name)?;
        self.0
            .iter()
            .try_for_each(|(k, v)| v.write::<B>(&group, k).map(|_| ()))?;
        Ok(DataContainer::Group(group))
    }
}

impl ReadData for Mapping {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        todo!()
    }
}