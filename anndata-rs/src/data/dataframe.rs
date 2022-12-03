use crate::backend::{Backend, GroupOp, DatasetOp, LocationOp, DataContainer};
use crate::data::data_traits::*;
use crate::data::array::{CategoricalArray, DynArray};

use ndarray::Array1;
use polars::prelude::{DataFrame, Series};
use polars::datatypes::{DataType, CategoricalChunkedBuilder};
use polars::prelude::IntoSeries;
use anyhow::{bail, Result};
use std::collections::HashMap;

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
        location.create_array_data(&self.index_name, &data, Default::default())?;
        Ok(DataContainer::Group(group))
    }
    fn overwrite<B: Backend>(&self, container: DataContainer<B>) -> Result<DataContainer<B>> {
        let index_name = container.read_str_attr("_index")?;
        container.as_group()?.delete(&index_name)?;
        container.write_str_attr("_index", &self.index_name)?;

        let data: Array1<String> = self.names.iter().map(|x| x.clone()).collect();
        let group = container.as_group()?;
        group.create_array_data(
            &self.index_name,
            &data,
            Default::default()
        )?;
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