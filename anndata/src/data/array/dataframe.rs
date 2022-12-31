use crate::backend::{Backend, DataContainer, DatasetOp, GroupOp, LocationOp};
use crate::data::array::slice::{SelectInfoElem, Shape};
use crate::data::array::{CategoricalArray, DynArray};
use crate::data::data_traits::*;
use crate::data::scalar::DynScalar;

use anyhow::{bail, Result};
use ndarray::Array1;
use polars::datatypes::{CategoricalChunkedBuilder, DataType};
use polars::prelude::IntoSeries;
use polars::prelude::{DataFrame, Series};
use std::collections::HashMap;

use super::{BoundedSelectInfo, BoundedSelectInfoElem};

impl WriteData for DataFrame {
    fn data_type(&self) -> crate::backend::DataType {
        crate::backend::DataType::DataFrame
    }
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
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
        group.write_array_attr("column-order", &columns)?;
        self.iter()
            .try_for_each(|x| x.write(&group, x.name()).map(|_| ()))?;

        // Create an index as python anndata package enforce it.
        //DataFrameIndex::from(self.height()).write(&container)?;
        Ok(DataContainer::Group(group))
    }

    fn overwrite<B: Backend>(&self, container: DataContainer<B>) -> Result<DataContainer<B>> {
        if let Ok(index_name) = container.read_str_attr("_index") {
            for obj in container.as_group()?.list()? {
                if obj != index_name {
                    container.as_group()?.delete(&obj)?;
                }
            }
        } else {
            for obj in container.as_group()?.list()? {
                container.as_group()?.delete(&obj)?;
            }
        }

        let columns: Array1<String> = self
            .get_column_names()
            .into_iter()
            .map(|x| x.to_owned())
            .collect();
        container.write_array_attr("column-order", &columns)?;
        self.iter()
            .try_for_each(|x| x.write(container.as_group()?, x.name()).map(|_| ()))?;

        Ok(container)
    }
}

impl ReadData for DataFrame {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let columns: Array1<String> = container.read_array_attr("column-order")?;
        columns
            .into_iter()
            .map(|x| {
                let name = x.as_str();
                let series_container = DataContainer::<B>::open(container.as_group()?, name)?;
                let mut series = Series::read::<B>(&series_container)?;
                series.rename(name);
                Ok(series)
            })
            .collect()
    }
}

impl HasShape for DataFrame {
    fn shape(&self) -> Shape {
        self.shape().into()
    }
}

impl ArrayOp for DataFrame {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        self[index[1]].get(&[index[0]])
    }

    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        if info.as_ref().len() != 2 {
            panic!("DataFrame only support 2D selection");
        }
        let columns = self.get_column_names();
        let select = BoundedSelectInfo::new(&info, &HasShape::shape(self));
        self.select(select.as_ref()[1].to_vec().into_iter().map(|i| columns[i]))
            .unwrap()
            .take_iter(select.as_ref()[0].to_vec().into_iter())
            .unwrap()
    }
}

impl ReadArrayData for DataFrame {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        let group = container.as_group()?;
        let index = group.read_str_attr("_index")?;
        let nrows = group.open_dataset(&index)?.shape()[0];
        let columns: Array1<String> = container.read_array_attr("column-order")?;
        Ok((nrows, columns.len()).into())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        let columns: Vec<String> = container.read_array_attr("column-order")?.to_vec();
        BoundedSelectInfoElem::new(&info.as_ref()[1], columns.len())
            .iter()
            .map(|i| {
                let name = &columns[i];
                let mut series = container
                    .as_group()?
                    .open_dataset(name)
                    .map(DataContainer::Dataset)
                    .and_then(|x| Series::read_select::<B, _>(&x, &info[..1]))?;
                series.rename(name);
                Ok(series)
            })
            .collect()
    }
}

impl WriteArrayData for DataFrame {}

impl WriteData for Series {
    fn data_type(&self) -> crate::backend::DataType {
        crate::backend::DataType::DataFrame
    }
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let array: DynArray = match self.dtype() {
            DataType::UInt8 => self
                .u8()?
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Array1<_>>()
                .into_dyn()
                .into(),
            DataType::UInt16 => self
                .u16()?
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Array1<_>>()
                .into_dyn()
                .into(),
            DataType::UInt32 => self
                .u32()?
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Array1<_>>()
                .into_dyn()
                .into(),
            DataType::UInt64 => self
                .u64()?
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Array1<_>>()
                .into_dyn()
                .into(),
            DataType::Int8 => self
                .i8()?
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Array1<_>>()
                .into_dyn()
                .into(),
            DataType::Int16 => self
                .i16()?
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Array1<_>>()
                .into_dyn()
                .into(),
            DataType::Int32 => self
                .i32()?
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Array1<_>>()
                .into_dyn()
                .into(),
            DataType::Int64 => self
                .i64()?
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Array1<_>>()
                .into_dyn()
                .into(),
            DataType::Float32 => self
                .f32()?
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Array1<_>>()
                .into_dyn()
                .into(),
            DataType::Float64 => self
                .f64()?
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Array1<_>>()
                .into_dyn()
                .into(),
            DataType::Boolean => self
                .bool()?
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Array1<_>>()
                .into_dyn()
                .into(),
            DataType::Utf8 => self
                .utf8()?
                .into_iter()
                .map(|x| x.unwrap().to_string())
                .collect::<Array1<_>>()
                .into_dyn()
                .into(),
            DataType::Categorical(_) => self
                .categorical()?
                .iter_str()
                .map(|x| x.unwrap())
                .collect::<CategoricalArray>()
                .into(),
            other => bail!("Unsupported series data type: {:?}", other),
        };
        array.write(location, name)
    }
}

impl ReadData for Series {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
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
}

impl HasShape for Series {
    fn shape(&self) -> Shape {
        self.len().into()
    }
}

impl ArrayOp for Series {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        todo!()
    }

    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        let i = BoundedSelectInfoElem::new(info.as_ref()[0].as_ref(), self.len());
        self.take_iter(&mut i.to_vec().into_iter()).unwrap()
    }
}

impl ReadArrayData for Series {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container.as_dataset()?.shape().into())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        Ok(Self::read(container)?.select(info))
    }
}

#[derive(Debug, Clone)]
pub struct DataFrameIndex {
    pub index_name: String,
    pub names: Vec<String>,
    pub index_map: HashMap<String, usize>,
}

impl DataFrameIndex {
    pub fn new() -> Self {
        Self {
            index_name: "index".to_string(),
            names: Vec::new(),
            index_map: HashMap::new(),
        }
    }

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

impl IntoIterator for DataFrameIndex {
    type Item = String;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.names.into_iter()
    }
}

impl WriteData for DataFrameIndex {
    fn data_type(&self) -> crate::backend::DataType {
        crate::backend::DataType::DataFrame
    }
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
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
        if let Ok(index_name) = container.read_str_attr("_index") {
            container.as_group()?.delete(&index_name)?;
        }
        container.write_str_attr("_index", &self.index_name)?;

        let data: Array1<String> = self.names.iter().map(|x| x.clone()).collect();
        let group = container.as_group()?;
        group.create_array_data(&self.index_name, &data, Default::default())?;
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

impl HasShape for DataFrameIndex {
    fn shape(&self) -> Shape {
        self.len().into()
    }
}

impl ArrayOp for DataFrameIndex {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        self.names.get(index[0]).map(|x| x.clone().into())
    }

    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        let mut index: DataFrameIndex =
            BoundedSelectInfoElem::new(info.as_ref()[0].as_ref(), self.len())
                .iter()
                .map(|x| self.names[x].clone())
                .collect();
        index.index_name = self.index_name.clone();
        index
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
