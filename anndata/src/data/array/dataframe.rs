use std::ops::Deref;

use crate::backend::{Backend, DataContainer, DatasetOp, GroupOp, AttributeOp};
use crate::data::array::slice::{SelectInfoElem, Shape};
use crate::data::array::{CategoricalArray, DynArray};
use crate::data::data_traits::*;
use crate::data::index::{Index, Interval};
use crate::data::scalar::DynScalar;

use log::warn;
use anyhow::{bail, Result};
use ndarray::{Array1, Array2};
use polars::chunked_array::ChunkedArray;
use polars::datatypes::{CategoricalChunkedBuilder, DataType};
use polars::prelude::IntoSeries;
use polars::prelude::{DataFrame, Series};

use super::{BoundedSelectInfo, BoundedSelectInfoElem};

impl WriteData for DataFrame {
    fn data_type(&self) -> crate::backend::DataType {
        crate::backend::DataType::DataFrame
    }
    fn write<B: Backend, G: GroupOp<B>>(
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

        let container = DataContainer::Group(group);

        // Create an index as the python anndata package enforce it. This is not used by this library
        DataFrameIndex::from(self.height()).overwrite(container)
    }

    fn overwrite<B: Backend>(&self, mut container: DataContainer<B>) -> Result<DataContainer<B>> {
        if let Ok(index_name) = container.read_str_attr("_index") {
            for obj in container.as_group()?.list()? {
                if obj != index_name {
                    container.as_group()?.delete(&obj)?;
                }
            }
            let n = self.height();
            if n != 0 && n != container.as_group()?.open_dataset(&index_name)?.shape()[0] {
                container = DataFrameIndex::from(self.height()).overwrite(container)?;
            }
        } else {
            for obj in container.as_group()?.list()? {
                container.as_group()?.delete(&obj)?;
            }
            container = DataFrameIndex::from(self.height()).overwrite(container)?;
        }

        let columns: Array1<String> = self
            .get_column_names()
            .into_iter()
            .map(|x| x.to_owned())
            .collect();
        container.write_array_attr("column-order", &columns)?;
        self.iter()
            .try_for_each(|x| x.write(container.as_group()?, x.name()).map(|_| ()))?;
        container.write_str_attr("encoding-type", "dataframe")?;
        container.write_str_attr("encoding-version", "0.2.0")?;

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
        let ridx = select.as_ref()[0].iter().map(|x| x as u32).collect();
        self.select(select.as_ref()[1].to_vec().into_iter().map(|i| columns[i]))
            .unwrap()
            .take(&ChunkedArray::from_vec("idx", ridx))
            .unwrap()
    }

    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        Ok(
            iter.reduce(|mut a, b| {
                a.vstack_mut(&b).unwrap();
                a
            }).unwrap_or(Self::empty())
        )
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
    fn write<B: Backend, G: GroupOp<B>>(
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
            DataType::String => self
                .str()?
                .into_iter()
                .map(|x| x.unwrap().to_string())
                .collect::<Array1<_>>()
                .into_dyn()
                .into(),
            DataType::Categorical(_,_) => self
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
                let result = CategoricalChunkedBuilder::new(
                    "", arr.codes.len(), polars::datatypes::CategoricalOrdering::Lexical
                ).drain_iter_and_finish(
                    arr.codes
                        .into_iter()
                        .map(|i| Some(arr.categories[i as usize].as_str())),
                ).into_series();
                Ok(result)
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
    fn get(&self, _index: &[usize]) -> Option<DynScalar> {
        todo!()
    }

    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        let i = BoundedSelectInfoElem::new(info.as_ref()[0].as_ref(), self.len())
            .iter().map(|x| x as u32).collect::<Vec<_>>();
        self.take(&ChunkedArray::from_vec("idx", i)).unwrap()
    }

    fn vstack<I: Iterator<Item = Self>>(_iter: I) -> Result<Self> {
        todo!("vstack not implemented for Series")
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
    index: Index,
}

impl std::cmp::PartialEq for DataFrameIndex {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl DataFrameIndex {
    pub fn empty() -> Self {
        Self {
            index_name: "index".to_string(),
            index: Index::empty(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn get_index(&self, k: &str) -> Option<usize> {
        self.index.get_index(k)
    }

    pub fn into_vec(self) -> Vec<String> {
        self.index.into_vec()
    }

    pub fn select(&self, select: &SelectInfoElem) -> Self {
        let index = self.index.select(select);
        Self {
            index_name: self.index_name.clone(),
            index,
        }
    }
}

impl IntoIterator for DataFrameIndex {
    type Item = String;
    type IntoIter = Box<dyn Iterator<Item = String>>;

    fn into_iter(self) -> Self::IntoIter {
        self.index.into_iter()
    }
}

impl WriteData for DataFrameIndex {
    fn data_type(&self) -> crate::backend::DataType {
        crate::backend::DataType::DataFrame
    }
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let group = if location.exists(name)? {
            location.open_group(name)?
        } else {
            location.create_group(name)?
        };
        self.overwrite(DataContainer::Group(group))
    }

    fn overwrite<B: Backend>(&self, container: DataContainer<B>) -> Result<DataContainer<B>> {
        if let Ok(index_name) = container.read_str_attr("_index") {
            container.as_group()?.delete(&index_name)?;
        }
        container.write_str_attr("_index", &self.index_name)?;
        let group = container.as_group()?;
        let arr: Array1<String> = self.clone().into_iter().collect();
        let data = group.create_array_data(&self.index_name, &arr, Default::default())?;
        match &self.index {
            Index::List(_) => { data.write_str_attr("index_type", "list")?; },
            Index::Intervals(intervals) => {
                let keys: Array1<String> = intervals.keys().cloned().collect();
                let vec: Vec<usize> = intervals.values().flat_map(|x| [x.start, x.end, x.size, x.step]).collect();
                let values = Array2::from_shape_vec((intervals.deref().len(), 4), vec)?;
                if data.write_array_attr("names", &keys).is_err() || data.write_array_attr("intervals", &values).is_err() { // fallback to "list"
                    data.write_str_attr("index_type", "list")?;
                    warn!("Failed to save interval index as attributes, fallback to list index");
                } else {
                    data.write_str_attr("index_type", "intervals")?;
                }
            },
            Index::Range(range) => {
                data.write_str_attr("index_type", "range")?;
                data.write_scalar_attr("start", range.start)?;
                data.write_scalar_attr("end", range.end)?;
            },
        }
        Ok(container)
    }
}

impl ReadData for DataFrameIndex {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let index_name = container.read_str_attr("_index")?;
        let dataset = container.as_group()?.open_dataset(&index_name)?;
        match dataset.read_str_attr("index_type").as_ref().map_or("list", |x| x.as_str()) {
            "list" => {
                let data = dataset.read_array()?;
                let mut index: DataFrameIndex = data.to_vec().into();
                index.index_name = index_name;
                Ok(index)
            },
            "intervals" => {
                let keys: Array1<String> = dataset.read_array_attr("names")?;
                let values: Array2<usize> = dataset.read_array_attr("intervals")?;
                Ok(keys.into_iter().zip(
                    values.rows().into_iter().map(|row| Interval {start: row[0], end: row[1], size: row[2], step: row[3]})
                ).collect())
            }
            "range" => {
                let start = dataset.read_scalar_attr("start")?;
                let end = dataset.read_scalar_attr("end")?;
                Ok((start..end).into())
            }
            x => bail!("Unknown index type: {}", x)
        }
    }
}

impl<D> From<D> for DataFrameIndex
where
    Index: From<D>,
{
    fn from(data: D) -> Self {
        Self {
            index_name: "index".to_owned(),
            index: data.into(),
        }
    }
}

impl<D> FromIterator<D> for DataFrameIndex
where
    Index: FromIterator<D>,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = D>,
    {
        Self {
            index_name: "index".to_owned(),
            index: iter.into_iter().collect(),
        }
    }
}