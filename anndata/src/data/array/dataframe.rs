use std::collections::HashMap;

use crate::backend::{AttributeOp, Backend, DataContainer, DatasetOp, GroupOp, ScalarType};
use crate::data::array::{
    slice::{SelectInfoElem, Shape},
    CategoricalArray, DynArray,
};
use crate::data::data_traits::*;
use crate::data::index::{Index, Interval};

use anyhow::{bail, Context, Result};
use log::warn;
use ndarray::Array1;
use polars::chunked_array::ChunkedArray;
use polars::datatypes::DataType;
use polars::prelude::{DataFrame, Series};

use super::{SelectInfoBounds, SelectInfoElemBounds};

impl Element for DataFrame {
    fn data_type(&self) -> crate::backend::DataType {
        crate::backend::DataType::DataFrame
    }

    fn metadata(&self) -> MetaData {
        let mut metadata = HashMap::new();

        let columns: Vec<String> = self
            .get_column_names()
            .into_iter()
            .map(|x| x.to_string())
            .collect();
        metadata.insert("column-order".to_string(), columns.into());

        MetaData::new("dataframe", "0.2.0", Some(metadata))
    }
}

impl Writable for DataFrame {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let mut group = if location.exists(name)? {
            location.open_group(name)?
        } else {
            location.new_group(name)?
        };
        self.metadata().save_metadata(&mut group)?;

        self.iter().try_for_each(|x| {
            write_series(x, &group, x.name())?;
            anyhow::Ok(())
        })?;

        let mut container = DataContainer::Group(group);

        // Create an index as the python anndata package enforce it. This is not used by this library
        DataFrameIndex::from(self.height()).overwrite(&mut container)?;

        Ok(container)
    }

    /// Overwrite the data inplace.
    fn overwrite<B: Backend>(&self, mut container: DataContainer<B>) -> Result<DataContainer<B>> {
        if let Ok(index_name) = container.get_attr::<String>("_index") {
            for obj in container.as_group()?.list()? {
                if obj != index_name {
                    container.as_group()?.delete(&obj)?;
                }
            }
            let n = self.height();
            if n != 0 && n != container.as_group()?.open_dataset(&index_name)?.shape()[0] {
                DataFrameIndex::from(self.height()).overwrite(&mut container)?;
            }
        } else {
            for obj in container.as_group()?.list()? {
                container.as_group()?.delete(&obj)?;
            }
            DataFrameIndex::from(self.height()).overwrite(&mut container)?;
        }

        self.iter().try_for_each(|x| {
            write_series(x, container.as_group()?, x.name())?;
            anyhow::Ok(())
        })?;
        self.metadata().save_metadata(&mut container)?;

        Ok(container)
    }
}

impl Readable for DataFrame {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let columns: Vec<String> = container.get_attr("column-order")?;
        columns
            .into_iter()
            .map(|name| {
                let name = name.as_str();
                let series_container = DataContainer::<B>::open(container.as_group()?, name)?;
                let mut series = read_series::<B>(&series_container)
                    .with_context(|| format!("Failed to read series: {}", name))?;
                series.rename(name.into());
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

impl Selectable for DataFrame {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        if info.as_ref().len() != 2 {
            panic!("DataFrame only support 2D selection");
        }
        let columns = self.get_column_names();
        let select = SelectInfoBounds::new(&info, &HasShape::shape(self));
        let ridx = select.as_ref()[0].iter().map(|x| x as u32).collect();
        self.select(
            select.as_ref()[1]
                .to_vec()
                .into_iter()
                .map(|i| columns[i].as_str()),
        )
        .unwrap()
        .take(&ChunkedArray::from_vec("idx".into(), ridx))
        .unwrap()
    }
}

impl Stackable for DataFrame {
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        Ok(iter
            .reduce(|mut a, b| {
                a.vstack_mut(&b).unwrap();
                a
            })
            .unwrap_or(Self::empty()))
    }
}

impl ReadableArray for DataFrame {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        let group = container.as_group()?;
        let index: String = group.get_attr("_index")?;
        let nrows = group.open_dataset(&index)?.shape()[0];
        let columns: Vec<String> = container.get_attr("column-order")?;
        Ok((nrows, columns.len()).into())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        let columns: Vec<String> = container.get_attr("column-order")?;
        SelectInfoElemBounds::new(&info.as_ref()[1], columns.len())
            .iter()
            .map(|i| {
                let name = &columns[i];
                let series = container
                    .as_group()?
                    .open_dataset(name)
                    .map(DataContainer::Dataset)
                    .and_then(|x| read_series::<B>(&x))
                    .with_context(|| format!("Failed to read series: {}", name))?;
 
                let indices: Vec<u32> = SelectInfoElemBounds::new(&info[0], series.len())
                    .iter().map(|x| x.try_into().unwrap()).collect();
                let mut series = series.take_slice(indices.as_slice())?;
                series.rename(name.into());
                Ok(series)
            })
            .collect()
    }
}

impl WritableArray for DataFrame {}

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

    pub(crate) fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let index_name: String = container.get_attr("_index")?;
        let dataset = container.as_group()?.open_dataset(&index_name)?;
        match dataset
            .get_attr::<String>("index_type")
            .as_ref()
            .map_or("list", |x| x.as_str())
        {
            "list" => {
                let data = dataset.read_array()?;
                let mut index: DataFrameIndex = data.to_vec().into();
                index.index_name = index_name;
                Ok(index)
            }
            "intervals" => {
                let keys: Vec<String> = dataset.get_attr("names")?;
                let values: Vec<Vec<u64>> = dataset.get_attr("intervals")?;
                Ok(keys
                    .into_iter()
                    .zip(values.into_iter().map(|row| Interval {
                        start: row[0] as usize,
                        end: row[1] as usize,
                        size: row[2] as usize,
                        step: row[3] as usize,
                    }))
                    .collect())
            }
            "range" => {
                let start: u64 = dataset.get_attr("start")?;
                let end: u64 = dataset.get_attr("end")?;
                Ok((start as usize..end as usize).into())
            }
            x => bail!("Unknown index type: {}", x),
        }
    }

    /// Overwrite the index inplace.
    pub(crate) fn overwrite<B: Backend>(&self, container: &mut DataContainer<B>) -> Result<()> {
        if let Ok(index_name) = container.get_attr::<String>("_index") {
            container.as_group()?.delete(&index_name)?;
        }
        container.new_attr("_index", self.index_name.clone())?;
        let group = container.as_group()?;
        let arr: Array1<String> = self.clone().into_iter().collect();
        let mut data = group.new_array_dataset(&self.index_name, arr.into(), Default::default())?;
        match &self.index {
            Index::List(_) => {
                data.new_attr("index_type", "list")?;
            }
            Index::Intervals(intervals) => {
                let keys: Vec<String> = intervals.keys().cloned().collect();
                let values: Vec<Vec<u64>> = intervals
                    .values()
                    .map(|x| vec![x.start as u64, x.end as u64, x.size as u64, x.step as u64])
                    .collect();
                if data.new_attr("names", keys).is_err()
                    || data.new_attr("intervals", values).is_err()
                {
                    // fallback to "list"
                    data.new_attr("index_type", "list")?;
                    warn!("Failed to save interval index as attributes, fallback to list index");
                } else {
                    data.new_attr("index_type", "intervals")?;
                }
            }
            Index::Range(range) => {
                data.new_attr("index_type", "range")?;
                data.new_attr("start", range.start as u64)?;
                data.new_attr("end", range.end as u64)?;
            }
        }
        Ok(())
    }
}

impl IntoIterator for DataFrameIndex {
    type Item = String;
    type IntoIter = Box<dyn Iterator<Item = String>>;

    fn into_iter(self) -> Self::IntoIter {
        self.index.into_iter()
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



////////////////////////////////////////////////////////////////////////////////
/// Helper functions
////////////////////////////////////////////////////////////////////////////////

fn write_series<B: Backend, G: GroupOp<B>>(
    series: &Series,
    location: &G,
    name: &str,
) -> Result<DataContainer<B>> {
    match series.dtype() {
        DataType::UInt8 => series
            .u8()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .write(location, name),
        DataType::UInt16 => series
            .u16()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .write(location, name),
        DataType::UInt32 => series
            .u32()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .write(location, name),
        DataType::UInt64 => series
            .u64()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .write(location, name),
        DataType::Int8 => series
            .i8()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .write(location, name),
        DataType::Int16 => series
            .i16()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .write(location, name),
        DataType::Int32 => series
            .i32()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .write(location, name),
        DataType::Int64 => series
            .i64()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .write(location, name),
        DataType::Float32 => series
            .f32()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .write(location, name),
        DataType::Float64 => series
            .f64()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .write(location, name),
        DataType::Boolean => series
            .bool()?
            .into_iter()
            .map(|x| x.unwrap())
            .collect::<Array1<_>>()
            .write(location, name),
        DataType::String => series
            .str()?
            .into_iter()
            .map(|x| x.unwrap().to_string())
            .collect::<Array1<_>>()
            .write(location, name),
        DataType::Categorical(_, _) => series
            .categorical()?
            .iter_str()
            .collect::<CategoricalArray>()
            .write(location, name),
        other => bail!("Unsupported series data type: {:?}", other),
    }
}

fn read_series<B: Backend>(container: &DataContainer<B>) -> Result<Series> {
    let ty = container.encoding_type()?;
    match ty {
        crate::backend::DataType::Categorical => {
            let categories = container.as_group()?.open_dataset("categories")?;
            let s = match categories.dtype()? {
                ScalarType::String => CategoricalArray::read(container)?.into(),
                _ => read_cat_as_series(container)?,
            };
            Ok(s)
        },
        crate::backend::DataType::Array(_) => Ok(DynArray::read(container)?.into()),
        _ => bail!("Unsupported data type: {:?}", ty),
    }
}

/// Used to read non-string categorical data into regular arrays. After all, such
/// data should not be stored as categorical data.
fn read_cat_as_series<B: Backend>(container: &DataContainer<B>) -> Result<Series> {
    let group = container.as_group()?;
    let codes: Array1<i32> = group.open_dataset("codes")?.read_array_cast()?;
    let codes = codes.mapv(|x| if x < 0 { None } else { Some(x as usize) });
    let categories = group.open_dataset("categories")?.read_dyn_array().unwrap();

    macro_rules! fun {
        ($variant:ident, $value:expr) => {
            codes.iter().map(|x| x.map(|i| $value[i].clone())).collect()
        };
    }

    let arr = crate::macros::dyn_map!(categories, DynArray, fun);
    Ok(arr)
}

impl HasShape for Series {
    fn shape(&self) -> Shape {
        self.len().into()
    }
}

impl Selectable for Series {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        let i = SelectInfoElemBounds::new(info.as_ref()[0].as_ref(), self.len())
            .iter()
            .map(|x| x as u32)
            .collect::<Vec<_>>();
        self.take(&ChunkedArray::from_vec("idx".into(), i)).unwrap()
    }
}
