mod dynamic;

pub use dynamic::{ArrayConvert, DynArray, DynCowArray, DynScalar};

use crate::{
    backend::*,
    data::{
        data_traits::*,
        slice::{SelectInfoElem, SelectInfoElemBounds, Shape},
    },
};

use anyhow::{anyhow, Result};
use ndarray::{Array, Array1, ArrayD, ArrayView, Axis, Dimension, RemoveAxis, SliceInfoElem};
use polars::{
    prelude::CategoricalChunkedBuilder,
    series::{IntoSeries, Series},
};
use std::collections::HashMap;
use std::ops::Index;

impl<'a, T: BackendData, D> Element for ArrayView<'a, T, D> {
    fn metadata(&self) -> MetaData {
        let encoding_type = match T::DTYPE {
            ScalarType::String => "string-array",
            _ => "array",
        };
        MetaData::new(encoding_type, "0.2.0", None)
    }

    fn data_type(&self) -> DataType {
        DataType::Array(T::DTYPE)
    }
}

impl<'a, T: BackendData, D: Dimension> Writable for ArrayView<'a, T, D> {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let dataset = location.new_array_dataset(name, self.into(), Default::default())?;
        let mut container = DataContainer::<B>::Dataset(dataset);
        self.metadata().save(&mut container)?;
        Ok(container)
    }
}

impl<'a, T, D: Dimension> HasShape for ArrayView<'a, T, D> {
    fn shape(&self) -> Shape {
        self.shape().to_vec().into()
    }
}

impl<T: BackendData, D> Element for Array<T, D> {
    fn metadata(&self) -> MetaData {
        let encoding_type = match T::DTYPE {
            ScalarType::String => "string-array",
            _ => "array",
        };
        MetaData::new(encoding_type, "0.2.0", None)
    }

    fn data_type(&self) -> DataType {
        DataType::Array(T::DTYPE)
    }
}

impl<T: BackendData, D: Dimension> Writable for Array<T, D> {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        self.view().write(location, name)
    }
}

impl<T, D: Dimension> HasShape for Array<T, D> {
    fn shape(&self) -> Shape {
        self.shape().to_vec().into()
    }
}

impl<T: BackendData, D: Dimension> Indexable for Array<T, D> {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        self.view().into_dyn().get(index).map(|x| x.into_dyn())
    }
}

impl<T: Clone, D: Dimension> Selectable for Array<T, D> {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        let arr = self.view().into_dyn();
        let slices = info
            .as_ref()
            .into_iter()
            .map(|x| match x.as_ref() {
                SelectInfoElem::Slice(slice) => Some(SliceInfoElem::from(slice.clone())),
                _ => None,
            })
            .collect::<Option<Vec<_>>>();
        if let Some(slices) = slices {
            arr.slice(slices.as_slice()).into_owned()
        } else {
            let shape = self.shape();
            let select: Vec<_> = info
                .as_ref()
                .into_iter()
                .zip(shape)
                .map(|(x, n)| SelectInfoElemBounds::new(x.as_ref(), *n))
                .collect();
            let new_shape = select.iter().map(|x| x.len()).collect::<Vec<_>>();
            ArrayD::from_shape_fn(new_shape, |idx| {
                let new_idx: Vec<_> = (0..idx.ndim())
                    .into_iter()
                    .map(|i| select[i].index(idx[i]))
                    .collect();
                arr.index(new_idx.as_slice()).clone()
            })
        }
        .into_dimensionality::<D>()
        .unwrap()
    }
}

impl<T: Clone, D: RemoveAxis> Stackable for Array<T, D> {
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        iter.reduce(|mut this, other| {
            this.append(Axis(0), other.view()).unwrap();
            this
        })
        .ok_or_else(|| anyhow!("Cannot vstack empty iterator"))
    }
}

impl<T: BackendData, D: Dimension> Readable for Array<T, D> {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        Ok(container.as_dataset()?.read_array::<T, D>()?)
    }
}

impl<T: BackendData, D: Dimension> ReadableArray for Array<T, D> {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container.as_dataset()?.shape().into())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        container.as_dataset()?.read_array_slice(info)
    }
}

impl<T: BackendData, D: Dimension> WritableArray for Array<T, D> {}
impl<T: BackendData, D: Dimension> WritableArray for &Array<T, D> {}
impl<'a, T: BackendData, D: Dimension> WritableArray for ArrayView<'a, T, D> {}

/// CategoricalArrays store discrete values.
/// These arrays encode the values as small width integers (codes), which map to
/// the original label set (categories). Each entry in the codes array is the
/// zero-based index of the encoded value in the categories array.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalArray {
    pub codes: ArrayD<Option<u32>>,
    pub categories: Array1<String>,
}

impl Into<Series> for CategoricalArray {
    fn into(self) -> Series {
        CategoricalChunkedBuilder::new(
            "".into(),
            self.codes.len(),
            polars::datatypes::CategoricalOrdering::Lexical,
        )
        .drain_iter_and_finish(
            self.codes
                .into_iter()
                .map(|i| Some(self.categories[i? as usize].as_str())),
        )
        .into_series()
    }
}

impl<'a> FromIterator<Option<&'a str>> for CategoricalArray {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Option<&'a str>>,
    {
        let mut str_to_id = HashMap::new();
        let mut counter = 0;
        let codes: Array1<Option<u32>> = iter
            .into_iter()
            .map(|x| {
                let str = x?.to_string();
                let idx = match str_to_id.get(&str) {
                    Some(v) => *v,
                    None => {
                        let v = counter;
                        str_to_id.insert(str, v);
                        counter += 1;
                        v
                    }
                };
                Some(idx)
            })
            .collect();
        let mut categories = str_to_id.drain().collect::<Vec<_>>();
        categories.sort_by_key(|x| x.1);
        CategoricalArray {
            codes: codes.into_dyn(),
            categories: categories.into_iter().map(|x| x.0).collect(),
        }
    }
}

impl Element for CategoricalArray {
    fn metadata(&self) -> MetaData {
        let mut metadata = HashMap::new();
        metadata.insert("ordered".to_string(), false.into());
        MetaData::new("categorical", "0.2.0", Some(metadata))
    }

    fn data_type(&self) -> DataType {
        DataType::Categorical
    }
}

impl Writable for CategoricalArray {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let mut group = location.new_group(name)?;
        self.metadata().save(&mut group)?;

        self.codes
            .map(|x| x.map_or(-1, |x| x as i32))
            .write(&group, "codes")?;
        self.categories.view().write(&group, "categories")?;

        Ok(DataContainer::Group(group))
    }
}

impl HasShape for CategoricalArray {
    fn shape(&self) -> Shape {
        self.codes.shape().to_vec().into()
    }
}

impl Indexable for CategoricalArray {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        let code = *self.codes.get(index)?;
        Some(self.categories[code? as usize].clone().into())
    }
}

impl Selectable for CategoricalArray {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        CategoricalArray {
            codes: Selectable::select(&self.codes, info),
            categories: self.categories.clone(),
        }
    }
}

impl WritableArray for CategoricalArray {}

impl Readable for CategoricalArray {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let group = container.as_group()?;
        let codes: ArrayD<i32> = group.open_dataset("codes")?.read_array_cast()?;
        let codes = codes.mapv(|x| if x < 0 { None } else { Some(x as u32) });
        let categories = group.open_dataset("categories")?.read_array().unwrap();
        Ok(CategoricalArray { codes, categories })
    }
}

impl ReadableArray for CategoricalArray {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        let group = container.as_group()?;
        let codes = group.open_dataset("codes")?.shape();
        Ok(codes.into())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        let group = container.as_group()?;
        let codes: ArrayD<i32> = group.open_dataset("codes")?.read_array_slice(info)?;
        let codes = codes.mapv(|x| if x < 0 { None } else { Some(x as u32) });
        let categories = group.open_dataset("categories")?.read_array()?;
        Ok(CategoricalArray { codes, categories })
    }
}
