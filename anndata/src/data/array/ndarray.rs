use crate::{
    backend::*,
    data::{
        data_traits::*,
        array::DynScalar,
        slice::{Shape, SelectInfoElem, SelectInfoElemBounds},
    },
};

use anyhow::{anyhow, Result};
use ndarray::{ArrayView, Array, Array1, ArrayD, RemoveAxis, SliceInfoElem, Dimension, Axis};
use std::collections::HashMap;
use std::ops::Index;

impl<'a, T: BackendData, D: Dimension> WriteData for ArrayView<'a, T, D> {
    fn data_type(&self) -> DataType {
        DataType::Array(T::DTYPE)
    }
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let dataset = location.new_array_dataset(name, self, Default::default())?;
        let encoding_type = if T::DTYPE == ScalarType::String {
            "string-array"
        } else {
            "array"
        };
        let mut container = DataContainer::<B>::Dataset(dataset);
        container.new_str_attr("encoding-type", encoding_type)?;
        container.new_str_attr("encoding-version", "0.2.0")?;
        Ok(container)
    }
}

impl<T: BackendData, D: Dimension> WriteData for Array<T, D> {
    fn data_type(&self) -> DataType {
        DataType::Array(T::DTYPE)
    }
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        self.view().write(location, name)
    }
}

impl<T: BackendData, D: Dimension> HasShape for Array<T, D> {
    fn shape(&self) -> Shape {
        self.shape().to_vec().into()
    }
}

impl<'a, T: BackendData, D: Dimension> HasShape for ArrayView<'a, T, D> {
    fn shape(&self) -> Shape {
        self.shape().to_vec().into()
    }
}

impl<T: BackendData, D: RemoveAxis> ArrayOp for Array<T, D> {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        self.view().into_dyn().get(index).map(|x| x.into_dyn())
    }

    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        let arr = self.view().into_dyn();
        let slices = info.as_ref().into_iter().map(|x| match x.as_ref() {
            SelectInfoElem::Slice(slice) => Some(SliceInfoElem::from(slice.clone())),
            _ => None,
        }).collect::<Option<Vec<_>>>();
        if let Some(slices) = slices {
            arr.slice(slices.as_slice()).into_owned()
        } else {
            let shape = self.shape();
            let select: Vec<_> = info.as_ref().into_iter().zip(shape)
                .map(|(x, n)| SelectInfoElemBounds::new(x.as_ref(), *n)).collect();
            let new_shape = select.iter().map(|x| x.len()).collect::<Vec<_>>();
            ArrayD::from_shape_fn(new_shape, |idx| {
                let new_idx: Vec<_> = (0..idx.ndim()).into_iter().map(|i| select[i].index(idx[i])).collect();
                arr.index(new_idx.as_slice()).clone()
            })
        }.into_dimensionality::<D>().unwrap()
    }

    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        iter.reduce(|mut this, other| {
            this.append(Axis(0), other.view()).unwrap();
            this
        }).ok_or_else(|| anyhow!("Cannot vstack empty iterator"))
    }
}

impl<T: BackendData, D: RemoveAxis> ReadData for Array<T, D> {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        Ok(container.as_dataset()?.read_array::<T, D>()?)
    }
}

impl<T: BackendData, D: RemoveAxis> ReadArrayData for Array<T, D> {
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


impl<T: BackendData, D: Dimension> WriteArrayData for Array<T, D> {}
impl<T: BackendData, D: Dimension> WriteArrayData for &Array<T, D> {}
impl<'a, T: BackendData, D: Dimension> WriteArrayData for ArrayView<'a, T, D> {}

#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalArray {
    pub codes: ArrayD<u32>,
    pub categories: Array1<String>,
}

impl<'a> FromIterator<&'a str> for CategoricalArray {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a str>,
    {
        let mut str_to_id = HashMap::new();
        let mut counter = 0;
        let codes: Array1<u32> = iter
            .into_iter()
            .map(|x| {
                let str = x.to_string();
                match str_to_id.get(&str) {
                    Some(v) => *v,
                    None => {
                        let v = counter;
                        str_to_id.insert(str, v);
                        counter += 1;
                        v
                    }
                }
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

impl WriteData for CategoricalArray {
    fn data_type(&self) -> DataType {
        DataType::Categorical
    }
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let mut group = location.new_group(name)?;
        group.new_str_attr("encoding-type", "categorical")?;
        group.new_str_attr("encoding-version", "0.2.0")?;
        group.new_scalar_attr("ordered", false)?;

        group.new_array_dataset("codes", &self.codes, Default::default())?;
        group.new_array_dataset("categories", &self.categories, Default::default())?;

        Ok(DataContainer::Group(group))
    }
}

impl HasShape for CategoricalArray {
    fn shape(&self) -> Shape {
        self.codes.shape().to_vec().into()
    }
}

impl WriteArrayData for CategoricalArray {}

impl ReadData for CategoricalArray {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let group = container.as_group()?;
        let codes = group.open_dataset("codes")?.read_array()?;
        let categories = group
            .open_dataset("categories")?
            .read_array()?;
        Ok(CategoricalArray { codes, categories })
    }
}

impl ReadArrayData for CategoricalArray {
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
        let codes = group.open_dataset("codes")?.read_array_slice(info)?;
        let categories = group
            .open_dataset("categories")?
            .read_array()?;
        Ok(CategoricalArray { codes, categories })
    }
}