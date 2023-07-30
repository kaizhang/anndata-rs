use crate::{
    backend::*,
    data::{
        data_traits::*,
        scalar::DynScalar,
        slice::{Shape, SelectInfoElem, BoundedSelectInfoElem},
    },
};

use anyhow::{bail, ensure, anyhow, Result};
use ndarray::{ArrayView, Array, Array1, ArrayD, RemoveAxis, SliceInfoElem, Dimension, Axis};
use std::collections::HashMap;
use std::ops::Index;

/// A dynamic-typed array.
#[derive(Debug, Clone, PartialEq)]
pub enum DynArray {
    I8(ArrayD<i8>),
    I16(ArrayD<i16>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    U8(ArrayD<u8>),
    U16(ArrayD<u16>),
    U32(ArrayD<u32>),
    U64(ArrayD<u64>),
    Usize(ArrayD<usize>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    Bool(ArrayD<bool>),
    String(ArrayD<String>),
    Categorical(CategoricalArray),
}

impl From<CategoricalArray> for DynArray {
    fn from(cat: CategoricalArray) -> Self {
        Self::Categorical(cat)
    }
}

impl TryFrom<DynArray> for CategoricalArray {
    type Error = anyhow::Error;

    fn try_from(v: DynArray) -> Result<Self> {
        match v {
            DynArray::Categorical(cat) => Ok(cat),
            _ => bail!("Cannot convert {:?} to CategoricalArray", v),
        }
    }
}

macro_rules! impl_dyn_array_convert {
    ($from_type:ty, $to_type:ident) => {
        impl<D: RemoveAxis> From<Array<$from_type, D>> for DynArray {
            fn from(data: Array<$from_type, D>) -> Self {
                DynArray::$to_type(data.into_dyn())
            }
        }

        impl<D: RemoveAxis> TryFrom<DynArray> for Array<$from_type, D> {
            type Error = anyhow::Error;
            fn try_from(v: DynArray) -> Result<Self, Self::Error> {
                match v {
                    DynArray::$to_type(data) => {
                        let arr: ArrayD<$from_type> = data.try_into()?;
                        if let Some(n) = D::NDIM {
                            ensure!(arr.ndim() == n, format!("RemoveAxis mismatch: {} (in) != {} (out)", arr.ndim(), n));
                        }
                        Ok(arr.into_dimensionality::<D>()?)
                    },
                    _ => bail!("Cannot convert {:?} to ArrayD<$from_type>", v),
                }
            }
        }
    };
}

impl_dyn_array_convert!(i8, I8);
impl_dyn_array_convert!(i16, I16);
impl_dyn_array_convert!(i32, I32);
impl_dyn_array_convert!(i64, I64);
impl_dyn_array_convert!(u8, U8);
impl_dyn_array_convert!(u16, U16);
impl_dyn_array_convert!(u32, U32);
impl_dyn_array_convert!(u64, U64);
impl_dyn_array_convert!(usize, Usize);
impl_dyn_array_convert!(f32, F32);
impl_dyn_array_convert!(f64, F64);
impl_dyn_array_convert!(bool, Bool);
impl_dyn_array_convert!(String, String);

impl WriteData for DynArray {
    fn data_type(&self) -> DataType {
        match self {
            Self::I8(arr) => arr.data_type(),
            Self::I16(arr) => arr.data_type(),
            Self::I32(arr) => arr.data_type(),
            Self::I64(arr) => arr.data_type(),
            Self::U8(arr) => arr.data_type(),
            Self::U16(arr) => arr.data_type(),
            Self::U32(arr) => arr.data_type(),
            Self::U64(arr) => arr.data_type(),
            Self::Usize(arr) => arr.data_type(),
            Self::F32(arr) => arr.data_type(),
            Self::F64(arr) => arr.data_type(),
            Self::Bool(arr) => arr.data_type(),
            Self::String(arr) => arr.data_type(),
            Self::Categorical(arr) => arr.data_type(),
        }
    }
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        match self {
            Self::I8(array) => array.write(location, name),
            Self::I16(array) => array.write(location, name),
            Self::I32(array) => array.write(location, name),
            Self::I64(array) => array.write(location, name),
            Self::U8(array) => array.write(location, name),
            Self::U16(array) => array.write(location, name),
            Self::U32(array) => array.write(location, name),
            Self::U64(array) => array.write(location, name),
            Self::Usize(array) => array.write(location, name),
            Self::F32(array) => array.write(location, name),
            Self::F64(array) => array.write(location, name),
            Self::Bool(array) => array.write(location, name),
            Self::String(array) => array.write(location, name),
            Self::Categorical(array) => array.write(location, name),
        }
    }
}

impl ReadData for DynArray {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container {
            DataContainer::Dataset(dataset) => match dataset.dtype()? {
                ScalarType::I8 => Ok(Self::I8(dataset.read_array()?)),
                ScalarType::I16 => Ok(Self::I16(dataset.read_array()?)),
                ScalarType::I32 => Ok(Self::I32(dataset.read_array()?)),
                ScalarType::I64 => Ok(Self::I64(dataset.read_array()?)),
                ScalarType::U8 => Ok(Self::U8(dataset.read_array()?)),
                ScalarType::U16 => Ok(Self::U16(dataset.read_array()?)),
                ScalarType::U32 => Ok(Self::U32(dataset.read_array()?)),
                ScalarType::U64 => Ok(Self::U64(dataset.read_array()?)),
                ScalarType::Usize => Ok(Self::Usize(dataset.read_array()?)),
                ScalarType::F32 => Ok(Self::F32(dataset.read_array()?)),
                ScalarType::F64 => Ok(Self::F64(dataset.read_array()?)),
                ScalarType::Bool => Ok(Self::Bool(dataset.read_array()?)),
                ScalarType::String => Ok(Self::String(dataset.read_array()?)),
            },
            DataContainer::Group(_) =>
                Ok(Self::Categorical(CategoricalArray::read(container)?)),
        }
    }
}

impl HasShape for DynArray {
    fn shape(&self) -> Shape {
        match self {
            DynArray::I8(array) => array.shape().to_vec(),
            DynArray::I16(array) => array.shape().to_vec(),
            DynArray::I32(array) => array.shape().to_vec(),
            DynArray::I64(array) => array.shape().to_vec(),
            DynArray::U8(array) => array.shape().to_vec(),
            DynArray::U16(array) => array.shape().to_vec(),
            DynArray::U32(array) => array.shape().to_vec(),
            DynArray::U64(array) => array.shape().to_vec(),
            DynArray::Usize(array) => array.shape().to_vec(),
            DynArray::F32(array) => array.shape().to_vec(),
            DynArray::F64(array) => array.shape().to_vec(),
            DynArray::Bool(array) => array.shape().to_vec(),
            DynArray::String(array) => array.shape().to_vec(),
            DynArray::Categorical(array) => array.codes.shape().to_vec(),
        }
        .into()
    }
}

impl ArrayOp for DynArray {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        match self {
            DynArray::I8(array) => array.get(index).map(|x| (*x).into()),
            DynArray::I16(array) => array.get(index).map(|x| (*x).into()),
            DynArray::I32(array) => array.get(index).map(|x| (*x).into()),
            DynArray::I64(array) => array.get(index).map(|x| (*x).into()),
            DynArray::U8(array) => array.get(index).map(|x| (*x).into()),
            DynArray::U16(array) => array.get(index).map(|x| (*x).into()),
            DynArray::U32(array) => array.get(index).map(|x| (*x).into()),
            DynArray::U64(array) => array.get(index).map(|x| (*x).into()),
            DynArray::Usize(array) => array.get(index).map(|x| (*x).into()),
            DynArray::F32(array) => array.get(index).map(|x| (*x).into()),
            DynArray::F64(array) => array.get(index).map(|x| (*x).into()),
            DynArray::Bool(array) => array.get(index).map(|x| (*x).into()),
            DynArray::String(array) => array.get(index).map(|x| x.clone().into()),
            DynArray::Categorical(array) => array
                .codes
                .get(index)
                .map(|x| array.categories[*x as usize].clone().into()),
        }
    }

    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        match self {
            DynArray::I8(array) => ArrayOp::select(array, info).into(),
            DynArray::I16(array) => ArrayOp::select(array, info).into(),
            DynArray::I32(array) => ArrayOp::select(array, info).into(),
            DynArray::I64(array) => ArrayOp::select(array, info).into(),
            DynArray::U8(array) => ArrayOp::select(array, info).into(),
            DynArray::U16(array) => ArrayOp::select(array, info).into(),
            DynArray::U32(array) => ArrayOp::select(array, info).into(),
            DynArray::U64(array) => ArrayOp::select(array, info).into(),
            DynArray::Usize(array) => ArrayOp::select(array, info).into(),
            DynArray::F32(array) => ArrayOp::select(array, info).into(),
            DynArray::F64(array) => ArrayOp::select(array, info).into(),
            DynArray::Bool(array) => ArrayOp::select(array, info).into(),
            DynArray::String(array) => ArrayOp::select(array, info).into(),
            DynArray::Categorical(array) => CategoricalArray {
                codes: ArrayOp::select(&array.codes, info),
                categories: array.categories.clone(),
            }
            .into(),
        }
    }

    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            DynArray::U8(_) => ArrayD::<u8>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::U16(_) => ArrayD::<u16>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::U32(_) => ArrayD::<u32>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::U64(_) => ArrayD::<u64>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::Usize(_) => ArrayD::<usize>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::I8(_) => ArrayD::<i8>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::I16(_) => ArrayD::<i16>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::I32(_) => ArrayD::<i32>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::I64(_) => ArrayD::<i64>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::F32(_) => ArrayD::<f32>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::F64(_) => ArrayD::<f64>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::Bool(_) => ArrayD::<bool>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::String(_) => ArrayD::<String>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::Categorical(_) => todo!(),
        }
    }
}

impl WriteArrayData for DynArray {}
impl ReadArrayData for DynArray {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container.as_dataset()?.shape().into())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        match container {
            DataContainer::Dataset(dataset) => match dataset.dtype()? {
                ScalarType::I8 => Ok(Self::I8(dataset.read_array_slice(info)?)),
                ScalarType::I16 => Ok(Self::I16(dataset.read_array_slice(info)?)),
                ScalarType::I32 => Ok(Self::I32(dataset.read_array_slice(info)?)),
                ScalarType::I64 => Ok(Self::I64(dataset.read_array_slice(info)?)),
                ScalarType::U8 => Ok(Self::U8(dataset.read_array_slice(info)?)),
                ScalarType::U16 => Ok(Self::U16(dataset.read_array_slice(info)?)),
                ScalarType::U32 => Ok(Self::U32(dataset.read_array_slice(info)?)),
                ScalarType::U64 => Ok(Self::U64(dataset.read_array_slice(info)?)),
                ScalarType::Usize => Ok(Self::Usize(dataset.read_array_slice(info)?)),
                ScalarType::F32 => Ok(Self::F32(dataset.read_array_slice(info)?)),
                ScalarType::F64 => Ok(Self::F64(dataset.read_array_slice(info)?)),
                ScalarType::Bool => Ok(Self::Bool(dataset.read_array_slice(info)?)),
                ScalarType::String => Ok(Self::String(dataset.read_array_slice(info)?)),
            },
            DataContainer::Group(_) =>
                Ok(Self::Categorical(CategoricalArray::read_select(container, info)?)),
        }
    }
 
}

impl<'a, T: BackendData, D: RemoveAxis> WriteData for ArrayView<'a, T, D> {
    fn data_type(&self) -> DataType {
        DataType::Array(T::DTYPE)
    }
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let dataset = location.create_array_data(name, self, Default::default())?;
        let encoding_type = if T::DTYPE == ScalarType::String {
            "string-array"
        } else {
            "array"
        };
        let container = DataContainer::<B>::Dataset(dataset);
        container.write_str_attr("encoding-type", encoding_type)?;
        container.write_str_attr("encoding-version", "0.2.0")?;
        Ok(container)
    }
}

impl<T: BackendData, D: RemoveAxis> WriteData for Array<T, D> {
    fn data_type(&self) -> DataType {
        DataType::Array(T::DTYPE)
    }
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        self.view().write(location, name)
    }
}

impl<T: BackendData, D: RemoveAxis> HasShape for Array<T, D> {
    fn shape(&self) -> Shape {
        self.shape().to_vec().into()
    }
}

impl<'a, T: BackendData, D: RemoveAxis> HasShape for ArrayView<'a, T, D> {
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
                .map(|(x, n)| BoundedSelectInfoElem::new(x.as_ref(), *n)).collect();
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


impl<T: BackendData, D: RemoveAxis> WriteArrayData for Array<T, D> {}
impl<T: BackendData, D: RemoveAxis> WriteArrayData for &Array<T, D> {}
impl<'a, T: BackendData, D: RemoveAxis> WriteArrayData for ArrayView<'a, T, D> {}

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
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let group = location.create_group(name)?;
        group.write_str_attr("encoding-type", "categorical")?;
        group.write_str_attr("encoding-version", "0.2.0")?;
        group.write_scalar_attr("ordered", false)?;

        group.create_array_data("codes", &self.codes, Default::default())?;
        group.create_array_data("categories", &self.categories, Default::default())?;

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