use crate::backend::*;
use crate::data::{SelectInfoElem, DynScalar, ReadData, WriteData};
use crate::data::slice::BoundedSelectInfoElem;

use itertools::Itertools;
use anyhow::{bail, ensure, Result};
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{ArrayView, Array, Array1, ArrayD, Dimension};
use std::collections::HashMap;
use std::ops::{Index, IndexMut};
use smallvec::SmallVec;

#[derive(Clone, Debug)]
pub struct Shape(SmallVec<[usize; 3]>);

impl Shape {
    pub fn ndim(&self) -> usize {
        self.0.len()
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.as_slice().iter().map(|x| x.to_string()).join(" x "))
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        &self.0
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl From<Vec<usize>> for Shape {
    fn from(shape: Vec<usize>) -> Self {
        Self(SmallVec::from_vec(shape))
    }
}


pub trait HasShape {
    fn shape(&self) -> Shape;
}

pub trait ArrayOp: HasShape {
    fn get(&self, index: &[usize]) -> Option<DynScalar>;
    fn select<S, E>(&self, info: S) -> Self
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>;
}

pub trait WriteArrayData: WriteData {}

pub trait ReadArrayData: ReadData {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape>;

    fn read_select<B, S, E>(container: &DataContainer<B>, info: S) -> Result<Self>
    where
        B: Backend,
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
        Self: Sized;
        //Self::read(container).map(|data| data.select(info))
}

/// A dynamic-typed array.
#[derive(Debug, Clone)]
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
        impl<D: Dimension> From<Array<$from_type, D>> for DynArray {
            fn from(data: Array<$from_type, D>) -> Self {
                DynArray::$to_type(data.into_dyn())
            }
        }

        impl<D: Dimension> TryFrom<DynArray> for Array<$from_type, D> {
            type Error = anyhow::Error;
            fn try_from(v: DynArray) -> Result<Self, Self::Error> {
                match v {
                    DynArray::$to_type(data) => {
                        let arr: ArrayD<$from_type> = data.try_into()?;
                        if let Some(n) = D::NDIM {
                            ensure!(arr.ndim() == n, format!("Dimension mismatch: {} (in) != {} (out)", arr.ndim(), n));
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

    fn select<S, E>(&self, info: S) -> Self
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
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
}

impl WriteArrayData for DynArray {}
impl ReadArrayData for DynArray {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container.as_dataset()?.shape()?.into())
    }

    fn read_select<B, S, E>(container: &DataContainer<B>, info: S) -> Result<Self>
    where
        B: Backend,
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
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

impl<'a, T: BackendData, D: Dimension> WriteData for ArrayView<'a, T, D> {
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let dataset = location.write_array(name, self)?;
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

impl<T: BackendData, D: Dimension> WriteData for Array<T, D> {
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        self.view().write(location, name)
    }
}

impl<T: BackendData, D: Dimension> WriteData for &Array<T, D> {
    fn write<B: Backend, G: GroupOp<Backend = B>>(
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

impl<T: BackendData, D: Dimension> HasShape for &Array<T, D> {
    fn shape(&self) -> Shape {
        (*self).shape().to_vec().into()
    }
}

impl<T: BackendData, D: Dimension> ArrayOp for Array<T, D> {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        self.view().into_dyn().get(index).map(|x| x.into_dyn())
    }

    fn select<S, E>(&self, info: S) -> Self
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
    {
        let arr = self.view().into_dyn();
        let slices = info.as_ref().into_iter().map(|x| match x.as_ref() {
            SelectInfoElem::Slice(slice) => Some(slice.clone()),
            _ => None,
        }).collect::<Option<Vec<_>>>();
        if let Some(slices) = slices {
            arr.slice(slices.as_slice()).into_owned()
        } else {
            let shape = self.shape();
            let select: Vec<_> = info.as_ref().into_iter().zip(shape)
                .map(|(x, n)| BoundedSelectInfoElem::new(x.as_ref(), *n).unwrap()).collect();
            let new_shape = select.iter().map(|x| x.len()).collect::<Vec<_>>();
            ArrayD::from_shape_fn(new_shape, |idx| {
                let new_idx: Vec<_> = (0..idx.ndim()).into_iter().map(|i| select[i].index(idx[i])).collect();
                arr.index(new_idx.as_slice()).clone()
            })
        }.into_dimensionality::<D>().unwrap()
        /*
        // Perform slice operation on the array.
        let slice_info: Vec<_> = info
            .as_ref()
            .iter()
            .map(|x| match x.as_ref() {
                SelectInfoElem::Index(_) => SLICE_FULL,
                SelectInfoElem::Slice(s) => s.clone(),
            })
            .collect();
        let arr = self.slice(slice_info.as_slice());

        // Perform selection on the array.
        info.as_ref()
            .iter()
            .enumerate()
            .fold(None::<ArrayD<T>>, |acc, (axis, sel)| {
                if let SelectInfoElem::Index(indices) = sel.as_ref() {
                    if let Some(acc) = acc {
                        Some(acc.select(Axis(axis), indices.as_slice()))
                    } else {
                        Some(arr.select(Axis(axis), indices.as_slice()))
                    }
                } else {
                    acc
                }
            })
            .unwrap_or(arr.to_owned())
        */
    }

}

impl<T: BackendData, D: Dimension> ReadData for Array<T, D> {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        Ok(container.as_dataset()?.read_array::<T, D>()?)
    }
}

impl<T: BackendData, D: Dimension> ReadArrayData for Array<T, D> {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container.as_dataset()?.shape()?.into())
    }

    fn read_select<B, S, E>(container: &DataContainer<B>, info: S) -> Result<Self>
        where
            B: Backend,
            S: AsRef<[E]>,
            E: AsRef<SelectInfoElem>,
    {
        container.as_dataset()?.read_array_slice(info)
    }
}


impl<T: BackendData, D: Dimension> WriteArrayData for Array<T, D> {}
impl<T: BackendData, D: Dimension> WriteArrayData for &Array<T, D> {}
impl<'a, T: BackendData, D: Dimension> WriteArrayData for ArrayView<'a, T, D> {}

#[derive(Debug, Clone)]
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
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let group = location.create_group(name)?;
        group.write_str_attr("encoding-type", "categorical")?;
        group.write_str_attr("encoding-version", "0.2.0")?;

        group.write_array("codes", &self.codes)?;
        group.write_array("categories", &self.categories)?;

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
        let codes = group.open_dataset("codes")?.shape()?;
        Ok(codes.into())
    }

    fn read_select<B, S, E>(container: &DataContainer<B>, info: S) -> Result<Self>
        where
            B: Backend,
            S: AsRef<[E]>,
            E: AsRef<SelectInfoElem>,
    {
        let group = container.as_group()?;
        let codes = group.open_dataset("codes")?.read_array_slice(info)?;
        let categories = group
            .open_dataset("categories")?
            .read_array()?;
        Ok(CategoricalArray { codes, categories })
    }
}


#[derive(Debug, Clone)]
pub enum DynCsrMatrix {
    I8(CsrMatrix<i8>),
    I16(CsrMatrix<i16>),
    I32(CsrMatrix<i32>),
    I64(CsrMatrix<i64>),
    U8(CsrMatrix<u8>),
    U16(CsrMatrix<u16>),
    U32(CsrMatrix<u32>),
    U64(CsrMatrix<u64>),
    Usize(CsrMatrix<usize>),
    F32(CsrMatrix<f32>),
    F64(CsrMatrix<f64>),
    Bool(CsrMatrix<bool>),
    String(CsrMatrix<String>),
}

macro_rules! impl_into_dyn_csr {
    ($from_type:ty, $to_type:ident) => {
        impl From<CsrMatrix<$from_type>> for DynCsrMatrix {
            fn from(data: CsrMatrix<$from_type>) -> Self {
                DynCsrMatrix::$to_type(data)
            }
        }
        impl TryFrom<DynCsrMatrix> for CsrMatrix<$from_type> {
            type Error = anyhow::Error;
            fn try_from(data: DynCsrMatrix) -> Result<Self> {
                match data {
                    DynCsrMatrix::$to_type(data) => Ok(data),
                    _ => bail!("Cannot convert to CsrMatrix<$from_type>"),
                }
            }
        }
    };
}

impl_into_dyn_csr!(i8, I8);
impl_into_dyn_csr!(i16, I16);
impl_into_dyn_csr!(i32, I32);
impl_into_dyn_csr!(i64, I64);
impl_into_dyn_csr!(u8, U8);
impl_into_dyn_csr!(u16, U16);
impl_into_dyn_csr!(u32, U32);
impl_into_dyn_csr!(u64, U64);
impl_into_dyn_csr!(usize, Usize);
impl_into_dyn_csr!(f32, F32);
impl_into_dyn_csr!(f64, F64);
impl_into_dyn_csr!(bool, Bool);
impl_into_dyn_csr!(String, String);

impl WriteData for DynCsrMatrix {
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        match self {
            DynCsrMatrix::I8(data) => data.write(location, name),
            DynCsrMatrix::I16(data) => data.write(location, name),
            DynCsrMatrix::I32(data) => data.write(location, name),
            DynCsrMatrix::I64(data) => data.write(location, name),
            DynCsrMatrix::U8(data) => data.write(location, name),
            DynCsrMatrix::U16(data) => data.write(location, name),
            DynCsrMatrix::U32(data) => data.write(location, name),
            DynCsrMatrix::U64(data) => data.write(location, name),
            DynCsrMatrix::Usize(data) => data.write(location, name),
            DynCsrMatrix::F32(data) => data.write(location, name),
            DynCsrMatrix::F64(data) => data.write(location, name),
            DynCsrMatrix::Bool(data) => data.write(location, name),
            DynCsrMatrix::String(data) => data.write(location, name),
        }
    }
}

impl ReadData for DynCsrMatrix {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container {
            DataContainer::Group(group) => match group.open_dataset("data")?.dtype()? {
                ScalarType::I8 => CsrMatrix::<i8>::read(container).map(DynCsrMatrix::I8),
                ScalarType::I16 => CsrMatrix::<i16>::read(container).map(DynCsrMatrix::I16),
                ScalarType::I32 => CsrMatrix::<i32>::read(container).map(DynCsrMatrix::I32),
                ScalarType::I64 => CsrMatrix::<i64>::read(container).map(DynCsrMatrix::I64),
                ScalarType::U8 => CsrMatrix::<u8>::read(container).map(DynCsrMatrix::U8),
                ScalarType::U16 => CsrMatrix::<u16>::read(container).map(DynCsrMatrix::U16),
                ScalarType::U32 => CsrMatrix::<u32>::read(container).map(DynCsrMatrix::U32),
                ScalarType::U64 => CsrMatrix::<u64>::read(container).map(DynCsrMatrix::U64),
                ScalarType::Usize => CsrMatrix::<usize>::read(container).map(DynCsrMatrix::Usize),
                ScalarType::F32 => CsrMatrix::<f32>::read(container).map(DynCsrMatrix::F32),
                ScalarType::F64 => CsrMatrix::<f64>::read(container).map(DynCsrMatrix::F64),
                ScalarType::Bool => CsrMatrix::<bool>::read(container).map(DynCsrMatrix::Bool),
                ScalarType::String => CsrMatrix::<String>::read(container).map(DynCsrMatrix::String),
            },
            _ => bail!("cannot read csr matrix from non-group container"),
        }
    }
}

impl HasShape for DynCsrMatrix {
    fn shape(&self) -> Shape {
        match self {
            DynCsrMatrix::I8(m) => m.shape(),
            DynCsrMatrix::I16(m) => m.shape(),
            DynCsrMatrix::I32(m) => m.shape(),
            DynCsrMatrix::I64(m) => m.shape(),
            DynCsrMatrix::U8(m) => m.shape(),
            DynCsrMatrix::U16(m) => m.shape(),
            DynCsrMatrix::U32(m) => m.shape(),
            DynCsrMatrix::U64(m) => m.shape(),
            DynCsrMatrix::Usize(m) => m.shape(),
            DynCsrMatrix::F32(m) => m.shape(),
            DynCsrMatrix::F64(m) => m.shape(),
            DynCsrMatrix::Bool(m) => m.shape(),
            DynCsrMatrix::String(m) => m.shape(),
        }
    }
}

impl ArrayOp for DynCsrMatrix {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        todo!()
    }

    fn select<S, E>(&self, info: S) -> Self
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
    {
        todo!()
    }
}

impl WriteArrayData for DynCsrMatrix {}
impl ReadArrayData for DynCsrMatrix {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container.as_group()?.read_arr_attr("shape")?.to_vec().into())
    }

    fn read_select<B, S, E>(container: &DataContainer<B>, info: S) -> Result<Self>
        where
            B: Backend,
            S: AsRef<[E]>,
            E: AsRef<SelectInfoElem>,
            Self: Sized {
        todo!()
    }
}

impl<T> HasShape for &CsrMatrix<T> {
    fn shape(&self) -> Shape {
        vec![self.nrows(), self.ncols()].into()
    }
}

impl<T> HasShape for CsrMatrix<T> {
    fn shape(&self) -> Shape {
        (&self).shape()
    }
}

impl<T: BackendData> WriteData for CsrMatrix<T> {
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        (&self).write(location, name)
    }
}
 
impl<T: BackendData> WriteData for &CsrMatrix<T> {
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let group = location.create_group(name)?;
        let shape = self.shape();

        group.write_str_attr("encoding-type", "csr_matrix")?;
        group.write_str_attr("encoding-version", "0.2.0")?;
        group.write_arr_attr("shape", shape.as_ref())?;

        group.write_array("data", &self.values())?;

        let num_cols = shape[1];
        // Use i32 or i64 as indices type in order to be compatible with scipy
        if TryInto::<i32>::try_into(num_cols.saturating_sub(1)).is_ok() {
            let try_convert_indptr: Option<Vec<i32>> = self
                .row_offsets()
                .iter()
                .map(|x| (*x).try_into().ok())
                .collect();
            if let Some(indptr_i32) = try_convert_indptr {
                group.write_array("indptr", &indptr_i32)?;
                group.write_array(
                    "indices",
                    self.col_indices()
                        .iter()
                        .map(|x| (*x) as i32)
                        .collect::<Vec<_>>()
                        .as_slice(),
                )?;
            } else {
                group.write_array(
                    "indptr",
                    self.row_offsets()
                        .iter()
                        .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                        .collect::<Vec<_>>()
                        .as_slice(),
                )?;
                group.write_array(
                    "indices",
                    self.col_indices()
                        .iter()
                        .map(|x| (*x) as i64)
                        .collect::<Vec<_>>()
                        .as_slice(),
                )?;
            }
        } else if TryInto::<i64>::try_into(num_cols.saturating_sub(1)).is_ok() {
            group.write_array(
                "indptr",
                self.row_offsets()
                    .iter()
                    .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )?;
            group.write_array(
                "indices",
                self.col_indices()
                    .iter()
                    .map(|x| (*x) as i64)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )?;
        } else {
            panic!(
                "The number of columns ({}) is too large to be stored as i64",
                num_cols
            );
        }

        Ok(DataContainer::Group(group))
    }
}


impl<T: BackendData> ReadData for CsrMatrix<T> {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let group = container.as_group()?;
        let shape: Vec<usize> = group.read_arr_attr("shape")?.to_vec();
        let data = group.open_dataset("data")?.read_array()?.to_vec();
        let indices: Vec<usize> = group.open_dataset("indices")?.read_array()?.to_vec();
        let indptr: Vec<usize> = group.open_dataset("indptr")?.read_array()?.to_vec();
        Ok(CsrMatrix::try_from_csr_data(shape[0], shape[1], indptr, indices, data).unwrap())
    }
}