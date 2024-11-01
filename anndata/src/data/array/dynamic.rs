use crate::{
    backend::*,
    data::{
        CategoricalArray,
        data_traits::*,
        slice::{SelectInfoElem, Shape},
    },
};

use anyhow::{bail, ensure, Result};
use ndarray::{arr0, Array, ArrayD, ArrayView, CowArray, Dimension, IxDyn};

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
    ($($from:ident, $to:ident),*) => {
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

            impl WriteData for $from {
                fn data_type(&self) -> DataType {
                    DataType::Scalar(ScalarType::$to)
                }
                fn write<B: Backend, G: GroupOp<B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
                    let dataset = location.new_scalar_dataset(name, self)?;
                    let mut container = DataContainer::Dataset(dataset);
                    let encoding_type = if $from::DTYPE == ScalarType::String {
                        "string"
                    } else {
                        "numeric-scalar"
                    };
                    container.new_str_attr("encoding-type", encoding_type)?;
                    container.new_str_attr("encoding-version", "0.2.0")?;
                    Ok(container)
                }
            }
        )*
    };
}

impl_from_dynscalar!(
    i8, I8, i16, I16, i32, I32, i64, I64, u8, U8, u16, U16, u32, U32, u64, U64, usize, Usize, f32,
    F32, f64, F64, bool, Bool, String, String
);

impl WriteData for DynScalar {
    fn data_type(&self) -> DataType {
        match self {
            DynScalar::I8(_) => DataType::Scalar(ScalarType::I8),
            DynScalar::I16(_) => DataType::Scalar(ScalarType::I16),
            DynScalar::I32(_) => DataType::Scalar(ScalarType::I32),
            DynScalar::I64(_) => DataType::Scalar(ScalarType::I64),
            DynScalar::U8(_) => DataType::Scalar(ScalarType::U8),
            DynScalar::U16(_) => DataType::Scalar(ScalarType::U16),
            DynScalar::U32(_) => DataType::Scalar(ScalarType::U32),
            DynScalar::U64(_) => DataType::Scalar(ScalarType::U64),
            DynScalar::Usize(_) => DataType::Scalar(ScalarType::Usize),
            DynScalar::F32(_) => DataType::Scalar(ScalarType::F32),
            DynScalar::F64(_) => DataType::Scalar(ScalarType::F64),
            DynScalar::Bool(_) => DataType::Scalar(ScalarType::Bool),
            DynScalar::String(_) => DataType::Scalar(ScalarType::String),
        }
    }

    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
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
            _ => bail!("Cannot convert {:?} to CategoricalArray", v.data_type()),
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
                            ensure!(
                                arr.ndim() == n,
                                format!("Dimension mismatch: {} (in) != {} (out)", arr.ndim(), n)
                            );
                        }
                        Ok(arr.into_dimensionality::<D>()?)
                    }
                    _ => bail!(
                        "Cannot convert {:?} to {} ArrayD",
                        v.data_type(),
                        stringify!($from_type)
                    ),
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
impl_dyn_array_convert!(f32, F32);
impl_dyn_array_convert!(f64, F64);
impl_dyn_array_convert!(bool, Bool);
impl_dyn_array_convert!(String, String);

impl<D: Dimension> From<Array<usize, D>> for DynArray {
    fn from(data: Array<usize, D>) -> Self {
        DynArray::Usize(data.into_dyn())
    }
}

impl<D: Dimension> TryFrom<DynArray> for Array<usize, D> {
    type Error = anyhow::Error;
    fn try_from(v: DynArray) -> Result<Self, Self::Error> {
        match v {
            DynArray::Usize(data) => {
                let arr: ArrayD<usize> = data.try_into()?;
                if let Some(n) = D::NDIM {
                    ensure!(
                        arr.ndim() == n,
                        format!("Dimension mismatch: {} (in) != {} (out)", arr.ndim(), n)
                    );
                }
                Ok(arr.into_dimensionality::<D>()?)
            }
            _ => bail!(
                "Cannot convert {:?} to {} ArrayD",
                v.data_type(),
                stringify!(usize)
            ),
        }
    }
}

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
    fn write<B: Backend, G: GroupOp<B>>(
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
            DataContainer::Group(_) => Ok(Self::Categorical(CategoricalArray::read(container)?)),
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
            DynArray::U8(_) => {
                ArrayD::<u8>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::U16(_) => {
                ArrayD::<u16>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::U32(_) => {
                ArrayD::<u32>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::U64(_) => {
                ArrayD::<u64>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::Usize(_) => {
                ArrayD::<usize>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::I8(_) => {
                ArrayD::<i8>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::I16(_) => {
                ArrayD::<i16>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::I32(_) => {
                ArrayD::<i32>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::I64(_) => {
                ArrayD::<i64>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::F32(_) => {
                ArrayD::<f32>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::F64(_) => {
                ArrayD::<f64>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::Bool(_) => {
                ArrayD::<bool>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            DynArray::String(_) => {
                ArrayD::<String>::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
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
            DataContainer::Group(_) => Ok(Self::Categorical(CategoricalArray::read_select(
                container, info,
            )?)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DynCowArray<'a> {
    I8(CowArray<'a, i8, IxDyn>),
    I16(CowArray<'a, i16, IxDyn>),
    I32(CowArray<'a, i32, IxDyn>),
    I64(CowArray<'a, i64, IxDyn>),
    U8(CowArray<'a, u8, IxDyn>),
    U16(CowArray<'a, u16, IxDyn>),
    U32(CowArray<'a, u32, IxDyn>),
    U64(CowArray<'a, u64, IxDyn>),
    Usize(CowArray<'a, usize, IxDyn>),
    F32(CowArray<'a, f32, IxDyn>),
    F64(CowArray<'a, f64, IxDyn>),
    Bool(CowArray<'a, bool, IxDyn>),
    String(CowArray<'a, String, IxDyn>),
}

impl From<DynScalar> for DynCowArray<'_> {
    fn from(scalar: DynScalar) -> Self {
        match scalar {
            DynScalar::I8(val) => DynCowArray::I8(arr0(val).into_dyn().into()),
            DynScalar::I16(val) => DynCowArray::I16(arr0(val).into_dyn().into()),
            DynScalar::I32(val) => DynCowArray::I32(arr0(val).into_dyn().into()),
            DynScalar::I64(val) => DynCowArray::I64(arr0(val).into_dyn().into()),
            DynScalar::U8(val) => DynCowArray::U8(arr0(val).into_dyn().into()),
            DynScalar::U16(val) => DynCowArray::U16(arr0(val).into_dyn().into()),
            DynScalar::U32(val) => DynCowArray::U32(arr0(val).into_dyn().into()),
            DynScalar::U64(val) => DynCowArray::U64(arr0(val).into_dyn().into()),
            DynScalar::Usize(val) => DynCowArray::Usize(arr0(val).into_dyn().into()),
            DynScalar::F32(val) => DynCowArray::F32(arr0(val).into_dyn().into()),
            DynScalar::F64(val) => DynCowArray::F64(arr0(val).into_dyn().into()),
            DynScalar::Bool(val) => DynCowArray::Bool(arr0(val).into_dyn().into()),
            DynScalar::String(val) => DynCowArray::String(arr0(val).into_dyn().into()),
        }
    }
}

impl DynCowArray<'_> {
    pub fn ndim(&self) -> usize {
        match self {
            DynCowArray::I8(arr) => arr.ndim(),
            DynCowArray::I16(arr) => arr.ndim(),
            DynCowArray::I32(arr) => arr.ndim(),
            DynCowArray::I64(arr) => arr.ndim(),
            DynCowArray::U8(arr) => arr.ndim(),
            DynCowArray::U16(arr) => arr.ndim(),
            DynCowArray::U32(arr) => arr.ndim(),
            DynCowArray::U64(arr) => arr.ndim(),
            DynCowArray::Usize(arr) => arr.ndim(),
            DynCowArray::F32(arr) => arr.ndim(),
            DynCowArray::F64(arr) => arr.ndim(),
            DynCowArray::Bool(arr) => arr.ndim(),
            DynCowArray::String(arr) => arr.ndim(),
        }
    }

    pub fn shape(&self) -> Shape {
        match self {
            DynCowArray::I8(arr) => arr.shape().to_vec(),
            DynCowArray::I16(arr) => arr.shape().to_vec(),
            DynCowArray::I32(arr) => arr.shape().to_vec(),
            DynCowArray::I64(arr) => arr.shape().to_vec(),
            DynCowArray::U8(arr) => arr.shape().to_vec(),
            DynCowArray::U16(arr) => arr.shape().to_vec(),
            DynCowArray::U32(arr) => arr.shape().to_vec(),
            DynCowArray::U64(arr) => arr.shape().to_vec(),
            DynCowArray::Usize(arr) => arr.shape().to_vec(),
            DynCowArray::F32(arr) => arr.shape().to_vec(),
            DynCowArray::F64(arr) => arr.shape().to_vec(),
            DynCowArray::Bool(arr) => arr.shape().to_vec(),
            DynCowArray::String(arr) => arr.shape().to_vec(),
        }
        .into()
    }

    pub fn len(&self) -> usize {
        match self {
            DynCowArray::I8(arr) => arr.len(),
            DynCowArray::I16(arr) => arr.len(),
            DynCowArray::I32(arr) => arr.len(),
            DynCowArray::I64(arr) => arr.len(),
            DynCowArray::U8(arr) => arr.len(),
            DynCowArray::U16(arr) => arr.len(),
            DynCowArray::U32(arr) => arr.len(),
            DynCowArray::U64(arr) => arr.len(),
            DynCowArray::Usize(arr) => arr.len(),
            DynCowArray::F32(arr) => arr.len(),
            DynCowArray::F64(arr) => arr.len(),
            DynCowArray::Bool(arr) => arr.len(),
            DynCowArray::String(arr) => arr.len(),
        }
    }
}

macro_rules! impl_dyn_cowarray_convert {
    ($from_type:ty, $to_type:ident) => {
        impl<D: Dimension> From<Array<$from_type, D>> for DynCowArray<'_> {
            fn from(data: Array<$from_type, D>) -> Self {
                DynCowArray::$to_type(data.into_dyn().into())
            }
        }

        impl<'a, D: Dimension> From<ArrayView<'a, $from_type, D>> for DynCowArray<'a> {
            fn from(data: ArrayView<'a, $from_type, D>) -> Self {
                DynCowArray::$to_type(data.into_dyn().into())
            }
        }

        impl<'a, D: Dimension> From<CowArray<'a, $from_type, D>> for DynCowArray<'a> {
            fn from(data: CowArray<'a, $from_type, D>) -> Self {
                DynCowArray::$to_type(data.into_dyn())
            }
        }

        impl<D: Dimension> TryFrom<DynCowArray<'_>> for Array<$from_type, D> {
            type Error = anyhow::Error;
            fn try_from(v: DynCowArray) -> Result<Self, Self::Error> {
                match v {
                    DynCowArray::$to_type(data) => {
                        let arr: ArrayD<$from_type> = data.into_owned();
                        if let Some(n) = D::NDIM {
                            ensure!(
                                arr.ndim() == n,
                                format!("Dimension mismatch: {} (in) != {} (out)", arr.ndim(), n)
                            );
                        }
                        Ok(arr.into_dimensionality::<D>()?)
                    }
                    _ => bail!(
                        "Cannot convert to {} ArrayD",
                        stringify!($from_type)
                    ),
                }
            }
        }
    };
}

impl_dyn_cowarray_convert!(i8, I8);
impl_dyn_cowarray_convert!(i16, I16);
impl_dyn_cowarray_convert!(i32, I32);
impl_dyn_cowarray_convert!(i64, I64);
impl_dyn_cowarray_convert!(u8, U8);
impl_dyn_cowarray_convert!(u16, U16);
impl_dyn_cowarray_convert!(u32, U32);
impl_dyn_cowarray_convert!(u64, U64);
impl_dyn_cowarray_convert!(usize, Usize);
impl_dyn_cowarray_convert!(f32, F32);
impl_dyn_cowarray_convert!(f64, F64);
impl_dyn_cowarray_convert!(bool, Bool);
impl_dyn_cowarray_convert!(String, String);