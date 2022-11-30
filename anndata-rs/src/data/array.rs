use crate::backend::{
    Backend, BackendData, DataContainer, DatasetOp, GroupOp, LocationOp, ScalarType, Selection,
};
use crate::data::other::{DynScalar, ReadData, WriteData};

use anyhow::Result;
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{ArrayView, Array, Array1, ArrayD, Axis, Dimension, SliceInfoElem};
use std::collections::HashMap;
use std::ops::Index;

#[derive(Debug)]
pub struct SelectInfo(Vec<SelectInfoElem>);

impl AsRef<[SelectInfoElem]> for SelectInfo {
    fn as_ref(&self) -> &[SelectInfoElem] {
        &self.0
    }
}

#[derive(Debug)]
pub enum SelectInfoElem {
    Index(Vec<usize>),
    Slice(SliceInfoElem),
}

impl AsRef<SelectInfoElem> for SelectInfoElem {
    fn as_ref(&self) -> &SelectInfoElem {
        self
    }
}

impl SelectInfoElem {
    pub fn output_len(&self, n: usize) -> usize {
        match self {
            SelectInfoElem::Index(idx) => idx.len(),
            SelectInfoElem::Slice(slice) => todo!(),
        }
    }

    pub fn is_index(&self) -> bool {
        matches!(self, SelectInfoElem::Index(_))
    }

    pub fn is_full_slice(&self) -> bool {
        matches!(
            self,
            SelectInfoElem::Slice(SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1
            })
        )
    }
}

pub const SLICE_FULL: SliceInfoElem = SliceInfoElem::Slice {
    start: 0,
    end: None,
    step: 1,
};

pub fn select_all<S, E>(selection: S) -> bool
where
    S: AsRef<[E]>,
    E: AsRef<SelectInfoElem>,
{
    selection
        .as_ref()
        .into_iter()
        .all(|x| x.as_ref().is_full_slice())
}

/// TODO: consider using smallvec
pub struct Shape(Vec<usize>);

impl From<Vec<usize>> for Shape {
    fn from(shape: Vec<usize>) -> Self {
        Self(shape)
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
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

pub trait ReadArrayData: ReadData + ArrayOp {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape>;

    fn read_select<B, S, E>(container: &DataContainer<B>, info: S) -> Result<Self>
    where
        B: Backend,
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
        Self: Sized,
    {
        Self::read(container).map(|data| data.select(info))
    }
}

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

        group.write_array("codes", &self.codes, Selection::All)?;
        group.write_array("categories", &self.categories, Selection::All)?;

        Ok(DataContainer::Group(group))
    }
}

impl WriteArrayData for CategoricalArray {}

/// Untyped array.
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
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    Bool(ArrayD<bool>),
    String(ArrayD<String>),
    Categorical(CategoricalArray),
}

macro_rules! impl_into_dyn_array {
    ($from_type:ty, $to_type:ident) => {
        impl From<$from_type> for DynArray {
            fn from(data: $from_type) -> Self {
                DynArray::$to_type(data)
            }
        }
    };
}

impl_into_dyn_array!(ArrayD<i8>, I8);
impl_into_dyn_array!(ArrayD<i16>, I16);
impl_into_dyn_array!(ArrayD<i32>, I32);
impl_into_dyn_array!(ArrayD<i64>, I64);
impl_into_dyn_array!(ArrayD<u8>, U8);
impl_into_dyn_array!(ArrayD<u16>, U16);
impl_into_dyn_array!(ArrayD<u32>, U32);
impl_into_dyn_array!(ArrayD<u64>, U64);
impl_into_dyn_array!(ArrayD<f32>, F32);
impl_into_dyn_array!(ArrayD<f64>, F64);
impl_into_dyn_array!(ArrayD<bool>, Bool);
impl_into_dyn_array!(ArrayD<String>, String);
impl_into_dyn_array!(CategoricalArray, Categorical);

impl<'a, T: BackendData, D: Dimension> WriteData for ArrayView<'a, T, D> {
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let dataset = location.write_array(name, self, Selection::All)?;
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


impl<T: BackendData> ArrayOp for ArrayD<T> {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        self.get(index).map(|x| x.into_dyn())
    }

    fn select<S, E>(&self, info: S) -> Self
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
    {
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
    }
}

impl<T: BackendData, D: Dimension> WriteArrayData for Array<T, D> {}
impl<T: BackendData, D: Dimension> WriteArrayData for &Array<T, D> {}
impl<'a, T: BackendData, D: Dimension> WriteArrayData for ArrayView<'a, T, D> {}

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
                ScalarType::I8 => Ok(Self::I8(dataset.read_array(Selection::All)?)),
                ScalarType::I16 => Ok(Self::I16(dataset.read_array(Selection::All)?)),
                ScalarType::I32 => Ok(Self::I32(dataset.read_array(Selection::All)?)),
                ScalarType::I64 => Ok(Self::I64(dataset.read_array(Selection::All)?)),
                ScalarType::U8 => Ok(Self::U8(dataset.read_array(Selection::All)?)),
                ScalarType::U16 => Ok(Self::U16(dataset.read_array(Selection::All)?)),
                ScalarType::U32 => Ok(Self::U32(dataset.read_array(Selection::All)?)),
                ScalarType::U64 => Ok(Self::U64(dataset.read_array(Selection::All)?)),
                ScalarType::F32 => Ok(Self::F32(dataset.read_array(Selection::All)?)),
                ScalarType::F64 => Ok(Self::F64(dataset.read_array(Selection::All)?)),
                ScalarType::Bool => Ok(Self::Bool(dataset.read_array(Selection::All)?)),
                ScalarType::String => Ok(Self::String(dataset.read_array(Selection::All)?)),
            },
            DataContainer::Group(group) => {
                let codes = group.open_dataset("codes")?.read_array(Selection::All)?;
                let categories = group
                    .open_dataset("categories")?
                    .read_array(Selection::All)?;
                Ok(Self::Categorical(CategoricalArray { codes, categories }))
            }
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
    F32(CsrMatrix<f32>),
    F64(CsrMatrix<f64>),
    Bool(CsrMatrix<bool>),
    String(CsrMatrix<String>),
}

macro_rules! impl_into_dyn_csr {
    ($from_type:ty, $to_type:ident) => {
        impl From<$from_type> for DynCsrMatrix {
            fn from(data: $from_type) -> Self {
                DynCsrMatrix::$to_type(data)
            }
        }
    };
}

impl_into_dyn_csr!(CsrMatrix<i8>, I8);
impl_into_dyn_csr!(CsrMatrix<i16>, I16);
impl_into_dyn_csr!(CsrMatrix<i32>, I32);
impl_into_dyn_csr!(CsrMatrix<i64>, I64);
impl_into_dyn_csr!(CsrMatrix<u8>, U8);
impl_into_dyn_csr!(CsrMatrix<u16>, U16);
impl_into_dyn_csr!(CsrMatrix<u32>, U32);
impl_into_dyn_csr!(CsrMatrix<u64>, U64);
impl_into_dyn_csr!(CsrMatrix<f32>, F32);
impl_into_dyn_csr!(CsrMatrix<f64>, F64);
impl_into_dyn_csr!(CsrMatrix<bool>, Bool);
impl_into_dyn_csr!(CsrMatrix<String>, String);

impl<T: BackendData> WriteData for CsrMatrix<T> {
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        todo!()
    }
}

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
            DynCsrMatrix::F32(data) => data.write(location, name),
            DynCsrMatrix::F64(data) => data.write(location, name),
            DynCsrMatrix::Bool(data) => data.write(location, name),
            DynCsrMatrix::String(data) => data.write(location, name),
        }
    }
}

impl ReadData for DynCsrMatrix {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        todo!()
    }
}

impl HasShape for DynCsrMatrix {
    fn shape(&self) -> Shape {
        fn csr_shape<T>(matrix: &CsrMatrix<T>) -> Shape {
            vec![matrix.nrows(), matrix.ncols()].into()
        }
        match self {
            DynCsrMatrix::I8(matrix) => csr_shape(matrix),
            DynCsrMatrix::I16(matrix) => csr_shape(matrix),
            DynCsrMatrix::I32(matrix) => csr_shape(matrix),
            DynCsrMatrix::I64(matrix) => csr_shape(matrix),
            DynCsrMatrix::U8(matrix) => csr_shape(matrix),
            DynCsrMatrix::U16(matrix) => csr_shape(matrix),
            DynCsrMatrix::U32(matrix) => csr_shape(matrix),
            DynCsrMatrix::U64(matrix) => csr_shape(matrix),
            DynCsrMatrix::F32(matrix) => csr_shape(matrix),
            DynCsrMatrix::F64(matrix) => csr_shape(matrix),
            DynCsrMatrix::Bool(matrix) => csr_shape(matrix),
            DynCsrMatrix::String(matrix) => csr_shape(matrix),
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
        todo!()
    }
}
