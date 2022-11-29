use crate::backend::{Backend, GroupOp, LocationOp, DatasetOp, BackendData, DataContainer, ScalarType, Selection};
use crate::data::other::{DynScalar, ReadData, WriteData};

use anyhow::Result;
use either::Either;
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{Array, Array1, ArrayD, Axis, Dim, Dimension, IxDynImpl, SliceInfo, SliceInfoElem};
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
        matches!(self, SelectInfoElem::Slice(SliceInfoElem::Slice { start: 0, end: None, step: 1}))
    }
}

pub const SLICE_FULL: SliceInfoElem = SliceInfoElem::Slice { start: 0, end: None, step: 1 };

pub fn select_all<S, E>(selection: S) -> bool
where
    S: AsRef<[E]>,
    E: AsRef<SelectInfoElem>,
{
    selection.as_ref().into_iter().all(|x| x.as_ref().is_full_slice())
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

pub trait ArrayOp {
    fn shape(&self) -> Shape;
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
    fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
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
    ArrayI8(ArrayD<i8>),
    ArrayI16(ArrayD<i16>),
    ArrayI32(ArrayD<i32>),
    ArrayI64(ArrayD<i64>),
    ArrayU8(ArrayD<u8>),
    ArrayU16(ArrayD<u16>),
    ArrayU32(ArrayD<u32>),
    ArrayU64(ArrayD<u64>),
    ArrayF32(ArrayD<f32>),
    ArrayF64(ArrayD<f64>),
    ArrayBool(ArrayD<bool>),
    ArrayString(ArrayD<String>),
    ArrayCategorical(CategoricalArray),
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

impl_into_dyn_array!(ArrayD<i8>, ArrayI8);
impl_into_dyn_array!(ArrayD<i16>, ArrayI16);
impl_into_dyn_array!(ArrayD<i32>, ArrayI32);
impl_into_dyn_array!(ArrayD<i64>, ArrayI64);
impl_into_dyn_array!(ArrayD<u8>, ArrayU8);
impl_into_dyn_array!(ArrayD<u16>, ArrayU16);
impl_into_dyn_array!(ArrayD<u32>, ArrayU32);
impl_into_dyn_array!(ArrayD<u64>, ArrayU64);
impl_into_dyn_array!(ArrayD<f32>, ArrayF32);
impl_into_dyn_array!(ArrayD<f64>, ArrayF64);
impl_into_dyn_array!(ArrayD<bool>, ArrayBool);
impl_into_dyn_array!(ArrayD<String>, ArrayString);
impl_into_dyn_array!(CategoricalArray, ArrayCategorical);

impl<T: BackendData, D: Dimension> WriteData for Array<T, D> {
    fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
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
        let slice_info: Vec<_> = info.as_ref().iter().map(|x| match x.as_ref() {
            SelectInfoElem::Index(_) => SLICE_FULL,
            SelectInfoElem::Slice(s) => s.clone(),
        }).collect();
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

    fn shape(&self) -> Shape {
        self.shape().to_vec().into()
    }
}

impl<T: BackendData, D: Dimension> WriteArrayData for Array<T, D> {}

impl WriteData for DynArray {
    fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
        match self {
            Self::ArrayI8(array) => array.write(location, name),
            Self::ArrayI16(array) => array.write(location, name),
            Self::ArrayI32(array) => array.write(location, name),
            Self::ArrayI64(array) => array.write(location, name),
            Self::ArrayU8(array) => array.write(location, name),
            Self::ArrayU16(array) => array.write(location, name),
            Self::ArrayU32(array) => array.write(location, name),
            Self::ArrayU64(array) => array.write(location, name),
            Self::ArrayF32(array) => array.write(location, name),
            Self::ArrayF64(array) => array.write(location, name),
            Self::ArrayBool(array) => array.write(location, name),
            Self::ArrayString(array) => array.write(location, name),
            Self::ArrayCategorical(array) => array.write(location, name),
        }
    }
}

impl ReadData for DynArray {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container {
            DataContainer::Dataset(dataset) => match dataset.dtype()? {
                ScalarType::I8 => Ok(Self::ArrayI8(dataset.read_array(Selection::All)?)),
                ScalarType::I16 => Ok(Self::ArrayI16(dataset.read_array(Selection::All)?)),
                ScalarType::I32 => Ok(Self::ArrayI32(dataset.read_array(Selection::All)?)),
                ScalarType::I64 => Ok(Self::ArrayI64(dataset.read_array(Selection::All)?)),
                ScalarType::U8 => Ok(Self::ArrayU8(dataset.read_array(Selection::All)?)),
                ScalarType::U16 => Ok(Self::ArrayU16(dataset.read_array(Selection::All)?)),
                ScalarType::U32 => Ok(Self::ArrayU32(dataset.read_array(Selection::All)?)),
                ScalarType::U64 => Ok(Self::ArrayU64(dataset.read_array(Selection::All)?)),
                ScalarType::F32 => Ok(Self::ArrayF32(dataset.read_array(Selection::All)?)),
                ScalarType::F64 => Ok(Self::ArrayF64(dataset.read_array(Selection::All)?)),
                ScalarType::Bool => Ok(Self::ArrayBool(dataset.read_array(Selection::All)?)),
                ScalarType::String => Ok(Self::ArrayString(dataset.read_array(Selection::All)?)),
            },
            DataContainer::Group(group) => {
                let codes = group.open_dataset("codes")?.read_array(Selection::All)?;
                let categories = group.open_dataset("categories")?.read_array(Selection::All)?;
                Ok(Self::ArrayCategorical(CategoricalArray {
                    codes,
                    categories,
                }))
            }
        }
    }
}

impl ArrayOp for DynArray {
    fn shape(&self) -> Shape {
        match self {
            DynArray::ArrayI8(array) => array.shape().to_vec(),
            DynArray::ArrayI16(array) => array.shape().to_vec(),
            DynArray::ArrayI32(array) => array.shape().to_vec(),
            DynArray::ArrayI64(array) => array.shape().to_vec(),
            DynArray::ArrayU8(array) => array.shape().to_vec(),
            DynArray::ArrayU16(array) => array.shape().to_vec(),
            DynArray::ArrayU32(array) => array.shape().to_vec(),
            DynArray::ArrayU64(array) => array.shape().to_vec(),
            DynArray::ArrayF32(array) => array.shape().to_vec(),
            DynArray::ArrayF64(array) => array.shape().to_vec(),
            DynArray::ArrayBool(array) => array.shape().to_vec(),
            DynArray::ArrayString(array) => array.shape().to_vec(),
            DynArray::ArrayCategorical(array) => array.codes.shape().to_vec(),
        }
        .into()
    }

    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        match self {
            DynArray::ArrayI8(array) => array.get(index).map(|x| (*x).into()),
            DynArray::ArrayI16(array) => array.get(index).map(|x| (*x).into()),
            DynArray::ArrayI32(array) => array.get(index).map(|x| (*x).into()),
            DynArray::ArrayI64(array) => array.get(index).map(|x| (*x).into()),
            DynArray::ArrayU8(array) => array.get(index).map(|x| (*x).into()),
            DynArray::ArrayU16(array) => array.get(index).map(|x| (*x).into()),
            DynArray::ArrayU32(array) => array.get(index).map(|x| (*x).into()),
            DynArray::ArrayU64(array) => array.get(index).map(|x| (*x).into()),
            DynArray::ArrayF32(array) => array.get(index).map(|x| (*x).into()),
            DynArray::ArrayF64(array) => array.get(index).map(|x| (*x).into()),
            DynArray::ArrayBool(array) => array.get(index).map(|x| (*x).into()),
            DynArray::ArrayString(array) => array.get(index).map(|x| x.clone().into()),
            DynArray::ArrayCategorical(array) => array
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
            DynArray::ArrayI8(array) => ArrayOp::select(array, info).into(),
            DynArray::ArrayI16(array) => ArrayOp::select(array, info).into(),
            DynArray::ArrayI32(array) => ArrayOp::select(array, info).into(),
            DynArray::ArrayI64(array) => ArrayOp::select(array, info).into(),
            DynArray::ArrayU8(array) => ArrayOp::select(array, info).into(),
            DynArray::ArrayU16(array) => ArrayOp::select(array, info).into(),
            DynArray::ArrayU32(array) => ArrayOp::select(array, info).into(),
            DynArray::ArrayU64(array) => ArrayOp::select(array, info).into(),
            DynArray::ArrayF32(array) => ArrayOp::select(array, info).into(),
            DynArray::ArrayF64(array) => ArrayOp::select(array, info).into(),
            DynArray::ArrayBool(array) => ArrayOp::select(array, info).into(),
            DynArray::ArrayString(array) => ArrayOp::select(array, info).into(),
            DynArray::ArrayCategorical(array) => CategoricalArray {
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
    CsrMatrixI8(CsrMatrix<i8>),
    CsrMatrixI16(CsrMatrix<i16>),
    CsrMatrixI32(CsrMatrix<i32>),
    CsrMatrixI64(CsrMatrix<i64>),
    CsrMatrixU8(CsrMatrix<u8>),
    CsrMatrixU16(CsrMatrix<u16>),
    CsrMatrixU32(CsrMatrix<u32>),
    CsrMatrixU64(CsrMatrix<u64>),
    CsrMatrixF32(CsrMatrix<f32>),
    CsrMatrixF64(CsrMatrix<f64>),
    CsrMatrixBool(CsrMatrix<bool>),
    CsrMatrixString(CsrMatrix<String>),
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

impl_into_dyn_csr!(CsrMatrix<i8>, CsrMatrixI8);
impl_into_dyn_csr!(CsrMatrix<i16>, CsrMatrixI16);
impl_into_dyn_csr!(CsrMatrix<i32>, CsrMatrixI32);
impl_into_dyn_csr!(CsrMatrix<i64>, CsrMatrixI64);
impl_into_dyn_csr!(CsrMatrix<u8>, CsrMatrixU8);
impl_into_dyn_csr!(CsrMatrix<u16>, CsrMatrixU16);
impl_into_dyn_csr!(CsrMatrix<u32>, CsrMatrixU32);
impl_into_dyn_csr!(CsrMatrix<u64>, CsrMatrixU64);
impl_into_dyn_csr!(CsrMatrix<f32>, CsrMatrixF32);
impl_into_dyn_csr!(CsrMatrix<f64>, CsrMatrixF64);
impl_into_dyn_csr!(CsrMatrix<bool>, CsrMatrixBool);
impl_into_dyn_csr!(CsrMatrix<String>, CsrMatrixString);

impl<T: BackendData> WriteData for CsrMatrix<T> {
    fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
        todo!()
    }
}

impl WriteData for DynCsrMatrix {
    fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
        match self {
            DynCsrMatrix::CsrMatrixI8(data) => data.write(location, name),
            DynCsrMatrix::CsrMatrixI16(data) => data.write(location, name),
            DynCsrMatrix::CsrMatrixI32(data) => data.write(location, name),
            DynCsrMatrix::CsrMatrixI64(data) => data.write(location, name),
            DynCsrMatrix::CsrMatrixU8(data) => data.write(location, name),
            DynCsrMatrix::CsrMatrixU16(data) => data.write(location, name),
            DynCsrMatrix::CsrMatrixU32(data) => data.write(location, name),
            DynCsrMatrix::CsrMatrixU64(data) => data.write(location, name),
            DynCsrMatrix::CsrMatrixF32(data) => data.write(location, name),
            DynCsrMatrix::CsrMatrixF64(data) => data.write(location, name),
            DynCsrMatrix::CsrMatrixBool(data) => data.write(location, name),
            DynCsrMatrix::CsrMatrixString(data) => data.write(location, name),
        }
    }
}

impl ReadData for DynCsrMatrix {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        todo!()
    }
}

impl ArrayOp for DynCsrMatrix {
    fn shape(&self) -> Shape {
        fn csr_shape<T>(matrix: &CsrMatrix<T>) -> Shape {
            vec![matrix.nrows(), matrix.ncols()].into()
        }
        match self {
            DynCsrMatrix::CsrMatrixI8(matrix) => csr_shape(matrix),
            DynCsrMatrix::CsrMatrixI16(matrix) => csr_shape(matrix),
            DynCsrMatrix::CsrMatrixI32(matrix) => csr_shape(matrix),
            DynCsrMatrix::CsrMatrixI64(matrix) => csr_shape(matrix),
            DynCsrMatrix::CsrMatrixU8(matrix) => csr_shape(matrix),
            DynCsrMatrix::CsrMatrixU16(matrix) => csr_shape(matrix),
            DynCsrMatrix::CsrMatrixU32(matrix) => csr_shape(matrix),
            DynCsrMatrix::CsrMatrixU64(matrix) => csr_shape(matrix),
            DynCsrMatrix::CsrMatrixF32(matrix) => csr_shape(matrix),
            DynCsrMatrix::CsrMatrixF64(matrix) => csr_shape(matrix),
            DynCsrMatrix::CsrMatrixBool(matrix) => csr_shape(matrix),
            DynCsrMatrix::CsrMatrixString(matrix) => csr_shape(matrix),
        }
    }

    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        todo!()
    }

    fn select<S, E>(&self, info: S) -> Self
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>
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
