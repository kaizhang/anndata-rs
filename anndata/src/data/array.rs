mod chunks;
pub mod dataframe;
mod dense;
pub mod slice;
mod sparse;
pub mod utils;

pub use chunks::ArrayChunk;
pub use dataframe::DataFrameIndex;
pub use dense::{ArrayConvert, CategoricalArray, DynArray, DynCowArray, DynScalar};
pub use slice::{SelectInfo, SelectInfoBounds, SelectInfoElem, SelectInfoElemBounds, Shape};
pub use sparse::{CsrNonCanonical, DynCscMatrix, DynCsrMatrix, DynCsrNonCanonical};

use crate::backend::*;
use crate::data::utils::from_csr_data;
use crate::data::{data_traits::*, DataType};

use ::ndarray::{Array, Ix1, RemoveAxis};
use anyhow::{bail, Result};
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use polars::prelude::DataFrame;

#[derive(Debug, Clone, PartialEq)]
pub enum ArrayData {
    Array(DynArray),
    CsrMatrix(DynCsrMatrix),
    CsrNonCanonical(DynCsrNonCanonical),
    CscMatrix(DynCscMatrix),
    DataFrame(DataFrame),
}

impl<T: Clone + Into<ArrayData>> From<&T> for ArrayData {
    fn from(data: &T) -> Self {
        data.clone().into()
    }
}

impl From<DataFrame> for ArrayData {
    fn from(data: DataFrame) -> Self {
        ArrayData::DataFrame(data)
    }
}

impl From<DynArray> for ArrayData {
    fn from(data: DynArray) -> Self {
        ArrayData::Array(data)
    }
}

impl From<DynCsrMatrix> for ArrayData {
    fn from(data: DynCsrMatrix) -> Self {
        ArrayData::CsrMatrix(data)
    }
}
impl From<DynCsrNonCanonical> for ArrayData {
    fn from(data: DynCsrNonCanonical) -> Self {
        ArrayData::CsrNonCanonical(data)
    }
}
impl From<DynCscMatrix> for ArrayData {
    fn from(data: DynCscMatrix) -> Self {
        ArrayData::CscMatrix(data)
    }
}

impl TryFrom<ArrayData> for DynArray {
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        match value {
            ArrayData::Array(data) => Ok(data),
            _ => bail!("Cannot convert {:?} to DynArray", value.data_type()),
        }
    }
}

impl TryFrom<ArrayData> for DynCsrMatrix {
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        match value {
            ArrayData::CsrMatrix(data) => Ok(data),
            _ => bail!("Cannot convert {:?} to DynCsrMatrix", value.data_type()),
        }
    }
}

impl TryFrom<ArrayData> for DynCsrNonCanonical {
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        match value {
            ArrayData::CsrNonCanonical(data) => Ok(data),
            ArrayData::CsrMatrix(data) => Ok(data.into()),
            _ => bail!(
                "Cannot convert {:?} to DynCsrNonCanonical",
                value.data_type()
            ),
        }
    }
}

impl TryFrom<ArrayData> for DynCscMatrix {
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        match value {
            ArrayData::CscMatrix(data) => Ok(data),
            _ => bail!("Cannot convert {:?} to DynCscMatrix", value.data_type()),
        }
    }
}

impl TryFrom<ArrayData> for DataFrame {
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        match value {
            ArrayData::DataFrame(data) => Ok(data),
            _ => bail!("Cannot convert {:?} to DataFrame", value.data_type()),
        }
    }
}

impl<T, D> TryFrom<ArrayData> for Array<T, D>
where
    Array<T, D>: TryFrom<DynArray, Error = anyhow::Error>,
{
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        DynArray::try_from(value)?.try_into()
    }
}

impl<T> TryFrom<ArrayData> for CsrMatrix<T>
where
    CsrMatrix<T>: TryFrom<DynCsrMatrix, Error = anyhow::Error>,
{
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        DynCsrMatrix::try_from(value)?.try_into()
    }
}

impl<T> TryFrom<ArrayData> for CscMatrix<T>
where
    CscMatrix<T>: TryFrom<DynCscMatrix, Error = anyhow::Error>,
{
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        DynCscMatrix::try_from(value)?.try_into()
    }
}

impl<T> TryFrom<ArrayData> for CsrNonCanonical<T>
where
    CsrNonCanonical<T>: TryFrom<DynCsrNonCanonical, Error = anyhow::Error>,
{
    type Error = anyhow::Error;
    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        DynCsrNonCanonical::try_from(value)?.try_into()
    }
}

impl<T, D> ArrayConvert<Array<T, D>> for ArrayData
where
    DynArray: ArrayConvert<Array<T, D>>,
{
    fn try_convert(self) -> Result<Array<T, D>> {
        DynArray::try_from(self)?.try_convert()
    }
}

/// macro for implementing From trait for Data from a list of types
macro_rules! impl_arraydata_traits {
    ($($ty:ty),*) => {
        $(
            impl<D: RemoveAxis> From<Array<$ty, D>> for ArrayData {
                fn from(data: Array<$ty, D>) -> Self {
                    ArrayData::Array(data.into())
                }
            }
            impl From<CsrMatrix<$ty>> for ArrayData {
                fn from(data: CsrMatrix<$ty>) -> Self {
                    ArrayData::CsrMatrix(data.into())
                }
            }
            impl From<CsrNonCanonical<$ty>> for ArrayData {
                fn from(data: CsrNonCanonical<$ty>) -> Self {
                    ArrayData::CsrNonCanonical(data.into())
                }
            }
            impl From<CscMatrix<$ty>> for ArrayData {
                fn from(data: CscMatrix<$ty>) -> Self {
                    ArrayData::CscMatrix(data.into())
                }
            }
        )*
    };
}

impl_arraydata_traits!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool, String);

impl Readable for ArrayData {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) => {
                DynArray::read(container).map(ArrayData::Array)
            }
            DataType::CsrMatrix(_) => read_csr(container),
            DataType::CscMatrix(_) => DynCscMatrix::read(container).map(ArrayData::CscMatrix),
            DataType::DataFrame => DataFrame::read(container).map(ArrayData::DataFrame),
            ty => bail!("Cannot read type '{:?}' as matrix data", ty),
        }
    }
}

impl Element for ArrayData {
    fn data_type(&self) -> DataType {
        match self {
            ArrayData::Array(data) => data.data_type(),
            ArrayData::CsrMatrix(data) => data.data_type(),
            ArrayData::CsrNonCanonical(data) => data.data_type(),
            ArrayData::CscMatrix(data) => data.data_type(),
            ArrayData::DataFrame(data) => data.data_type(),
        }
    }

    fn metadata(&self) -> MetaData {
        match self {
            ArrayData::Array(data) => data.metadata(),
            ArrayData::CsrMatrix(data) => data.metadata(),
            ArrayData::CsrNonCanonical(data) => data.metadata(),
            ArrayData::CscMatrix(data) => data.metadata(),
            ArrayData::DataFrame(data) => data.metadata(),
        }
    }
}

impl Writable for ArrayData {
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        match self {
            ArrayData::Array(data) => data.write(location, name),
            ArrayData::CsrMatrix(data) => data.write(location, name),
            ArrayData::CsrNonCanonical(data) => data.write(location, name),
            ArrayData::CscMatrix(data) => data.write(location, name),
            ArrayData::DataFrame(data) => data.write(location, name),
        }
    }
}

impl HasShape for ArrayData {
    fn shape(&self) -> Shape {
        match self {
            ArrayData::Array(data) => data.shape(),
            ArrayData::CsrMatrix(data) => data.shape(),
            ArrayData::CsrNonCanonical(data) => data.shape(),
            ArrayData::CscMatrix(data) => data.shape(),
            ArrayData::DataFrame(data) => HasShape::shape(data),
        }
    }
}

impl Indexable for ArrayData {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        match self {
            ArrayData::Array(data) => data.get(index),
            _ => todo!(),
        }
    }
}

impl Selectable for ArrayData {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        match self {
            ArrayData::Array(data) => data.select(info).into(),
            ArrayData::CsrMatrix(data) => data.select(info).into(),
            ArrayData::CsrNonCanonical(data) => data.select(info).into(),
            ArrayData::CscMatrix(data) => data.select(info).into(),
            ArrayData::DataFrame(data) => Selectable::select(data, info).into(),
        }
    }
}

impl Stackable for ArrayData {
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        let item = iter.peek();
        if item.is_none() {
            bail!("Cannot stack empty iterator");
        }
        match item.unwrap() {
            ArrayData::Array(_) => {
                DynArray::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            ArrayData::CsrMatrix(_) => {
                DynCsrNonCanonical::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| {
                    match x.canonicalize() {
                        Ok(x) => x.into(),
                        Err(x) => x.into(),
                    }
                })
            }
            ArrayData::CsrNonCanonical(_) => {
                DynCsrNonCanonical::vstack(iter.map(|x| x.try_into().unwrap())).map(|x| x.into())
            }
            ArrayData::CscMatrix(_) => todo!(),
            ArrayData::DataFrame(_) => {
                <DataFrame as Stackable>::vstack(iter.map(|x| x.try_into().unwrap()))
                    .map(|x| x.into())
            }
        }
    }
}

impl ReadableArray for ArrayData {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) => DynArray::get_shape(container),
            DataType::CsrMatrix(_) => DynCsrMatrix::get_shape(container),
            DataType::CscMatrix(_) => DynCscMatrix::get_shape(container),
            DataType::DataFrame => DataFrame::get_shape(container),
            ty => bail!("Cannot read shape information from type '{}'", ty),
        }
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        match container.encoding_type()? {
            DataType::Categorical | DataType::Array(_) => {
                DynArray::read_select(container, info).map(ArrayData::Array)
            }
            DataType::CsrMatrix(_) => read_csr_select(container, info),
            DataType::CscMatrix(_) => {
                DynCscMatrix::read_select(container, info).map(ArrayData::CscMatrix)
            }
            DataType::DataFrame => {
                DataFrame::read_select(container, info).map(ArrayData::DataFrame)
            }
            ty => bail!("Cannot read type '{:?}' as matrix data", ty),
        }
    }
}
impl WritableArray for ArrayData {}

impl WritableArray for &ArrayData {}

// Helper

fn read_csr<B: Backend>(container: &DataContainer<B>) -> Result<ArrayData> {
    fn _read_csr<B: Backend, T: BackendData>(container: &DataContainer<B>) -> Result<ArrayData>
    where
        CsrMatrix<T>: Into<ArrayData>,
        CsrNonCanonical<T>: Into<ArrayData>,
    {
        let group = container.as_group()?;
        let shape: Vec<u64> = group.get_attr("shape")?;
        let data = group
            .open_dataset("data")?
            .read_array::<_, Ix1>()?
            .into_raw_vec_and_offset()
            .0;
        let indptr: Vec<usize> = group
            .open_dataset("indptr")?
            .read_array_cast::<_, Ix1>()?
            .into_raw_vec_and_offset()
            .0;
        let indices: Vec<usize> = group
            .open_dataset("indices")?
            .read_array_cast::<_, Ix1>()?
            .into_raw_vec_and_offset()
            .0;
        from_csr_data::<T>(shape[0] as usize, shape[1] as usize, indptr, indices, data)
    }

    match container {
        DataContainer::Group(group) => match group.open_dataset("data")?.dtype()? {
            ScalarType::I8 => _read_csr::<B, i8>(container),
            ScalarType::I16 => _read_csr::<B, i16>(container),
            ScalarType::I32 => _read_csr::<B, i32>(container),
            ScalarType::I64 => _read_csr::<B, i64>(container),
            ScalarType::U8 => _read_csr::<B, u8>(container),
            ScalarType::U16 => _read_csr::<B, u16>(container),
            ScalarType::U32 => _read_csr::<B, u32>(container),
            ScalarType::U64 => _read_csr::<B, u64>(container),
            ScalarType::F32 => _read_csr::<B, f32>(container),
            ScalarType::F64 => _read_csr::<B, f64>(container),
            ScalarType::Bool => _read_csr::<B, bool>(container),
            ScalarType::String => _read_csr::<B, String>(container),
        },
        _ => bail!("cannot read csr matrix from non-group container"),
    }
}

fn read_csr_select<B: Backend, S>(container: &DataContainer<B>, info: &[S]) -> Result<ArrayData>
where
    B: Backend,
    S: AsRef<SelectInfoElem>,
{
    fn _read_csr<B: Backend, T: BackendData, S>(
        container: &DataContainer<B>,
        info: &[S],
    ) -> Result<ArrayData>
    where
        CsrMatrix<T>: Into<ArrayData>,
        CsrNonCanonical<T>: Into<ArrayData>,
        S: AsRef<SelectInfoElem>,
    {
        if info.as_ref().len() != 2 {
            panic!("index must have length 2");
        }

        if info.iter().all(|s| s.as_ref().is_full()) {
            return read_csr(container);
        }

        let data = if let SelectInfoElem::Slice(s) = info[0].as_ref() {
            let group = container.as_group()?;
            let shape: Vec<u64> = group.get_attr("shape")?;
            let indptr_slice = if let Some(end) = s.end {
                SelectInfoElem::from(s.start..end + 1)
            } else {
                SelectInfoElem::from(s.start..)
            };
            let mut indptr: Vec<usize> = group
                .open_dataset("indptr")?
                .read_array_slice_cast(&[indptr_slice])?
                .to_vec();
            let lo = indptr[0];
            let slice = SelectInfoElem::from(lo..indptr[indptr.len() - 1]);
            let data: Vec<T> = group
                .open_dataset("data")?
                .read_array_slice(&[&slice])?
                .to_vec();
            let indices: Vec<usize> = group
                .open_dataset("indices")?
                .read_array_slice_cast(&[&slice])?
                .to_vec();
            indptr.iter_mut().for_each(|x| *x -= lo);

            from_csr_data::<T>(indptr.len() - 1, shape[1] as usize, indptr, indices, data)
                .unwrap()
                .select_axis(1, info[1].as_ref())
        } else {
            read_csr(container)?.select(info)
        };
        Ok(data)
    }

    match container {
        DataContainer::Group(group) => match group.open_dataset("data")?.dtype()? {
            ScalarType::I8 => _read_csr::<B, i8, _>(container, info),
            ScalarType::I16 => _read_csr::<B, i16, _>(container, info),
            ScalarType::I32 => _read_csr::<B, i32, _>(container, info),
            ScalarType::I64 => _read_csr::<B, i64, _>(container, info),
            ScalarType::U8 => _read_csr::<B, u8, _>(container, info),
            ScalarType::U16 => _read_csr::<B, u16, _>(container, info),
            ScalarType::U32 => _read_csr::<B, u32, _>(container, info),
            ScalarType::U64 => _read_csr::<B, u64, _>(container, info),
            ScalarType::F32 => _read_csr::<B, f32, _>(container, info),
            ScalarType::F64 => _read_csr::<B, f64, _>(container, info),
            ScalarType::Bool => _read_csr::<B, bool, _>(container, info),
            ScalarType::String => _read_csr::<B, String, _>(container, info),
        },
        _ => bail!("cannot read csr matrix from non-group container"),
    }
}
