use crate::backend::*;
use crate::data::{
    array::utils::{cs_major_index, cs_major_minor_index, cs_major_slice},
    data_traits::*,
    array::DynScalar,
    slice::{SelectInfoElem, Shape},
    SelectInfoBounds, SelectInfoElemBounds,
};

use anyhow::{bail, Context, Result};
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::pattern::SparsityPattern;
use ndarray::Ix1;
use num::FromPrimitive;

use super::super::slice::SliceBounds;

#[derive(Debug, Clone, PartialEq)]
pub enum DynCscMatrix {
    I8(CscMatrix<i8>),
    I16(CscMatrix<i16>),
    I32(CscMatrix<i32>),
    I64(CscMatrix<i64>),
    U8(CscMatrix<u8>),
    U16(CscMatrix<u16>),
    U32(CscMatrix<u32>),
    U64(CscMatrix<u64>),
    Usize(CscMatrix<usize>),
    F32(CscMatrix<f32>),
    F64(CscMatrix<f64>),
    Bool(CscMatrix<bool>),
    String(CscMatrix<String>),
}

macro_rules! impl_into_dyn_csc {
    ($from_type:ty, $to_type:ident) => {
        impl From<CscMatrix<$from_type>> for DynCscMatrix {
            fn from(data: CscMatrix<$from_type>) -> Self {
                DynCscMatrix::$to_type(data)
            }
        }
        impl TryFrom<DynCscMatrix> for CscMatrix<$from_type> {
            type Error = anyhow::Error;
            fn try_from(data: DynCscMatrix) -> Result<Self> {
                match data {
                    DynCscMatrix::$to_type(data) => Ok(data),
                    _ => bail!(
                        "Cannot convert {:?} to {} CscMatrix",
                        data.data_type(),
                        stringify!($from_type)
                    ),
                }
            }
        }
    };
}

impl From<CscMatrix<u32>> for DynCscMatrix {
    fn from(data: CscMatrix<u32>) -> Self {
        DynCscMatrix::U32(data)
    }
}

impl TryFrom<DynCscMatrix> for CscMatrix<u32> {
    type Error = anyhow::Error;
    fn try_from(data: DynCscMatrix) -> Result<Self> {
        match data {
            DynCscMatrix::U32(data) => Ok(data),
            DynCscMatrix::I8(data) => Ok(cast_csc(data)?),
            DynCscMatrix::I16(data) => Ok(cast_csc(data)?),
            DynCscMatrix::I32(data) => Ok(cast_csc(data)?),
            DynCscMatrix::I64(data) => Ok(from_i64_csc(data)?),
            DynCscMatrix::U8(data) => Ok(cast_csc(data)?),
            DynCscMatrix::U16(data) => Ok(cast_csc(data)?),
            DynCscMatrix::U64(data) => Ok(cast_csc(data)?),
            DynCscMatrix::Usize(data) => Ok(cast_csc(data)?),
            DynCscMatrix::F32(_) => bail!("Cannot convert f32 to u32"),
            DynCscMatrix::F64(_) => bail!("Cannot convert f64 to u32"),
            DynCscMatrix::Bool(_) => bail!("Cannot convert bool to f64"),
            DynCscMatrix::String(_) => bail!("Cannot convert string to f64"),
        }
    }
}

impl From<CscMatrix<f64>> for DynCscMatrix {
    fn from(data: CscMatrix<f64>) -> Self {
        DynCscMatrix::F64(data)
    }
}

impl TryFrom<DynCscMatrix> for CscMatrix<f64> {
    type Error = anyhow::Error;
    fn try_from(data: DynCscMatrix) -> Result<Self> {
        match data {
            DynCscMatrix::F64(data) => Ok(data),
            DynCscMatrix::I8(data) => Ok(cast_csc(data)?),
            DynCscMatrix::I16(data) => Ok(cast_csc(data)?),
            DynCscMatrix::I32(data) => Ok(cast_csc(data)?),
            DynCscMatrix::I64(data) => Ok(from_i64_csc(data)?),
            DynCscMatrix::U8(data) => Ok(cast_csc(data)?),
            DynCscMatrix::U16(data) => Ok(cast_csc(data)?),
            DynCscMatrix::U32(data) => Ok(cast_csc(data)?),
            DynCscMatrix::U64(_) => bail!("Cannot convert u64 to f64"),
            DynCscMatrix::Usize(_) => bail!("Cannot convert usize to f64"),
            DynCscMatrix::F32(data) => Ok(cast_csc(data)?),
            DynCscMatrix::Bool(_) => bail!("Cannot convert bool to f64"),
            DynCscMatrix::String(_) => bail!("Cannot convert string to f64"),
        }
    }
}

impl_into_dyn_csc!(i8, I8);
impl_into_dyn_csc!(i16, I16);
impl_into_dyn_csc!(i32, I32);
impl_into_dyn_csc!(i64, I64);
impl_into_dyn_csc!(u8, U8);
impl_into_dyn_csc!(u16, U16);
impl_into_dyn_csc!(u64, U64);
impl_into_dyn_csc!(usize, Usize);
impl_into_dyn_csc!(f32, F32);
impl_into_dyn_csc!(bool, Bool);
impl_into_dyn_csc!(String, String);

macro_rules! impl_dyn_csc_matrix {
    ($self:expr, $fun:ident) => {
        match $self {
            DynCscMatrix::I8(data) => $fun!(data),
            DynCscMatrix::I16(data) => $fun!(data),
            DynCscMatrix::I32(data) => $fun!(data),
            DynCscMatrix::I64(data) => $fun!(data),
            DynCscMatrix::U8(data) => $fun!(data),
            DynCscMatrix::U16(data) => $fun!(data),
            DynCscMatrix::U32(data) => $fun!(data),
            DynCscMatrix::U64(data) => $fun!(data),
            DynCscMatrix::Usize(data) => $fun!(data),
            DynCscMatrix::F32(data) => $fun!(data),
            DynCscMatrix::F64(data) => $fun!(data),
            DynCscMatrix::Bool(data) => $fun!(data),
            DynCscMatrix::String(data) => $fun!(data),
        }
    };
}

impl WriteData for DynCscMatrix {
    fn data_type(&self) -> DataType {
        match self {
            DynCscMatrix::I8(csc) => csc.data_type(),
            DynCscMatrix::I16(csc) => csc.data_type(),
            DynCscMatrix::I32(csc) => csc.data_type(),
            DynCscMatrix::I64(csc) => csc.data_type(),
            DynCscMatrix::U8(csc) => csc.data_type(),
            DynCscMatrix::U16(csc) => csc.data_type(),
            DynCscMatrix::U32(csc) => csc.data_type(),
            DynCscMatrix::U64(csc) => csc.data_type(),
            DynCscMatrix::Usize(csc) => csc.data_type(),
            DynCscMatrix::F32(csc) => csc.data_type(),
            DynCscMatrix::F64(csc) => csc.data_type(),
            DynCscMatrix::Bool(csc) => csc.data_type(),
            DynCscMatrix::String(csc) => csc.data_type(),
        }
    }
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        macro_rules! write_data {
            ($data:expr) => {
                $data.write(location, name)
            };
        }
        impl_dyn_csc_matrix!(self, write_data)
    }
}

impl ReadData for DynCscMatrix {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container {
            DataContainer::Group(group) => match group.open_dataset("data")?.dtype()? {
                ScalarType::I8 => CscMatrix::<i8>::read(container).map(DynCscMatrix::I8),
                ScalarType::I16 => CscMatrix::<i16>::read(container).map(DynCscMatrix::I16),
                ScalarType::I32 => CscMatrix::<i32>::read(container).map(DynCscMatrix::I32),
                ScalarType::I64 => CscMatrix::<i64>::read(container).map(DynCscMatrix::I64),
                ScalarType::U8 => CscMatrix::<u8>::read(container).map(DynCscMatrix::U8),
                ScalarType::U16 => CscMatrix::<u16>::read(container).map(DynCscMatrix::U16),
                ScalarType::U32 => CscMatrix::<u32>::read(container).map(DynCscMatrix::U32),
                ScalarType::U64 => CscMatrix::<u64>::read(container).map(DynCscMatrix::U64),
                ScalarType::Usize => CscMatrix::<usize>::read(container).map(DynCscMatrix::Usize),
                ScalarType::F32 => CscMatrix::<f32>::read(container).map(DynCscMatrix::F32),
                ScalarType::F64 => CscMatrix::<f64>::read(container).map(DynCscMatrix::F64),
                ScalarType::Bool => CscMatrix::<bool>::read(container).map(DynCscMatrix::Bool),
                ScalarType::String => {
                    CscMatrix::<String>::read(container).map(DynCscMatrix::String)
                }
            },
            _ => bail!("cannot read csc matrix from non-group container"),
        }
    }
}

impl HasShape for DynCscMatrix {
    fn shape(&self) -> Shape {
        macro_rules! shape {
            ($data:expr) => {
                $data.shape()
            };
        }
        impl_dyn_csc_matrix!(self, shape)
    }
}

impl ArrayOp for DynCscMatrix {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        macro_rules! get {
            ($data:expr) => {
                $data.get(index)
            };
        }
        impl_dyn_csc_matrix!(self, get)
    }

    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        macro_rules! select {
            ($data:expr) => {
                $data.select(info).into()
            };
        }
        impl_dyn_csc_matrix!(self, select)
    }

    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            DynCscMatrix::U8(_) => Ok(DynCscMatrix::U8(CscMatrix::<u8>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCscMatrix::U16(_) => Ok(DynCscMatrix::U16(CscMatrix::<u16>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCscMatrix::U32(_) => Ok(DynCscMatrix::U32(CscMatrix::<u32>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCscMatrix::U64(_) => Ok(DynCscMatrix::U64(CscMatrix::<u64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCscMatrix::Usize(_) => Ok(DynCscMatrix::Usize(CscMatrix::<usize>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCscMatrix::I8(_) => Ok(DynCscMatrix::I8(CscMatrix::<i8>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCscMatrix::I16(_) => Ok(DynCscMatrix::I16(CscMatrix::<i16>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCscMatrix::I32(_) => Ok(DynCscMatrix::I32(CscMatrix::<i32>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCscMatrix::I64(_) => Ok(DynCscMatrix::I64(CscMatrix::<i64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCscMatrix::F32(_) => Ok(DynCscMatrix::F32(CscMatrix::<f32>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCscMatrix::F64(_) => Ok(DynCscMatrix::F64(CscMatrix::<f64>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCscMatrix::Bool(_) => Ok(DynCscMatrix::Bool(CscMatrix::<bool>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCscMatrix::String(_) => Ok(DynCscMatrix::String(CscMatrix::<String>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
        }
    }
}

impl WriteArrayData for DynCscMatrix {}
impl ReadArrayData for DynCscMatrix {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_array_attr("shape")?
            .to_vec()
            .into())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        if let DataType::CscMatrix(ty) = container.encoding_type()? {
            match ty {
                ScalarType::I8 => CscMatrix::<i8>::read_select(container, info).map(Into::into),
                ScalarType::I16 => CscMatrix::<i16>::read_select(container, info).map(Into::into),
                ScalarType::I32 => CscMatrix::<i32>::read_select(container, info).map(Into::into),
                ScalarType::I64 => CscMatrix::<i64>::read_select(container, info).map(Into::into),
                ScalarType::U8 => CscMatrix::<u8>::read_select(container, info).map(Into::into),
                ScalarType::U16 => CscMatrix::<u16>::read_select(container, info).map(Into::into),
                ScalarType::U32 => CscMatrix::<u32>::read_select(container, info).map(Into::into),
                ScalarType::U64 => CscMatrix::<u64>::read_select(container, info).map(Into::into),
                ScalarType::Usize => {
                    CscMatrix::<usize>::read_select(container, info).map(Into::into)
                }
                ScalarType::F32 => CscMatrix::<f32>::read_select(container, info).map(Into::into),
                ScalarType::F64 => CscMatrix::<f64>::read_select(container, info).map(Into::into),
                ScalarType::Bool => CscMatrix::<bool>::read_select(container, info).map(Into::into),
                ScalarType::String => {
                    CscMatrix::<String>::read_select(container, info).map(Into::into)
                }
            }
        } else {
            bail!("the container does not contain a csc matrix");
        }
    }
}

impl<T> HasShape for CscMatrix<T> {
    fn shape(&self) -> Shape {
        vec![self.nrows(), self.ncols()].into()
    }
}

impl<T: BackendData + Clone> ArrayOp for CscMatrix<T> {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        if index.len() != 2 {
            panic!("index must have length 2");
        }
        todo!()
        //self.get_entry(index[0], index[1]).map(|x| DynScalar::from(x.into_value()))
    }

    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        let info = SelectInfoBounds::new(&info, &self.shape());
        if info.ndim() != 2 {
            panic!("index must have length 2");
        }
        let row_idx = &info.as_ref()[0];
        let col_idx = &info.as_ref()[1];
        let (col_offsets, row_indices, data) = self.csc_data();
        let (new_col_offsets, new_row_indices, new_data) = if row_idx.is_full(info.in_shape()[0]) {
            match col_idx {
                &SelectInfoElemBounds::Slice(SliceBounds { step, start, end }) => {
                    if step == 1 {
                        let (offsets, indices, data) =
                            cs_major_slice(start, end, col_offsets, row_indices, data);
                        (
                            offsets,
                            indices.iter().copied().collect(),
                            data.iter().cloned().collect(),
                        )
                    } else if step < 0 {
                        cs_major_index(
                            (start..end).step_by(step.abs() as usize).rev(),
                            col_offsets,
                            row_indices,
                            data,
                        )
                    } else {
                        cs_major_index(
                            (start..end).step_by(step as usize),
                            col_offsets,
                            row_indices,
                            data,
                        )
                    }
                }
                SelectInfoElemBounds::Index(idx) => {
                    cs_major_index(idx.iter().copied(), col_offsets, row_indices, data)
                }
            }
        } else {
            // row_idx not full
            match col_idx {
                &SelectInfoElemBounds::Slice(SliceBounds {
                    start: col_start,
                    end: col_end,
                    step: col_step,
                }) => {
                    if col_step < 0 {
                        // col_idx is major, row_idx is minor
                        match row_idx {
                            &SelectInfoElemBounds::Slice(row) => {
                                if row.step < 0 {
                                    cs_major_minor_index(
                                        (col_start..col_end).step_by(col_step.abs() as usize).rev(),
                                        (row.start..row.end).step_by(row.step.abs() as usize).rev(),
                                        self.nrows(),
                                        col_offsets,
                                        row_indices,
                                        data,
                                    )
                                } else {
                                    cs_major_minor_index(
                                        (col_start..col_end).step_by(col_step.abs() as usize).rev(),
                                        (row.start..row.end).step_by(row.step as usize),
                                        self.nrows(),
                                        col_offsets,
                                        row_indices,
                                        data,
                                    )
                                }
                            }
                            SelectInfoElemBounds::Index(idx) => cs_major_minor_index(
                                (col_start..col_end).step_by(col_step.abs() as usize).rev(),
                                idx.iter().copied(),
                                self.nrows(),
                                col_offsets,
                                row_indices,
                                data,
                            ),
                        }
                    } else {
                        // col_step >0, col_idx is major, row_idx is minor
                        match row_idx {
                            &SelectInfoElemBounds::Slice(row) => {
                                if row.step < 0 {
                                    cs_major_minor_index(
                                        (col_start..col_end).step_by(col_step as usize),
                                        (row.start..row.end).step_by(row.step.abs() as usize).rev(),
                                        self.nrows(),
                                        col_offsets,
                                        row_indices,
                                        data,
                                    )
                                } else {
                                    cs_major_minor_index(
                                        (col_start..col_end).step_by(col_step as usize),
                                        (row.start..row.end).step_by(row.step as usize),
                                        self.nrows(),
                                        col_offsets,
                                        row_indices,
                                        data,
                                    )
                                }
                            }
                            SelectInfoElemBounds::Index(idx) => cs_major_minor_index(
                                (col_start..col_end).step_by(col_step as usize),
                                idx.iter().copied(),
                                self.nrows(),
                                col_offsets,
                                row_indices,
                                data,
                            ),
                        }
                    }
                }
                SelectInfoElemBounds::Index(i) => match row_idx {
                    &SelectInfoElemBounds::Slice(row) => {
                        if row.step < 0 {
                            cs_major_minor_index(
                                i.iter().copied(),
                                (row.start..row.end).step_by(row.step.abs() as usize).rev(),
                                self.nrows(),
                                col_offsets,
                                row_indices,
                                data,
                            )
                        } else {
                            cs_major_minor_index(
                                i.iter().copied(),
                                (row.start..row.end).step_by(row.step as usize),
                                self.nrows(),
                                col_offsets,
                                row_indices,
                                data,
                            )
                        }
                    }
                    SelectInfoElemBounds::Index(j) => cs_major_minor_index(
                        i.iter().copied(),
                        j.iter().copied(),
                        self.nrows(),
                        col_offsets,
                        row_indices,
                        data,
                    ),
                },
            }
        };
        let out_shape = info.out_shape();
        let pattern = unsafe {
            SparsityPattern::from_offset_and_indices_unchecked(
                out_shape[1],
                out_shape[0],
                new_col_offsets,
                new_row_indices,
            )
        };
        CscMatrix::try_from_pattern_and_values(pattern, new_data).unwrap()
    }

    fn vstack<I: Iterator<Item = Self>>(_iter: I) -> Result<Self>
    where
        Self: Sized,
    {
        todo!()
    }
}

impl<T: BackendData> WriteData for CscMatrix<T> {
    fn data_type(&self) -> DataType {
        DataType::CscMatrix(T::DTYPE)
    }
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let mut group = location.new_group(name)?;
        let shape = self.shape();

        group.new_str_attr("encoding-type", "csc_matrix")?;
        group.new_str_attr("encoding-version", "0.1.0")?;
        group.new_array_attr("shape", shape.as_ref())?;

        group.new_array_dataset("data", self.values().into(), Default::default())?;

        let num_rows = shape[0];
        // Use i32 or i64 as indices type in order to be compatible with scipy
        if TryInto::<i32>::try_into(num_rows.saturating_sub(1)).is_ok() {
            let try_convert_indptr: Option<Vec<i32>> = self
                .col_offsets()
                .iter()
                .map(|x| (*x).try_into().ok())
                .collect();
            if let Some(indptr_i32) = try_convert_indptr {
                group.new_array_dataset("indptr", indptr_i32.into(), Default::default())?;
                group.new_array_dataset(
                    "indices",
                    self.row_indices()
                        .iter()
                        .map(|x| (*x) as i32)
                        .collect::<Vec<_>>()
                        .into(),
                    Default::default(),
                )?;
            } else {
                group.new_array_dataset(
                    "indptr",
                    self.col_offsets()
                        .iter()
                        .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                        .collect::<Vec<_>>()
                        .into(),
                    Default::default(),
                )?;
                group.new_array_dataset(
                    "indices",
                    self.row_indices()
                        .iter()
                        .map(|x| (*x) as i64)
                        .collect::<Vec<_>>()
                        .into(),
                    Default::default(),
                )?;
            }
        } else if TryInto::<i64>::try_into(num_rows.saturating_sub(1)).is_ok() {
            group.new_array_dataset(
                "indptr",
                self.col_offsets()
                    .iter()
                    .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                    .collect::<Vec<_>>()
                    .into(),
                Default::default(),
            )?;
            group.new_array_dataset(
                "indices",
                self.row_indices()
                    .iter()
                    .map(|x| (*x) as i64)
                    .collect::<Vec<_>>()
                    .into(),
                Default::default(),
            )?;
        } else {
            panic!(
                "The number of rows ({}) is too large to be stored as i64",
                num_rows
            );
        }

        Ok(DataContainer::Group(group))
    }
}

impl<T: BackendData> ReadData for CscMatrix<T> {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let data_type = container.encoding_type()?;
        if let DataType::CscMatrix(_) = data_type {
            let group = container.as_group()?;
            let shape: Vec<usize> = group.get_array_attr("shape")?.to_vec();
            let data = group
                .open_dataset("data")?
                .read_array::<_, Ix1>()?
                .into_raw_vec_and_offset()
                .0;
            let indptr: Vec<usize> = group
                .open_dataset("indptr")?
                .read_array::<_, Ix1>()?
                .into_raw_vec_and_offset()
                .0;
            let indices: Vec<usize> = group
                .open_dataset("indices")?
                .read_array::<_, Ix1>()?
                .into_raw_vec_and_offset()
                .0;
            CscMatrix::try_from_csc_data(shape[0], shape[1], indptr, indices, data)
                .map_err(|e| anyhow::anyhow!("{}", e))
        } else {
            bail!(
                "cannot read csc matrix from container with data type {:?}",
                data_type
            )
        }
    }
}

impl<T: BackendData> ReadArrayData for CscMatrix<T> {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_array_attr("shape")?
            .to_vec()
            .into())
    }

    // TODO: efficient implementation for slice
    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        let data_type = container.encoding_type()?;
        if let DataType::CscMatrix(_) = data_type {
            if info.as_ref().len() != 2 {
                panic!("index must have length 2");
            }

            if info.iter().all(|s| s.as_ref().is_full()) {
                return Self::read(container);
            }

            let data = if let SelectInfoElem::Slice(s) = info[1].as_ref() {
                let group = container.as_group()?;
                let indptr_slice = if let Some(end) = s.end {
                    SelectInfoElem::from(s.start..end + 1)
                } else {
                    SelectInfoElem::from(s.start..)
                };
                let mut indptr: Vec<usize> = group
                    .open_dataset("indptr")?
                    .read_array_slice(&[indptr_slice])?
                    .to_vec();
                let lo = indptr[0];
                let slice = SelectInfoElem::from(lo..indptr[indptr.len() - 1]);
                let data: Vec<T> = group
                    .open_dataset("data")?
                    .read_array_slice(&[&slice])?
                    .to_vec();
                let indices: Vec<usize> = group
                    .open_dataset("indices")?
                    .read_array_slice(&[&slice])?
                    .to_vec();
                indptr.iter_mut().for_each(|x| *x -= lo);
                CscMatrix::try_from_csc_data(
                    Self::get_shape(container)?[0],
                    indptr.len() - 1,
                    indptr,
                    indices,
                    data,
                )
                .unwrap()
                .select_axis(0, info[0].as_ref()) // selct axis 1, then select axis 0 elements
            } else {
                Self::read(container)?.select(info)
            };
            Ok(data)
        } else {
            bail!(
                "cannot read csc matrix from container with data type {:?}",
                data_type
            )
        }
    }
}

impl<T: BackendData> WriteArrayData for &CscMatrix<T> {}
impl<T: BackendData> WriteArrayData for CscMatrix<T> {}

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

fn cast_csc<T, U>(csc: CscMatrix<T>) -> Result<CscMatrix<U>>
where
    T: TryInto<U>,
    <T as TryInto<U>>::Error: std::error::Error + Sync + Send + 'static,
{
    let (pattern, values) = csc.into_pattern_and_values();
    let out = CscMatrix::try_from_pattern_and_values(
        pattern,
        values
            .into_iter()
            .map(|x| x.try_into())
            .collect::<Result<_, _>>()?,
    )
    .unwrap();
    Ok(out)
}

fn from_i64_csc<U: FromPrimitive>(csc: CscMatrix<i64>) -> Result<CscMatrix<U>> {
    let (pattern, values) = csc.into_pattern_and_values();
    let out = CscMatrix::try_from_pattern_and_values(
        pattern,
        values
            .into_iter()
            .map(|x| U::from_i64(x).context("cannot convert from i64"))
            .collect::<Result<_>>()?,
    )
    .unwrap();
    Ok(out)
}

#[cfg(test)]
mod csc_matrix_index_tests {
    use super::*;
    use crate::s;
    use nalgebra::base::DMatrix;
    use nalgebra_sparse::CooMatrix;
    use ndarray::Array;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn csc_select<I1, I2>(csc: &CscMatrix<i64>, row_indices: I1, col_indices: I2) -> CscMatrix<i64>
    where
        I1: Iterator<Item = usize>,
        I2: Iterator<Item = usize>,
    {
        let i = row_indices.collect::<Vec<_>>();
        let j = col_indices.collect::<Vec<_>>();
        let mut dm = DMatrix::<i64>::zeros(csc.nrows(), csc.ncols());
        csc.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
        CscMatrix::from(&dm.select_rows(&i).select_columns(&j))
    }

    fn csc_select_rows<I>(csc: &CscMatrix<i64>, row_indices: I) -> CscMatrix<i64>
    where
        I: Iterator<Item = usize>,
    {
        let i = row_indices.collect::<Vec<_>>();
        let mut dm = DMatrix::<i64>::zeros(csc.nrows(), csc.ncols());
        csc.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
        CscMatrix::from(&dm.select_rows(&i))
    }

    fn csc_select_cols<I>(csc: &CscMatrix<i64>, col_indices: I) -> CscMatrix<i64>
    where
        I: Iterator<Item = usize>,
    {
        let j = col_indices.collect::<Vec<_>>();
        let mut dm = DMatrix::<i64>::zeros(csc.nrows(), csc.ncols());
        csc.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
        CscMatrix::from(&dm.select_columns(&j))
    }

    #[test]
    fn test_csc() {
        let n: usize = 200;
        let m: usize = 200;
        let nnz: usize = 1000;

        let ridx = Array::random(220, Uniform::new(0, n)).to_vec();
        let cidx = Array::random(100, Uniform::new(0, m)).to_vec();

        let row_indices = Array::random(nnz, Uniform::new(0, n)).to_vec();
        let col_indices = Array::random(nnz, Uniform::new(0, m)).to_vec();
        let values = Array::random(nnz, Uniform::new(-10000, 10000)).to_vec();

        let csc_matrix: CscMatrix<i64> =
            (&CooMatrix::try_from_triplets(n, m, row_indices, col_indices, values).unwrap()).into();

        // Row slice
        assert_eq!(
            csc_matrix.select(s![2..177, ..].as_ref()),
            csc_select_rows(&csc_matrix, 2..177),
        );
        assert_eq!(
            csc_matrix.select(s![0..2, ..].as_ref()),
            csc_select_rows(&csc_matrix, 0..2),
        );

        // Row fancy indexing
        assert_eq!(
            csc_matrix.select(s![&ridx, ..].as_ref()),
            csc_select_rows(&csc_matrix, ridx.iter().cloned()),
        );

        // Column slice
        assert_eq!(
            csc_matrix.select(s![.., 77..200].as_ref()),
            csc_select_cols(&csc_matrix, 77..200),
        );

        // Column fancy indexing
        assert_eq!(
            csc_matrix.select(s![.., &cidx].as_ref()),
            csc_select_cols(&csc_matrix, cidx.iter().cloned()),
        );

        // Both
        assert_eq!(
            csc_matrix.select(s![2..49, 0..77].as_ref()),
            csc_select(&csc_matrix, 2..49, 0..77),
        );

        assert_eq!(
            csc_matrix.select(s![2..177, &cidx].as_ref()),
            csc_select(&csc_matrix, 2..177, cidx.iter().cloned()),
        );

        assert_eq!(
            csc_matrix.select(s![&ridx, &cidx].as_ref()),
            csc_select(&csc_matrix, ridx.iter().cloned(), cidx.iter().cloned()),
        );
    }
}
