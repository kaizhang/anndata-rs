use crate::backend::*;
use crate::data::{
    array::utils::{cs_major_minor_index, cs_major_index, cs_major_slice},
    data_traits::*,
    scalar::DynScalar,
    slice::{SelectInfoElem, Shape},
    BoundedSelectInfo, BoundedSelectInfoElem,
};

use anyhow::{bail, anyhow, Context, Result};
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::pattern::SparsityPattern;
use ndarray::Ix1;
use num::FromPrimitive;

use super::super::slice::BoundedSlice;

#[derive(Debug, Clone, PartialEq)]
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
                    _ => bail!("Cannot convert {:?} to {} CsrMatrix", data.data_type(), stringify!($from_type)),
                }
            }
        }
    };
}

impl From<CsrMatrix<u32>> for DynCsrMatrix {
    fn from(data: CsrMatrix<u32>) -> Self {
        DynCsrMatrix::U32(data)
    }
}

impl TryFrom<DynCsrMatrix> for CsrMatrix<u32> {
    type Error = anyhow::Error;
    fn try_from(data: DynCsrMatrix) -> Result<Self> {
        match data {
            DynCsrMatrix::U32(data) => Ok(data),
            DynCsrMatrix::I8(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::I16(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::I32(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::I64(data) => Ok(from_i64_csr(data)?),
            DynCsrMatrix::U8(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::U16(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::U64(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::Usize(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::F32(_) => bail!("Cannot convert f32 to u32"),
            DynCsrMatrix::F64(_) => bail!("Cannot convert f64 to u32"),
            DynCsrMatrix::Bool(_) => bail!("Cannot convert bool to f64"),
            DynCsrMatrix::String(_) => bail!("Cannot convert string to f64"),
        }
    }
}


impl From<CsrMatrix<f64>> for DynCsrMatrix {
    fn from(data: CsrMatrix<f64>) -> Self {
        DynCsrMatrix::F64(data)
    }
}

impl TryFrom<DynCsrMatrix> for CsrMatrix<f64> {
    type Error = anyhow::Error;
    fn try_from(data: DynCsrMatrix) -> Result<Self> {
        match data {
            DynCsrMatrix::F64(data) => Ok(data),
            DynCsrMatrix::I8(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::I16(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::I32(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::I64(data) => Ok(from_i64_csr(data)?),
            DynCsrMatrix::U8(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::U16(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::U32(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::U64(_) => bail!("Cannot convert u64 to f64"),
            DynCsrMatrix::Usize(_) => bail!("Cannot convert usize to f64"),
            DynCsrMatrix::F32(data) => Ok(cast_csr(data)?),
            DynCsrMatrix::Bool(_) => bail!("Cannot convert bool to f64"),
            DynCsrMatrix::String(_) => bail!("Cannot convert string to f64"),
        }
    }
}

impl_into_dyn_csr!(i8, I8);
impl_into_dyn_csr!(i16, I16);
impl_into_dyn_csr!(i32, I32);
impl_into_dyn_csr!(i64, I64);
impl_into_dyn_csr!(u8, U8);
impl_into_dyn_csr!(u16, U16);
impl_into_dyn_csr!(u64, U64);
impl_into_dyn_csr!(usize, Usize);
impl_into_dyn_csr!(f32, F32);
impl_into_dyn_csr!(bool, Bool);
impl_into_dyn_csr!(String, String);

macro_rules! impl_dyn_csr_matrix {
    ($self:expr, $fun:ident) => {
        match $self {
            DynCsrMatrix::I8(data) => $fun!(data),
            DynCsrMatrix::I16(data) => $fun!(data),
            DynCsrMatrix::I32(data) => $fun!(data),
            DynCsrMatrix::I64(data) => $fun!(data),
            DynCsrMatrix::U8(data) => $fun!(data),
            DynCsrMatrix::U16(data) => $fun!(data),
            DynCsrMatrix::U32(data) => $fun!(data),
            DynCsrMatrix::U64(data) => $fun!(data),
            DynCsrMatrix::Usize(data) => $fun!(data),
            DynCsrMatrix::F32(data) => $fun!(data),
            DynCsrMatrix::F64(data) => $fun!(data),
            DynCsrMatrix::Bool(data) => $fun!(data),
            DynCsrMatrix::String(data) => $fun!(data),
        }
    };
}

impl WriteData for DynCsrMatrix {
    fn data_type(&self) -> DataType {
        macro_rules! data_type{
            ($data:expr) => {
                $data.data_type()
            };
        }
        impl_dyn_csr_matrix!(self, data_type)
    }
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        macro_rules! write_data {
            ($data:expr) => {
                $data.write(location, name)
            };
        }
        impl_dyn_csr_matrix!(self, write_data)
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
                ScalarType::String => {
                    CsrMatrix::<String>::read(container).map(DynCsrMatrix::String)
                }
            },
            _ => bail!("cannot read csr matrix from non-group container"),
        }
    }
}

impl HasShape for DynCsrMatrix {
    fn shape(&self) -> Shape {
        macro_rules! shape {
            ($data:expr) => {
                $data.shape()
            };
        }
        impl_dyn_csr_matrix!(self, shape)
    }
}

impl ArrayOp for DynCsrMatrix {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        macro_rules! get {
            ($data:expr) => {
                $data.get(index)
            };
        }
        impl_dyn_csr_matrix!(self, get)
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
        impl_dyn_csr_matrix!(self, select)
    }

    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            DynCsrMatrix::U8(_) => Ok(DynCsrMatrix::U8(CsrMatrix::<u8>::vstack(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::U16(_) => Ok(DynCsrMatrix::U16(CsrMatrix::<u16>::vstack(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::U32(_) => Ok(DynCsrMatrix::U32(CsrMatrix::<u32>::vstack(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::U64(_) => Ok(DynCsrMatrix::U64(CsrMatrix::<u64>::vstack(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::Usize(_) => Ok(DynCsrMatrix::Usize(CsrMatrix::<usize>::vstack(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::I8(_) => Ok(DynCsrMatrix::I8(CsrMatrix::<i8>::vstack(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::I16(_) => Ok(DynCsrMatrix::I16(CsrMatrix::<i16>::vstack(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::I32(_) => Ok(DynCsrMatrix::I32(CsrMatrix::<i32>::vstack(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::I64(_) => Ok(DynCsrMatrix::I64(CsrMatrix::<i64>::vstack(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::F32(_) => Ok(DynCsrMatrix::F32(CsrMatrix::<f32>::vstack(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::F64(_) => Ok(DynCsrMatrix::F64(CsrMatrix::<f64>::vstack(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::Bool(_) => Ok(DynCsrMatrix::Bool(CsrMatrix::<bool>::vstack(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::String(_) => Ok(DynCsrMatrix::String(CsrMatrix::<String>::vstack(iter.map(|x| x.try_into().unwrap()))?)),
        }
    }
}

impl WriteArrayData for DynCsrMatrix {}
impl ReadArrayData for DynCsrMatrix {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .read_array_attr("shape")?
            .to_vec()
            .into())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        if let DataType::CsrMatrix(ty) = container.encoding_type()? {
            match ty {
                ScalarType::I8 => CsrMatrix::<i8>::read_select(container, info)
                    .map(Into::into),
                ScalarType::I16 => CsrMatrix::<i16>::read_select(container, info)
                    .map(Into::into),
                ScalarType::I32 => CsrMatrix::<i32>::read_select(container, info)
                    .map(Into::into),
                ScalarType::I64 => CsrMatrix::<i64>::read_select(container, info)
                    .map(Into::into),
                ScalarType::U8 => CsrMatrix::<u8>::read_select(container, info)
                    .map(Into::into),
                ScalarType::U16 => CsrMatrix::<u16>::read_select(container, info)
                    .map(Into::into),
                ScalarType::U32 => CsrMatrix::<u32>::read_select(container, info)
                    .map(Into::into),
                ScalarType::U64 => CsrMatrix::<u64>::read_select(container, info) 
                    .map(Into::into),
                ScalarType::Usize => CsrMatrix::<usize>::read_select(container, info)
                    .map(Into::into),
                ScalarType::F32 => CsrMatrix::<f32>::read_select(container, info)
                    .map(Into::into),
                ScalarType::F64 => CsrMatrix::<f64>::read_select(container, info)
                    .map(Into::into),
                ScalarType::Bool => CsrMatrix::<bool>::read_select(container, info)
                    .map(Into::into),
                ScalarType::String => CsrMatrix::<String>::read_select(container, info)
                    .map(Into::into),
            }
        } else {
            bail!("the container does not contain a csr matrix");
        }
    }
}

impl<T> HasShape for CsrMatrix<T> {
    fn shape(&self) -> Shape {
        vec![self.nrows(), self.ncols()].into()
    }
}

impl<T: BackendData + Clone> ArrayOp for CsrMatrix<T> {
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
        let info = BoundedSelectInfo::new(&info, &self.shape());
        if info.ndim() != 2 {
            panic!("index must have length 2");
        }
        let row_idx = &info.as_ref()[0];
        let col_idx = &info.as_ref()[1];
        let (row_offsets, col_indices, data) = self.csr_data();
        let (new_row_offsets, new_col_indices, new_data) = if col_idx.is_full(info.in_shape()[1]) {
            match row_idx {
                &BoundedSelectInfoElem::Slice(BoundedSlice { step, start, end }) => {
                    if step == 1 {
                        let (offsets, indices, data) =
                            cs_major_slice(start, end, row_offsets, col_indices, data);
                        (
                            offsets,
                            indices.iter().copied().collect(),
                            data.iter().cloned().collect(),
                        )
                    } else if step < 0 {
                        cs_major_index(
                            (start..end).step_by(step.abs() as usize).rev(),
                            row_offsets,
                            col_indices,
                            data,
                        )
                    } else {
                        cs_major_index(
                            (start..end).step_by(step as usize),
                            row_offsets,
                            col_indices,
                            data,
                        )
                    }
                }
                BoundedSelectInfoElem::Index(idx) => {
                    cs_major_index(idx.iter().copied(), row_offsets, col_indices, data)
                }
            }
        } else {
            match row_idx {
                &BoundedSelectInfoElem::Slice(BoundedSlice { start: row_start,end: row_end, step: row_step }) => {
                    if row_step < 0 {
                        match col_idx {
                            &BoundedSelectInfoElem::Slice(col) => {
                                if col.step < 0 {
                                    cs_major_minor_index(
                                        (row_start..row_end).step_by(row_step.abs() as usize).rev(),
                                        (col.start..col.end).step_by(col.step.abs() as usize).rev(),
                                        self.ncols(),
                                        row_offsets,
                                        col_indices,
                                        data,
                                    )
                                } else {
                                    cs_major_minor_index(
                                        (row_start..row_end).step_by(row_step.abs() as usize).rev(),
                                        (col.start..col.end).step_by(col.step as usize),
                                        self.ncols(),
                                        row_offsets,
                                        col_indices,
                                        data,
                                    )
                                }
                            }
                            BoundedSelectInfoElem::Index(idx) => cs_major_minor_index(
                                (row_start..row_end).step_by(row_step.abs() as usize).rev(),
                                idx.iter().copied(),
                                self.ncols(),
                                row_offsets,
                                col_indices,
                                data,
                            ),
                        }
                    } else {
                        match col_idx {
                            &BoundedSelectInfoElem::Slice(col) => {
                                if col.step < 0 {
                                    cs_major_minor_index(
                                        (row_start..row_end).step_by(row_step as usize),
                                        (col.start..col.end).step_by(col.step.abs() as usize).rev(),
                                        self.ncols(),
                                        row_offsets,
                                        col_indices,
                                        data,
                                    )
                                } else {
                                    cs_major_minor_index(
                                        (row_start..row_end).step_by(row_step as usize),
                                        (col.start..col.end).step_by(col.step as usize),
                                        self.ncols(),
                                        row_offsets,
                                        col_indices,
                                        data,
                                    )
                                }
                            }
                            BoundedSelectInfoElem::Index(idx) => cs_major_minor_index(
                                (row_start..row_end).step_by(row_step as usize),
                                idx.iter().copied(),
                                self.ncols(),
                                row_offsets,
                                col_indices,
                                data,
                            ),
                        }
                    }
                }
                BoundedSelectInfoElem::Index(i) => match col_idx {
                    &BoundedSelectInfoElem::Slice(col) => {
                        if col.step < 0 {
                            cs_major_minor_index(
                                i.iter().copied(),
                                (col.start..col.end).step_by(col.step.abs() as usize).rev(),
                                self.ncols(),
                                row_offsets,
                                col_indices,
                                data,
                            )
                        } else {
                            cs_major_minor_index(
                                i.iter().copied(),
                                (col.start..col.end).step_by(col.step as usize),
                                self.ncols(),
                                row_offsets,
                                col_indices,
                                data,
                            )
                        }
                    }
                    BoundedSelectInfoElem::Index(j) => cs_major_minor_index(
                        i.iter().copied(),
                        j.iter().copied(),
                        self.ncols(),
                        row_offsets,
                        col_indices,
                        data,
                    ),
                },
            }
        };
        let out_shape = info.out_shape();
        let pattern = unsafe {
            SparsityPattern::from_offset_and_indices_unchecked(
                out_shape[0],
                out_shape[1],
                new_row_offsets,
                new_col_indices,
            )
        };
        CsrMatrix::try_from_pattern_and_values(pattern, new_data).unwrap()
    }

    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        fn vstack_csr<T: Clone>(this: CsrMatrix<T>, other: CsrMatrix<T>) -> CsrMatrix<T> {
            let num_cols = this.ncols();
            let num_rows = this.nrows() + other.nrows();
            let nnz = this.nnz();
            let (mut indptr, mut indices, mut data) = this.disassemble();
            let (indptr2, indices2, data2) = other.csr_data();
            indices.extend_from_slice(indices2);
            data.extend_from_slice(data2);
            indptr2.iter().skip(1).for_each(|&i| indptr.push(i + nnz));

            let pattern = unsafe {
                SparsityPattern::from_offset_and_indices_unchecked(num_rows, num_cols, indptr, indices)
            };
            CsrMatrix::try_from_pattern_and_values(pattern, data).unwrap()
        }

        Ok(iter.reduce(|acc, x| vstack_csr(acc, x)).unwrap())
    }
}


impl<T: BackendData> WriteData for CsrMatrix<T> {
    fn data_type(&self) -> DataType {
        DataType::CsrMatrix(T::DTYPE)
    }
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let group = location.create_group(name)?;
        let shape = self.shape();

        group.write_str_attr("encoding-type", "csr_matrix")?;
        group.write_str_attr("encoding-version", "0.1.0")?;
        group.write_array_attr("shape", shape.as_ref())?;

        group.create_array_data("data", &self.values(), Default::default())?;

        let num_cols = shape[1];
        // Use i32 or i64 as indices type in order to be compatible with scipy
        if TryInto::<i32>::try_into(num_cols.saturating_sub(1)).is_ok() {
            let try_convert_indptr: Option<Vec<i32>> = self
                .row_offsets()
                .iter()
                .map(|x| (*x).try_into().ok())
                .collect();
            if let Some(indptr_i32) = try_convert_indptr {
                group.create_array_data("indptr", &indptr_i32, Default::default())?;
                group.create_array_data(
                    "indices",
                    self.col_indices()
                        .iter()
                        .map(|x| (*x) as i32)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    Default::default(),
                )?;
            } else {
                group.create_array_data(
                    "indptr",
                    self.row_offsets()
                        .iter()
                        .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                        .collect::<Vec<_>>()
                        .as_slice(),
                    Default::default(),
                )?;
                group.create_array_data(
                    "indices",
                    self.col_indices()
                        .iter()
                        .map(|x| (*x) as i64)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    Default::default(),
                )?;
            }
        } else if TryInto::<i64>::try_into(num_cols.saturating_sub(1)).is_ok() {
            group.create_array_data(
                "indptr",
                self.row_offsets()
                    .iter()
                    .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
                Default::default(),
            )?;
            group.create_array_data(
                "indices",
                self.col_indices()
                    .iter()
                    .map(|x| (*x) as i64)
                    .collect::<Vec<_>>()
                    .as_slice(),
                Default::default(),
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
        let data_type = container.encoding_type()?;
        if let DataType::CsrMatrix(_) = data_type {
            let group = container.as_group()?;
            let shape: Vec<usize> = group.read_array_attr("shape")?.to_vec();
            let data = group.open_dataset("data")?.read_array::<_, Ix1>()?.into_raw_vec();
            let indptr: Vec<usize> = group.open_dataset("indptr")?.read_array::<_, Ix1>()?.into_raw_vec();
            let indices: Vec<usize> = group.open_dataset("indices")?.read_array::<_, Ix1>()?.into_raw_vec();
            CsrMatrix::try_from_csr_data(
                shape[0], shape[1], indptr, indices, data
            ).map_err(|e| anyhow!("cannot read csr matrix: {}", e))
        } else {
            bail!("cannot read csr matrix from container with data type {:?}", data_type)
        }
    }
}

impl<T: BackendData> ReadArrayData for CsrMatrix<T> {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .read_array_attr("shape")?
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
        if let DataType::CsrMatrix(_) = data_type {
            if info.as_ref().len() != 2 {
                panic!("index must have length 2");
            }

            if info.iter().all(|s| s.as_ref().is_full()) {
                return Self::read(container);
            }

            let data = if let SelectInfoElem::Slice(s) = info[0].as_ref()  {
                let group = container.as_group()?;
                let indptr_slice = if let Some(end) = s.end {
                    SelectInfoElem::from(s.start .. end + 1)
                } else {
                    SelectInfoElem::from(s.start ..)
                };
                let mut indptr: Vec<usize> = group 
                    .open_dataset("indptr")?
                    .read_array_slice(&[indptr_slice])?
                    .to_vec();
                let lo = indptr[0];
                let slice = SelectInfoElem::from(lo .. indptr[indptr.len() - 1]);
                let data: Vec<T> = group.open_dataset("data")?.read_array_slice(&[&slice])?.to_vec();
                let indices: Vec<usize> = group.open_dataset("indices")?.read_array_slice(&[&slice])?.to_vec();
                indptr.iter_mut().for_each(|x| *x -= lo);
                CsrMatrix::try_from_csr_data(
                    indptr.len() - 1,
                    Self::get_shape(container)?[1],
                    indptr,
                    indices,
                    data,
                ).unwrap().select_axis(1, info[1].as_ref())
            } else {
                Self::read(container)?.select(info)
            };
            Ok(data)
        } else {
            bail!("cannot read csr matrix from container with data type {:?}", data_type)
        }
    }
}

impl<T: BackendData> WriteArrayData for &CsrMatrix<T> {}
impl<T: BackendData> WriteArrayData for CsrMatrix<T> {}


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

fn cast_csr<T, U>(csr: CsrMatrix<T>) -> Result<CsrMatrix<U>>
where
    T: TryInto<U>,
    <T as TryInto<U>>::Error: std::error::Error + Sync + Send + 'static,
{
    let (pattern, values) = csr.into_pattern_and_values();
    let out = CsrMatrix::try_from_pattern_and_values(
        pattern,
        values.into_iter().map(|x| x.try_into()).collect::<Result<_, _>>()?,
    ).unwrap();
    Ok(out)
}

fn from_i64_csr<U: FromPrimitive>(csr: CsrMatrix<i64>) -> Result<CsrMatrix<U>>
{
    let (pattern, values) = csr.into_pattern_and_values();
    let out = CsrMatrix::try_from_pattern_and_values(
        pattern,
        values.into_iter().map(|x| U::from_i64(x).context("cannot convert from i64")).collect::<Result<_>>()?,
    ).unwrap();
    Ok(out)
}

#[cfg(test)]
mod csr_matrix_index_tests {
    use super::*;
    use crate::s;
    use nalgebra::base::DMatrix;
    use nalgebra_sparse::CooMatrix;
    use ndarray::Array;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn csr_select<I1, I2>(
        csr: &CsrMatrix<i64>,
        row_indices: I1,
        col_indices: I2,
    ) -> CsrMatrix<i64>
    where
        I1: Iterator<Item = usize>,
        I2: Iterator<Item = usize>,
    {
        let i = row_indices.collect::<Vec<_>>();
        let j = col_indices.collect::<Vec<_>>();
        let mut dm = DMatrix::<i64>::zeros(csr.nrows(), csr.ncols());
        csr.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
        CsrMatrix::from(&dm.select_rows(&i).select_columns(&j))
    }

    fn csr_select_rows<I>(csr: &CsrMatrix<i64>, row_indices: I) -> CsrMatrix<i64>
    where
        I: Iterator<Item = usize>,
    {
        let i = row_indices.collect::<Vec<_>>();
        let mut dm = DMatrix::<i64>::zeros(csr.nrows(), csr.ncols());
        csr.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
        CsrMatrix::from(&dm.select_rows(&i))
    }

    fn csr_select_cols<I>(csr: &CsrMatrix<i64>, col_indices: I) -> CsrMatrix<i64>
    where
        I: Iterator<Item = usize>,
    {
        let j = col_indices.collect::<Vec<_>>();
        let mut dm = DMatrix::<i64>::zeros(csr.nrows(), csr.ncols());
        csr.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
        CsrMatrix::from(&dm.select_columns(&j))
    }

    #[test]
    fn test_c() {
        let dense = DMatrix::from_row_slice(3, 3, &[1, 0, 3, 2, 0, 1, 0, 0, 4]);
        let csr = CsrMatrix::from(&dense);

        // Column fancy indexing
        let cidx = [1, 2, 0, 1, 1, 2];
        let mut expected = DMatrix::from_row_slice(
            3,
            6,
            &[0, 3, 1, 0, 0, 3, 0, 1, 2, 0, 0, 1, 0, 4, 0, 0, 0, 4],
        );
        let mut expected_csr = CsrMatrix::from(&expected);
        assert_eq!(csr.select(s![.., cidx.as_ref()].as_ref()), expected_csr,);

        expected = DMatrix::from_row_slice(3, 2, &[1, 0, 2, 0, 0, 0]);
        expected_csr = CsrMatrix::from(&expected);
        assert_eq!(csr.select(s![.., 0..2].as_ref()), expected_csr);

        let ridx = [1, 2, 0, 1];
        expected = DMatrix::from_row_slice(
            4,
            6,
            &[
                0, 1, 2, 0, 0, 1, 0, 4, 0, 0, 0, 4, 0, 3, 1, 0, 0, 3, 0, 1, 2, 0, 0, 1,
            ],
        );
        expected_csr = CsrMatrix::from(&expected);
        let (new_offsets, new_indices, new_data) = cs_major_minor_index(
            ridx.into_iter(),
            cidx.into_iter(),
            csr.ncols(),
            csr.row_offsets(),
            csr.col_indices(),
            csr.values(),
        );

        assert_eq!(new_offsets.as_slice(), expected_csr.row_offsets());
        assert_eq!(new_indices.as_slice(), expected_csr.col_indices());
        assert_eq!(new_data.as_slice(), expected_csr.values());
    }

    #[test]
    fn test_csr() {
        let n: usize = 200;
        let m: usize = 200;
        let nnz: usize = 1000;

        let ridx = Array::random(220, Uniform::new(0, n)).to_vec();
        let cidx = Array::random(100, Uniform::new(0, m)).to_vec();

        let row_indices = Array::random(nnz, Uniform::new(0, n)).to_vec();
        let col_indices = Array::random(nnz, Uniform::new(0, m)).to_vec();
        let values = Array::random(nnz, Uniform::new(-10000, 10000)).to_vec();

        let csr_matrix: CsrMatrix<i64> =
            (&CooMatrix::try_from_triplets(n, m, row_indices, col_indices, values).unwrap()).into();

        // Row slice
        assert_eq!(
            csr_matrix.select(s![2..177, ..].as_ref()),
            csr_select_rows(&csr_matrix, 2..177),
        );

        // Row fancy indexing
        assert_eq!(
            csr_matrix.select(s![&ridx, ..].as_ref()),
            csr_select_rows(&csr_matrix, ridx.iter().cloned()),
        );

        // Column slice
        assert_eq!(
            csr_matrix.select(s![.., 77..200].as_ref()),
            csr_select_cols(&csr_matrix, 77..200),
        );

        // Column fancy indexing
        assert_eq!(
            csr_matrix.select(s![.., &cidx].as_ref()),
            csr_select_cols(&csr_matrix, cidx.iter().cloned()),
        );

        // Both
        assert_eq!(
            csr_matrix.select(s![2..49, 0..77].as_ref()),
            csr_select(&csr_matrix, 2..49, 0..77),
        );

        assert_eq!(
            csr_matrix.select(s![2..177, &cidx].as_ref()),
            csr_select(&csr_matrix, 2..177, cidx.iter().cloned()),
        );

        assert_eq!(
            csr_matrix.select(s![&ridx, &cidx].as_ref()),
            csr_select(&csr_matrix, ridx.iter().cloned(), cidx.iter().cloned()),
        );
    }
}
