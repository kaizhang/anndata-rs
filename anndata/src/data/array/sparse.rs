use crate::backend::*;
use crate::data::{
    array::utils::{cs_major_minor_index, cs_major_index, cs_major_slice},
    data_traits::*,
    scalar::DynScalar,
    slice::{SelectInfoElem, Shape},
    BoundedSelectInfo, BoundedSelectInfoElem,
};

use anyhow::{bail, Context, Result};
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::pattern::SparsityPattern;
use ndarray::Ix1;
use num::FromPrimitive;

use super::slice::BoundedSlice;

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
                    _ => bail!("Cannot convert to CsrMatrix<$from_type>"),
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
        match self {
            DynCsrMatrix::I8(csr) => csr.data_type(),
            DynCsrMatrix::I16(csr) => csr.data_type(),
            DynCsrMatrix::I32(csr) => csr.data_type(),
            DynCsrMatrix::I64(csr) => csr.data_type(),
            DynCsrMatrix::U8(csr) => csr.data_type(),
            DynCsrMatrix::U16(csr) => csr.data_type(),
            DynCsrMatrix::U32(csr) => csr.data_type(),
            DynCsrMatrix::U64(csr) => csr.data_type(),
            DynCsrMatrix::Usize(csr) => csr.data_type(),
            DynCsrMatrix::F32(csr) => csr.data_type(),
            DynCsrMatrix::F64(csr) => csr.data_type(),
            DynCsrMatrix::Bool(csr) => csr.data_type(),
            DynCsrMatrix::String(csr) => csr.data_type(),
        }
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
        let group = container.as_group()?;
        let shape: Vec<usize> = group.read_array_attr("shape")?.to_vec();
        let data = group.open_dataset("data")?.read_array::<_, Ix1>()?.into_raw_vec();
        let indptr: Vec<usize> = group.open_dataset("indptr")?.read_array::<_, Ix1>()?.into_raw_vec();
        let indices: Vec<usize> = group.open_dataset("indices")?.read_array::<_, Ix1>()?.into_raw_vec();
        Ok(CsrMatrix::try_from_csr_data(shape[0], shape[1], indptr, indices, data).unwrap())
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
                    _ => bail!("Cannot convert to CscMatrix<$from_type>"),
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
}

impl WriteArrayData for DynCscMatrix {}
impl ReadArrayData for DynCscMatrix {
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
        if let DataType::CscMatrix(ty) = container.encoding_type()? {
            match ty {
                ScalarType::I8 => CscMatrix::<i8>::read_select(container, info)
                    .map(Into::into),
                ScalarType::I16 => CscMatrix::<i16>::read_select(container, info)
                    .map(Into::into),
                ScalarType::I32 => CscMatrix::<i32>::read_select(container, info)
                    .map(Into::into),
                ScalarType::I64 => CscMatrix::<i64>::read_select(container, info)
                    .map(Into::into),
                ScalarType::U8 => CscMatrix::<u8>::read_select(container, info)
                    .map(Into::into),
                ScalarType::U16 => CscMatrix::<u16>::read_select(container, info)
                    .map(Into::into),
                ScalarType::U32 => CscMatrix::<u32>::read_select(container, info)
                    .map(Into::into),
                ScalarType::U64 => CscMatrix::<u64>::read_select(container, info) 
                    .map(Into::into),
                ScalarType::Usize => CscMatrix::<usize>::read_select(container, info)
                    .map(Into::into),
                ScalarType::F32 => CscMatrix::<f32>::read_select(container, info)
                    .map(Into::into),
                ScalarType::F64 => CscMatrix::<f64>::read_select(container, info)
                    .map(Into::into),
                ScalarType::Bool => CscMatrix::<bool>::read_select(container, info)
                    .map(Into::into),
                ScalarType::String => CscMatrix::<String>::read_select(container, info)
                    .map(Into::into),
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

    /// TODO!
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
        let (col_offsets, row_indices, data) = self.csc_data();
        let (new_col_offsets, new_row_indices, new_data) = if row_idx.is_full(info.in_shape()[1]) {
            match col_idx {
                &BoundedSelectInfoElem::Slice(BoundedSlice { step, start, end }) => {
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
                BoundedSelectInfoElem::Index(idx) => {
                    cs_major_index(idx.iter().copied(), col_offsets, row_indices, data)
                }
            }
        } else {
            // row_idx not full
            match col_idx {
                &BoundedSelectInfoElem::Slice(BoundedSlice { start: col_start,end: col_end, step: col_step }) => {
                    if col_step < 0 {
                        // col_idx is major, row_idx is minor
                        match row_idx {
                            &BoundedSelectInfoElem::Slice(row) => {
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
                            BoundedSelectInfoElem::Index(idx) => cs_major_minor_index(
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
                            &BoundedSelectInfoElem::Slice(row) => {
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
                            BoundedSelectInfoElem::Index(idx) => cs_major_minor_index(
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
                BoundedSelectInfoElem::Index(i) => match row_idx {
                    &BoundedSelectInfoElem::Slice(row) => {
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
                    BoundedSelectInfoElem::Index(j) => cs_major_minor_index(
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
}


impl<T: BackendData> WriteData for CscMatrix<T> {
    fn data_type(&self) -> DataType {
        DataType::CscMatrix(T::DTYPE)
    }
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let group = location.create_group(name)?;
        let shape = self.shape();

        group.write_str_attr("encoding-type", "csc_matrix")?;
        group.write_str_attr("encoding-version", "0.1.0")?;
        group.write_array_attr("shape", shape.as_ref())?;

        group.create_array_data("data", &self.values(), Default::default())?;

        let num_cols = shape[1];
        // Use i32 or i64 as indices type in order to be compatible with scipy
        if TryInto::<i32>::try_into(num_cols.saturating_sub(1)).is_ok() {
            let try_convert_indptr: Option<Vec<i32>> = self
                .col_offsets()
                .iter()
                .map(|x| (*x).try_into().ok())
                .collect();
            if let Some(indptr_i32) = try_convert_indptr {
                group.create_array_data("indptr", &indptr_i32, Default::default())?;
                group.create_array_data(
                    "indices",
                    self.row_indices()
                        .iter()
                        .map(|x| (*x) as i32)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    Default::default(),
                )?;
            } else {
                group.create_array_data(
                    "indptr",
                    self.col_offsets()
                        .iter()
                        .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                        .collect::<Vec<_>>()
                        .as_slice(),
                    Default::default(),
                )?;
                group.create_array_data(
                    "indices",
                    self.row_indices()
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
                self.col_offsets()
                    .iter()
                    .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
                Default::default(),
            )?;
            group.create_array_data(
                "indices",
                self.row_indices()
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

impl<T: BackendData> ReadData for CscMatrix<T> {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let group = container.as_group()?;
        let shape: Vec<usize> = group.read_array_attr("shape")?.to_vec();
        let data = group.open_dataset("data")?.read_array::<_, Ix1>()?.into_raw_vec();
        let indptr: Vec<usize> = group.open_dataset("indptr")?.read_array::<_, Ix1>()?.into_raw_vec();
        let indices: Vec<usize> = group.open_dataset("indices")?.read_array::<_, Ix1>()?.into_raw_vec();
        Ok(CscMatrix::try_from_csc_data(shape[0], shape[1], indptr, indices, data).unwrap())
    }
}

impl<T: BackendData> ReadArrayData for CscMatrix<T> {
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
            CscMatrix::try_from_csc_data(
                Self::get_shape(container)?[0],
                indptr.len() - 1,
                indptr,
                indices,
                data,
            ).unwrap().select_axis(1, info[1].as_ref())
        } else {
            Self::read(container)?.select(info)
        };
        Ok(data)
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
        values.into_iter().map(|x| x.try_into()).collect::<Result<_, _>>()?,
    ).unwrap();
    Ok(out)
}

fn from_i64_csc<U: FromPrimitive>(csc: CscMatrix<i64>) -> Result<CscMatrix<U>>
{
    let (pattern, values) = csc.into_pattern_and_values();
    let out = CscMatrix::try_from_pattern_and_values(
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
