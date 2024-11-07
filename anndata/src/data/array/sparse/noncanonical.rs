use crate::backend::*;
use crate::data::{
    array::utils::{cs_major_index, cs_major_minor_index, cs_major_slice},
    data_traits::*,
    slice::{SelectInfoElem, Shape},
    SelectInfoBounds, SelectInfoElemBounds,
};

use anyhow::{bail, Result};
use nalgebra_sparse::pattern::SparsityPattern;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use ndarray::Ix1;

use super::super::slice::SliceBounds;
use super::DynCsrMatrix;

#[derive(Debug, Clone, PartialEq)]
pub enum DynCsrNonCanonical {
    I8(CsrNonCanonical<i8>),
    I16(CsrNonCanonical<i16>),
    I32(CsrNonCanonical<i32>),
    I64(CsrNonCanonical<i64>),
    U8(CsrNonCanonical<u8>),
    U16(CsrNonCanonical<u16>),
    U32(CsrNonCanonical<u32>),
    U64(CsrNonCanonical<u64>),
    F32(CsrNonCanonical<f32>),
    F64(CsrNonCanonical<f64>),
    Bool(CsrNonCanonical<bool>),
    String(CsrNonCanonical<String>),
}

impl DynCsrNonCanonical {
    pub fn canonicalize(self) -> Result<DynCsrMatrix, Self> {
        match self {
            DynCsrNonCanonical::I8(data) => data
                .canonicalize()
                .map(DynCsrMatrix::I8)
                .map_err(Into::into),
            DynCsrNonCanonical::I16(data) => data
                .canonicalize()
                .map(DynCsrMatrix::I16)
                .map_err(Into::into),
            DynCsrNonCanonical::I32(data) => data
                .canonicalize()
                .map(DynCsrMatrix::I32)
                .map_err(Into::into),
            DynCsrNonCanonical::I64(data) => data
                .canonicalize()
                .map(DynCsrMatrix::I64)
                .map_err(Into::into),
            DynCsrNonCanonical::U8(data) => data
                .canonicalize()
                .map(DynCsrMatrix::U8)
                .map_err(Into::into),
            DynCsrNonCanonical::U16(data) => data
                .canonicalize()
                .map(DynCsrMatrix::U16)
                .map_err(Into::into),
            DynCsrNonCanonical::U32(data) => data
                .canonicalize()
                .map(DynCsrMatrix::U32)
                .map_err(Into::into),
            DynCsrNonCanonical::U64(data) => data
                .canonicalize()
                .map(DynCsrMatrix::U64)
                .map_err(Into::into),
            DynCsrNonCanonical::F32(data) => data
                .canonicalize()
                .map(DynCsrMatrix::F32)
                .map_err(Into::into),
            DynCsrNonCanonical::F64(data) => data
                .canonicalize()
                .map(DynCsrMatrix::F64)
                .map_err(Into::into),
            DynCsrNonCanonical::Bool(data) => data
                .canonicalize()
                .map(DynCsrMatrix::Bool)
                .map_err(Into::into),
            DynCsrNonCanonical::String(data) => data
                .canonicalize()
                .map(DynCsrMatrix::String)
                .map_err(Into::into),
        }
    }
}

macro_rules! impl_noncanonicalcsr_traits {
    ($($from_type:ty, $to_type:ident),*) => {
        $(
            impl From<CsrNonCanonical<$from_type>> for DynCsrNonCanonical {
                fn from(data: CsrNonCanonical<$from_type>) -> Self {
                    DynCsrNonCanonical::$to_type(data)
                }
            }
            impl TryFrom<DynCsrNonCanonical> for CsrNonCanonical<$from_type> {
                type Error = anyhow::Error;
                fn try_from(data: DynCsrNonCanonical) -> Result<Self> {
                    match data {
                        DynCsrNonCanonical::$to_type(data) => Ok(data),
                        _ => bail!(
                            "Cannot convert {:?} to {} CsrNonCanonical",
                            data.data_type(),
                            stringify!($from_type)
                        ),
                    }
                }
            }
        )*
    };
}

impl_noncanonicalcsr_traits!(
    i8, I8, i16, I16, i32, I32, i64, I64, u8, U8, u16, U16, u32, U32, u64, U64, f32, F32, f64, F64,
    bool, Bool, String, String
);

impl From<DynCsrMatrix> for DynCsrNonCanonical {
    fn from(value: DynCsrMatrix) -> Self {
        macro_rules! fun {
            ($variant:ident, $data:expr) => {
                DynCsrNonCanonical::$variant($data.into())
            };
        }
        crate::macros::dyn_map!(value, DynCsrMatrix, fun)
    }
}

impl Writable for DynCsrNonCanonical {
    fn data_type(&self) -> DataType {
        crate::macros::dyn_map_fun!(self, DynCsrNonCanonical, data_type)
    }

    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        crate::macros::dyn_map_fun!(self, DynCsrNonCanonical, write, location, name)
    }
}

impl Readable for DynCsrNonCanonical {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container {
            DataContainer::Group(group) => {
                macro_rules! fun {
                    ($variant:ident) => {
                        CsrNonCanonical::read(container).map(DynCsrNonCanonical::$variant)
                    };
                }
                crate::macros::dyn_match!(group.open_dataset("data")?.dtype()?, ScalarType, fun)
            },
            _ => bail!("cannot read csr matrix from non-group container"),
        }
    }
}

impl HasShape for DynCsrNonCanonical {
    fn shape(&self) -> Shape {
        crate::macros::dyn_map_fun!(self, DynCsrNonCanonical, shape)
    }
}

impl Selectable for DynCsrNonCanonical {
    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        macro_rules! fun {
            ($variant:ident, $data:expr) => {
                $data.select(info).into()
            };
        }
        crate::macros::dyn_map!(self, DynCsrNonCanonical, fun)
    }
}

impl Stackable for DynCsrNonCanonical {
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            DynCsrNonCanonical::U8(_) => Ok(DynCsrNonCanonical::U8(CsrNonCanonical::<u8>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrNonCanonical::U16(_) => Ok(DynCsrNonCanonical::U16(
                CsrNonCanonical::<u16>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::U32(_) => Ok(DynCsrNonCanonical::U32(
                CsrNonCanonical::<u32>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::U64(_) => Ok(DynCsrNonCanonical::U64(
                CsrNonCanonical::<u64>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::I8(_) => Ok(DynCsrNonCanonical::I8(CsrNonCanonical::<i8>::vstack(
                iter.map(|x| x.try_into().unwrap()),
            )?)),
            DynCsrNonCanonical::I16(_) => Ok(DynCsrNonCanonical::I16(
                CsrNonCanonical::<i16>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::I32(_) => Ok(DynCsrNonCanonical::I32(
                CsrNonCanonical::<i32>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::I64(_) => Ok(DynCsrNonCanonical::I64(
                CsrNonCanonical::<i64>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::F32(_) => Ok(DynCsrNonCanonical::F32(
                CsrNonCanonical::<f32>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::F64(_) => Ok(DynCsrNonCanonical::F64(
                CsrNonCanonical::<f64>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::Bool(_) => Ok(DynCsrNonCanonical::Bool(
                CsrNonCanonical::<bool>::vstack(iter.map(|x| x.try_into().unwrap()))?,
            )),
            DynCsrNonCanonical::String(_) => {
                Ok(DynCsrNonCanonical::String(
                    CsrNonCanonical::<String>::vstack(iter.map(|x| x.try_into().unwrap()))?,
                ))
            }
        }
    }
}

impl WritableArray for DynCsrNonCanonical {}
impl ReadableArray for DynCsrNonCanonical {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into_iter()
            .collect())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        if let DataType::CsrMatrix(ty) = container.encoding_type()? {
            macro_rules! fun {
                ($variant:ident) => {
                    CsrNonCanonical::read_select(container, info).map(DynCsrNonCanonical::$variant)
                };
            }
            crate::macros::dyn_match!(ty, ScalarType, fun)
        } else {
            bail!("the container does not contain a csr matrix");
        }
    }
}

/// Compressed sparse row matrix with potentially duplicate column indices.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrNonCanonical<T> {
    offsets: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<T>,
    num_rows: usize,
    num_cols: usize,
}

impl<T> CsrNonCanonical<T> {
    pub fn nrows(&self) -> usize {
        self.num_rows
    }

    pub fn ncols(&self) -> usize {
        self.num_cols
    }

    pub fn row_offsets(&self) -> &[usize] {
        &self.offsets
    }

    pub fn col_indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }

    pub fn csr_data(&self) -> (&[usize], &[usize], &[T]) {
        (&self.offsets, &self.indices, &self.values)
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn disassemble(self) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        (self.offsets, self.indices, self.values)
    }

    pub fn from_csr_data(
        num_rows: usize,
        num_cols: usize,
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
        data: Vec<T>,
    ) -> Self {
        Self {
            offsets: row_offsets,
            indices: col_indices,
            values: data,
            num_rows,
            num_cols,
        }
    }

    pub fn canonicalize(self) -> Result<CsrMatrix<T>, Self> {
        let nrows = self.nrows();
        let ncols = self.ncols();
        if crate::data::utils::check_format(nrows, ncols, self.row_offsets(), self.col_indices())
            .is_ok()
        {
            let pattern = unsafe {
                SparsityPattern::from_offset_and_indices_unchecked(
                    nrows,
                    ncols,
                    self.offsets,
                    self.indices,
                )
            };
            Ok(CsrMatrix::try_from_pattern_and_values(pattern, self.values).unwrap())
        } else {
            Err(self)
        }
    }
}

impl<T> From<CsrMatrix<T>> for CsrNonCanonical<T> {
    fn from(csr: CsrMatrix<T>) -> Self {
        let num_rows = csr.nrows();
        let num_cols = csr.ncols();
        let (row_offsets, col_indices, data) = csr.disassemble();
        Self::from_csr_data(num_rows, num_cols, row_offsets, col_indices, data)
    }
}

impl<T: Clone + num::Zero> From<&CooMatrix<T>> for CsrNonCanonical<T> {
    fn from(coo: &CooMatrix<T>) -> Self {
        let major_dim = coo.nrows();
        let major_indices = coo.row_indices();
        let minor_indices = coo.col_indices();
        let values = coo.values();
        assert_eq!(major_indices.len(), minor_indices.len());
        assert_eq!(minor_indices.len(), values.len());
        let nnz = major_indices.len();

        let (unsorted_major_offsets, unsorted_minor_idx, unsorted_vals) = {
            let mut offsets = vec![0usize; major_dim + 1];
            let mut minor_idx = vec![0usize; nnz];
            let mut vals = vec![T::zero(); nnz];
            crate::data::utils::coo_to_unsorted_cs(
                &mut offsets,
                &mut minor_idx,
                &mut vals,
                major_dim,
                major_indices,
                minor_indices,
                values,
            );
            (offsets, minor_idx, vals)
        };

        // At this point, assembly is essentially complete. However, we must ensure
        // that minor indices are sorted within each lane and without duplicates.
        let mut sorted_major_offsets = Vec::new();
        let mut sorted_minor_idx = Vec::new();
        let mut sorted_vals = Vec::new();

        sorted_major_offsets.push(0);

        // We need some temporary storage when working with each lane. Since lanes often have a
        // very small number of non-zero entries, we try to amortize allocations across
        // lanes by reusing workspace vectors
        let mut idx_workspace = Vec::new();
        let mut perm_workspace = Vec::new();
        let mut values_workspace = Vec::new();

        for lane in 0..major_dim {
            let begin = unsorted_major_offsets[lane];
            let end = unsorted_major_offsets[lane + 1];
            let count = end - begin;
            let range = begin..end;

            // Ensure that workspaces can hold enough data
            perm_workspace.resize(count, 0);
            idx_workspace.resize(count, 0);
            values_workspace.resize(count, T::zero());
            crate::data::utils::sort_lane(
                &mut idx_workspace[..count],
                &mut values_workspace[..count],
                &unsorted_minor_idx[range.clone()],
                &unsorted_vals[range.clone()],
                &mut perm_workspace[..count],
            );

            let sorted_ja_current_len = sorted_minor_idx.len();

            for i in range {
                sorted_minor_idx.push(unsorted_minor_idx[i]);
                sorted_vals.push(unsorted_vals[i].clone());
            }

            let new_col_count = sorted_minor_idx.len() - sorted_ja_current_len;
            sorted_major_offsets.push(sorted_major_offsets.last().unwrap() + new_col_count);
        }

        Self::from_csr_data(
            coo.nrows(),
            coo.ncols(),
            sorted_major_offsets,
            sorted_minor_idx,
            sorted_vals,
        )
    }
}

impl<T: Clone> From<&CsrNonCanonical<T>> for CooMatrix<T> {
    fn from(csr: &CsrNonCanonical<T>) -> Self {
        let mut coo: CooMatrix<T> = CooMatrix::new(csr.nrows(), csr.ncols());
        for row in 0..csr.nrows() {
            let start = csr.row_offsets()[row];
            let end = csr.row_offsets()[row + 1];
            for i in start..end {
                coo.push(row, csr.col_indices()[i], csr.values()[i].clone());
            }
        }
        coo
    }
}

impl<T> HasShape for CsrNonCanonical<T> {
    fn shape(&self) -> Shape {
        vec![self.nrows(), self.ncols()].into()
    }
}

impl<T: Clone> Selectable for CsrNonCanonical<T> {
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
        let (row_offsets, col_indices, data) = self.csr_data();
        let (new_row_offsets, new_col_indices, new_data) = if col_idx.is_full(info.in_shape()[1]) {
            match row_idx {
                &SelectInfoElemBounds::Slice(SliceBounds { step, start, end }) => {
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
                SelectInfoElemBounds::Index(idx) => {
                    cs_major_index(idx.iter().copied(), row_offsets, col_indices, data)
                }
            }
        } else {
            match row_idx {
                &SelectInfoElemBounds::Slice(SliceBounds {
                    start: row_start,
                    end: row_end,
                    step: row_step,
                }) => {
                    if row_step < 0 {
                        match col_idx {
                            &SelectInfoElemBounds::Slice(col) => {
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
                            SelectInfoElemBounds::Index(idx) => cs_major_minor_index(
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
                            &SelectInfoElemBounds::Slice(col) => {
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
                            SelectInfoElemBounds::Index(idx) => cs_major_minor_index(
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
                SelectInfoElemBounds::Index(i) => match col_idx {
                    &SelectInfoElemBounds::Slice(col) => {
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
                    SelectInfoElemBounds::Index(j) => cs_major_minor_index(
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
        Self::from_csr_data(
            out_shape[0],
            out_shape[1],
            new_row_offsets,
            new_col_indices,
            new_data,
        )
    }
}

impl<T: Clone> Stackable for CsrNonCanonical<T> {
    fn vstack<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        fn vstack_csr<T: Clone>(
            this: CsrNonCanonical<T>,
            other: CsrNonCanonical<T>,
        ) -> CsrNonCanonical<T> {
            let num_cols = this.ncols();
            let num_rows = this.nrows() + other.nrows();
            let nnz = this.nnz();
            let (mut indptr, mut indices, mut data) = this.disassemble();
            let (indptr2, indices2, data2) = other.csr_data();
            indices.extend_from_slice(indices2);
            data.extend_from_slice(data2);
            indptr2.iter().skip(1).for_each(|&i| indptr.push(i + nnz));

            CsrNonCanonical::from_csr_data(num_rows, num_cols, indptr, indices, data)
        }

        Ok(iter.reduce(|acc, x| vstack_csr(acc, x)).unwrap())
    }
}

impl<T: BackendData> Writable for CsrNonCanonical<T> {
    fn data_type(&self) -> DataType {
        DataType::CsrMatrix(T::DTYPE)
    }
    fn write<B: Backend, G: GroupOp<B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let mut group = location.new_group(name)?;
        let shape = self.shape();

        group.new_attr("encoding-type", "csr_matrix")?;
        group.new_attr("encoding-version", "0.1.0")?;
        group.new_attr(
            "shape",
            shape
                .as_ref()
                .iter()
                .map(|x| *x as u64)
                .collect::<Vec<_>>(),
        )?;

        group.new_array_dataset("data", self.values().into(), Default::default())?;

        let num_cols = shape[1];
        // Use i32 or i64 as indices type in order to be compatible with scipy
        if TryInto::<i32>::try_into(num_cols.saturating_sub(1)).is_ok() {
            let try_convert_indptr: Option<Vec<i32>> = self
                .row_offsets()
                .iter()
                .map(|x| (*x).try_into().ok())
                .collect();
            if let Some(indptr_i32) = try_convert_indptr {
                group.new_array_dataset("indptr", indptr_i32.into(), Default::default())?;
                group.new_array_dataset(
                    "indices",
                    self.col_indices()
                        .iter()
                        .map(|x| (*x) as i32)
                        .collect::<Vec<_>>()
                        .into(),
                    Default::default(),
                )?;
            } else {
                group.new_array_dataset(
                    "indptr",
                    self.row_offsets()
                        .iter()
                        .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                        .collect::<Vec<_>>()
                        .into(),
                    Default::default(),
                )?;
                group.new_array_dataset(
                    "indices",
                    self.col_indices()
                        .iter()
                        .map(|x| (*x) as i64)
                        .collect::<Vec<_>>()
                        .into(),
                    Default::default(),
                )?;
            }
        } else if TryInto::<i64>::try_into(num_cols.saturating_sub(1)).is_ok() {
            group.new_array_dataset(
                "indptr",
                self.row_offsets()
                    .iter()
                    .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                    .collect::<Vec<_>>()
                    .into(),
                Default::default(),
            )?;
            group.new_array_dataset(
                "indices",
                self.col_indices()
                    .iter()
                    .map(|x| (*x) as i64)
                    .collect::<Vec<_>>()
                    .into(),
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

impl<T: BackendData> Readable for CsrNonCanonical<T> {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
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
        Ok(Self::from_csr_data(
            shape[0] as usize,
            shape[1] as usize,
            indptr,
            indices,
            data,
        ))
    }
}

impl<T: BackendData> ReadableArray for CsrNonCanonical<T> {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .get_attr::<Vec<usize>>("shape")?
            .into_iter()
            .collect())
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

        let data = if let SelectInfoElem::Slice(s) = info[0].as_ref() {
            let group = container.as_group()?;
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
            Self::from_csr_data(
                indptr.len() - 1,
                Self::get_shape(container)?[1],
                indptr,
                indices,
                data,
            )
            .select_axis(1, info[1].as_ref())
        } else {
            Self::read(container)?.select(info)
        };
        Ok(data)
    }
}

impl<T: BackendData> WritableArray for &CsrNonCanonical<T> {}
impl<T: BackendData> WritableArray for CsrNonCanonical<T> {}

#[cfg(test)]
mod csr_noncanonical_index_tests {
    use super::*;
    use crate::s;
    use nalgebra_sparse::CooMatrix;

    fn csr_eq<T: std::cmp::PartialEq + std::fmt::Debug + Clone>(
        a: &CsrNonCanonical<T>,
        b: &CooMatrix<T>,
    ) {
        assert_eq!(&CooMatrix::from(a), b);
    }

    #[test]
    fn test_csr_noncanonical() {
        let coo = CooMatrix::try_from_triplets(
            5,
            4,
            vec![0, 1, 1, 1, 2, 3, 4],
            vec![0, 0, 0, 2, 3, 1, 3],
            vec![1, 2, 3, 4, 5, 6, 7],
        )
        .unwrap();
        let csr = CsrNonCanonical::from(&coo);

        csr_eq(&csr, &coo);

        csr_eq(
            &csr.select(s![vec![0, 1], ..].as_ref()),
            &CooMatrix::try_from_triplets(
                2,
                4,
                vec![0, 1, 1, 1],
                vec![0, 0, 0, 2],
                vec![1, 2, 3, 4],
            )
            .unwrap(),
        );

        csr_eq(
            &csr.select(s![.., vec![0, 0, 1]].as_ref()),
            &CooMatrix::try_from_triplets(
                5,
                3,
                vec![0, 0, 1, 1, 1, 1, 3],
                vec![0, 1, 0, 0, 1, 1, 2],
                vec![1, 1, 2, 3, 2, 3, 6],
            )
            .unwrap(),
        );

        csr_eq(
            &csr.select(s![vec![0, 1, 1], ..].as_ref()),
            &CooMatrix::try_from_triplets(
                3,
                4,
                vec![0, 1, 1, 1, 2, 2, 2],
                vec![0, 0, 0, 2, 0, 0, 2],
                vec![1, 2, 3, 4, 2, 3, 4],
            )
            .unwrap(),
        );

        csr_eq(
            &csr.select(s![vec![0, 1, 1], vec![0, 1]].as_ref()),
            &CooMatrix::try_from_triplets(
                3,
                2,
                vec![0, 1, 1, 2, 2],
                vec![0, 0, 0, 0, 0],
                vec![1, 2, 3, 2, 3],
            )
            .unwrap(),
        );
    }
}
