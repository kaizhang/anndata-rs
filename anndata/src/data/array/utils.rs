#![allow(dead_code, unused)]

use crate::backend::{Backend, BackendData, DatasetOp, GroupOp, WriteConfig};
use crate::data::{ArrayData, DynArray, DynCsrMatrix, DynCscMatrix};
use crate::data::{SelectInfoElem, Shape};

use anyhow::{bail, Result};
use itertools::Itertools;
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::na::Scalar;
use nalgebra_sparse::pattern::SparsityPattern;
use ndarray::{Array, Axis, RemoveAxis};
use ndarray::{ArrayView, Dimension};
use smallvec::SmallVec;

pub struct ExtendableDataset<B: Backend, T> {
    dataset: B::Dataset,
    capacity: Shape,
    size: Shape,
    elem_type: std::marker::PhantomData<T>,
}

impl<B: Backend, T: BackendData> ExtendableDataset<B, T> {
    pub fn with_capacity<G>(group: &G, name: &str, capacity: Shape) -> Result<Self>
    where
        G: GroupOp<Backend = B>,
    {
        let block_size = vec![1000; capacity.ndim()].into();
        let dataset = group.new_dataset::<T>(name, &capacity, WriteConfig { block_size: Some(block_size), ..Default::default() })?;
        Ok(Self {
            dataset,
            size: std::iter::repeat(0).take(capacity.ndim()).collect(),
            capacity,
            elem_type: std::marker::PhantomData,
        })
    }

    fn reserve(&mut self, additional: &Shape) -> Result<()> {
        self.capacity
            .as_mut()
            .iter_mut()
            .zip(additional.as_ref())
            .for_each(|(x, add)| *x += *add);
        self.dataset.reshape(&self.capacity)
    }

    fn check_or_grow(&mut self, size: &Shape, default: usize) -> Result<()> {
        let additional: Shape = self
            .capacity
            .as_ref()
            .iter()
            .zip(size.as_ref())
            .map(|(cap, size)| {
                if *cap < *size {
                    default.max(*size - *cap)
                } else {
                    0
                }
            })
            .collect();

        if additional.as_ref().iter().any(|x| *x != 0) {
            self.reserve(&additional)?;
        }
        Ok(())
    }

    pub fn extend<'a, D: Dimension>(
        &mut self,
        axis: usize,
        data: ArrayView<'a, T, D>,
    ) -> Result<()> {
        if !data.is_empty() {
            let (new_size, slice): (Vec<usize>, SmallVec<[SelectInfoElem; 3]>) = self
                .size
                .as_ref()
                .iter()
                .zip(data.shape())
                .enumerate()
                .map(|(i, (x, y))| if i == axis {
                    let s = *x + *y;
                    (s, (*x..s).into())
                } else if x == y || *x == 0 {
                    (*y, (0..*y).into())
                } else {
                    panic!("Cannot concatenate arrays of different shapes");
                }).unzip();
            let new_size = new_size.into();
            self.check_or_grow(&new_size, 10000)?;
            self.dataset.write_array_slice(data, slice.as_ref())?;
            self.size = new_size;
        }
        Ok(())
    }

    pub fn finish(self) -> Result<B::Dataset> {
        self.dataset.reshape(&self.size)?;
        Ok(self.dataset)
    }
}

macro_rules! impl_vstack_array {
    ($this:expr, $this_ty:ident, $other:expr, $other_ty:ident, $func:expr) => {{
        match ($this, $other) {
            ($this_ty::Bool(this), $other_ty::Bool(other)) => $func(this, other).into(),
            ($this_ty::U8(this), $other_ty::U8(other)) => $func(this, other).into(),
            ($this_ty::U16(this), $other_ty::U16(other)) => $func(this, other).into(),
            ($this_ty::U32(this), $other_ty::U32(other)) => $func(this, other).into(),
            ($this_ty::U64(this), $other_ty::U64(other)) => $func(this, other).into(),
            ($this_ty::Usize(this), $other_ty::Usize(other)) => $func(this, other).into(),
            ($this_ty::I8(this), $other_ty::I8(other)) => $func(this, other).into(),
            ($this_ty::I16(this), $other_ty::I16(other)) => $func(this, other).into(),
            ($this_ty::I32(this), $other_ty::I32(other)) => $func(this, other).into(),
            ($this_ty::I64(this), $other_ty::I64(other)) => $func(this, other).into(),
            ($this_ty::F32(this), $other_ty::F32(other)) => $func(this, other).into(),
            ($this_ty::F64(this), $other_ty::F64(other)) => $func(this, other).into(),
            ($this_ty::String(this), $other_ty::String(other)) => $func(this, other).into(),
            _ => bail!("Cannot concatenate arrays of different types"),
        }
    }};
}

/// Vertically concatenate an iterator of arrays into a single array.
pub fn concat_array_data<I>(arrays: I) -> Result<ArrayData>
where
    I: IntoIterator<Item = ArrayData>,
{
    Ok(arrays.into_iter().try_reduce(vstack_array_data)?.unwrap())
}

fn vstack_array_data(this: ArrayData, other: ArrayData) -> Result<ArrayData> {
    let array = match (this, other) {
        (ArrayData::Array(this), ArrayData::Array(other)) => {
            impl_vstack_array!(this, DynArray, other, DynArray, vstack_arr)
        }
        (ArrayData::CsrMatrix(this), ArrayData::CsrMatrix(other)) => {
            impl_vstack_array!(this, DynCsrMatrix, other, DynCsrMatrix, vstack_csr)
        }
        (ArrayData::CscMatrix(this), ArrayData::CscMatrix(other)) => {
            impl_vstack_array!(this, DynCscMatrix, other, DynCscMatrix, vstack_csc)
        }
        _ => bail!("Cannot concatenate arrays of different types"),
    };
    Ok(array)
}

pub fn vstack_arr<T: Clone, D: RemoveAxis>(mut this: Array<T, D>, other: Array<T, D>) -> Array<T, D> {
    this.append(Axis(0), other.view()).unwrap();
    this
}

/// Row concatenation of sparse row matrices.
/// Stack csr in sequence vertically (row wise).
pub fn vstack_csr<T: Clone>(this: CsrMatrix<T>, other: CsrMatrix<T>) -> CsrMatrix<T> {
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

/// Column concatenation of sparse row matrices.
pub fn hstack_csr<T: Clone + Scalar>(this: CsrMatrix<T>, other: CsrMatrix<T>) -> CsrMatrix<T> {
    vstack_csr(this.transpose(), other.transpose()).transpose()
}

/// Column concatenation of sparse column matrices.
pub fn hstack_csc<T: Clone>(this: CscMatrix<T>, other: CscMatrix<T>) -> CscMatrix<T> {

    let num_cols = this.ncols()  + other.ncols();
    let num_rows = this.nrows();
    let nnz = this.nnz();
    let (mut indptr, mut indices, mut data) = this.disassemble();
    let (indptr2, indices2, data2) = other.csc_data();
    indices.extend_from_slice(indices2);
    data.extend_from_slice(data2);
    indptr2.iter().skip(1).for_each(|&i| indptr.push(i + nnz));

    let pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(num_cols, num_rows, indptr, indices)
    };
    CscMatrix::try_from_pattern_and_values(pattern, data).unwrap()
}

/// Row concatenation of sparse column matrices.
pub fn vstack_csc<T: Clone + Scalar>(this: CscMatrix<T>, other: CscMatrix<T>) -> CscMatrix<T> {
    hstack_csc(this.transpose(), other.transpose()).transpose()
}


pub fn from_csr_rows<I, In, T>(iter: I, num_cols: usize) -> CsrMatrix<T>
where
    I: IntoIterator<IntoIter = In>,
    In: ExactSizeIterator<Item = Vec<(usize, T)>>,
{
    let rows = iter.into_iter();
    let num_rows = rows.len();
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(num_rows + 1);
    let mut nnz = 0;
    for row in rows {
        indptr.push(nnz);
        for (col, val) in row {
            data.push(val);
            indices.push(col);
            nnz += 1;
        }
    }
    indptr.push(nnz);
    CsrMatrix::try_from_csr_data(num_rows, num_cols, indptr, indices, data).unwrap()
}


pub fn from_csc_cols<I, In, T>(iter: I, num_rows: usize) -> CscMatrix<T>
where
    I: IntoIterator<IntoIter = In>,
    In: ExactSizeIterator<Item = Vec<(usize, T)>>,
{
    let cols = iter.into_iter();
    let num_cols = cols.len();
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(num_cols + 1);
    let mut nnz = 0;
    for col in cols {
        indptr.push(nnz);
        for (row, val) in col {
            data.push(val);
            indices.push(row);
            nnz += 1;
        }
    }
    indptr.push(nnz);
    CscMatrix::try_from_csc_data(num_rows, num_cols, indptr, indices, data).unwrap()
}



/// select rows of csr_matrix, or columns of csc_matrix
/// - major_indices: row_indices/col_indices of csr/csc matrix
/// - offset: indptr
pub fn cs_major_index<I, T>(
    major_indices: I,
    offsets: &[usize],
    indices: &[usize],
    data: &[T],
) -> (Vec<usize>, Vec<usize>, Vec<T>)
where
    I: Iterator<Item = usize>,
    T: Clone,
{
    let mut new_offsets = vec![0];
    let mut new_indices = Vec::new();
    let mut new_data = Vec::new();
    let mut nnz = 0;
    major_indices.for_each(|major| {
        let start = offsets[major];
        let end = offsets[major + 1];
        nnz += end - start;
        new_offsets.push(nnz);
        new_indices.extend_from_slice(&indices[start..end]);
        new_data.extend_from_slice(&data[start..end]);
    });
    (new_offsets, new_indices, new_data)
}


/// slicing rows of csr_matrix, or columns of csc_matrix
/// - start, end: slice buound of row_indices/col_indices of csr/csc matrix
/// - offset: indptr
pub fn cs_major_slice<'a, T>(
    start: usize,
    end: usize,
    offsets: &'a [usize],
    indices: &'a [usize],
    data: &'a [T],
) -> (Vec<usize>, &'a [usize], &'a [T]) {
    let i = offsets[start];
    let j = offsets[end];
    let new_offsets = offsets[start..end + 1].iter().map(|&x| x - i).collect();
    (new_offsets, &indices[i..j], &data[i..j])
}


/// row and column indexing of csr_matrix
/// - major_idx: row_idx of csr_matrix, col_idx of csc_matrix
/// - minor_idx: col_idx of csr_matrix, row_idx of csc_matrix
/// - len_minor: ncols of csr_matrix
/// - offset: indptr
/// - indices: 
/// - data:
pub fn cs_major_minor_index<I1, I2, T>(
    major_idx: I1,
    minor_idx: I2,
    len_minor: usize,
    offsets: &[usize],
    indices: &[usize],
    data: &[T],
) -> (Vec<usize>, Vec<usize>, Vec<T>)
where
    I1: ExactSizeIterator<Item = usize> + Clone,
    I2: Iterator<Item = usize> + Clone,
    T: Clone,
{
    // Compute the occurrence of each minor index
    let mut minor_idx_count = vec![0; len_minor];
    minor_idx.clone().for_each(|j| minor_idx_count[j] += 1);

    // Compute new indptr
    let mut new_nnz = 0;
    let new_offsets = std::iter::once(0)
        .chain(major_idx.clone().map(|i| {
            (offsets[i]..offsets[i + 1]).for_each(|jj| new_nnz += minor_idx_count[indices[jj]]);
            new_nnz
        }))
        .collect();

    // cumsum in-place -> calc new offset
    (1..len_minor).for_each(|j| minor_idx_count[j] += minor_idx_count[j - 1]);

    let col_order: Vec<usize> = minor_idx
        .enumerate()
        .sorted_by_key(|(_, k)| *k)
        .map(|(j, _)| j)
        .collect();

    // populates indices/data entries for selected columns.
    let mut new_indices = vec![0; new_nnz];
    let mut new_values: Vec<T> = Vec::with_capacity(new_nnz);
    let mut n = 0;

    // iter major then minor
    major_idx.for_each(|i| {
        let new_start = n;
        let start = offsets[i];
        let end = offsets[i + 1];
        (start..end).for_each(|jj| {
            let j = indices[jj];
            let v = &data[jj];
            let offset = minor_idx_count[j];
            let prev_offset = if j == 0 { 0 } else { minor_idx_count[j - 1] };
            (prev_offset..offset).for_each(|k| {
                new_indices[n] = col_order[k];
                new_values.push(v.clone());
                n += 1;
            });
        });
        let mut permutation = permutation::sort(&new_indices[new_start..n]);
        permutation.apply_slice_in_place(&mut new_indices[new_start..n]);
        permutation.apply_slice_in_place(&mut new_values[new_start..n]);
    });

    (new_offsets, new_indices, new_values)
}