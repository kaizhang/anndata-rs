use crate::backend::{Backend, BackendData, DatasetOp};
use crate::data::{ArrayData, DynArray, DynCsrMatrix};
use crate::data::{SelectInfoElem, Shape};

use anyhow::{bail, Result};
use itertools::Itertools;
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::pattern::SparsityPattern;
use ndarray::{Array, Axis, IxDyn};
use ndarray::{ArrayView, Dimension};
use smallvec::SmallVec;

pub struct ExtendableDataset<B: Backend> {
    dataset: B::Dataset,
    capacity: Shape,
    size: Shape,
}

impl<B: Backend> ExtendableDataset<B> {
    pub fn with_capacity(dataset: B::Dataset, capacity: Shape) -> Self {
        Self {
            dataset,
            size: std::iter::repeat(0).take(capacity.ndim()).collect(),
            capacity,
        }
    }

    pub fn reserve(&mut self, additional: &Shape) -> Result<()> {
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

    pub fn extend<'a, T: BackendData, D: Dimension>(
        &mut self,
        data: ArrayView<'a, T, D>,
    ) -> Result<()> {
        let new_size = self
            .size
            .as_ref()
            .iter()
            .zip(data.shape())
            .map(|(x, y)| *x + *y)
            .collect();
        self.check_or_grow(&new_size, 100000)?;
        let slice: SmallVec<[SelectInfoElem; 3]> = self
            .size
            .as_ref()
            .iter()
            .zip(new_size.as_ref())
            .map(|(x, y)| (*x..*y).into())
            .collect();
        self.dataset.write_array_slice(data, slice)?;
        self.size = new_size;
        Ok(())
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

pub(crate) fn concat_array_data<I>(arrays: I) -> Result<ArrayData>
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
        _ => bail!("Cannot concatenate arrays of different types"),
    };
    Ok(array)
}

fn vstack_arr<T: Clone>(mut this: Array<T, IxDyn>, other: Array<T, IxDyn>) -> Array<T, IxDyn> {
    this.append(Axis(0), other.view()).unwrap();
    this
}

/// Row concatenation of sparse row matrices.
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

    // cumsum in-place
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
