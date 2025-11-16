use crate::backend::{Backend, BackendData, DatasetOp, GroupOp, WriteConfig};
use crate::data::{SelectInfoElem, Shape};
use crate::ArrayData;

use anyhow::{anyhow, Result};
use itertools::Itertools;
use nalgebra_sparse::{
    pattern::{SparsityPattern, SparsityPatternFormatError},
    CsrMatrix,
};
use ndarray::{Array2, ArrayView, RemoveAxis};
use smallvec::SmallVec;

use super::CsrNonCanonical;

pub struct ExtendableDataset<B: Backend, T> {
    dataset: B::Dataset,
    capacity: Shape,
    size: Shape,
    elem_type: std::marker::PhantomData<T>,
}

impl<B: Backend, T: BackendData> ExtendableDataset<B, T> {
    pub fn with_capacity<G>(group: &G, name: &str, capacity: Shape) -> Result<Self>
    where
        G: GroupOp<B>,
    {
        let block_size = vec![1000; capacity.ndim()].into();
        let dataset = group.new_empty_dataset::<T>(
            name,
            &capacity,
            WriteConfig {
                block_size: Some(block_size),
                ..Default::default()
            },
        )?;
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

    pub fn extend<'a, D: RemoveAxis>(
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
                .map(|(i, (x, y))| {
                    if i == axis {
                        let s = *x + *y;
                        (s, (*x..s).into())
                    } else if x == y || *x == 0 {
                        (*y, (0..*y).into())
                    } else {
                        panic!("Cannot concatenate arrays of different shapes");
                    }
                })
                .unzip();
            let new_size = new_size.into();
            self.check_or_grow(&new_size, 10000)?;
            self.dataset
                .write_array_slice(data.into(), slice.as_ref())?;
            self.size = new_size;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<B::Dataset> {
        self.dataset.reshape(&self.size)?;
        Ok(self.dataset)
    }
}

/// select rows of csr_matrix, or columns of csc_matrix
/// - major_indices: row_indices/col_indices of csr/csc matrix
/// - offset: indptr
pub(crate) fn cs_major_index<I, T>(
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
pub(crate) fn cs_major_slice<'a, T>(
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
/// - len_minor: number of columns/rows of in the csr_matrix/csc_matrix
/// - offset: offsets (indptr)
/// - indices: minor indices
/// - data: values in the matrix
pub(crate) fn cs_major_minor_index<I1, I2, T>(
    major_idx: I1,
    minor_idx: I2,
    len_minor: usize,
    offsets: &[usize],
    indices: &[usize],
    data: &[T],
) -> (Vec<usize>, Vec<usize>, Vec<T>)
where
    I1: Iterator<Item = usize> + Clone,
    I2: Iterator<Item = usize> + Clone,
    T: Clone,
{
    // Compute the occurrence of each minor index, as the same index can occur multiple times
    let mut minor_idx_count = vec![0; len_minor];
    minor_idx.clone().for_each(|j| minor_idx_count[j] += 1);

    // Compute new offsets (this is the row/column pointer array for the new matrix)
    let mut new_nnz = 0;
    let new_offsets = std::iter::once(0)
        .chain(major_idx.clone().map(|i| {
            (offsets[i]..offsets[i + 1]).for_each(|jj| new_nnz += minor_idx_count[indices[jj]]);
            new_nnz
        }))
        .collect();

    // Get the permutation that sorts the minor indices.
    // Position in col_order corresponds to the sorted minor index.
    // The values in col_order can be used to sort the original minor indices.
    let col_order: Vec<usize> = minor_idx
        .enumerate()
        .sorted_by_key(|(_, k)| *k)
        .map(|(j, _)| j)
        .collect();

    // Cumsum in-place to calculate the new index of each original index, assuming
    // that the minor indices are already sorted.
    // From the resultant vector: the index of each minor index j can be query by v[j-1].
    // Note that v[j] - v[j-1] == 0 if j is not present in the selection.
    (1..len_minor).for_each(|j| minor_idx_count[j] += minor_idx_count[j - 1]);

    // populates indices/data entries for selected columns.
    let mut new_indices = vec![0; new_nnz];
    let mut new_values: Vec<T> = Vec::with_capacity(new_nnz);
    let mut n = 0;

    // iterate over the row indices
    major_idx.for_each(|i| {
        let new_start = n;
        // iterate over the columns indices of the current row from the original matrix
        (offsets[i]..offsets[i + 1]).for_each(|jj| {
            let j = indices[jj]; // column index
            let v = &data[jj]; // value

            // we need to compute the new indices for the current row
            let idx_offset = minor_idx_count[j];
            let prev_offset = if j == 0 { 0 } else { minor_idx_count[j - 1] };
            (prev_offset..idx_offset).for_each(|k| {
                // Note we use col_order[k] to get the permutation of the k-th sorted index.
                // We later use sort this permutation to get the correct order of minor indices.
                new_indices[n] = col_order[k];
                new_values.push(v.clone());
                n += 1;
            });
        });

        // Now we need to actually sort the indices and values of the current row
        let mut permutation = permutation::sort(&new_indices[new_start..n]);
        permutation.apply_slice_in_place(&mut new_indices[new_start..n]);
        permutation.apply_slice_in_place(&mut new_values[new_start..n]);
    });

    (new_offsets, new_indices, new_values)
}

/// Like `cs_major_minor_index`, but allows for the addition of empty columns/rows.
/// Empty columns/rows are added by setting the corresponding index to `None`.
pub(crate) fn cs_major_minor_index2<T: Clone>(
    major_idx: &[Option<usize>],
    minor_idx: &[Option<usize>],
    len_minor: usize,
    offsets: &[usize],
    indices: &[usize],
    data: &[T],
) -> (Vec<usize>, Vec<usize>, Vec<T>)
{
    let num_added_minors = minor_idx.iter().filter(|&j| j.is_none()).count();
    let len_minor = len_minor + num_added_minors;

    // Compute the occurrence of each minor index, as the same index can occur multiple times
    let mut minor_idx_count = vec![0; len_minor];
    minor_idx_count[(len_minor-num_added_minors) .. len_minor].fill(1);
    minor_idx.iter().for_each(|j| {
        if let Some(j) = j {
            minor_idx_count[*j] += 1;
        }
    });

    // Compute new offsets (this is the row/column pointer array for the new matrix)
    let mut new_nnz = 0;
    let new_offsets = std::iter::once(0)
        .chain(major_idx.iter().map(|i| {
            if let Some(i) = i {
                (offsets[*i]..offsets[i + 1]).for_each(|jj| new_nnz += minor_idx_count[indices[jj]]);
            } 
            new_nnz
        }))
        .collect();

    // Get the permutation that sorts the minor indices.
    // col_order[k] gets the unsorted minor index of the k-th sorted minor index
    let mut t = len_minor - num_added_minors;
    let col_order: Vec<usize> = minor_idx
        .iter()
        .map(|i| {
            if let Some(i) = i {
                *i
            } else {
                let i = t;
                t += 1;
                i
            }
        })
        .enumerate()
        .sorted_by_key(|(_, k)| *k)
        .map(|(j, _)| j)
        .collect();

    // Cumsum in-place to calculate the new index of each original index, assuming
    // that the minor indices are already sorted.
    // From the resultant vector: the index of each minor index j can be query by v[j-1].
    // Note that v[j] - v[j-1] == 0 if j is not present in the selection.
    (1..len_minor).for_each(|j| minor_idx_count[j] += minor_idx_count[j - 1]);

    // populates indices/data entries for selected columns.
    let mut new_indices = vec![0; new_nnz];
    let mut new_values: Vec<T> = Vec::with_capacity(new_nnz);
    let mut n = 0;

    // iterate over the row indices
    major_idx.iter().flatten().for_each(|&i| {
        let new_start = n;
        // iterate over the columns indices of the current row from the original matrix
        (offsets[i]..offsets[i + 1]).for_each(|jj| {
            let j = indices[jj]; // column index
            let v = &data[jj]; // value

            // we need to compute the new indices for the current row
            let idx_offset = minor_idx_count[j];
            let prev_offset = if j == 0 { 0 } else { minor_idx_count[j - 1] };
            (prev_offset..idx_offset).for_each(|k| {
                // Note we use col_order[k] to get the original index of the k-th sorted index
                new_indices[n] = col_order[k];
                new_values.push(v.clone());
                n += 1;
            });
        });

        // Now we need to actually sort the indices and values of the current row
        let mut permutation = permutation::sort(&new_indices[new_start..n]);
        permutation.apply_slice_in_place(&mut new_indices[new_start..n]);
        permutation.apply_slice_in_place(&mut new_values[new_start..n]);
    });

    (new_offsets, new_indices, new_values)
}

/// Converts matrix data given in triplet format to unsorted CSR/CSC, retaining any duplicated
/// indices.
///
/// Here `major/minor` is `row/col` for CSR and `col/row` for CSC.
pub(crate) fn coo_to_unsorted_cs<T: Clone>(
    major_offsets: &mut [usize],
    cs_minor_idx: &mut [usize],
    cs_values: &mut [T],
    major_dim: usize,
    major_indices: &[usize],
    minor_indices: &[usize],
    coo_values: &[T],
) {
    assert_eq!(major_offsets.len(), major_dim + 1);
    assert_eq!(cs_minor_idx.len(), cs_values.len());
    assert_eq!(cs_values.len(), major_indices.len());
    assert_eq!(major_indices.len(), minor_indices.len());
    assert_eq!(minor_indices.len(), coo_values.len());

    // Count the number of occurrences of each row
    for major_idx in major_indices {
        major_offsets[*major_idx] += 1;
    }

    convert_counts_to_offsets(major_offsets);

    {
        // TODO: Instead of allocating a whole new vector storing the current counts,
        // I think it's possible to be a bit more clever by storing each count
        // in the last of the column indices for each row
        let mut current_counts = vec![0usize; major_dim + 1];
        let triplet_iter = major_indices.iter().zip(minor_indices).zip(coo_values);
        for ((i, j), value) in triplet_iter {
            let current_offset = major_offsets[*i] + current_counts[*i];
            cs_minor_idx[current_offset] = *j;
            cs_values[current_offset] = value.clone();
            current_counts[*i] += 1;
        }
    }
}

fn convert_counts_to_offsets(counts: &mut [usize]) {
    // Convert the counts to an offset
    let mut offset = 0;
    for i_offset in counts.iter_mut() {
        let count = *i_offset;
        *i_offset = offset;
        offset += count;
    }
}

/// Sort the indices of the given lane.
///
/// The indices and values in `minor_idx` and `values` are sorted according to the
/// minor indices and stored in `minor_idx_result` and `values_result` respectively.
///
/// All input slices are expected to be of the same length. The contents of mutable slices
/// can be arbitrary, as they are anyway overwritten.
pub(crate) fn sort_lane<T: Clone>(
    minor_idx_result: &mut [usize],
    values_result: &mut [T],
    minor_idx: &[usize],
    values: &[T],
    workspace: &mut [usize],
) {
    assert_eq!(minor_idx_result.len(), values_result.len());
    assert_eq!(values_result.len(), minor_idx.len());
    assert_eq!(minor_idx.len(), values.len());
    assert_eq!(values.len(), workspace.len());

    let permutation = workspace;
    compute_sort_permutation(permutation, minor_idx);

    apply_permutation(minor_idx_result, minor_idx, permutation);
    apply_permutation(values_result, values, permutation);
}

/// Helper functions for sparse matrix computations

/// permutes entries of in_slice according to permutation slice and puts them to out_slice
#[inline]
pub(crate) fn apply_permutation<T: Clone>(
    out_slice: &mut [T],
    in_slice: &[T],
    permutation: &[usize],
) {
    assert_eq!(out_slice.len(), in_slice.len());
    assert_eq!(out_slice.len(), permutation.len());
    for (out_element, old_pos) in out_slice.iter_mut().zip(permutation) {
        *out_element = in_slice[*old_pos].clone();
    }
}

/// computes permutation by using provided indices as keys
#[inline]
pub(crate) fn compute_sort_permutation(permutation: &mut [usize], indices: &[usize]) {
    assert_eq!(permutation.len(), indices.len());
    // Set permutation to identity
    for (i, p) in permutation.iter_mut().enumerate() {
        *p = i;
    }

    // Compute permutation needed to bring minor indices into sorted order
    // Note: Using sort_unstable here avoids internal allocations, which is crucial since
    // each lane might have a small number of elements
    permutation.sort_unstable_by_key(|idx| indices[*idx]);
}

pub fn from_csr_data<T>(
    nrows: usize,
    ncols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<T>,
) -> anyhow::Result<ArrayData>
where
    CsrMatrix<T>: Into<ArrayData>,
    CsrNonCanonical<T>: Into<ArrayData>,
{
    match check_format(nrows, ncols, &indptr, &indices) {
        Ok(_) => {
            let pattern = unsafe {
                SparsityPattern::from_offset_and_indices_unchecked(nrows, ncols, indptr, indices)
            };
            let csr = CsrMatrix::try_from_pattern_and_values(pattern, data).unwrap();
            Ok(csr.into())
        }
        Err(e) => match e {
            SparsityPatternFormatError::DuplicateEntry => {
                let csr = CsrNonCanonical::from_csr_data(nrows, ncols, indptr, indices, data);
                Ok(csr.into())
            }
            _ => Err(anyhow!("cannot read csr matrix: {}", e)),
        },
    }
}

pub(crate) fn check_format(
    nrows: usize,
    ncols: usize,
    indptr: &[usize],
    indices: &[usize],
) -> std::result::Result<(), SparsityPatternFormatError> {
    use SparsityPatternFormatError::*;

    if indptr.len() != nrows + 1 {
        return Err(InvalidOffsetArrayLength);
    }

    // Check that the first and last offsets conform to the specification
    {
        let first_offset_ok = *indptr.first().unwrap() == 0;
        let last_offset_ok = *indptr.last().unwrap() == indices.len();
        if !first_offset_ok || !last_offset_ok {
            return Err(InvalidOffsetFirstLast);
        }
    }

    // Test that each lane has strictly monotonically increasing minor indices, i.e.
    // minor indices within a lane are sorted, unique. In addition, each minor index
    // must be in bounds with respect to the minor dimension.
    let mut has_duplicate_entries = false;
    {
        for lane_idx in 0..nrows {
            let range_start = indptr[lane_idx];
            let range_end = indptr[lane_idx + 1];

            // Test that major offsets are monotonically increasing
            if range_start > range_end {
                return Err(NonmonotonicOffsets);
            }

            let indices = &indices[range_start..range_end];

            // We test for in-bounds, uniqueness and monotonicity at the same time
            // to ensure that we only visit each minor index once
            let mut iter = indices.iter();
            let mut prev = None;

            while let Some(next) = iter.next().copied() {
                if next >= ncols {
                    return Err(MinorIndexOutOfBounds);
                }

                if let Some(prev) = prev {
                    if prev > next {
                        return Err(NonmonotonicMinorIndices);
                    } else if prev == next {
                        has_duplicate_entries = true;
                    }
                }
                prev = Some(next);
            }
        }
    }

    if has_duplicate_entries {
        Err(DuplicateEntry)
    } else {
        Ok(())
    }
}

pub fn to_csr_data<I, In, T>(
    iter: I,
    num_cols: usize,
) -> (usize, usize, Vec<usize>, Vec<usize>, Vec<T>)
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

    (num_rows, num_cols, indptr, indices, data)
}

pub(crate) fn array_major_minor_index<T: Clone>(
    major_idx: &[Option<usize>],
    minor_idx: &[Option<usize>],
    data: &Array2<T>,
    fill_value: &T,
) -> Array2<T>
{
    Array2::from_shape_fn(
        (major_idx.len(), minor_idx.len()),
        |(i, j)| {
            if let (Some(i), Some(j)) = (major_idx[i], minor_idx[j]) {
                data.get((i, j)).unwrap().clone()
            } else {
                fill_value.clone()
            }
        },
    )
}

pub(crate) fn array_major_minor_index_default<T: Default + Clone>(
    major_idx: &[Option<usize>],
    minor_idx: &[Option<usize>],
    data: &Array2<T>,
) -> Array2<T>
{
    array_major_minor_index(major_idx, minor_idx, data, &T::default())
}