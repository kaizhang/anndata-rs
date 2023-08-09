use crate::ArrayData;
use anyhow::anyhow;

use nalgebra_sparse::{CsrMatrix, pattern::{ SparsityPattern, SparsityPatternFormatError}};

use super::CsrNonCanonical;

/// Converts matrix data given in triplet format to unsorted CSR/CSC, retaining any duplicated
/// indices.
///
/// Here `major/minor` is `row/col` for CSR and `col/row` for CSC.
pub fn coo_to_unsorted_cs<T: Clone>(
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
pub fn sort_lane<T: Clone>(
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
pub fn apply_permutation<T: Clone>(out_slice: &mut [T], in_slice: &[T], permutation: &[usize]) {
    assert_eq!(out_slice.len(), in_slice.len());
    assert_eq!(out_slice.len(), permutation.len());
    for (out_element, old_pos) in out_slice.iter_mut().zip(permutation) {
        *out_element = in_slice[*old_pos].clone();
    }
}

/// computes permutation by using provided indices as keys
#[inline]
pub fn compute_sort_permutation(permutation: &mut [usize], indices: &[usize]) {
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
        },
        Err(e) => {
            match e {
                SparsityPatternFormatError::DuplicateEntry => {
                    let csr = CsrNonCanonical::from_csr_data(
                        nrows, ncols, indptr, indices, data
                    );
                    Ok(csr.into())
                },
                _ => Err(anyhow!("cannot read csr matrix: {}", e))
            }
        }
    }
}

fn check_format(nrows: usize, ncols: usize, indptr: &[usize], indices: &[usize]) -> std::result::Result<(), SparsityPatternFormatError>
{
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