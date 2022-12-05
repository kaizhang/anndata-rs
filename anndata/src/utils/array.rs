use crate::data::ArrayData;
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{Array, Ix1, Dimension};

pub(crate) fn concat_array_data<I>(arrays: I) -> ArrayData
where
    I: IntoIterator<Item = ArrayData>,
{
    todo!()
}

fn hstack_array_data(this: ArrayData, other: ArrayData) -> ArrayData {
    todo!()
}

fn hstack_arr<T, D>(this: Array<T, D>, other: Array<T, D>) -> Array<T, D>
{
    todo!()
}

/// Row concatenation of sparse row matrices.
fn hstack_csr<T>(this: CsrMatrix<T>, other: CsrMatrix<T>) -> CsrMatrix<T>
where
    T: Clone,
{
    todo!()
    /*
    let mut num_rows = 0;
    let mut num_cols = 0;

    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_offsets = Vec::new();
    let nnz = mats.fold(0, |acc, mat| {
        num_rows += mat.nrows();
        num_cols = mat.ncols();
        mat.row_iter().fold(acc, |cidx, row| {
            row_offsets.push(cidx);
            values.extend_from_slice(row.values());
            col_indices.extend_from_slice(row.col_indices());
            cidx + row.nnz()
        })
    });
    row_offsets.push(nnz);
    CsrMatrix::try_from_csr_data(num_rows, num_cols, row_offsets, col_indices, values).unwrap()
    */
}