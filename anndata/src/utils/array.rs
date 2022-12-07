use crate::data::{ArrayData, DynArray, DynCsrMatrix};

use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::pattern::SparsityPattern;
use ndarray::{Axis, Array, IxDyn};
use anyhow::{Result, bail};

macro_rules! impl_vstack_array {
    ($this:expr, $this_ty:ident, $other:expr, $other_ty:ident, $func:expr) => {{
        match ($this, $other) {
            ($this_ty::Bool(this), $other_ty::Bool(other)) =>
                $func(this, other).into(),
            ($this_ty::U8(this), $other_ty::U8(other)) =>
                $func(this, other).into(),
            ($this_ty::U16(this), $other_ty::U16(other)) =>
                $func(this, other).into(),
            ($this_ty::U32(this), $other_ty::U32(other)) =>
                $func(this, other).into(),
            ($this_ty::U64(this), $other_ty::U64(other)) =>
                $func(this, other).into(),
            ($this_ty::Usize(this), $other_ty::Usize(other)) =>
                $func(this, other).into(),
            ($this_ty::I8(this), $other_ty::I8(other)) =>
                $func(this, other).into(),
            ($this_ty::I16(this), $other_ty::I16(other)) =>
                $func(this, other).into(),
            ($this_ty::I32(this), $other_ty::I32(other)) =>
                $func(this, other).into(),
            ($this_ty::I64(this), $other_ty::I64(other)) =>
                $func(this, other).into(),
            ($this_ty::F32(this), $other_ty::F32(other)) =>
                $func(this, other).into(),
            ($this_ty::F64(this), $other_ty::F64(other)) =>
                $func(this, other).into(),
            ($this_ty::String(this), $other_ty::String(other)) =>
                $func(this, other).into(),
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
        (ArrayData::Array(this), ArrayData::Array(other)) =>
            impl_vstack_array!(this, DynArray, other, DynArray, vstack_arr),
        (ArrayData::CsrMatrix(this), ArrayData::CsrMatrix(other)) =>
            impl_vstack_array!(this, DynCsrMatrix, other, DynCsrMatrix, vstack_csr),
        _ => bail!("Cannot concatenate arrays of different types"),
    };
    Ok(array)
}

fn vstack_arr<T: Clone>(mut this: Array<T, IxDyn>, other: Array<T, IxDyn>) -> Array<T, IxDyn>
{
    this.append(Axis(0), other.view()).unwrap();
    this
}

/// Row concatenation of sparse row matrices.
fn vstack_csr<T: Clone>(this: CsrMatrix<T>, other: CsrMatrix<T>) -> CsrMatrix<T>
{
    let num_cols = this.ncols();
    let num_rows = this.nrows() + other.nrows();
    let nnz = this.nnz();
    let (mut indptr, mut indices, mut data) = this.disassemble();
    let (indptr2, indices2, data2) = other.csr_data();
    indices.extend_from_slice(indices2);
    data.extend_from_slice(data2);
    indptr2.iter().skip(1).for_each(|&i| indptr.push(i + nnz));

    let pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(
            num_rows,
            num_cols,
            indptr,
            indices,
        )
    };
    CsrMatrix::try_from_pattern_and_values(pattern, data).unwrap()
}