pub(crate) mod base;
pub(crate) mod collection;

pub use base::{
    DataFrameElem, InnerDataFrameElem, DataFrameIndex, Slot, Inner,
    Elem, MatrixElem, AccumLength, StackedMatrixElem, StackedDataFrame,
};
pub use collection::{ElemCollection, AxisArrays, Axis, StackedAxisArrays};