pub(crate) mod base;
pub(crate) mod collection;

pub use base::{
    DataFrameElem, DataFrameIndex, Elem, Inner, MatrixElem, Slot, StackedDataFrame,
    StackedMatrixElem,
};
pub use collection::{Axis, AxisArrays, ElemCollection, StackedAxisArrays};