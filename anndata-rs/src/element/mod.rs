pub(crate) mod base;
pub(crate) mod collection;

pub use base::{DataFrameElem, DataFrameIndex, Slot, Inner, Elem, MatrixElem, StackedMatrixElem, StackedDataFrame};
pub use collection::{ElemCollection, AxisArrays, Axis, StackedAxisArrays};