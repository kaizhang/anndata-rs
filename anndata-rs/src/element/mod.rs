/// Raw elements which are not clonable and cannot be shared with Python
pub(crate) mod base;

/// Collections containing multiple elements
pub(crate) mod collection;

pub use base::{
    DataFrameElem, InnerDataFrameElem, DataFrameIndex, Slot, Inner,
    Elem, MatrixElem, AccumLength, StackedMatrixElem, StackedDataFrame,
};
pub use collection::{ElemCollection, AxisArrays, Axis, StackedAxisArrays};