/// Raw elements which are not clonable and cannot be shared with Python
pub(crate) mod base;

/// Arc and Mutex wrapped elements that can be shared with Python
pub(crate) mod element;

/// Collections containing multiple elements
pub(crate) mod collection;

pub use base::{RawElem, RawMatrixElem, DataFrameElem, InnerDataFrameElem, DataFrameIndex, Slot, Inner};
pub use element::{
    Elem, MatrixElem,
    AccumLength, Stacked, StackedDataFrame,
};
pub use collection::{ElemCollection, AxisArrays, Axis, StackedAxisArrays};