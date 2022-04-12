/// Raw elements which are not clonable and cannot be shared with Python
mod base;

/// Arc and Mutex wrapped elements that can be shared with Python
mod element;

/// Collections containing multiple elements
mod collection;

pub use base::{RawElem, RawMatrixElem};
pub use element::{ElemTrait, Elem, MatrixElem, DataFrameElem, Stacked};
pub use collection::{ElemCollection, AxisArrays, Axis, StackedAxisArrays};