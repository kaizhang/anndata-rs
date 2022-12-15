pub(crate) mod base;
pub(crate) mod collection;

pub use base::{
    InnerDataFrameElem, DataFrameElem, Elem, Inner, ArrayElem, Slot,
    StackedDataFrame, StackedArrayElem, ChunkedArrayElem, StackedChunkedArrayElem,
};
pub use collection::{Axis, AxisArrays, ElemCollection, StackedAxisArrays};