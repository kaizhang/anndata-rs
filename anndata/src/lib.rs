mod anndata;
pub mod concat;
pub mod traits;
pub mod backend;
pub mod data;
pub mod container;
pub mod reader;
mod macros;

pub use traits::{AnnDataOp, AxisArraysOp, ElemCollectionOp, ArrayElemOp};
pub use crate::anndata::{AnnData, AnnDataSet, StackedAnnData};
pub use backend::Backend;
pub use data::{HasShape, Data, Readable, Writable, ArrayData, WritableArray, ReadableArray, Selectable};
pub use container::{
    AxisArrays, DataFrameElem, Elem, ElemCollection, ArrayElem, 
    StackedAxisArrays, StackedDataFrame, StackedArrayElem,
};