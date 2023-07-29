#![feature(iterator_try_reduce)]

mod anndata;
pub mod traits;
pub mod backend;
pub mod data;
pub mod container;
pub mod reader;

pub use traits::{AnnDataOp, AxisArraysOp, ElemCollectionOp, ArrayElemOp};
pub use crate::anndata::{AnnData, AnnDataSet, StackedAnnData};
pub use backend::Backend;
pub use data::{HasShape, Data, ReadData, WriteData, ArrayData, WriteArrayData, ReadArrayData, ArrayOp};
pub use container::{
    AxisArrays, DataFrameElem, Elem, ElemCollection, ArrayElem, 
    StackedAxisArrays, StackedDataFrame, StackedArrayElem,
};