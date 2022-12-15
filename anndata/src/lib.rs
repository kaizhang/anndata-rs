#![feature(iterator_try_reduce)]

mod anndata;
pub mod traits;
pub mod backend;
pub mod data;
pub mod container;

pub use traits::AnnDataOp;
pub use crate::anndata::{AnnData, AnnDataSet};
pub use backend::Backend;
pub use data::{HasShape, Data, ReadData, WriteData, ArrayData, WriteArrayData, ReadArrayData, ArrayOp};
pub use container::{
    AxisArrays, DataFrameElem, Elem, ElemCollection, ArrayElem, 
    StackedAxisArrays, StackedDataFrame, StackedArrayElem,
};
//pub use iterator::{AnnDataIterator, ChunkedMatrix, StackedChunkedMatrix};

/*
/// Implementation's prelude. Common types used everywhere.
mod imp_prelude {
    pub use crate::dimension::DimensionExt;
    pub use crate::prelude::*;
    pub use crate::ArcArray;
    pub use crate::{
        CowRepr, Data, DataMut, DataOwned, DataShared, Ix, Ixs, RawData, RawDataMut, RawViewRepr,
        RemoveAxis, ViewRepr,
    };
}
*/