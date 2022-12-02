pub mod traits;
pub mod anndata;
pub mod backend;
pub mod data;
pub mod element;
//pub mod io;
//pub mod iterator;
pub(crate) mod utils;

pub use traits::AnnDataOp;
pub use anndata::AnnData;
pub use data::{Data, ArrayData, WriteArrayData, ReadArrayData, ArrayOp};
pub use element::{
    AxisArrays, DataFrameElem, Elem, ElemCollection, ArrayElem, 
    //StackedAxisArrays, StackedDataFrame, StackedMatrixElem,
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