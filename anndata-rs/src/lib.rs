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