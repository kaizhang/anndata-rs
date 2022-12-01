use crate::{
    backend::{
        Backend, BackendData, DatasetOp, FileOp, GroupOp, LocationOp, ScalarType, Selection, DynArrayView,
    },
    data::{DynScalar, DynArray},
};

use anyhow::{bail, Result};
use ndarray::{Array, Array2, ArrayView, Dimension, IxDyn, IxDynImpl};
use std::{
    fmt::write,
    ops::Deref,
    path::{Path, PathBuf},
    sync::Arc,
};

pub struct N5;

impl Backend for N5 {
    type File = N5Filesystem;

    type Group = Group;

    /// datasets contain arrays.
    type Dataset = Dataset;

    fn create<P: AsRef<Path>>(path: P) -> Result<Self::File> {
        Ok(File::create(path)?)
    }
}