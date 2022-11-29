use crate::backend::Backend;

use std::{ops::Deref, path::{Path, PathBuf}};
use hdf5::{
    File, Group, Location,
    dataset::Dataset,
    types::{TypeDescriptor, VarLenAscii, VarLenUnicode},
    Error, Extent, Group, H5Type, Location, Selection,
};
use anyhow::{bail, Result};

pub struct H5;

impl Backend for H5 {
    type File = File;

    type Group = Group;

    /// datasets contain arrays.
    type Dataset = Dataset;

    type Location = Location;

    fn create<P: AsRef<Path>>(path: P) -> Result<Self::File> {todo!()}
    fn filename(file: &Self::File) -> PathBuf {todo!()}
    fn close(file: Self::File) -> Result<()> {todo!()}

    fn list(group: &Self::Group) -> Result<Vec<String>> {todo!()}
    fn create_group(group: &Self::Group, name: &str) -> Result<Self::Group> {todo!()}
}