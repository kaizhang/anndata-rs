use anndata_rs::anndata::{AnnData, AnnDataOp};
use anndata_rs::backend::hdf5::H5;
use anndata_rs::data::*;

use proptest::prelude::*;

use anyhow::Result;
use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::fmt::Debug;
use tempfile::tempdir;
use hdf5::File;
use std::path::PathBuf;

pub fn with_tmp_dir<T, F: FnMut(PathBuf) -> T>(mut func: F) -> T {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    func(path)
}

fn with_tmp_path<T, F: Fn(PathBuf) -> T>(func: F) -> T {
    with_tmp_dir(|dir| func(dir.join("foo.h5")))
}

fn with_tmp_file<T, F: Fn(File) -> T>(func: F) -> T {
    with_tmp_path(|path| {
        let file = File::create(&path).unwrap();
        func(file)
    })
}

fn uns_io<T>(input: T)
where
    T: Eq + Debug + Into<Data> + Clone,
    Data: TryInto<T>,
    <Data as TryInto<T>>::Error: Debug,
{
    with_tmp_path(|file| {
        let adata: AnnData<H5> = AnnData::new(file, 0, 0).unwrap();
        adata.add_uns_item("test", Data::from(&input)).unwrap();
        assert_eq!(input, adata.read_uns_item("test").unwrap().unwrap().try_into().unwrap());
    });
}

#[test]
fn test_basic() -> Result<()> {
    with_tmp_path(|file| -> Result<()> {
        let adata: AnnData<H5> = AnnData::new(file, 0, 0)?;

        adata.add_uns_item("test", 3i32)?;
        assert_eq!(3i32, adata.read_uns_item("test")?.unwrap().try_into()?);
        adata.add_uns_item("test", 3.0f32)?;
        assert_eq!(3.0f32, adata.read_uns_item("test")?.unwrap().try_into()?);

        let arr = Array::random((2, 5), Uniform::new(0, 100)).into_dyn();
        adata.add_obsm_item("test", &arr)?;
        adata.add_varm_item("test", &arr)?;

        let arr_x = Array::random((2, 2), Uniform::new(0.0, 100.0)).into_dyn();
        adata.set_x(arr_x.clone())?;

        assert!(adata.add_obsp_item("test", &arr).is_err());

        Ok(())
    })
}

#[test]
fn test_uns_io_array() {
    let arr = Array::random((2, 5), Uniform::new(0, 100)).into_dyn();
    uns_io(arr);
}

proptest! {
    #[test]
    fn test_uns_io(input: (i64, String)) {
        uns_io(input.0);
        uns_io(input.1);
    }
}