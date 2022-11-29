use anndata_rs::anndata::{AnnData, AnnDataOp};
use anndata_rs::backend::hdf5::H5;
use anndata_rs::data::*;

use proptest::prelude::*;

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

proptest! {
    #[test]
    fn uns_io(input: (i64, String)) {
        with_tmp_path(|file| {
            let adata: AnnData<H5> = AnnData::new(file, 0, 0).unwrap();
            adata.add_uns_item("test", input.0).unwrap();
            if let Data::Scalar(DynScalar::I64(i)) = adata.read_uns_item("test").unwrap().unwrap() {
                assert_eq!(i, input.0);
            } else {
                panic!("return type different from input type");
            }
            adata.add_uns_item("test", input.1.clone()).unwrap();
            if let Data::Scalar(DynScalar::String(i)) = adata.read_uns_item("test").unwrap().unwrap() {
                assert_eq!(i, input.1);
            } else {
                panic!("return type different from input type");
            }
        })
    }
}