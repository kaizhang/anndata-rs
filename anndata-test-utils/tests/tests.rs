use anndata_test_utils as utils;
use anndata_test_utils::with_tmp_dir;
use anndata_hdf5::H5;
use anndata_zarr::Zarr;
use anndata::AnnData;

#[test]
fn test_basic() {
    utils::test_basic::<H5>();
    utils::test_basic::<Zarr>();
}

#[test]
fn test_save() {
    utils::test_save::<H5>();
    utils::test_save::<Zarr>();
}

#[test]
fn test_speacial_cases() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_speacial_cases(|| adata_gen());

        let file = dir.join("test.zarr");
        let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        utils::test_speacial_cases(|| adata_gen());
    })
}

#[test]
fn test_noncanonical() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_noncanonical(|| adata_gen());

        let file = dir.join("test.zarr");
        let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        utils::test_noncanonical(|| adata_gen());
    })
}

#[test]
fn test_io() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_io(|| adata_gen());

        let file = dir.join("test.zarr");
        let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        utils::test_io(|| adata_gen());
    })
}

#[test]
fn test_index_h5() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_index(|| adata_gen());

        let file = dir.join("test.zarr");
        let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        utils::test_index(|| adata_gen());
    })
}

#[test]
fn test_iterator() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        utils::test_iterator(|| adata_gen());

        let file = dir.join("test.zarr");
        let adata_gen = || AnnData::<Zarr>::new(&file).unwrap();
        utils::test_iterator(|| adata_gen());
    })
}