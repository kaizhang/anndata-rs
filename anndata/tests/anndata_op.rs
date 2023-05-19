#![allow(dead_code, unused)]

mod common;
use common::*;

use ndarray::Array2;
use proptest::prelude::*;
use anndata::*;
use anndata_hdf5::H5;
use std::path::Path;
use nalgebra_sparse::{CscMatrix, CsrMatrix};


fn test_speacial_cases<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let adata = adata_gen();
    
    let arr = Array2::<i32>::zeros((0, 0));
    adata.set_x(&arr).unwrap();

    // Adding matrices with wrong shapes should fail
    let arr2 = Array2::<i32>::zeros((10, 20));
    assert!(adata.obsm().add("test", &arr2).is_err());

    // Automatical data type casting
    adata.x().get::<Array2<f64>>().unwrap().unwrap();
}

fn test_io<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let arrays = proptest::collection::vec(0 as usize ..50, 2..4).prop_flat_map(|shape| array_strat(&shape));
    proptest!(ProptestConfig::with_cases(256), |(x in arrays)| {
        let adata = adata_gen();
        adata.set_x(&x).unwrap();
        prop_assert_eq!(adata.x().get::<ArrayData>().unwrap().unwrap(), x);
    });
}

fn test_index<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let arrays = proptest::collection::vec(0 as usize ..50, 2..4)
        .prop_flat_map(|shape| array_slice_strat(&shape));
    proptest!(ProptestConfig::with_cases(256), |((x, select) in arrays)| {
        let adata = adata_gen();
        adata.set_x(&x).unwrap();
        prop_assert_eq!(
            adata.x().slice::<ArrayData, _>(&select).unwrap().unwrap(),
            array_select(&x, select.as_slice())
        );

        adata.obsm().add("test", &x).unwrap();
        prop_assert_eq!(
            adata.obsm().get_item_slice::<ArrayData, _>("test", &select).unwrap().unwrap(),
            array_select(&x, select.as_slice())
        );
    });
}

fn test_iterator<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let arrays = proptest::collection::vec(20 as usize ..50, 2..3)
        .prop_flat_map(|shape| array_strat(&shape));
    proptest!(ProptestConfig::with_cases(10), |(x in arrays)| {
        let adata = adata_gen();
        adata.obsm().add_iter("test", array_chunks(&x, 7)).unwrap();
        prop_assert_eq!(adata.obsm().get_item::<ArrayData>("test").unwrap().unwrap(), x.clone());

        adata.obsm().add_iter("test2", adata.obsm().get_item_iter::<ArrayData>("test", 7).unwrap().map(|x| x.0)).unwrap();
        prop_assert_eq!(adata.obsm().get_item::<ArrayData>("test2").unwrap().unwrap(), x);
    });
}


////////////////////////////////////////////////////////////////////////////////
/// Test HDF5 backend
////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_speacial_cases_h5() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        test_speacial_cases(|| adata_gen());
    })
}

#[test]
fn test_io_h5() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        test_io(|| adata_gen());
    })
}

#[test]
fn test_index_h5() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        test_index(|| adata_gen());
    })
}

#[test]
fn test_iterator_h5() {
    with_tmp_dir(|dir| {
        let file = dir.join("test.h5");
        let adata_gen = || AnnData::<H5>::new(&file).unwrap();
        test_iterator(|| adata_gen());
    })
}

#[test]
fn test_io_h5_csc() {
    let path = Path::new("/home/fangzq/github/anndata-rs/LLP.csc.h5ad");
    let file = H5::open(path.to_path_buf()).unwrap();
    let adata = AnnData::<H5>::open(file).unwrap();
    // println!("adata csc_matrix: {:?}", &adata);
    // // Automatical data type casting
    // adata.x().get::<Array2<f64>>().unwrap().unwrap();
    // adata.x().get::<ArrayData>().unwrap().unwrap()
    let idxx = vec![0, 3, 5];
    let gixx = s![.., idxx];
    // let inner = adata.get_x().inner();
    // let gex = inner.select::<CscMatrix<f32>, _>(gixx.as_ref()).unwrap();
    let gex = adata.x().slice::<CscMatrix<f32>, _>(gixx).unwrap().unwrap();// CSCMatrix

    // println!("{}", inner.dtype());
    // println!("{}", inner.shape());
    // let x = adata.x().get::<CscMatrix<f32>>().unwrap().unwrap();
    println!("{:?}", gex.shape());
    // let data = inner.data::<CsrMatrix<f32>>()?;
    // println!("{:?}", data);


}