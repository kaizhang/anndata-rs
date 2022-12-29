mod common;
use common::*;

use ndarray::Array2;
use proptest::prelude::*;
use anndata::*;
use anndata_hdf5::H5;

fn test_speacial_cases<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let mut adata = adata_gen();
    let arr = Array2::<i32>::zeros((0, 0));
    let arr2 = Array2::<i32>::zeros((10, 20));
    adata.set_x(&arr).unwrap();
    assert!(adata.obsm().add("test", &arr2).is_err());
}

/*
fn test_iterator<T: AnnDataOp>(adata: T) {
    let (csr, chunks) = rand_chunks(997, 20, 41);

    adata.set_x_from_iter(chunks)?;

    assert_eq!(adata.n_obs(), 997);
    assert_eq!(adata.n_vars(), 20);
    assert_eq!(adata.read_x::<CsrMatrix<i64>>()?.unwrap(), csr);

    adata.add_obsm_from_iter::<_, CsrMatrix<i64>>("key", adata.read_x_iter(111).map(|x| x.0))?;
    assert_eq!(adata.fetch_obsm::<CsrMatrix<i64>>("key")?.unwrap(), csr);
}
*/

fn test_io<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let arrays = proptest::collection::vec(0 as usize ..50, 2..4).prop_flat_map(|shape| array_strat(&shape));
    proptest!(ProptestConfig::with_cases(999), |(x in arrays)| {
        let adata = adata_gen();
        adata.set_x(&x).unwrap();
        prop_assert_eq!(adata.read_x::<ArrayData>().unwrap().unwrap(), x);
    });
}

fn test_index<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let arrays = proptest::collection::vec(0 as usize ..50, 2..4)
        .prop_flat_map(|shape| array_slice_strat(&shape));
    proptest!(ProptestConfig::with_cases(999), |((x, select) in arrays)| {
        let adata = adata_gen();
        adata.set_x(&x).unwrap();
        prop_assert_eq!(
            adata.read_x_slice::<ArrayData, _>(&select).unwrap().unwrap(),
            array_select(&x, select.as_slice())
        );

        adata.obsm().add("test", &x).unwrap();
        prop_assert_eq!(
            adata.obsm().get_slice::<ArrayData, _>("test", &select).unwrap().unwrap(),
            array_select(&x, select.as_slice())
        );
    });
}

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