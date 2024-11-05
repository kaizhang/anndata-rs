mod common;
pub use common::*;

use anndata::{data::CsrNonCanonical, *};
use data::ArrayConvert;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use ndarray::Array2;
use proptest::prelude::*;

pub fn test_basic<B: Backend>() {
    with_tmp_dir(|dir| {
        let ann1 = AnnData::<B>::new(dir.join("test1")).unwrap();
        let csc = rand_csc::<i32>(10, 5, 3, 1, 100);
        ann1.obsm().add("csc", &csc).unwrap();
        assert!(ann1.obsm().get_item::<CsrMatrix<i32>>("csc").is_err());

        let ann2 = AnnData::<B>::new(dir.join("test2")).unwrap();
        AnnDataSet::<B>::new(
            [("ann1", ann1), ("ann2", ann2)],
            dir.join("dataset"),
            "sample",
        )
        .unwrap();
    })
}

pub fn test_save<B: Backend>() {
    with_tmp_dir(|dir| {
        let input = dir.join("input");
        let output = dir.join("output");
        let anndatas = ((0 as usize..100), (0 as usize..100)).prop_flat_map(|(n_obs, n_vars)| {
            (
                anndata_strat::<B, _>(&input, n_obs, n_vars),
                select_strat(n_obs),
                select_strat(n_vars),
            )
        });
        proptest!(ProptestConfig::with_cases(100), |((adata, slice_obs, slice_var) in anndatas)| {
            adata.write::<B, _>(&output).unwrap();
            let adata_in = AnnData::<B>::open(B::open(&output).unwrap()).unwrap();
            prop_assert!(anndata_eq(&adata, &adata_in).unwrap());
            adata_in.close().unwrap();

            let index = adata.obs_names().select(&slice_obs);
            assert_eq!(index.len(), index.into_vec().len());

            let select = [slice_obs, slice_var];
            adata.write_select::<B, _, _>(&select, &output).unwrap();
            adata.subset(&select).unwrap();
            let adata_in = AnnData::<B>::open(B::open(&output).unwrap()).unwrap();
            prop_assert!(anndata_eq(&adata, &adata_in).unwrap());
            adata_in.close().unwrap();
        });
    });
}

pub fn test_speacial_cases<F, T>(adata_gen: F)
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

    // Data type casting
    let _: Array2<f64> = adata.x().get::<ArrayData>().unwrap().unwrap().try_convert().expect("data type casting failed");
}

pub fn test_noncanonical<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let adata = adata_gen();
    let coo: CooMatrix<i32> = CooMatrix::try_from_triplets(
        5,
        4,
        vec![0, 1, 1, 1, 2, 3, 4],
        vec![0, 0, 0, 2, 3, 1, 3],
        vec![1, 2, 3, 4, 5, 6, 7],
    )
    .unwrap();
    adata.set_x(&CsrNonCanonical::from(&coo)).unwrap();
    assert!(adata.x().get::<CsrMatrix<i32>>().is_err());
    adata.x().get::<CsrNonCanonical<i32>>().unwrap().unwrap();
    adata.x().get::<ArrayData>().unwrap().unwrap();
}

pub fn test_io<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let arrays =
        proptest::collection::vec(0 as usize..50, 2..4).prop_flat_map(|shape| array_strat(&shape));
    proptest!(ProptestConfig::with_cases(256), |(x in arrays)| {
        let adata = adata_gen();
        adata.set_x(&x).unwrap();
        prop_assert_eq!(adata.x().get::<ArrayData>().unwrap().unwrap(), x);
    });
}

pub fn test_index<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let arrays = proptest::collection::vec(0 as usize..50, 2..4)
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

pub fn test_iterator<F, T>(adata_gen: F)
where
    F: Fn() -> T,
    T: AnnDataOp,
{
    let arrays =
        proptest::collection::vec(20 as usize..50, 2..3).prop_flat_map(|shape| array_strat(&shape));
    proptest!(ProptestConfig::with_cases(10), |(x in arrays)| {
        if let ArrayData::CscMatrix(_) = x {
        } else {
            let adata = adata_gen();
            adata.obsm().add_iter("test", array_chunks(&x, 7)).unwrap();
            prop_assert_eq!(adata.obsm().get_item::<ArrayData>("test").unwrap().unwrap(), x.clone());

            adata.obsm().add_iter("test2", adata.obsm().get_item_iter("test", 7).unwrap().map(|x| x.0)).unwrap();
            prop_assert_eq!(adata.obsm().get_item::<ArrayData>("test2").unwrap().unwrap(), x);
        }
    });
}
