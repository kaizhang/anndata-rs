mod common;
use common::*;

use proptest::prelude::*;
use anndata::*;
use anndata_hdf5::H5;


fn anndata_op_test() {
    proptest!(ProptestConfig::with_cases(999), |(x in array_data_strat())| {
        with_empty_adata::<H5, _, _>(|adata| {
            adata.set_x(&x).unwrap();
            assert_eq!(adata.read_x::<ArrayData>().unwrap().unwrap(), x);
            adata.del_x().unwrap();
        })
        /*
        });
        */
    });
}

#[test]
fn anndata_op_test_h5() {
    anndata_op_test();
}