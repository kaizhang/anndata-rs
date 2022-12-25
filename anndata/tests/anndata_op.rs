mod common;
use common::*;

use proptest::prelude::*;
use anndata::*;
use anndata_hdf5::H5;


fn test_io<T: AnnDataOp>(adata: T) {
    proptest!(ProptestConfig::with_cases(999), |(x in array_data_strat())| {
        adata.set_x(&x).unwrap();
        prop_assert_eq!(adata.read_x::<ArrayData>().unwrap().unwrap(), x);
        adata.del_x().unwrap();
    });
}

#[test]
fn test_io_h5() {
    with_empty_adata::<H5, _, _>(|adata| test_io(adata));
}