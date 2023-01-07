mod common;
use common::*;

use proptest::prelude::*;
use anndata::*;
use anndata_hdf5::H5;

fn test_save<B: Backend>() {
    with_tmp_dir(|dir| {
        let input = dir.join("input.h5ad");
        let output = dir.join("output.h5ad");
        let anndatas = ((0 as usize ..100), (0 as usize ..100))
            .prop_flat_map(|(n_obs, n_vars)|
                ( anndata_strat::<B, _>(&input, n_obs, n_vars),
                  select_strat(n_obs),
                  select_strat(n_vars),
                )
            );
        proptest!(ProptestConfig::with_cases(100), |((adata, slice_obs, slice_var) in anndatas)| {
            adata.write::<B, _>(&output).unwrap();
            let mut adata_in = AnnData::<B>::open(B::open(&output).unwrap()).unwrap();
            prop_assert!(anndata_eq(&adata, &adata_in).unwrap());
            adata_in.close().unwrap();

            let index = adata.obs_names().select(&slice_obs);
            assert_eq!(index.len(), index.into_vec().len());

            let select = [slice_obs, slice_var];
            adata.write_select::<B, _, _>(&select, &output).unwrap();
            adata.subset(&select).unwrap();
            adata_in = AnnData::<B>::open(B::open(&output).unwrap()).unwrap();
            prop_assert!(anndata_eq(&adata, &adata_in).unwrap());
            adata_in.close().unwrap();
        });
    });
}

#[test]
fn test_save_h5() {
    test_save::<H5>()
}