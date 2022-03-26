pub mod anndata_trait;
pub mod base;
pub mod element;
pub mod utils;

#[cfg(test)]
mod tests {
    use crate::backed::AnnData;

    #[test]
    fn test() {
        let adata: AnnData = AnnData::read("data.h5ad").unwrap();
        println!(
            "{:?}",
            adata.obsm.get("X_spectral").unwrap().arr_f64().read_data(),
        );

        let adata_subset = adata.subset_obs(&[1,2,3]);
        println!(
            "{:?}",
            adata_subset.obsm.get("X_spectral").unwrap().arr_f64().read_data(),
        );

        adata_subset.write("out.h5ad").unwrap();

        //assert_eq!(beds, expected);
    }
}