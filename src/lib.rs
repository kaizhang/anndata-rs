pub mod anndata_trait;
pub mod base;
pub mod backed;
pub mod utils;

/*
pub fn read_dataframe_index(group: &Group) -> Result<Elem<dyn AnnDataType>> {
    let index_name = read_str_attr(group, "_index")?;
    let elem = Box::new(group.dataset(&index_name)?);
    Elem::new(elem)
}
*/

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