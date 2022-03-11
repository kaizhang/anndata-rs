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

