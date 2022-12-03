use crate::backend::{Backend, BackendData, DataContainer, GroupOp, LocationOp};
use crate::data::data_traits::{ReadData, WriteData};

use anyhow::Result;

pub struct CsrIterator<I> {
    iterator: I,
    num_cols: usize,
}

impl<I> CsrIterator<I> {
    pub fn new(iterator: I, num_cols: usize) -> Self {
        Self { iterator, num_cols }
    }
}

impl<I, D> WriteData for CsrIterator<I>
where
    I: Iterator<Item = Vec<Vec<(usize, D)>>>,
    D: BackendData,
{
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let group = location.create_group(name)?;
        group.write_str_attr("encoding_type", "csr_matrix")?;
        group.write_str_attr("encoding-version", "0.2.0")?;
        group.write_str_attr("h5sparse_format", "csr")?;

        let chunk_size: usize = 50000;
        let data: ResizableVectorData<D> = ResizableVectorData::new(&group, "data", chunk_size)?;
        // Use i64 as the indices type to be compatible with scipy.
        let indices: ResizableVectorData<i64> =
            ResizableVectorData::new(&group, "indices", chunk_size)?;

        let mut indptr: Vec<i64> = Vec::new();

        let nnz: Result<i64> = self.iterator.try_fold(0, |accum, chunk| {
            data.extend(chunk.iter().flatten().map(|x| x.1))?;
            indices.extend(
                chunk
                    .iter()
                    .flatten()
                    .map(|x| -> i64 { x.0.try_into().unwrap() }),
            )?;
            Ok(chunk.iter().fold(accum, |accum_, vec| {
                indptr.push(accum_);
                accum_ + (vec.len() as i64)
            }))
        });
        indptr.push(nnz?);

        let num_rows = indptr.len() - 1;
        group
            .new_attr_builder()
            .with_data(&arr1(&[num_rows, self.num_cols]))
            .create("shape")?;
        group
            .new_dataset_builder()
            .deflate(COMPRESSION)
            .with_data(&Array::from_vec(indptr))
            .create("indptr")?;

        Ok((DataContainer::H5Group(group), num_rows))
    }
}
