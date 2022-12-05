use crate::backend::{WriteConfig, Backend, BackendData, DataContainer, DatasetOp, GroupOp, LocationOp};
use crate::data::data_traits::{ReadData, WriteData};
use crate::s;

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

        let config = WriteConfig { block_size: Some(50000.into()), ..Default::default() };
        let data = group.new_dataset::<D>("data", &0.into(), config.clone())?;
        let indices = group.new_dataset::<i64>("indices", &0.into(), config)?;
        let mut indptr: Vec<i64> = Vec::new();

        let mut write_pos= 0;

        let nnz: Result<i64> = self.iterator.try_fold(0, |accum, chunk| {
            let mut data_chunk = Vec::new();
            let mut indices_chunk = Vec::new();

            let acc = chunk.into_iter().fold(accum, |accum_, vec| {
                indptr.push(accum_);
                let acc = accum_ + (vec.len() as i64);
                vec.into_iter().for_each(|(i, d)| {
                    indices_chunk.push(i as i64);
                    data_chunk.push(d);
                });
                acc
            });

            let end_pos = write_pos + data_chunk.len();
            data.reshape(&end_pos.into())?;
            indices.reshape(&end_pos.into())?;
            data.write_array_slice(&data_chunk, s![write_pos..end_pos])?;
            indices.write_array_slice(&indices_chunk, s![write_pos..end_pos])?;
            write_pos = end_pos;
            Ok(acc)
        });
        indptr.push(nnz?);

        let num_rows = indptr.len() - 1;
        group.write_arr_attr("shape", &[num_rows, self.num_cols])?;
        group.create_array_data("indptr", &indptr, Default::default())?;
        Ok(DataContainer::Group(group))
    }
}