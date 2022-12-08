use crate::backend::{WriteConfig, Backend, BackendData, DataContainer, DatasetOp, GroupOp, LocationOp};
use crate::data::data_traits::{ReadData, WriteData};
use crate::data::{Shape, SelectInfoElem};
use crate::s;

use anyhow::Result;
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{ArrayView, Dimension};
use smallvec::SmallVec;

pub struct CsrIterator<I>(I);

impl<I: Iterator<Item = CsrMatrix<T>>, T: BackendData> CsrIterator<I> {
    pub fn new(iterator: I) -> Self {
        Self(iterator)
    }
}

struct ExtendableDataset<B: Backend> {
    dataset: B::Dataset,
    capacity: Shape,
    size: Shape,
}

impl<B: Backend> ExtendableDataset<B> {
    fn with_capacity(dataset: B::Dataset, capacity: Shape) -> Self {
        Self {
            dataset,
            size: std::iter::repeat(0).take(capacity.ndim()).collect(),
            capacity,
        }
    }

    fn reserve(&mut self, additional: &Shape) -> Result<()> {
        self.capacity.as_mut().iter_mut()
            .zip(additional.as_ref())
            .for_each(|(x, add)| *x += *add);
        self.dataset.reshape(&self.capacity)
    }

    fn check_or_grow(&mut self, size: &Shape, default: usize) -> Result<()> {
        let additional: Shape = self.capacity.as_ref().iter().zip(size.as_ref()).map(|(cap, size)|
            if *cap < *size {
                default.max(*size - *cap)
            } else {
                0
            }
        ).collect();

        if additional.as_ref().iter().any(|x| *x != 0) {
            self.reserve(&additional)?;
        }
        Ok(())
    }

    fn extend<'a, T: BackendData, D: Dimension>(&mut self, data: ArrayView<'a, T, D>) -> Result<()> {
        let new_size = self.size.as_ref().iter().zip(data.shape())
            .map(|(x, y)| *x + *y).collect();
        self.check_or_grow(&new_size, 100000)?;
        let slice: SmallVec<[SelectInfoElem; 3]> = self.size.as_ref().iter().zip(new_size.as_ref())
            .map(|(x, y)| (*x..*y).into()).collect();
        self.dataset.write_array_slice(data, slice)?;
        self.size = new_size;
        Ok(())
    }
}

/*
impl<I, T> WriteData for CsrIterator<I>
where
    I: Iterator<Item = CsrMatrix<T>>,
    T: BackendData,
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
*/