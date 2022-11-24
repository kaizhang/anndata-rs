use hdf5::{
    dataset::Dataset,
    types::{TypeDescriptor, VarLenAscii, VarLenUnicode},
    Error, Extent, Group, H5Type, Location, Result, Selection,
};
use itertools::Itertools;
use ndarray::{Array, Array1, ArrayView, Dimension};
use std::marker::PhantomData;

pub const COMPRESSION: u8 = 2;

pub fn create_dataset<'d, A, T, D>(location: &Group, name: &str, data: A) -> Result<Dataset>
where
    A: Into<ArrayView<'d, T, D>>,
    T: H5Type,
    D: Dimension,
{
    let arr = data.into();
    let shape = arr.shape();
    let chunk_size = if shape.len() == 1 {
        vec![shape[0].min(100000)]
    } else {
        shape.iter().map(|&x| x.min(100)).collect()
    };
    if arr.len() > 100 {
        location
            .new_dataset_builder()
            .deflate(COMPRESSION)
            .chunk(chunk_size)
            .with_data(arr)
            .create(name)
    } else {
        location.new_dataset_builder().with_data(arr).create(name)
    }
}

pub fn create_str_attr(location: &Location, name: &str, value: &str) -> Result<()> {
    let value_: VarLenUnicode = value.parse().unwrap();
    let attr = location
        .attr(name)
        .or(location.new_attr::<VarLenUnicode>().create(name))?;
    attr.write_scalar(&value_)
}

pub fn read_str_attr(location: &Location, name: &str) -> Result<String> {
    let container = location.attr(name)?;
    match container.dtype()?.to_descriptor()? {
        TypeDescriptor::VarLenAscii => {
            let attr: VarLenAscii = container.read_scalar()?;
            Ok(attr.parse().unwrap())
        }
        TypeDescriptor::VarLenUnicode => {
            let attr: VarLenUnicode = container.read_scalar()?;
            Ok(attr.parse().unwrap())
        }
        ty => {
            panic!("Cannot read string from type '{}'", ty);
        }
    }
}

pub fn read_str_vec_attr(location: &Location, name: &str) -> Result<Vec<String>> {
    let container = location.attr(name)?;
    if container.size() == 0 {
        Ok(Vec::new())
    } else {
        let arr: Array1<VarLenUnicode> = container.read()?;
        Ok(arr
            .into_raw_vec()
            .into_iter()
            .map(|x| x.as_str().to_string())
            .collect())
    }
}

pub struct ResizableVectorData<T> {
    dataset: Dataset,
    dataset_type: PhantomData<T>,
}

impl<T: H5Type> ResizableVectorData<T> {
    pub fn new(group: &Group, name: &str, chunk_size: usize) -> Result<Self> {
        let dataset = group
            .new_dataset::<T>()
            .deflate(COMPRESSION)
            .chunk(chunk_size)
            .shape(Extent::resizable(0))
            .create(name)?;
        Ok(ResizableVectorData {
            dataset,
            dataset_type: PhantomData,
        })
    }

    /// Returns the current size of the vector.
    pub fn size(&self) -> usize {
        self.dataset.shape()[0]
    }

    /// Resizes the dataset to a new length.
    pub fn resize(&self, size: usize) -> Result<()> {
        self.dataset.resize(size)
    }

    /// Returns the chunk size of the vector.
    pub fn chunk_size(&self) -> usize {
        self.dataset.chunk().unwrap()[0]
    }

    pub fn extend<I>(&self, iter: I) -> Result<()>
    where
        I: Iterator<Item = T>,
    {
        let arr = Array::from_iter(iter);
        let n = arr.raw_dim().size();
        let old_size = self.size();
        let new_size = old_size + n;
        self.resize(new_size)?;
        self.write_slice(&arr, old_size..new_size)
    }

    pub fn extend_by<I>(&self, iter: I, step: usize) -> Result<()>
    where
        I: Iterator<Item = T>,
    {
        for chunk in &iter.chunks(step) {
            self.extend(chunk)?;
        }
        Ok(())
    }

    fn write_slice<'a, A, S, D>(&self, arr: A, selection: S) -> Result<()>
    where
        A: Into<ArrayView<'a, T, D>>,
        S: TryInto<Selection>,
        Error: From<S::Error>,
        D: Dimension,
    {
        self.dataset.write_slice(arr, selection)
    }
}

pub fn read_str_vec(dataset: &Dataset) -> Result<Vec<String>> {
    let arr: Array1<VarLenUnicode> = dataset.read()?;
    Ok(arr
        .into_raw_vec()
        .into_iter()
        .map(|x| x.as_str().to_string())
        .collect())
}

pub struct Chunks<'a, T> {
    data: &'a Dataset,
    chunk_size: usize,
    position: usize,
    length: usize,
    phantom: PhantomData<T>,
}

impl<'a, T: hdf5::H5Type> Iterator for Chunks<'a, T> {
    type Item = Array1<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.length {
            None
        } else {
            let i = self.position;
            let j = (i + self.chunk_size).min(self.length);
            self.position = j;
            Some(self.data.read_slice_1d(i..j).unwrap())
        }
    }
}

pub fn read_chunks_1d<T>(dataset: &Dataset) -> Chunks<'_, T> {
    assert!(dataset.ndim() <= 1);
    let length = dataset.size();
    Chunks {
        data: dataset,
        chunk_size: dataset.chunk().map_or(length, |x| x[0]),
        position: 0,
        length,
        phantom: PhantomData,
    }
}
