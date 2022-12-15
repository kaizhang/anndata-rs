use crate::backend::{Backend, GroupOp, DataContainer, iter_containers};
use crate::data::{Data, ReadData, WriteData};

use std::collections::HashMap;
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct Mapping(HashMap<String, Data>);

impl WriteData for Mapping {
    fn write<B: Backend, G: GroupOp<Backend = B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
        let group = location.create_group(name)?;
        self.0
            .iter()
            .try_for_each(|(k, v)| v.write(&group, k).map(|_| ()))?;
        Ok(DataContainer::Group(group))
    }
}

impl ReadData for Mapping {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let data: Result<_> = iter_containers::<B>(container.as_group()?).map(|(k, v)| {
            Ok((k.to_owned(), Data::read(&v)?))
        }).collect();
        Ok(Mapping(data?))
    }
}