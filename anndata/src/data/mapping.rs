use crate::backend::*;
use crate::data::data_traits::*;
use crate::data::Data;

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
        todo!()
    }
}