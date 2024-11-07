use crate::backend::{Backend, GroupOp, DataContainer, iter_containers, DataType};
use crate::data::{Data, Readable, Writable};

use std::collections::HashMap;
use std::ops::Deref;
use anyhow::Result;

use super::{Element, Encoding};

#[derive(Debug, Clone, PartialEq)]
pub struct Mapping(HashMap<String, Data>);

impl Into<HashMap<String, Data>> for Mapping {
    fn into(self) -> HashMap<String, Data> {
        self.0
    }
}

impl From<HashMap<String, Data>> for Mapping {
    fn from(data: HashMap<String, Data>) -> Self {
        Self(data)
    }
}

impl Deref for Mapping {
    type Target = HashMap<String, Data>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Element for Mapping {
    fn encoding(&self) -> Encoding {
        Encoding{
            encoding_type: "dict",
            version: "0.1.0",
            attributes: None,
        }
    }
}

impl Readable for Mapping {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let data: Result<_> = iter_containers::<B>(container.as_group()?).map(|(k, v)| {
            Ok((k.to_owned(), Data::read(&v)?))
        }).collect();
        Ok(Mapping(data?))
    }
}

impl Writable for Mapping {
    fn data_type(&self) -> DataType {
        DataType::Mapping
    }
    fn write<B: Backend, G: GroupOp<B>>(&self, location: &G, name: &str) -> Result<DataContainer<B>> {
        let group = location.new_group(name)?;
        self.0
            .iter()
            .try_for_each(|(k, v)| v.write(&group, k).map(|_| ()))?;
        Ok(DataContainer::Group(group))
    }
}