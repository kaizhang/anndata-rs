use crate::anndata_trait::*;

use std::collections::HashMap;
use hdf5::{File, Result, Group}; 
use std::boxed::Box;

pub trait BoxedData {
    fn new(container: Box<dyn DataContainer>) -> Result<Self> where Self: Sized;
    fn write(&self, location: &Group, name: &str) -> Result<()>;
}

pub trait BoxedDataSubRow {
    fn subset_rows(self, idx: &[usize]) -> Self where Self: Sized;
}

pub trait BoxedDataSubCol {
    fn subset_cols(self, idx: &[usize]) -> Self where Self: Sized;
}

pub trait BoxedDataSub2D: BoxedDataSubRow + BoxedDataSubCol {
    fn subset(self, ridx: &[usize], cidx: &[usize]) -> Self where Self: Sized;
}

pub struct AnnDataBase<X, O, V> {
    file: File,
    pub x: X,
    pub obsm: HashMap<String, O>,
    pub varm: HashMap<String, V>,
}

impl<X, O, V> AnnDataBase<X, O, V> {
    pub fn read(path: &str) -> Result<Self>
    where
        X: BoxedData,
        O: BoxedData,
        V: BoxedData,
    {
        let file = File::open(path)?;
        let x = BoxedData::new(Box::new(file.group("X")?))?;
        let obsm = file.group("obsm").as_ref().map_or(Ok(HashMap::new()), |group|
            get_all_data(group))?;
        let varm = file.group("varm").as_ref().map_or(Ok(HashMap::new()), |group|
            get_all_data(group))?;
        Ok(Self { file, x, obsm, varm })
    }

    pub fn write(&self, filename: &str) -> Result<()>
    where
        X: BoxedData,
        O: BoxedData,
        V: BoxedData,
    {
        let file = File::create(filename)?;
        self.x.write(&file, "X")?;
        let obsm = file.create_group("obsm")?;
        for (key, val) in self.obsm.iter() {
            val.write(&obsm, key)?;
        }
        let varm = file.create_group("varm")?;
        for (key, val) in self.varm.iter() {
            val.write(&varm, key)?;
        }
        Ok(())
    }

    pub fn subset_obs(mut self, idx: &[usize]) -> Self
    where
        X: BoxedDataSubRow,
        O: BoxedDataSubRow,
    {
        AnnDataBase {
            file: self.file,
            x: self.x.subset_rows(idx),
            obsm: self.obsm.drain().map(|(k, v)| (k, v.subset_rows(idx))).collect(),
            varm: self.varm,
        }
    }

    pub fn subset_var(mut self, idx: &[usize]) -> Self
    where
        X: BoxedDataSubCol,
        V: BoxedDataSubCol,
    {
        AnnDataBase {
            file: self.file,
            x: self.x.subset_cols(idx),
            obsm: self.obsm,
            varm: self.varm.drain().map(|(k, v)| (k, v.subset_cols(idx))).collect(),
        }
    }

    pub fn subset(mut self, ridx: &[usize], cidx: &[usize]) -> Self
    where
        X: BoxedDataSub2D,
        O: BoxedDataSubRow,
        V: BoxedDataSubCol,
    {
        AnnDataBase {
            file: self.file,
            x: self.x.subset(ridx, cidx),
            obsm: self.obsm.drain().map(|(k, v)| (k, v.subset_rows(ridx))).collect(),
            varm: self.varm.drain().map(|(k, v)| (k, v.subset_cols(cidx))).collect(),
        }
    }
}

fn get_all_data<B: BoxedData>(group: &Group) -> Result<HashMap<String, B>> {
    let get_name = |x: String| std::path::Path::new(&x).file_name()
        .unwrap().to_str().unwrap().to_string();
    Ok(group.groups()?.into_iter().map(|x|
        (get_name(x.name()), BoxedData::new(Box::new(x)).unwrap())
    ).chain(group.datasets()?.into_iter().map(|x|
        (get_name(x.name()), BoxedData::new(Box::new(x)).unwrap())
    )).collect())
}