use crate::anndata_trait::*;

use std::collections::HashMap;
use hdf5::{File, Result, Group}; 

pub trait BoxedData {
    fn new(container: DataContainer) -> Result<Self> where Self: Sized;
    fn write(&self, location: &Group, name: &str) -> Result<()>;
}

pub trait BoxedDataSubRow {
    fn subset_rows(&self, idx: &[usize]) -> Self where Self: Sized;
}

pub trait BoxedDataSubCol {
    fn subset_cols(&self, idx: &[usize]) -> Self where Self: Sized;
}

pub trait BoxedDataSub2D: BoxedDataSubRow + BoxedDataSubCol {
    fn subset(&self, ridx: &[usize], cidx: &[usize]) -> Self where Self: Sized;
}

pub struct AnnDataBase<X, O, V> {
    file: File,
    pub x: X,
    pub obs: O,
    pub obsm: HashMap<String, O>,
    pub var: V,
    pub varm: HashMap<String, V>,
}

impl<X, O, V> AnnDataBase<X, O, V> {
    pub fn read(file: File) -> Result<Self>
    where
        X: BoxedData,
        O: BoxedData,
        V: BoxedData,
    {
        let x = BoxedData::new(DataContainer::H5Group(file.group("X")?))?;
        let obs = BoxedData::new(DataContainer::H5Group(file.group("obs")?))?;
        let obsm = file.group("obsm").as_ref().map_or(Ok(HashMap::new()), |group|
            get_all_data(group))?;
        let var = BoxedData::new(DataContainer::H5Group(file.group("var")?))?;
        let varm = file.group("varm").as_ref().map_or(Ok(HashMap::new()), |group|
            get_all_data(group))?;
        Ok(Self { file, x, obs, obsm, var, varm })
    }

    pub fn write(&self, filename: &str) -> Result<()>
    where
        X: BoxedData,
        O: BoxedData,
        V: BoxedData,
    {
        let file = File::create(filename)?;
        self.x.write(&file, "X")?;
        self.obs.write(&file, "obs")?;
        self.var.write(&file, "var")?;
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

    pub fn subset_obs(&self, idx: &[usize]) -> Self
    where
        X: BoxedDataSubRow,
        O: BoxedDataSubRow,
        V: Clone,
    {
        AnnDataBase {
            file: self.file.clone(),
            x: self.x.subset_rows(idx),
            obs: self.obs.subset_rows(idx),
            obsm: self.obsm.iter().map(|(k, v)| (k.clone(), v.subset_rows(idx))).collect(),
            var: self.var.clone(),
            varm: self.varm.clone(),
        }
    }

    pub fn subset_var(&self, idx: &[usize]) -> Self
    where
        X: BoxedDataSubCol,
        O: Clone,
        V: BoxedDataSubCol,
    {
        AnnDataBase {
            file: self.file.clone(),
            x: self.x.subset_cols(idx),
            obs: self.obs.clone(),
            obsm: self.obsm.clone(),
            var: self.var.subset_cols(idx),
            varm: self.varm.iter().map(|(k, v)| (k.clone(), v.subset_cols(idx))).collect(),
        }
    }

    pub fn subset(&self, ridx: &[usize], cidx: &[usize]) -> Self
    where
        X: BoxedDataSub2D,
        O: BoxedDataSubRow,
        V: BoxedDataSubCol,
    {
        AnnDataBase {
            file: self.file.clone(),
            x: self.x.subset(ridx, cidx),
            obs: self.obs.subset_rows(ridx),
            obsm: self.obsm.iter().map(|(k, v)| (k.clone(), v.subset_rows(ridx))).collect(),
            var: self.var.subset_cols(cidx),
            varm: self.varm.iter().map(|(k, v)| (k.clone(), v.subset_cols(cidx))).collect(),
        }
    }
}

fn get_all_data<B: BoxedData>(group: &Group) -> Result<HashMap<String, B>> {
    let get_name = |x: String| std::path::Path::new(&x).file_name()
        .unwrap().to_str().unwrap().to_string();
    Ok(group.groups()?.into_iter().map(|x|
        (get_name(x.name()), BoxedData::new(DataContainer::H5Group(x)).unwrap())
    ).chain(group.datasets()?.into_iter().map(|x|
        (get_name(x.name()), BoxedData::new(DataContainer::H5Dataset(x)).unwrap())
    )).collect())
}