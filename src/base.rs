use crate::anndata_trait::*;
use crate::element::{Elem, MatrixElem, DataFrameElem};

use std::collections::HashMap;
use hdf5::{File, Result, Group}; 
use polars::frame::DataFrame;

#[derive(Clone)]
pub struct AnnData {
    file: File,
    pub n_obs: usize,
    pub n_vars: usize,
    pub x: Option<MatrixElem>,
    pub obs: Option<DataFrameElem>,
    pub obsm: HashMap<String, MatrixElem>,
    pub var: Option<DataFrameElem>,
    pub varm: HashMap<String, MatrixElem>,
    pub uns: HashMap<String, Elem>,
}

impl AnnData {
    pub fn set_x(&mut self, data: &Box<dyn WritePartialData>) -> Result<()> {
        assert!(
            self.n_obs == data.nrows(),
            "Number of observations mismatched, expecting {}, but found {}",
            self.n_obs, data.nrows(),
        );
        assert!(
            self.n_vars == data.ncols(),
            "Number of variables mismatched, expecting {}, but found {}",
            self.n_vars, data.ncols(),
        );
        if self.x.is_some() { self.file.unlink("X")?; }
        let container = data.write(&self.file, "X")?;
        self.x = Some(MatrixElem::new(container)?);
        Ok(())
    }

    pub fn set_obs(&mut self, obs: &DataFrame) -> Result<()> {
        assert!(
            self.n_obs == obs.nrows(),
            "Number of observations mismatched, expecting {}, but found {}",
            self.n_obs, obs.nrows(),
        );

        if self.file.group("obs").is_ok() { self.file.unlink("obs")?; }
        let container = obs.write(&self.file, "obs")?;
        self.obs = Some(DataFrameElem::new(container)?);
        Ok(())
    }

    pub fn set_obsm(&mut self, obsm: &HashMap<String, Box<dyn WritePartialData>>) -> Result<()> {
        if self.file.group("obsm").is_ok() { self.file.unlink("obsm")?; }
        for (key, data) in obsm.iter() {
            self.add_obsm(key.as_str(), data)?;
        }
        Ok(())
    }

    pub fn add_obsm(&mut self, key: &str, data: &Box<dyn WritePartialData>) -> Result<()> {
        assert!(
            self.n_obs == data.nrows(),
            "Number of observations mismatched, expecting {}, but found {}",
            self.n_obs, data.nrows(),
        );

        let obsm = match self.file.group("obsm") {
            Ok(x) => x,
            _ => self.file.create_group("obsm").unwrap(),
        };
        if self.obsm.contains_key(key) { obsm.unlink(key)?; } 
        let container = data.write(&obsm, key)?;
        let elem = MatrixElem::new(container)?;
        self.obsm.insert(key.to_string(), elem);
        Ok(())
    }

    pub fn set_var(&mut self, var: &DataFrame) -> Result<()> {
        assert!(
            self.n_vars == var.nrows(),
            "Number of variables mismatched, expecting {}, but found {}",
            self.n_vars, var.nrows(),
        );

        if self.file.group("var").is_ok() { self.file.unlink("var")?; }
        let container = var.write(&self.file, "var")?;
        self.var = Some(DataFrameElem::new(container)?);
        Ok(())
    }

    pub fn set_varm(&mut self, varm: &HashMap<String, Box<dyn WritePartialData>>) -> Result<()> {
        if self.file.group("varm").is_ok() { self.file.unlink("varm")?; }
        for (key, data) in varm.iter() {
            self.add_varm(key.as_str(), data)?;
        }
        Ok(())
    }

    pub fn add_varm(&mut self, key: &str, data: &Box<dyn WritePartialData>) -> Result<()> {
        assert!(
            self.n_vars == data.ncols(),
            "Number of variables mismatched, expecting {}, but found {}",
            self.n_vars, data.ncols(),
        );
        let varm = match self.file.group("varm") {
            Ok(x) => x,
            _ => self.file.create_group("varm").unwrap(),
        };
        if self.varm.contains_key(key) { varm.unlink(key)?; } 
        let container = data.write(&varm, key)?;
        let elem = MatrixElem::new(container)?;
        self.varm.insert(key.to_string(), elem);
        Ok(())
    }

    pub fn set_uns(&mut self, uns: &HashMap<String, Box<dyn WriteData>>) -> Result<()> {
        if self.file.group("uns").is_ok() { self.file.unlink("uns")?; }
        for (key, data) in uns.iter() {
            self.add_uns(key.as_str(), data)?;
        }
        Ok(())
    }

    pub fn add_uns(&mut self, key: &str, data: &Box<dyn WriteData>) -> Result<()> {
        let uns = match self.file.group("uns") {
            Ok(x) => x,
            _ => self.file.create_group("uns").unwrap(),
        };
        if self.uns.contains_key(key) { uns.unlink(key)?; } 
        let container = data.write(&uns, key)?;
        let elem = Elem::new(container)?;
        self.uns.insert(key.to_string(), elem);
        Ok(())
    }

    pub fn new(filename: &str, n_obs: usize, n_vars: usize) -> Result<Self> {
        let file = hdf5::File::create_excl(filename)?;
        Ok(Self { file, n_obs, n_vars, x: None, obs: None,
            obsm: HashMap::new(), var: None, varm: HashMap::new(),
            uns: HashMap::new(),
        })
    }

    pub fn read(file: File) -> Result<Self>
    {
        let mut n_obs = None;
        let mut n_vars = None;

        // Read X
        let x = if file.link_exists("X") {
            let x = MatrixElem::new(DataContainer::open(&file, "X")?)?;
            n_obs = Some(x.nrows());
            n_vars = Some(x.ncols());
            Some(x)
        } else {
            None
        };

        // Read obs
        let obs = if file.link_exists("obs") {
            let obs = DataFrameElem::new(DataContainer::open(&file, "obs")?)?;
            if n_obs.is_none() { n_obs = Some(obs.nrows()); }
            assert!(n_obs.unwrap() == obs.nrows(),
                "Inconsistent number of observations: {} (X) != {} (obs)",
                n_obs.unwrap(), obs.nrows(),
            );
            Some(obs)
        } else {
            None
        };

        // Read obsm
        let obsm = file.group("obsm").as_ref().map_or(HashMap::new(), |group|
            get_all_data(group).map(|(k, v)| (k, MatrixElem::new(v).unwrap())).collect()
        );
        for (k, v) in obsm.iter() {
            if n_obs.is_none() { n_obs = Some(v.nrows()); }
            assert!(n_obs.unwrap() == v.nrows(), 
                "Inconsistent number of observations: {} (X) != {} ({})",
                n_obs.unwrap(), v.nrows(), k,
            );
        }

        // Read var
        let var = if file.link_exists("var") {
            let var = DataFrameElem::new(DataContainer::open(&file, "var")?)?;
            if n_vars.is_none() { n_vars = Some(var.ncols()); }
            assert!(n_vars.unwrap() == var.ncols(),
                "Inconsistent number of variables: {} (X) != {} (var)",
                n_vars.unwrap(), var.ncols(),
            );
            Some(var)
        } else {
            None
        };

        // Read varm
        let varm = file.group("varm").as_ref().map_or(HashMap::new(), |group|
            get_all_data(group).map(|(k, v)| (k, MatrixElem::new(v).unwrap())).collect()
        );
        for (k, v) in varm.iter() {
            if n_vars.is_none() { n_vars = Some(v.ncols()); }
            assert!(n_vars.unwrap() == v.ncols(), 
                "Inconsistent number of variables: {} (X) != {} ({})",
                n_vars.unwrap(), v.ncols(), k,
            );
        }

        // Read uns
        let uns = file.group("uns").as_ref().map_or(HashMap::new(), |group|
            get_all_data(group).map(|(k, v)| (k, Elem::new(v).unwrap())).collect()
        );

        Ok(Self {
            file,
            n_obs: n_obs.unwrap_or(0),
            n_vars: n_vars.unwrap_or(0),
            x, obs, obsm, var, varm, uns,
        })
    }

    pub fn write(&self, filename: &str) -> Result<()>
    {
        let file = File::create(filename)?;

        if let Some(x) = &self.x { x.write(&file, "X")?; }
        if let Some(obs) = &self.obs { obs.write(&file, "obs")?; }
        if let Some(var) = &self.var { var.write(&file, "var")?; }
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

    pub fn subset_obs(&mut self, idx: &[usize])
    {
        self.n_obs = idx.len();
        self.x.as_mut().map(|x| x.subset_rows(idx));
        self.obs.as_mut().map(|x| x.subset_rows(idx));
        self.obsm.values_mut().for_each(|obsm| obsm.subset_rows(idx));
    }

    pub fn subset_var(&mut self, idx: &[usize])
    {
        self.n_vars = idx.len();
        self.x.as_mut().map(|x| x.subset_cols(idx));
        self.var.as_mut().map(|x| x.subset_cols(idx));
        self.varm.values_mut().for_each(|varm| varm.subset_cols(idx));
    }

    pub fn subset(&mut self, ridx: &[usize], cidx: &[usize])
    {
        self.n_obs = ridx.len();
        self.n_vars = cidx.len();
        self.x.as_mut().map(|x| x.subset(ridx, cidx));
        self.obs.as_mut().map(|x| x.subset_rows(ridx));
        self.obsm.values_mut().for_each(|obsm| obsm.subset_rows(ridx));
        self.var.as_mut().map(|x| x.subset_cols(cidx));
        self.varm.values_mut().for_each(|varm| varm.subset_cols(cidx));
    }
}

fn get_all_data(group: &Group) -> impl Iterator<Item=(String, DataContainer)> {
    let get_name = |x: String| std::path::Path::new(&x).file_name()
        .unwrap().to_str().unwrap().to_string();
    group.groups().unwrap().into_iter().map(move |x|
        (get_name(x.name()), DataContainer::H5Group(x))
    ).chain(group.datasets().unwrap().into_iter().map(move |x|
        (get_name(x.name()), DataContainer::H5Dataset(x))
    ))
}