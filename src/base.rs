use crate::anndata_trait::*;
use crate::element::MatrixElem;

use std::collections::HashMap;
use hdf5::{File, Result, Group}; 

pub struct AnnData {
    file: File,
    pub n_obs: usize,
    pub n_vars: usize,
    pub x: Option<MatrixElem>,
    pub obs: Option<MatrixElem>,
    pub obsm: HashMap<String, MatrixElem>,
    pub var: Option<MatrixElem>,
    pub varm: HashMap<String, MatrixElem>,
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

    pub fn new(filename: &str, n_obs: usize, n_vars: usize) -> Result<Self> {
        let file = hdf5::File::create_excl(filename)?;
        Ok(Self { file, n_obs, n_vars, x: None, obs: None,
            obsm: HashMap::new(), var: None, varm: HashMap::new(),
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
            let obs = MatrixElem::new(DataContainer::open(&file, "obs")?)?;
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
        let obsm = file.group("obsm").as_ref().map_or(Ok(HashMap::new()), |group|
            get_all_data(group))?;
        for (k, v) in obsm.iter() {
            if n_obs.is_none() { n_obs = Some(v.nrows()); }
            assert!(n_obs.unwrap() == v.nrows(), 
                "Inconsistent number of observations: {} (X) != {} ({})",
                n_obs.unwrap(), v.nrows(), k,
            );
        }

        // Read var
        let var = if file.link_exists("var") {
            let var = MatrixElem::new(DataContainer::open(&file, "var")?)?;
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
        let varm = file.group("varm").as_ref().map_or(Ok(HashMap::new()), |group|
            get_all_data(group))?;
        for (k, v) in varm.iter() {
            if n_vars.is_none() { n_vars = Some(v.ncols()); }
            assert!(n_vars.unwrap() == v.ncols(), 
                "Inconsistent number of variables: {} (X) != {} ({})",
                n_vars.unwrap(), v.ncols(), k,
            );
        }

        Ok(Self {
            file,
            n_obs: n_obs.unwrap_or(0),
            n_vars: n_vars.unwrap_or(0),
            x, obs, obsm, var, varm
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

    pub fn subset_obs(&self, idx: &[usize]) -> Self
    {
        Self {
            file: self.file.clone(),
            n_obs: idx.len(),
            n_vars: self.n_vars,
            x: self.x.as_ref().map(|x| x.subset_rows(idx)),
            obs: self.obs.as_ref().map(|x| x.subset_rows(idx)),
            obsm: self.obsm.iter().map(|(k, v)| (k.clone(), v.subset_rows(idx))).collect(),
            var: self.var.clone(),
            varm: self.varm.clone(),
        }
    }

    pub fn subset_var(&self, idx: &[usize]) -> Self
    {
        Self {
            file: self.file.clone(),
            n_obs: self.n_obs,
            n_vars: idx.len(),
            x: self.x.as_ref().map(|x| x.subset_cols(idx)),
            obs: self.obs.clone(),
            obsm: self.obsm.clone(),
            var: self.var.as_ref().map(|x| x.subset_cols(idx)),
            varm: self.varm.iter().map(|(k, v)| (k.clone(), v.subset_cols(idx))).collect(),
        }
    }

    pub fn subset(&self, ridx: &[usize], cidx: &[usize]) -> Self
    {
        Self {
            file: self.file.clone(),
            n_obs: ridx.len(),
            n_vars: cidx.len(),
            x: self.x.as_ref().map(|x| x.subset(ridx, cidx)),
            obs: self.obs.as_ref().map(|x| x.subset_rows(ridx)),
            obsm: self.obsm.iter().map(|(k, v)| (k.clone(), v.subset_rows(ridx))).collect(),
            var: self.var.as_ref().map(|x| x.subset_cols(cidx)),
            varm: self.varm.iter().map(|(k, v)| (k.clone(), v.subset_cols(cidx))).collect(),
        }
    }
}

fn get_all_data(group: &Group) -> Result<HashMap<String, MatrixElem>> {
    let get_name = |x: String| std::path::Path::new(&x).file_name()
        .unwrap().to_str().unwrap().to_string();
    Ok(group.groups()?.into_iter().map(|x|
        (get_name(x.name()), MatrixElem::new(DataContainer::H5Group(x)).unwrap())
    ).chain(group.datasets()?.into_iter().map(|x|
        (get_name(x.name()), MatrixElem::new(DataContainer::H5Dataset(x)).unwrap())
    )).collect())
}