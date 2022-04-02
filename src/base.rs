use crate::anndata_trait::*;
use crate::element::{Elem, MatrixElem, MatrixElemOptional, DataFrameElem};
use std::sync::{Arc, Mutex};

use std::collections::HashMap;
use hdf5::{File, Result, Group}; 
use polars::frame::DataFrame;
use std::ops::Deref;
use std::ops::DerefMut;

#[derive(Clone)]
pub struct AnnData {
    pub(crate) file: File,
    pub n_obs: Arc<Mutex<usize>>,
    pub n_vars: Arc<Mutex<usize>>,
    pub x: MatrixElemOptional,
    pub obs: DataFrameElem,
    pub obsm: HashMap<String, MatrixElem>,
    pub obsp: HashMap<String, MatrixElem>,
    pub var: DataFrameElem,
    pub varm: HashMap<String, MatrixElem>,
    pub varp: HashMap<String, MatrixElem>,
    pub uns: HashMap<String, Elem>,
}

impl AnnData {
    pub fn n_obs(&self) -> usize { *self.n_obs.lock().unwrap().deref() }

    pub fn set_n_obs(&self, n: usize) {
        *self.n_obs.lock().unwrap().deref_mut() = n;
    }

    pub fn n_vars(&self) -> usize { *self.n_vars.lock().unwrap().deref() }

    pub fn set_n_vars(&self, n: usize) {
        *self.n_vars.lock().unwrap().deref_mut() = n;
    }

    pub fn filename(&self) -> String { self.file.filename() }

    pub fn close(self) -> Result<()> { self.file.close() }

    pub fn set_x(&self, data: &Box<dyn WritePartialData>) -> Result<()> {
        assert!(
            self.n_obs() == data.nrows(),
            "Number of observations mismatched, expecting {}, but found {}",
            self.n_obs(), data.nrows(),
        );
        assert!(
            self.n_vars() == data.ncols(),
            "Number of variables mismatched, expecting {}, but found {}",
            self.n_vars(), data.ncols(),
        );
        if !self.x.is_empty() { self.file.unlink("X")?; }
        self.x.insert(data.write(&self.file, "X")?)?;
        Ok(())
    }

    pub fn set_obs(&self, obs: &DataFrame) -> Result<()> {
        assert!(
            self.n_obs() == obs.nrows(),
            "Number of observations mismatched, expecting {}, but found {}",
            self.n_obs(), obs.nrows(),
        );
        if self.obs.is_empty() {
            self.obs.insert(obs.write(&self.file, "obs")?)?;
        } else {
            self.obs.update(obs);
        }
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
            self.n_obs() == data.nrows(),
            "Number of observations mismatched, expecting {}, but found {}",
            self.n_obs(), data.nrows(),
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

    pub fn set_obsp(&mut self, obsp: &HashMap<String, Box<dyn WritePartialData>>) -> Result<()> {
        if self.file.group("obsp").is_ok() { self.file.unlink("obsp")?; }
        for (key, data) in obsp.iter() {
            self.add_obsp(key.as_str(), data)?;
        }
        Ok(())
    }

    pub fn add_obsp(&mut self, key: &str, data: &Box<dyn WritePartialData>) -> Result<()> {
        assert!(
            self.n_obs() == data.nrows(),
            "Number of observations mismatched, expecting {}, but found {}",
            self.n_obs(), data.nrows(),
        );
        assert!(
            data.ncols() == data.nrows(),
            "Not a square matrix, nrows: {}, ncols: {}",
            data.nrows(), data.ncols(),
        );

        let obsp = match self.file.group("obsp") {
            Ok(x) => x,
            _ => self.file.create_group("obsp").unwrap(),
        };
        if self.obsp.contains_key(key) { obsp.unlink(key)?; } 
        let container = data.write(&obsp, key)?;
        let elem = MatrixElem::new(container)?;
        self.obsp.insert(key.to_string(), elem);
        Ok(())
    }

    pub fn set_var(&self, var: &DataFrame) -> Result<()> {
        assert!(
            self.n_vars() == var.nrows(),
            "Number of variables mismatched, expecting {}, but found {}",
            self.n_vars(), var.nrows(),
        );
        if self.var.is_empty() {
            self.var.insert(var.write(&self.file, "var")?)?;
        } else {
            self.var.update(var);
        }
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
            self.n_vars() == data.ncols(),
            "Number of variables mismatched, expecting {}, but found {}",
            self.n_vars(), data.ncols(),
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

    pub fn set_varp(&mut self, varp: &HashMap<String, Box<dyn WritePartialData>>) -> Result<()> {
        if self.file.group("varp").is_ok() { self.file.unlink("varp")?; }
        for (key, data) in varp.iter() {
            self.add_varp(key.as_str(), data)?;
        }
        Ok(())
    }

    pub fn add_varp(&mut self, key: &str, data: &Box<dyn WritePartialData>) -> Result<()> {
        assert!(
            self.n_vars() == data.ncols(),
            "Number of variables mismatched, expecting {}, but found {}",
            self.n_vars(), data.ncols(),
        );
        assert!(
            data.ncols() == data.nrows(),
            "Not a square matrix, nrows: {}, ncols: {}",
            data.nrows(), data.ncols(),
        );

        let varp = match self.file.group("varp") {
            Ok(x) => x,
            _ => self.file.create_group("varp").unwrap(),
        };
        if self.varp.contains_key(key) { varp.unlink(key)?; } 
        let container = data.write(&varp, key)?;
        let elem = MatrixElem::new(container)?;
        self.varp.insert(key.to_string(), elem);
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
        let file = hdf5::File::create(filename)?;
        Ok(Self { file,
            n_obs: Arc::new(Mutex::new(n_obs)), n_vars: Arc::new(Mutex::new(n_vars)),
            x: MatrixElemOptional::empty(),
            obs: DataFrameElem::empty(), obsm: HashMap::new(), obsp: HashMap::new(),
            var: DataFrameElem::empty(), varm: HashMap::new(), varp: HashMap::new(),
            uns: HashMap::new(),
        })
    }

    pub fn read(file: File) -> Result<Self>
    {
        let mut n_obs = None;
        let mut n_vars = None;

        // Read X
        let x = if file.link_exists("X") {
            let x = MatrixElemOptional::new(DataContainer::open(&file, "X")?)?;
            n_obs = x.nrows();
            n_vars = x.ncols();
            x
        } else {
            MatrixElemOptional::empty()
        };

        // Read obs
        let obs = if file.link_exists("obs") {
            let obs = DataFrameElem::new(DataContainer::open(&file, "obs")?)?;
            if n_obs.is_none() { n_obs = obs.nrows(); }
            assert!(n_obs == obs.nrows(),
                "Inconsistent number of observations: {} (X) != {} (obs)",
                n_obs.unwrap(), obs.nrows().unwrap(),
            );
            obs
        } else {
            DataFrameElem::empty()
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

        // Read obsp
        let obsp = file.group("obsp").as_ref().map_or(HashMap::new(), |group|
            get_all_data(group).map(|(k, v)| (k, MatrixElem::new(v).unwrap())).collect()
        );
        for (k, v) in obsp.iter() {
            if n_obs.is_none() { n_obs = Some(v.nrows()); }
            assert!(n_obs.unwrap() == v.nrows(), 
                "Inconsistent number of observations: {} (X) != {} ({})",
                n_obs.unwrap(), v.nrows(), k,
            );
            assert!(v.ncols() == v.nrows(), 
                "Not a square matrix: {}", k
            );
        }

        // Read var
        let var = if file.link_exists("var") {
            let var = DataFrameElem::new(DataContainer::open(&file, "var")?)?;
            if n_vars.is_none() { n_vars = var.ncols(); }
            assert!(n_vars == var.ncols(),
                "Inconsistent number of variables: {} (X) != {} (var)",
                n_vars.unwrap(), var.ncols().unwrap(),
            );
            var
        } else {
            DataFrameElem::empty()
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

        // Read varp
        let varp = file.group("varp").as_ref().map_or(HashMap::new(), |group|
            get_all_data(group).map(|(k, v)| (k, MatrixElem::new(v).unwrap())).collect()
        );
        for (k, v) in varp.iter() {
            if n_vars.is_none() { n_vars = Some(v.ncols()); }
            assert!(n_vars.unwrap() == v.ncols(), 
                "Inconsistent number of variables: {} (X) != {} ({})",
                n_vars.unwrap(), v.ncols(), k,
            );
            assert!(v.ncols() == v.nrows(), 
                "Not a square matrix: {}", k
            );
        }

        // Read uns
        let uns = file.group("uns").as_ref().map_or(HashMap::new(), |group|
            get_all_data(group).map(|(k, v)| (k, Elem::new(v).unwrap())).collect()
        );

        Ok(Self {
            file,
            n_obs: Arc::new(Mutex::new(n_obs.unwrap_or(0))),
            n_vars: Arc::new(Mutex::new(n_vars.unwrap_or(0))),
            x, obs, obsm, obsp, var, varm, varp, uns,
        })
    }

    pub fn write(&self, filename: &str) -> Result<()>
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

    pub fn subset_obs(&mut self, idx: &[usize])
    {
        self.set_n_obs(idx.len());
        self.x.subset_rows(idx);
        self.obs.subset_rows(idx);
        self.obsm.values_mut().for_each(|obsm| obsm.subset_rows(idx));
        self.obsp.values_mut().for_each(|obsp| obsp.subset(idx, idx));
    }

    pub fn subset_var(&mut self, idx: &[usize])
    {
        self.set_n_vars(idx.len());
        self.x.subset_cols(idx);
        self.var.subset_cols(idx);
        self.varm.values_mut().for_each(|varm| varm.subset_cols(idx));
        self.varp.values_mut().for_each(|varp| varp.subset(idx, idx));
    }

    pub fn subset(&mut self, ridx: &[usize], cidx: &[usize])
    {
        self.set_n_obs(ridx.len());
        self.set_n_vars(cidx.len());
        self.x.subset(ridx, cidx);
        self.obs.subset_rows(ridx);
        self.obsm.values_mut().for_each(|obsm| obsm.subset_rows(ridx));
        self.obsp.values_mut().for_each(|obsp| obsp.subset(ridx, ridx));
        self.var.subset_cols(cidx);
        self.varm.values_mut().for_each(|varm| varm.subset_cols(cidx));
        self.varp.values_mut().for_each(|varp| varp.subset(cidx, cidx));
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