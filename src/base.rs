use crate::anndata_trait::*;
use crate::element::MatrixElem;

use std::collections::HashMap;
use hdf5::{File, Result, Group}; 

pub struct AnnData {
    file: File,
    pub n_obs: usize,
    pub n_var: usize,
    pub x: MatrixElem,
    pub obs: MatrixElem,
    pub obsm: HashMap<String, MatrixElem>,
    pub var: MatrixElem,
    pub varm: HashMap<String, MatrixElem>,
}

impl AnnData {
    pub fn set_x<D>(&mut self, data: &D) -> Result<()>
    where
        D: DataSubset2D,
    {
        assert!(
            self.n_obs == data.nrows(),
            "number of observations mismatched, expecting {}, but found {}",
            self.n_obs, data.nrows(),
        );
        assert!(
            self.n_var == data.ncols(),
            "number of variables mismatched, expecting {}, but found {}",
            self.n_var, data.ncols(),
        );
        self.file.unlink("X")?;
        let container = data.write(&self.file, "X")?;
        self.x = MatrixElem::new(container)?;
        Ok(())
    }

    pub fn read(file: File) -> Result<Self>
    {
        let x = MatrixElem::new(DataContainer::H5Group(file.group("X")?))?;
        let n_obs = x.nrows();
        let n_var = x.ncols();

        let obs = MatrixElem::new(DataContainer::H5Group(file.group("obs")?))?;
        assert!(n_obs == obs.nrows(),
            "inconsistent number of observations: {} (X) != {} (obs)", n_obs, obs.nrows(),
        );

        let obsm = file.group("obsm").as_ref().map_or(Ok(HashMap::new()), |group|
            get_all_data(group))?;
        for (k, v) in obsm.iter() {
            assert!(n_obs == v.nrows(), 
                "inconsistent number of observations: {} (X) != {} ({})",
                n_obs, v.nrows(), k,
            );
        }

        let var = MatrixElem::new(DataContainer::H5Group(file.group("var")?))?;
        assert!(n_var == var.ncols(),
            "inconsistent number of variables: {} (X) != {} (var)", n_var, var.ncols(),
        );

        let varm = file.group("varm").as_ref().map_or(Ok(HashMap::new()), |group|
            get_all_data(group))?;
        for (k, v) in varm.iter() {
            assert!(n_var == var.ncols(), 
                "inconsistent number of variables: {} (X) != {} ({})",
                n_var, var.ncols(), k,
            );
        }

        Ok(Self { file, n_obs, n_var, x, obs, obsm, var, varm })
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

    pub fn subset_obs(&self, idx: &[usize]) -> Self
    {
        Self {
            file: self.file.clone(),
            n_obs: idx.len(),
            n_var: self.n_var,
            x: self.x.subset_rows(idx),
            obs: self.obs.subset_rows(idx),
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
            n_var: idx.len(),
            x: self.x.subset_cols(idx),
            obs: self.obs.clone(),
            obsm: self.obsm.clone(),
            var: self.var.subset_cols(idx),
            varm: self.varm.iter().map(|(k, v)| (k.clone(), v.subset_cols(idx))).collect(),
        }
    }

    pub fn subset(&self, ridx: &[usize], cidx: &[usize]) -> Self
    {
        Self {
            file: self.file.clone(),
            n_obs: ridx.len(),
            n_var: cidx.len(),
            x: self.x.subset(ridx, cidx),
            obs: self.obs.subset_rows(ridx),
            obsm: self.obsm.iter().map(|(k, v)| (k.clone(), v.subset_rows(ridx))).collect(),
            var: self.var.subset_cols(cidx),
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