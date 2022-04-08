use crate::{
    anndata_trait::*,
    element::{
        ElemTrait, MatrixElem, DataFrameElem,
        ElemCollection, AxisArrays, Axis, Stacked,
    },
};

use std::sync::{Arc, Mutex};
use std::collections::{HashMap, HashSet};
use hdf5::{File, Result}; 
use polars::frame::DataFrame;
use std::ops::Deref;
use std::ops::DerefMut;
use indexmap::map::IndexMap;

#[derive(Clone)]
pub struct AnnData {
    pub(crate) file: File,
    pub n_obs: Arc<Mutex<usize>>,
    pub n_vars: Arc<Mutex<usize>>,
    pub x: Arc<Mutex<Option<MatrixElem>>>,
    pub obs: Arc<Mutex<Option<DataFrameElem>>>,
    pub obsm: AxisArrays,
    pub obsp: AxisArrays,
    pub var: Arc<Mutex<Option<DataFrameElem>>>,
    pub varm: AxisArrays,
    pub varp: AxisArrays,
    pub uns: ElemCollection,
}

/*
impl std::fmt::Display for AnnData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            &mut f,
            "AnnData object with n_obs x n_vars = {} x {} backed at '{}'",
            self.n_obs(),
            self.n_vars(),
            self.filename(),
        ).unwrap();

        if !self.obs.is_empty {
        if let Some(obs) = self.obs.0 {
            write!(
                &mut f,
                "\n    obs: {}",
                self.obs.0.get_column_names().unwrap().join(", "),
            ).unwrap();
        }
        if let Some(var) = self.get_var() {
            write!(
                &mut f,
                "\n    var: {}",
                var.0.get_column_names().unwrap().join(", "),
            ).unwrap();
        }

        let obsm = self.get_obsm().keys();
        if obsm.len() > 0 {
            write!(&mut descr, "\n    obsm: {}", obsm.join(", ")).unwrap();
        }
        let obsp = self.get_obsp().keys();
        if obsp.len() > 0 {
            write!(&mut descr, "\n    obsp: {}", obsp.join(", ")).unwrap();
        }
        let varm = self.get_varm().keys();
        if varm.len() > 0 {
            write!(&mut descr, "\n    varm: {}", varm.join(", ")).unwrap();
        }
        let varp = self.get_varp().keys();
        if varp.len() > 0 {
            write!(&mut descr, "\n    varp: {}", varp.join(", ")).unwrap();
        }
        let uns = self.get_uns().keys();
        if uns.len() > 0 {
            write!(&mut descr, "\n    uns: {}", uns.join(", ")).unwrap();
        }
        descr
    }






        let elem = self.0.lock().unwrap();
        write!(f, "Elem with {}, cache_enabled: {}, cached: {}",
            elem.dtype,
            if elem.cache_enabled { "yes" } else { "no" },
            if elem.element.is_some() { "yes" } else { "no" },
        )
    }
}
*/

impl AnnData {
    pub fn n_obs(&self) -> usize { *self.n_obs.lock().unwrap().deref() }

    pub(crate) fn set_n_obs(&self, n: usize) {
        *self.n_obs.lock().unwrap().deref_mut() = n;
    }

    pub fn n_vars(&self) -> usize { *self.n_vars.lock().unwrap().deref() }

    pub(crate) fn set_n_vars(&self, n: usize) {
        *self.n_vars.lock().unwrap().deref_mut() = n;
    }

    pub fn filename(&self) -> String { self.file.filename() }

    pub fn close(self) -> Result<()> { self.file.close() }

    pub fn set_x(&self, data_: Option<&Box<dyn DataPartialIO>>) -> Result<()> {
        let mut x_guard = self.x.lock().unwrap();
        match data_ {
            Some(data) => {
                let n = self.n_obs();
                let m = self.n_vars();
                assert!(
                    n == 0 || n == data.nrows(),
                    "Number of observations mismatched, expecting {}, but found {}",
                    n, data.nrows(),
                );
                assert!(
                    m == 0 || m == data.ncols(),
                    "Number of variables mismatched, expecting {}, but found {}",
                    m, data.ncols(),
                );
                if x_guard.is_some() { self.file.unlink("X")?; }
                *x_guard = Some(MatrixElem::new(data.write(&self.file, "X")?)?);
                self.set_n_obs(data.nrows());
                self.set_n_vars(data.ncols());
            },
            None => if x_guard.is_some() {
                self.file.unlink("X")?;
                *x_guard = None;
            },
        }
        Ok(())
    }

    pub fn set_obs(&self, obs_: Option<&DataFrame>) -> Result<()> {
        let mut obs_guard = self.obs.lock().unwrap();
        match obs_ {
            None => if obs_guard.is_some() {
                self.file.unlink("obs")?;
                *obs_guard = None;
            },
            Some(obs) => {
                let n = self.n_obs();
                assert!(
                    n == 0 || n == obs.nrows(),
                    "Number of observations mismatched, expecting {}, but found {}",
                    n, obs.nrows(),
                );
                match obs_guard.as_ref() {
                    Some(x) => x.update(obs),
                    None => {
                        *obs_guard = Some(DataFrameElem::new(obs.write(&self.file, "obs")?)?);
                    },
                }
                self.set_n_obs(obs.nrows());
            },
        }
        Ok(())
    }

    pub fn set_obsm(&mut self, obsm: &HashMap<String, Box<dyn DataPartialIO>>) -> Result<()> {
        if self.file.group("obsm").is_ok() { self.file.unlink("obsm")?; }
        self.obsm = {
            let container = self.file.create_group("obsm")?;
            AxisArrays::new(container, Axis::Row, self.n_obs.clone())
        };
        for (key, data) in obsm.iter() {
            self.obsm.insert(key, data)?;
        }
        Ok(())
    }

    pub fn set_obsp(&mut self, obsp: &HashMap<String, Box<dyn DataPartialIO>>) -> Result<()> {
        if self.file.group("obsp").is_ok() { self.file.unlink("obsp")?; }
        self.obsp = {
            let container = self.file.create_group("obsp")?;
            AxisArrays::new(container, Axis::Both, self.n_obs.clone())
        };
        for (key, data) in obsp.iter() {
            self.obsp.insert(key, data)?;
        }
        Ok(())
    }

    pub fn set_var(&self, var_: Option<&DataFrame>) -> Result<()> {
        let mut var_guard = self.var.lock().unwrap();
        match var_ {
            None => if var_guard.is_some() {
                self.file.unlink("var")?;
                *var_guard = None;
            },
            Some(var) => {
                let n = self.n_vars();
                assert!(
                    n == 0 || n == var.nrows(),
                    "Number of variables mismatched, expecting {}, but found {}",
                    n, var.nrows(),
                );
                match var_guard.as_ref() {
                    Some(x) => x.update(var),
                    None => {
                        *var_guard = Some(DataFrameElem::new(var.write(&self.file, "var")?)?);
                    },
                }
                self.set_n_vars(var.nrows());
            },
        }
        Ok(())
    }

    pub fn set_varm(&mut self, varm: &HashMap<String, Box<dyn DataPartialIO>>) -> Result<()> {
        if self.file.group("varm").is_ok() { self.file.unlink("varm")?; }
        self.varm = {
            let container = self.file.create_group("varm")?;
            AxisArrays::new(container, Axis::Column, self.n_vars.clone())
        };
        for (key, data) in varm.iter() {
            self.varm.insert(key, data)?;
        }
        Ok(())
    }

    pub fn set_varp(&mut self, varp: &HashMap<String, Box<dyn DataPartialIO>>) -> Result<()> {
        if self.file.group("varp").is_ok() { self.file.unlink("varp")?; }
        self.varp = {
            let container = self.file.create_group("varp")?;
            AxisArrays::new(container, Axis::Both, self.n_vars.clone())
        };
        for (key, data) in varp.iter() {
            self.varp.insert(key, data)?;
        }
        Ok(())
    }

    pub fn set_uns(&mut self, uns: &HashMap<String, Box<dyn DataIO>>) -> Result<()> {
        if self.file.group("uns").is_ok() { self.file.unlink("uns")?; }
        self.uns = {
            let container = self.file.create_group("uns")?;
            ElemCollection::new(container)
        };
        for (key, data) in uns.iter() {
            self.uns.insert(key, data)?;
        }
        Ok(())
    }

    pub fn new(filename: &str, n_obs: usize, n_vars: usize) -> Result<Self> {
        let file = hdf5::File::create(filename)?;
        let n_obs = Arc::new(Mutex::new(n_obs));
        let n_vars = Arc::new(Mutex::new(n_vars));
        let obsm = {
            let container = file.create_group("obsm")?;
            AxisArrays::new(container, Axis::Row, n_obs.clone())
        };
        let obsp = {
            let container = file.create_group("obsp")?;
            AxisArrays::new(container, Axis::Both, n_obs.clone())
        };
        let varm = {
            let container = file.create_group("varm")?;
            AxisArrays::new(container, Axis::Column, n_vars.clone())
        };
        let varp = {
            let container = file.create_group("varp")?;
            AxisArrays::new(container, Axis::Both, n_vars.clone())
        };
        let uns = {
            let container = file.create_group("uns")?;
            ElemCollection::new(container)
        };
        Ok(Self { file, n_obs, n_vars,
            x: Arc::new(Mutex::new(None)),
            obs: Arc::new(Mutex::new(None)),
            obsm, obsp,
            var: Arc::new(Mutex::new(None)),
            varm, varp,
            uns,
        })
    }

    pub fn subset_obs(&self, idx: &[usize])
    {
        self.x.lock().unwrap().as_ref().map(|x| x.subset_rows(idx));
        self.obs.lock().unwrap().as_ref().map(|x| x.subset_rows(idx));
        self.obsm.subset(idx);
        self.obsp.subset(idx);
        self.set_n_obs(idx.len());
    }

    pub fn subset_var(&self, idx: &[usize])
    {
        self.x.lock().unwrap().as_ref().map(|x| x.subset_cols(idx));
        self.var.lock().unwrap().as_ref().map(|x| x.subset_cols(idx));
        self.varm.subset(idx);
        self.varp.subset(idx);
        self.set_n_vars(idx.len());
    }

    pub fn subset(&self, ridx: &[usize], cidx: &[usize])
    {
        self.x.lock().unwrap().as_ref().map(|x| x.subset(ridx, cidx));
        self.obs.lock().unwrap().as_ref().map(|x| x.subset_rows(ridx));
        self.obsm.subset(ridx);
        self.obsp.subset(ridx);
        self.var.lock().unwrap().as_ref().map(|x| x.subset_cols(cidx));
        self.varm.subset(cidx);
        self.varp.subset(cidx);
        self.set_n_obs(ridx.len());
        self.set_n_vars(cidx.len());
    }
}

pub struct AnnDataSet {
    annotation: AnnData,
    pub x: Stacked<MatrixElem>,
    pub anndatas: IndexMap<String, AnnData>,
}

impl AnnDataSet {
    pub fn new(anndatas: IndexMap<String, AnnData>, filename: &str) -> Result<Self> {
        //if !anndatas.values().map(|x| x.var.read().unwrap().unwrap()[0]).all_equal() {
        //    panic!("var not equal");
        //}

        let n_obs = anndatas.values().map(|x| x.n_obs()).sum();
        let n_vars = anndatas.values().next().map(|x| x.n_vars()).unwrap_or(0);

        let annotation = AnnData::new(filename, n_obs, n_vars)?;
        if let Some(obs) = anndatas.values()
            .map(|x| x.obs.lock().unwrap().as_ref().map(|d| d.read().unwrap()[0].clone()))
            .collect::<Option<Vec<_>>>()
        {
            annotation.set_obs(Some(&DataFrame::new(vec![
            obs.into_iter().reduce(|mut accum, item| {
                accum.append(&item).unwrap();
                accum
            }).unwrap().clone()
            ]).unwrap()))?;
        }
        if let Some(var) = anndatas.values().next().unwrap().var.lock().unwrap().as_ref() {
            annotation.set_var(Some(&DataFrame::new(vec![var.read()?[0].clone()]).unwrap()))?;
        }
        
        /*
        let obs = intersections(anndatas.values().map(|x|
                x.obs.read().unwrap().unwrap().get_column_names().into_iter()
                .map(|s| s.to_string()).collect()).collect());
        let obsm = intersections(anndatas.values().map(
                |x| x.obsm.data.lock().unwrap().keys().map(Clone::clone).collect()).collect());
        */

        let x = Stacked::new(anndatas.values().map(|x|
            x.x.lock().unwrap().as_ref().unwrap().clone()).collect())?;

        Ok(Self { annotation, x, anndatas })
    }

    pub fn n_obs(&self) -> usize { self.annotation.n_obs() }

    pub fn n_vars(&self) -> usize { self.annotation.n_vars() }

    pub fn get_obsm(&self) -> &AxisArrays {
        &self.annotation.obsm
    }

    pub fn set_obsm(&mut self, obsm: &HashMap<String, Box<dyn DataPartialIO>>) -> Result<()> {
        self.annotation.set_obsm(obsm)
    }
}

fn intersections(mut sets: Vec<HashSet<String>>) -> HashSet<String> {
    {
        let (intersection, others) = sets.split_at_mut(1);
        let intersection = &mut intersection[0];
        for other in others {
            intersection.retain(|e| other.contains(e));
        }
    }
    sets[0].clone()
}