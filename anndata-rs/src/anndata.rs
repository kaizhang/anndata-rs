use crate::{
    anndata_trait::*,
    element::*,
};

use std::sync::Arc;
use std::path::Path;
use parking_lot::Mutex;
use std::collections::HashMap;
use hdf5::File; 
use anyhow::{anyhow, Result};
use polars::prelude::{NamedFrom, DataFrame, Series};
use std::ops::Deref;
use indexmap::map::IndexMap;
use itertools::Itertools;
use paste::paste;
use rayon::iter::{
    ParallelIterator, IntoParallelIterator, IndexedParallelIterator,
    IntoParallelRefIterator,
};

#[derive(Clone)]
pub struct AnnData {
    pub(crate) file: File,
    pub n_obs: Arc<Mutex<usize>>,
    pub n_vars: Arc<Mutex<usize>>,
    pub(crate) x: MatrixElem,
    pub(crate) obs: DataFrameElem,
    pub(crate) obsm: Slot<AxisArrays>,
    pub(crate) obsp: Slot<AxisArrays>,
    pub(crate) var: DataFrameElem,
    pub(crate) varm: Slot<AxisArrays>,
    pub(crate) varp: Slot<AxisArrays>,
    pub(crate) uns: Slot<ElemCollection>,
}

impl std::fmt::Display for AnnData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AnnData object with n_obs x n_vars = {} x {} backed at '{}'",
            self.n_obs(), self.n_vars(), self.filename(),
        )?;

        macro_rules! fmt_df {
            ($($df:ident),*) => {
                $(
                if !self.$df.is_empty() {
                    write!(
                        f, "\n    {}: {}", stringify!($df),
                        self.$df.get_column_names().unwrap().join(", "),
                    )?;
                }
                )*
            }
        }
        fmt_df!(obs, var);

        macro_rules! fmt_item {
            ($($item:ident),*) => {
                $(
                if let Some($item) = self.$item.inner().0.as_ref() {
                    let data: String = $item.keys().
                        map(|x| x.as_str()).intersperse(", ").collect();
                    if !data.is_empty() {
                        write!(f, "\n    {}: {}", stringify!($item), data)?;
                    }
                }
                )*
            }
        }
        fmt_item!(obsm, obsp, varm, varp, uns);

        Ok(())
    }
}

macro_rules! anndata_setter_row {
    ($($field:ident),*) => {
        paste! {
            $(
            pub fn [<set_ $field>](
                &self,
                data_: Option<&HashMap<String, Box<dyn DataPartialIO>>>,
            ) -> Result<()>
            {
                let mut guard = self.$field.inner();
                let field = stringify!($field);
                if guard.0.is_some() { self.file.unlink(field)?; }
                match data_ {
                    None => { *guard.0 = None; },
                    Some(data) => {
                        let container = self.file.create_group(field)?;
                        let mut item = AxisArrays::new(container, Axis::Row, self.n_obs.clone());
                        for (key, val) in data.iter() {
                            item.add_data(key, val)?;
                        }
                        *guard.0 = Some(item);
                    },
                }
                Ok(())
            }
            )*
        }
    }
}

macro_rules! anndata_setter_col {
    ($($field:ident),*) => {
        paste! {
            $(
            pub fn [<set_ $field>](
                &self,
                data_: Option<&HashMap<String, Box<dyn DataPartialIO>>>,
            ) -> Result<()>
            {
                let mut guard = self.$field.inner();
                let field = stringify!($field);
                if guard.0.is_some() { self.file.unlink(field)?; }
                match data_ {
                    None => { *guard.0 = None; },
                    Some(data) => {
                        let container = self.file.create_group(field)?;
                        let mut item = AxisArrays::new(container, Axis::Column, self.n_vars.clone());
                        for (key, val) in data.iter() {
                            item.add_data(key, val)?;
                        }
                        *guard.0 = Some(item);
                    },
                }
                Ok(())
            }
            )*
        }
    }
}

impl AnnData {
    pub fn n_obs(&self) -> usize { *self.n_obs.lock() }

    pub fn n_vars(&self) -> usize { *self.n_vars.lock() }

    pub fn obs_names(&self) -> Result<Vec<String>> { Ok(self.obs.get_index()?) }

    pub fn var_names(&self) -> Result<Vec<String>> { Ok(self.var.get_index()?) }

    pub fn get_x(&self) -> &MatrixElem { &self.x }
    pub fn get_obs(&self) -> &DataFrameElem { &self.obs }
    pub fn get_var(&self) -> &DataFrameElem { &self.var }
    pub fn get_obsm(&self) -> &Slot<AxisArrays> { &self.obsm }
    pub fn get_obsp(&self) -> &Slot<AxisArrays> { &self.obsp }
    pub fn get_varm(&self) -> &Slot<AxisArrays> { &self.varm }
    pub fn get_varp(&self) -> &Slot<AxisArrays> { &self.varp }
    pub fn get_uns(&self) -> &Slot<ElemCollection> { &self.uns }

    pub fn set_x(&self, data_: Option<&Box<dyn DataPartialIO>>) -> Result<()> {
        match data_ {
            Some(data) => {
                self.set_n_obs(data.nrows());
                self.set_n_vars(data.ncols());
                if !self.x.is_empty() { self.file.unlink("X")?; }
                *self.x.inner().0 = Some(RawMatrixElem::new(data.write(&self.file, "X")?)?);
            },
            None => if !self.x.is_empty() {
                self.file.unlink("X")?;
                *self.x.inner().0 = None;
            },
        }
        Ok(())
    }

    pub fn set_n_obs(&self, n: usize) {
        let mut n_obs = self.n_obs.lock();
        if *n_obs != n {
            let obs_is_empty = self.obs.is_empty() && self.x.is_empty()
                && (self.obsm.is_empty() || self.obsm.inner().is_empty())
                && (self.obsp.is_empty() || self.obsp.inner().is_empty());
            if obs_is_empty {
                *n_obs = n;
            } else {
                panic!(
                    "fail to set n_obs to {}: \
                    obs, obsm, obsp, X must be empty so that we can change n_obs",
                    n,
                );
            }
        }
    }

    pub fn set_n_vars(&self, n: usize) {
        let mut n_vars = self.n_vars.lock();
        if *n_vars != n {
            let var_is_empty = self.var.is_empty() && self.x.is_empty()
                && (self.varm.is_empty() || self.varm.inner().is_empty())
                && (self.varp.is_empty() || self.varp.inner().is_empty());
            if var_is_empty {
                *n_vars = n;
            } else {
                panic!(
                    "fail to set n_vars to {}: \
                    var, varm, varp, X must be empty so that we can change n_vars",
                    n,
                );
            }
        }
    }

    pub fn set_obs(&self, obs_: Option<&DataFrame>) -> Result<()> {
        match obs_ {
            None => if !self.obs.is_empty() {
                self.file.unlink("obs")?;
                *self.obs.inner().0 = None;
            },
            Some(obs) => {
                self.set_n_obs(obs.nrows());
                let mut obs_guard = self.obs.inner();
                match obs_guard.0.as_mut() {
                    Some(x) => x.update(obs)?,
                    None => {
                        let mut elem = RawMatrixElem::<DataFrame>::new_elem(
                            obs.write(&self.file, "obs")?
                        )?;
                        elem.enable_cache();
                        *obs_guard.0 = Some(elem);
                    },
                }
            },
        }
        Ok(())
    }

    pub fn set_var(&self, var_: Option<&DataFrame>) -> Result<()> {
        match var_ {
            None => if !self.var.is_empty() {
                self.file.unlink("var")?;
                *self.var.inner().0 = None;
            },
            Some(var) => {
                self.set_n_vars(var.nrows());
                let mut var_guard = self.var.inner();
                match var_guard.0.as_mut() {
                    Some(x) => x.update(var)?,
                    None => {
                        let mut elem = RawMatrixElem::<DataFrame>::new_elem(
                            var.write(&self.file, "var")?
                        )?;
                        elem.enable_cache();
                        *var_guard.0 = Some(elem);
                    },
                }
            },
        }
        Ok(())
    }

    anndata_setter_row!(obsm, obsp);
    anndata_setter_col!(varm, varp);

    pub fn set_uns(&mut self, uns_: Option<&HashMap<String, Box<dyn DataIO>>>) -> Result<()> {
        let mut guard = self.uns.inner();
        if guard.0.is_some() { self.file.unlink("uns")?; }
        match uns_ {
            None => { *guard.0 = None; },
            Some(uns) => {
                let container = self.file.create_group("uns")?;
                let mut item = ElemCollection::new(container);
                for (key, data) in uns.iter() {
                    item.add_data(key, data)?;
                }
                *guard.0 = Some(item);
            },
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
        Ok(Self { file, n_obs, n_vars, x: Slot::empty(), uns: Slot::new(uns),
            obs: Slot::empty(), obsm: Slot::new(obsm), obsp: Slot::new(obsp),
            var: Slot::empty(), varm: Slot::new(varm), varp: Slot::new(varp),
        })
    }

    pub fn filename(&self) -> String { self.file.filename() }

    pub fn close(self) -> Result<()> {
        macro_rules! close {
            ($($name:ident),*) => {
                $(
                self.$name.inner().0.as_ref().map(|x| x.values().for_each(|x| x.drop()));
                self.$name.drop();
                )*
            };
        }
        self.x.drop();
        self.obs.drop();
        self.var.drop();
        self.uns.drop();
        close!(obsm, obsp, varm, varp);
        self.file.close()?;
        Ok(())
    }

    pub fn subset(&self, obs_idx: Option<&[usize]>, var_idx: Option<&[usize]>) {
        match (obs_idx, var_idx) {
            (Some(i), Some(j)) => self.x.inner().0.as_mut().map(|x| x.subset(i, j)),
            (Some(i), None) => self.x.inner().0.as_mut().map(|x| x.subset_rows(i)),
            (None, Some(j)) => self.x.inner().0.as_mut().map(|x| x.subset_cols(j)),
            (None, None) => None,
        };
        
        if let Some(i) = obs_idx {
            self.obs.subset_rows(i);
            self.obsm.inner().0.as_mut().map(|x| x.subset(i));
            self.obsp.inner().0.as_mut().map(|x| x.subset(i));
            *self.n_obs.lock() = i.len();
        }

        if let Some(j) = var_idx {
            self.var.subset_rows(j);
            self.varm.inner().0.as_mut().map(|x| x.subset(j));
            self.varp.inner().0.as_mut().map(|x| x.subset(j));
            *self.n_vars.lock() = j.len();
        }
    }
}

pub struct StackedAnnData {
    anndatas: IndexMap<String, AnnData>,
    accum: Arc<Mutex<AccumLength>>,
    n_obs: Arc<Mutex<usize>>,
    n_vars: Arc<Mutex<usize>>,
    pub x: Stacked<MatrixElem>,
    pub obs: StackedDataFrame,
    pub obsm: StackedAxisArrays,
}

impl std::fmt::Display for StackedAnnData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stacked AnnData objects:")?;
        let obs: String = self.obs.keys.iter()
            .map(|x| x.as_str()).intersperse(", ").collect();
        write!(f, "\n    obs: {}", obs)?;
        let obsm: String = self.obsm.data.keys()
            .map(|x| x.as_str()).intersperse(", ").collect();
        write!(f, "\n    obsm: {}", obsm)?;
        Ok(())
    }
}

impl StackedAnnData {
    pub fn new(adatas: IndexMap<String, AnnData>) -> Result<Self> {
        if !adatas.values().map(|x|
            if x.var.is_empty() { None } else { x.var_names().ok() }
            ).all_equal()
        {
            return Err(anyhow!("var names mismatch"));
        }

        let accum_: AccumLength = adatas.values().map(|x| x.n_obs()).collect();
        let n_obs = Arc::new(Mutex::new(accum_.size()));
        let accum = Arc::new(Mutex::new(accum_));
        let n_vars = adatas.values().next().unwrap().n_vars.clone();

        let x = Stacked::new(
            adatas.values().map(|x| x.get_x().clone()).collect(),
            n_obs.clone(),
            n_vars.clone(),
            accum.clone(),
        )?;

        let obs = if adatas.values().any(|x| x.obs.is_empty()) {
            StackedDataFrame::new(Vec::new())
        } else {
            StackedDataFrame::new(adatas.values().map(|x| x.obs.clone()).collect())
        }?;

        let obsm = if adatas.values().any(|x| x.obsm.is_empty()) {
            Ok(StackedAxisArrays { axis: Axis::Row, data: HashMap::new() })
        } else {
            let obsm_guard: Vec<_> = adatas.values().map(|x| x.obsm.inner()).collect();
            StackedAxisArrays::new(
                obsm_guard.iter().map(|x| x.deref()).collect(),
                &n_obs,
                &n_vars,
                &accum,
            )
        }?;

        Ok(Self { anndatas: adatas, x, obs, obsm, accum, n_obs, n_vars })
    }

    pub fn len(&self) -> usize { self.anndatas.len() }

    pub fn keys(&self) -> indexmap::map::Keys<'_, String, AnnData> { self.anndatas.keys() }

    pub fn iter(&self) -> indexmap::map::Iter<'_, String, AnnData> {
        self.anndatas.iter()
    }

    fn write_subset<P>(
        &self,
        obs_idx: Option<&[usize]>,
        var_idx: Option<&[usize]>,
        dir: P,
    ) -> HashMap<String, String>
    where
        P: AsRef<Path> + std::marker::Sync,
    {
        let obs_idx_ = match obs_idx {
            Some(i) => self.accum.lock().normalize_indices(i),
            None => HashMap::new(),
        };
        self.anndatas.par_iter().enumerate().map(|(i, (k, data))| {
            let file = dir.as_ref().join(k.to_string() + ".h5ad");
            match obs_idx_.get(&i) {
                None => data.write_subset(Some(&[]), var_idx, file.clone()),
                Some(idx) => data.write_subset(Some(idx.as_slice()), var_idx, file.clone()),
            }.unwrap();
            (k.clone(), file.to_str().unwrap().to_string())
        }).collect()
    }

    /// Subsetting an AnnDataSet will not rearrange the data between
    /// AnnData objects.
    pub fn subset(&self, obs_idx: Option<&[usize]>, var_idx: Option<&[usize]>) {
        let mut accum_len = self.accum.lock();

        let obs_idx_ = match obs_idx {
            Some(i) => {
                *self.n_obs.lock() = i.len();
                accum_len.normalize_indices(i)
            },
            None => HashMap::new(),
        };
        if let Some(j) = var_idx {
            *self.n_vars.lock() = j.len();
        }
        self.anndatas.par_values().enumerate().for_each(|(i, data)|
            match obs_idx_.get(&i) {
                None => data.subset(Some(&[]), var_idx),
                Some(idx) => data.subset(Some(idx.as_slice()), var_idx),
            }
        );
        *accum_len = self.anndatas.values().map(|x| x.n_obs()).collect();
    }
}

pub struct AnnDataSet {
    annotation: AnnData,
    pub anndatas: Slot<StackedAnnData>,
}

impl std::fmt::Display for AnnDataSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let adatas = self.anndatas.inner();
        write!(
            f,
            "AnnDataSet object with n_obs x n_vars = {} x {} backed at '{}'",
            self.n_obs(),
            self.n_vars(),
            self.annotation.filename(),
        )?;
        write!(
            f,
            "\ncontains {} AnnData objects with keys: {}",
            adatas.0.as_ref().map_or(0, |x| x.len()),
            adatas.0.as_ref().unwrap().keys()
                .map(|x| x.as_str()).intersperse(", ").collect::<String>(),
        )?;

        if !self.annotation.obs.is_empty() {
            write!(f, "\n    obs: {}",
                self.annotation.obs.get_column_names().unwrap().join(", "),
            )?;
        }
        if !self.annotation.var.is_empty() {
            write!(f, "\n    var: {}",
                self.annotation.var.get_column_names().unwrap().join(", "),
            )?;
        }

        macro_rules! fmt_item {
            ($($item:ident),*) => {
                $(
                if let Some($item) = self.annotation.$item.inner().0.as_ref() {
                    let data: String = $item.keys().
                        map(|x| x.as_str()).intersperse(", ").collect();
                    if !data.is_empty() {
                        write!(f, "\n    {}: {}", stringify!($item), data)?;
                    }
                }
                )*
            }
        }
        fmt_item!(obsm, obsp, varm, varp);

        Ok(())
    }
}

macro_rules! def_accessor {
    ($get_type:ty, $set_type:ty, { $($field:ident),* }) => {
        paste! {
            $(
                pub fn [<get_ $field>](&self) -> &Slot<$get_type> {
                    &self.annotation.$field
                }

                pub fn [<set_ $field>](&mut self, $field: $set_type) -> Result<()> {
                    self.annotation.[<set_ $field>]($field)
                }
            )*
        }
    }
}


fn update_anndata_locations(ann: &AnnData, new_locations: HashMap<String, String>) -> Result<()> {
    let df: Box<DataFrame> = {
        ann.get_uns().inner().get_mut("AnnDataSet").unwrap().read()?
            .into_any().downcast().unwrap()
    };
    let keys = df.column("keys").unwrap();
    let filenames = df.column("file_path").unwrap().utf8()
        .unwrap().into_iter().collect::<Option<Vec<_>>>().unwrap();
    let new_files: Vec<_> = keys.utf8().unwrap().into_iter().zip(filenames)
        .map(|(k, v)| new_locations.get(k.unwrap()).map_or(
            v.to_string(), |x| x.clone()
        )).collect();
    let data: Box<dyn DataIO> = Box::new(DataFrame::new(vec![
        keys.clone(),
        Series::new("file_path", new_files),
    ]).unwrap());
    ann.get_uns().inner().add_data("AnnDataSet", &data)?;
    Ok(())
}

impl AnnDataSet {
    pub fn new(anndatas: IndexMap<String, AnnData>, filename: &str, add_key: &str) -> Result<Self> {
        // Compute n_obs and n_vars
        let n_obs = anndatas.values().map(|x| x.n_obs()).sum();
        let n_vars = anndatas.values().next().map(|x| x.n_vars()).unwrap_or(0);

        let annotation = AnnData::new(filename, n_obs, n_vars)?;

        // Set UNS. UNS includes children anndata locations.
        {
            let (keys, filenames): (Vec<_>, Vec<_>) = anndatas.iter()
                .map(|(k, v)| (k.clone(), v.filename())).unzip();
            let data: Box<dyn DataIO> = Box::new(DataFrame::new(vec![
                Series::new("keys", keys),
                Series::new("file_path", filenames),
            ]).unwrap());
            annotation.get_uns().inner().add_data("AnnDataSet", &data)?;
        }

        // Set OBS.
        {
            let keys = Series::new(
                add_key,
                anndatas.iter().map(|(k, v)| vec![k.clone(); v.n_obs()])
                .flatten().collect::<Series>(),
            );
            let df = if anndatas.values().all(|x| x.get_obs().is_empty()) {
                DataFrame::new(vec![keys]).unwrap()
            } else {
                DataFrame::new(vec![
                    Series::new(
                        anndatas.values().next().unwrap().obs.get_column_names()?[0].as_str(),
                        anndatas.values().map(|x| x.obs.get_index().unwrap())
                            .flatten().collect::<Series>(),
                    ),
                    keys,
                ]).unwrap()
            };
            annotation.set_obs(Some(&df))?;
        }

        // Set VAR.
        {
            let var = anndatas.values().next().unwrap().get_var();
            if !var.is_empty() {
                annotation.set_var(Some(&DataFrame::new(vec![
                    var.read()?[0].clone()
                ]).unwrap()))?;
            }
        }
        let stacked = StackedAnnData::new(anndatas)?;
        Ok(Self { annotation, anndatas: Slot::new(stacked), })
    }

    pub fn n_obs(&self) -> usize { self.annotation.n_obs() }

    pub fn n_vars(&self) -> usize { self.annotation.n_vars() }

    pub fn obs_names(&self) -> Result<Vec<String>> {
        self.annotation.obs_names()
    }

    pub fn var_names(&self) -> Result<Vec<String>> {
        self.annotation.var_names()
    }

    def_accessor!(
        RawMatrixElem<DataFrame>,
        Option<&DataFrame>,
        { obs, var }
    );

    def_accessor!(
        AxisArrays,
        Option<&HashMap<String, Box<dyn DataPartialIO>>>,
        { obsm, obsp, varm, varp }
    );

    def_accessor!(
        ElemCollection,
        Option<&HashMap<String, Box<dyn DataIO>>>,
        { uns }
    );

    pub fn read(file: File, adata_files_: Option<HashMap<&str, &str>>) -> Result<Self> {
        let annotation = AnnData::read(file)?;
        let df: Box<DataFrame> = annotation.get_uns().inner().get_mut("AnnDataSet").unwrap()
            .read()?.into_any().downcast().unwrap();
        let keys = df.column("keys").unwrap().utf8().unwrap()
            .into_iter().collect::<Option<Vec<_>>>().unwrap();
        let filenames = df.column("file_path").unwrap().utf8()
            .unwrap().into_iter().collect::<Option<Vec<_>>>().unwrap();
        let adata_files = adata_files_.unwrap_or(HashMap::new());
        let anndatas = keys.into_par_iter().zip(filenames).map(|(k, v)| {
                let f = adata_files.get(k).map_or(v, |x| *x);
                Ok((k.to_string(), AnnData::read(File::open(f)?)?))
            }).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            annotation,
            anndatas: Slot::new(StackedAnnData::new(
                anndatas.into_iter().collect()
            )?),
        })
    }

    /// Subsetting an AnnDataSet will not rearrange the data between
    /// AnnData objects.
    pub fn subset(&self, obs_idx: Option<&[usize]>, var_idx: Option<&[usize]>) {
        match self.anndatas.inner().0.deref() {
            None => self.annotation.subset(obs_idx, var_idx),
            Some(ann) => {
                let obs_idx_ = obs_idx.map(|x|
                    ann.accum.lock().sort_index_to_buckets(x)
                );
                let i = obs_idx_.as_ref().map(|x| x.as_slice());
                self.annotation.subset(i, var_idx);
                ann.subset(i, var_idx);
            }
        }
    }

    pub fn write_subset<P: AsRef<Path>>(
        &self,
        obs_idx: Option<&[usize]>,
        var_idx: Option<&[usize]>,
        dir: P,
    ) -> Result<()>
    {
        let file = dir.as_ref().join("_dataset.h5ads");
        let anndata_dir = dir.as_ref().join("anndatas");
        std::fs::create_dir_all(anndata_dir.clone())?;
        match self.anndatas.inner().0.deref() {
            None => self.annotation.write_subset(obs_idx, var_idx, file)?,
            Some(ann) => {
                let obs_idx_ = obs_idx.map(|x|
                    ann.accum.lock().sort_index_to_buckets(x)
                );
                let i = obs_idx_.as_ref().map(|x| x.as_slice());

                let filenames = ann.write_subset(i, var_idx, anndata_dir);
                let adata = self.annotation.copy_subset(i, var_idx, file)?;
                update_anndata_locations(&adata, filenames)?;
                adata.close()?;
            }
        }
        Ok(())
    }

    /// Copy and save the AnnDataSet to a new directory.
    /// This will copy all children AnnData files.
    pub fn copy_subset<P: AsRef<Path>>(
        &self,
        obs_idx: Option<&[usize]>,
        var_idx: Option<&[usize]>,
        dir: P
    ) -> Result<Self>
    {
        let file = dir.as_ref().join("_dataset.h5ads");
        self.write_subset(obs_idx, var_idx, dir)?;
        AnnDataSet::read(File::open_rw(file)?, None)
    }

    /// Copy and save the AnnDataSet to a new directory.
    /// This will copy all children AnnData files.
    pub fn copy<P: AsRef<Path>>(&self, dir: P) -> Result<Self> {
        let anndata_dir = dir.as_ref().join("anndatas");
        std::fs::create_dir_all(anndata_dir)?;
        todo!()
    }

    pub fn close(self) -> Result<()> {
        self.annotation.close()?;
        if let Some(anndatas) = self.anndatas.extract() {
            for ann in anndatas.anndatas.into_values() { ann.close()?; }
        }
        Ok(())
    }
}
