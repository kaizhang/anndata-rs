use crate::{
    anndata_trait::*,
    element::*,
};

use std::sync::Arc;
use parking_lot::Mutex;
use std::collections::HashMap;
use hdf5::File; 
use anyhow::{anyhow, Result};
use polars::prelude::{NamedFrom, DataFrame, Series};
use std::ops::Deref;
use std::ops::DerefMut;
use indexmap::map::IndexMap;
use itertools::Itertools;
use paste::paste;

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
    pub fn n_obs(&self) -> usize { *self.n_obs.lock().deref() }

    pub fn n_vars(&self) -> usize { *self.n_vars.lock().deref() }

    pub fn obs_names(&self) -> Result<Vec<String>> { Ok(self.obs.get_index()?) }

    pub fn var_names(&self) -> Result<Vec<String>> { Ok(self.var.get_index()?) }

    pub fn set_n_obs(&self, n: usize) {
        let mut n_obs = self.n_obs.lock();
        if *n_obs != n {
            let obs_is_empty = self.obs.is_empty() && self.x.is_empty()
                && self.obsm.is_empty() && self.obsp.is_empty();
            if obs_is_empty {
                *n_obs = n;
            } else {
                panic!("obs, obsm, obsp, X must be empty so that we can change n_obs");
            }
        }
    }

    pub fn set_n_vars(&self, n: usize) {
        let mut n_vars = self.n_vars.lock();
        if *n_vars != n {
            let var_is_empty = self.var.is_empty() && self.x.is_empty()
                && self.varm.is_empty() && self.varp.is_empty();
            if var_is_empty {
                *n_vars = n;
            } else {
                panic!("var, varm, varp, X must be empty so that we can change n_vars");
            }
        }
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

    pub fn get_x(&self) -> &MatrixElem { &self.x }
    pub fn get_obs(&self) -> &DataFrameElem { &self.obs }
    pub fn get_var(&self) -> &DataFrameElem { &self.var }
    pub fn get_obsm(&self) -> &Slot<AxisArrays> { &self.obsm }
    pub fn get_obsp(&self) -> &Slot<AxisArrays> { &self.obsp }
    pub fn get_varm(&self) -> &Slot<AxisArrays> { &self.varm }
    pub fn get_varp(&self) -> &Slot<AxisArrays> { &self.varp }
    pub fn get_uns(&self) -> &Slot<ElemCollection> { &self.uns }

    pub fn set_x(&self, data_: Option<&Box<dyn DataPartialIO>>) -> Result<()> {
        let mut x_guard = self.x.inner();
        match data_ {
            Some(data) => {
                self.set_n_obs(data.nrows());
                self.set_n_vars(data.ncols());
                if x_guard.0.is_some() { self.file.unlink("X")?; }
                *x_guard.0 = Some(RawMatrixElem::new(data.write(&self.file, "X")?)?);
            },
            None => if x_guard.0.is_some() {
                self.file.unlink("X")?;
                *x_guard.0 = None;
            },
        }
        Ok(())
    }

    pub fn set_obs(&self, obs_: Option<&DataFrame>) -> Result<()> {
        let mut obs_guard = self.obs.inner();
        match obs_ {
            None => if obs_guard.0.is_some() {
                self.file.unlink("obs")?;
                *obs_guard.0 = None;
            },
            Some(obs) => {
                self.set_n_obs(obs.nrows());
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
        let mut var_guard = self.var.inner();
        match var_ {
            None => if var_guard.0.is_some() {
                self.file.unlink("var")?;
                *var_guard.0 = None;
            },
            Some(var) => {
                self.set_n_vars(var.nrows());
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

    pub fn subset_obs(&self, idx: &[usize])
    {
        self.x.inner().0.as_mut().map(|x| x.subset_rows(idx));
        self.obs.inner().0.as_mut().map(|x| x.subset_rows(idx));
        self.obsm.inner().0.as_mut().map(|x| x.subset(idx));
        self.obsp.inner().0.as_mut().map(|x| x.subset(idx));
        *self.n_obs.lock() = idx.len();
    }

    pub fn subset_var(&self, idx: &[usize])
    {
        self.x.inner().0.as_mut().map(|x| x.subset_cols(idx));
        self.var.inner().0.as_mut().map(|x| x.subset_cols(idx));
        self.varm.inner().0.as_mut().map(|x| x.subset(idx));
        self.varp.inner().0.as_mut().map(|x| x.subset(idx));
        *self.n_vars.lock() = idx.len();
    }

    pub fn subset(&self, ridx: &[usize], cidx: &[usize])
    {
        self.x.inner().0.as_mut().map(|x| x.subset(ridx, cidx));
        self.obs.inner().0.as_mut().map(|x| x.subset_rows(ridx));
        self.obsm.inner().0.as_mut().map(|x| x.subset(ridx));
        self.obsp.inner().0.as_mut().map(|x| x.subset(ridx));
        self.var.inner().0.as_mut().map(|x| x.subset_cols(cidx));
        self.varm.inner().0.as_mut().map(|x| x.subset(cidx));
        self.varp.inner().0.as_mut().map(|x| x.subset(cidx));
        *self.n_obs.lock() = ridx.len();
        *self.n_vars.lock() = cidx.len();
    }
}

pub struct StackedAnnData {
    anndatas: IndexMap<String, AnnData>,
    pub x: Stacked<MatrixElem>,
    pub obsm: StackedAxisArrays,
}

impl std::fmt::Display for StackedAnnData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stacked AnnData objects:")?;
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
        let x = Stacked::new(adatas.values().map(|x| x.get_x().clone()).collect())?;
        let obsm = if adatas.values().any(|x| x.obsm.is_empty()) {
            Ok(StackedAxisArrays { axis: Axis::Row, data: HashMap::new() })
        } else {
            let obsm_guard: Vec<_> = adatas.values().map(|x| x.obsm.inner()).collect();
            StackedAxisArrays::new(obsm_guard.iter().map(|x| x.deref()).collect())
        }?;
        Ok(Self { anndatas: adatas, x, obsm })
    }

    pub fn len(&self) -> usize { self.anndatas.len() }

    pub fn keys(&self) -> indexmap::map::Keys<'_, String, AnnData> { self.anndatas.keys() }

    pub fn iter(&self) -> indexmap::map::Iter<'_, String, AnnData> {
        self.anndatas.iter()
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

    pub fn read(file: File, adata_files_: Option<HashMap<&str, &str>>) -> Result<Self> {
        let annotation = AnnData::read(file)?;
        let df: Box<DataFrame> = annotation.get_uns().inner().get_mut("AnnDataSet").unwrap()
            .read()?.into_any().downcast().unwrap();
        let keys = df.column("keys").unwrap().utf8().unwrap();
        let filenames = df.column("file_path").unwrap().utf8().unwrap();
        let adata_files = adata_files_.unwrap_or(HashMap::new());
        let anndatas = keys.into_iter().zip(filenames).map(|(k, v)| {
                let f = adata_files.get(k.unwrap()).map_or(v.unwrap(), |x| *x);
                Ok((k.unwrap().to_string(), AnnData::read(File::open(f)?)?))
            }).collect::<Result<IndexMap<_, _>>>()?;
        Ok(Self {
            annotation,
            anndatas: Slot::new(StackedAnnData::new(anndatas)?),
        })
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

    pub fn close(self) -> Result<()> {
        self.annotation.close()?;
        if let Some(anndatas) = self.anndatas.extract() {
            for ann in anndatas.anndatas.into_values() { ann.close()?; }
        }
        Ok(())
    }
}