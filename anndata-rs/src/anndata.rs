use crate::{
    anndata_trait::*,
    element::{
        ElemTrait, MatrixElem, DataFrameElem,
        ElemCollection, AxisArrays, Axis, Stacked,
    },
};

use std::sync::Arc;
use parking_lot::{Mutex, MutexGuard};
use std::collections::{HashMap, HashSet};
use hdf5::{File, Result}; 
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
    pub x: Arc<Mutex<Option<MatrixElem>>>,
    pub(crate) obs: Arc<Mutex<Option<DataFrameElem>>>,
    pub(crate) obsm: Arc<Mutex<Option<AxisArrays>>>,
    pub(crate) obsp: Arc<Mutex<Option<AxisArrays>>>,
    pub(crate) var: Arc<Mutex<Option<DataFrameElem>>>,
    pub(crate) varm: Arc<Mutex<Option<AxisArrays>>>,
    pub(crate) varp: Arc<Mutex<Option<AxisArrays>>>,
    pub(crate) uns: Arc<Mutex<Option<ElemCollection>>>,
}

pub struct ElementGuard<'a, T>(pub MutexGuard<'a, Option<T>>);

impl<T> Deref for ElementGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &self.0.deref() {
            None => panic!("accessing an empty element"),
            Some(x) => x,
        }
    }
}

impl<T> DerefMut for ElementGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self.0.deref_mut() {
            None => panic!("accessing an empty element"),
            Some(ref mut x) => x,
        }
    }
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
                if let Some($df) = self.$df.lock().as_ref() {
                    write!(
                        f, "\n    {}: {}", stringify!($df),
                        $df.get_column_names().unwrap().join(", "),
                    )?;
                }
                )*
            }
        }
        fmt_df!(obs, var);

        macro_rules! fmt_item {
            ($($item:ident),*) => {
                $(
                if let Some($item) = self.$item.lock().as_ref() {
                    let data: String = $item.data.lock().keys().
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

/// define get_* functions
macro_rules! anndata_getter {
    ($get_type:ty, { $($field:ident),* }) => {
        paste! {
            $(
                pub fn [<get_ $field>](&self) -> ElementGuard<'_, $get_type> {
                    ElementGuard(self.$field.lock())
                }
            )*
        }
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
                let mut guard = self.$field.lock();
                let field = stringify!($field);
                if guard.is_some() { self.file.unlink(field)?; }
                match data_ {
                    None => { *guard = None; },
                    Some(data) => {
                        let container = self.file.create_group(field)?;
                        let item = AxisArrays::new(container, Axis::Row, self.n_obs.clone());
                        for (key, val) in data.iter() {
                            item.insert(key, val)?;
                        }
                        *guard = Some(item);
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
                let mut guard = self.$field.lock();
                let field = stringify!($field);
                if guard.is_some() { self.file.unlink(field)?; }
                match data_ {
                    None => { *guard = None; },
                    Some(data) => {
                        let container = self.file.create_group(field)?;
                        let item = AxisArrays::new(container, Axis::Column, self.n_vars.clone());
                        for (key, val) in data.iter() {
                            item.insert(key, val)?;
                        }
                        *guard = Some(item);
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

    pub(crate) fn set_n_obs(&self, n: usize) {
        *self.n_obs.lock().deref_mut() = n;
    }

    pub(crate) fn set_n_vars(&self, n: usize) {
        *self.n_vars.lock().deref_mut() = n;
    }

    pub fn filename(&self) -> String { self.file.filename() }

    pub fn close(self) -> Result<()> { self.file.close() }

    anndata_getter!(MatrixElem, { x });
    anndata_getter!(DataFrameElem, { obs, var });
    anndata_getter!(AxisArrays, { obsm, obsp, varm, varp });
    anndata_getter!(ElemCollection, { uns });

    pub fn set_x(&self, data_: Option<&Box<dyn DataPartialIO>>) -> Result<()> {
        let mut x_guard = self.x.lock();
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
        let mut obs_guard = self.obs.lock();
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

    pub fn set_var(&self, var_: Option<&DataFrame>) -> Result<()> {
        let mut var_guard = self.var.lock();
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

    anndata_setter_row!(obsm, obsp);
    anndata_setter_col!(varm, varp);

    pub fn set_uns(&mut self, uns_: Option<&HashMap<String, Box<dyn DataIO>>>) -> Result<()> {
        let mut guard = self.uns.lock();
        if guard.is_some() { self.file.unlink("uns")?; }
        match uns_ {
            None => { *guard = None; },
            Some(uns) => {
                let container = self.file.create_group("uns")?;
                let item = ElemCollection::new(container);
                for (key, data) in uns.iter() {
                    item.insert(key, data)?;
                }
                *guard = Some(item);
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
        Ok(Self { file, n_obs, n_vars,
            x: Arc::new(Mutex::new(None)),
            obs: Arc::new(Mutex::new(None)),
            obsm: Arc::new(Mutex::new(Some(obsm))),
            obsp: Arc::new(Mutex::new(Some(obsp))),
            var: Arc::new(Mutex::new(None)),
            varm: Arc::new(Mutex::new(Some(varm))),
            varp: Arc::new(Mutex::new(Some(varp))),
            uns: Arc::new(Mutex::new(Some(uns))),
        })
    }

    pub fn subset_obs(&self, idx: &[usize])
    {
        self.x.lock().as_ref().map(|x| x.subset_rows(idx));
        self.obs.lock().as_ref().map(|x| x.subset_rows(idx));
        self.obsm.lock().as_ref().map(|x| x.subset(idx));
        self.obsp.lock().as_ref().map(|x| x.subset(idx));
        self.set_n_obs(idx.len());
    }

    pub fn subset_var(&self, idx: &[usize])
    {
        self.x.lock().as_ref().map(|x| x.subset_cols(idx));
        self.var.lock().as_ref().map(|x| x.subset_cols(idx));
        self.varm.lock().as_ref().map(|x| x.subset(idx));
        self.varp.lock().as_ref().map(|x| x.subset(idx));
        self.set_n_vars(idx.len());
    }

    pub fn subset(&self, ridx: &[usize], cidx: &[usize])
    {
        self.x.lock().as_ref().map(|x| x.subset(ridx, cidx));
        self.obs.lock().as_ref().map(|x| x.subset_rows(ridx));
        self.obsm.lock().as_ref().map(|x| x.subset(ridx));
        self.obsp.lock().as_ref().map(|x| x.subset(ridx));
        self.var.lock().as_ref().map(|x| x.subset_cols(cidx));
        self.varm.lock().as_ref().map(|x| x.subset(cidx));
        self.varp.lock().as_ref().map(|x| x.subset(cidx));
        self.set_n_obs(ridx.len());
        self.set_n_vars(cidx.len());
    }
}

pub struct AnnDataSet {
    annotation: AnnData,
    pub x: Stacked<MatrixElem>,
    pub anndatas: IndexMap<String, AnnData>,
}

impl std::fmt::Display for AnnDataSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
            self.anndatas.len(),
            self.anndatas.keys().map(|x| x.as_str()).intersperse(", ").collect::<String>(),
        )?;

        if let Some(obs) = self.annotation.obs.lock().as_ref() {
            write!(
                f,
                "\n    obs: {}",
                obs.get_column_names().unwrap().join(", "),
            )?;
        }
        if let Some(var) = self.annotation.var.lock().as_ref() {
            write!(
                f,
                "\n    var: {}",
                var.get_column_names().unwrap().join(", "),
            )?;
        }

        macro_rules! fmt_item {
            ($($item:ident),*) => {
                $(
                if let Some($item) = self.annotation.$item.lock().as_ref() {
                    let data: String = $item.data.lock().keys().
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

macro_rules! def_accessor {
    ($get_type:ty, $set_type:ty, { $($field:ident),* }) => {
        paste! {
            $(
                pub fn [<get_ $field>](&self) -> ElementGuard<'_, $get_type> {
                    ElementGuard(self.annotation.$field.lock())
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
        //if !anndatas.values().map(|x| x.var.read().unwrap().unwrap()[0]).all_equal() {
        //    panic!("var not equal");
        //}

        let n_obs = anndatas.values().map(|x| x.n_obs()).sum();
        let n_vars = anndatas.values().next().map(|x| x.n_vars()).unwrap_or(0);

        let annotation = AnnData::new(filename, n_obs, n_vars)?;
        {
            let (keys, filenames): (Vec<_>, Vec<_>) = anndatas.iter()
                .map(|(k, v)| (k.clone(), v.filename())).unzip();
            let data: Box<dyn DataIO> = Box::new(DataFrame::new(vec![
                Series::new("keys", keys),
                Series::new("file_path", filenames),
            ]).unwrap());
            annotation.get_uns().insert("AnnDataSet", &data)?;
        }

        if let Some(obs) = anndatas.values()
            .map(|x| x.obs.lock().as_ref().map(|d| d.read().unwrap()[0].clone()))
            .collect::<Option<Vec<_>>>()
        {
            annotation.set_obs(Some(&DataFrame::new(vec![
                obs.into_iter().reduce(|mut accum, item| {
                    accum.append(&item).unwrap();
                    accum
                }).unwrap().clone(),
                Series::new(
                    add_key,
                    anndatas.iter().map(|(k, v)| vec![k.clone(); v.n_obs()])
                    .flatten().collect::<Series>(),
                ),
            ]).unwrap()))?;
        }
        if let Some(var) = anndatas.values().next().unwrap().var.lock().as_ref() {
            annotation.set_var(Some(&DataFrame::new(vec![var.read()?[0].clone()]).unwrap()))?;
        }
        
        /*
        let obs = intersections(anndatas.values().map(|x|
                x.obs.read().unwrap().unwrap().get_column_names().into_iter()
                .map(|s| s.to_string()).collect()).collect());
        let obsm = intersections(anndatas.values().map(
                |x| x.obsm.data.lock().unwrap().keys().map(Clone::clone).collect()).collect());
        */

        let x = Stacked::new(anndatas.values().map(|x| x.get_x().clone()).collect())?;

        Ok(Self { annotation, x, anndatas })
    }

    pub fn read(file: File, adata_files_: Option<HashMap<&str, &str>>) -> Result<Self> {
        let annotation = AnnData::read(file)?;
        let df: Box<DataFrame> = annotation.get_uns().data.lock()
            .get("AnnDataSet").unwrap().read()?.into_any().downcast().unwrap();
        let keys = df.column("keys").unwrap().utf8().unwrap();
        let filenames = df.column("file_path").unwrap().utf8().unwrap();
        let adata_files = adata_files_.unwrap_or(HashMap::new());
        let anndatas = keys.into_iter().zip(filenames).map(|(k, v)| {
                let f = adata_files.get(k.unwrap()).map_or(v.unwrap(), |x| *x);
                Ok((k.unwrap().to_string(), AnnData::read(File::open(f)?)?))
            }).collect::<Result<IndexMap<_, _>>>()?;
        let x = Stacked::new(anndatas.values().map(|x| x.get_x().clone()).collect())?;
        Ok(Self { annotation, x, anndatas })
    }

    pub fn n_obs(&self) -> usize { self.annotation.n_obs() }

    pub fn n_vars(&self) -> usize { self.annotation.n_vars() }

    def_accessor!(
        DataFrameElem,
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
        for ann in self.anndatas.into_values() {
            ann.close()?;
        }
        Ok(())
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