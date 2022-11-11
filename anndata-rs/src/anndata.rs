use crate::{
    data::*, element::*,
    element::{base::{InnerDataFrameElem, InnerMatrixElem, AccumLength}, collection::{InnerAxisArrays, AxisArrays}},
};

use std::{sync::Arc, path::Path, collections::HashMap, ops::Deref};
use parking_lot::Mutex;
use hdf5::File; 
use anyhow::{bail, anyhow, ensure, Result, Context};
use polars::prelude::{NamedFrom, DataFrame, Series};
use indexmap::map::IndexMap;
use itertools::Itertools;
use paste::paste;

pub struct AnnData {
    pub(crate) file: File,
    pub(crate) n_obs: Arc<Mutex<usize>>,
    pub(crate) n_vars: Arc<Mutex<usize>>,
    pub(crate) x: MatrixElem,
    pub(crate) obs: DataFrameElem,
    pub(crate) obsm: AxisArrays,
    pub(crate) obsp: AxisArrays,
    pub(crate) var: DataFrameElem,
    pub(crate) varm: AxisArrays,
    pub(crate) varp: AxisArrays,
    pub(crate) uns: ElemCollection,
}

impl std::fmt::Display for AnnData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f, "AnnData object with n_obs x n_vars = {} x {} backed at '{}'",
            self.n_obs(), self.n_vars(), self.filename(),
        )?;
        if let Some(obs) = self.obs.get_column_names() {
            if !obs.is_empty() { write!(f, "\n    obs: '{}'", obs.into_iter().join("', '"))?; }
        }
        if let Some(var) = self.var.get_column_names() {
            if !var.is_empty() { write!(f, "\n    var: '{}'", var.into_iter().join("', '"))?; }
        }
        if let Some(keys) = self.uns.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() { write!(f, "\n    uns: '{}'", keys)?; }
        }
        if let Some(keys) = self.obsm.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() { write!(f, "\n    obsm: '{}'", keys)?; }
        }
        if let Some(keys) = self.obsp.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() { write!(f, "\n    obsp: '{}'", keys)?; }
        }
        if let Some(keys) = self.varm.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() { write!(f, "\n    varm: '{}'", keys)?; }
        }
        if let Some(keys) = self.varp.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() { write!(f, "\n    varp: '{}'", keys)?; }
        }
        Ok(())
    }
}

macro_rules! anndata_setter {
    ($(($field:ident, $ty:ty, $n:ident)),*) => {
        paste! {
            $(
            pub fn [<set_ $field>](
                &self,
                data_: Option<HashMap<String, Box<dyn MatrixData>>>,
            ) -> Result<()>
            {
                let mut guard = self.$field.inner();
                let field = stringify!($field);
                if guard.0.is_some() { self.file.unlink(field)?; }
                match data_ {
                    None => { *guard.0 = None; },
                    Some(data) => {
                        let container = self.file.create_group(field)?;
                        let item = Slot::new(InnerAxisArrays::new(container, $ty, self.$n.clone()));
                        for (key, val) in data.into_iter() {
                            item.add_data(&key, val)?;
                        }
                        *guard.0 = Some(item.extract().unwrap());
                    },
                }
                Ok(())
            }
            )*
        }
    }
}

impl AnnData {
    pub fn get_x(&self) -> &MatrixElem { &self.x }
    pub fn get_obs(&self) -> &DataFrameElem { &self.obs }
    pub fn get_var(&self) -> &DataFrameElem { &self.var }
    pub fn get_obsm(&self) -> &AxisArrays { &self.obsm }
    pub fn get_obsp(&self) -> &AxisArrays { &self.obsp }
    pub fn get_varm(&self) -> &AxisArrays { &self.varm }
    pub fn get_varp(&self) -> &AxisArrays { &self.varp }
    pub fn get_uns(&self) -> &ElemCollection { &self.uns }

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

    anndata_setter!(
        (obsm, Axis::Row, n_obs),
        (obsp, Axis::Both, n_obs),
        (varm, Axis::Row, n_vars),
        (varp, Axis::Both, n_vars)
    );

    pub fn set_uns(&self, uns_: Option<HashMap<String, Box<dyn Data>>>) -> Result<()> {
        if !self.uns.is_empty() { self.file.unlink("uns")?; }
        match uns_ {
            None => { self.uns.drop(); },
            Some(uns) => {
                let container = self.file.create_group("uns")?;
                let elem = ElemCollection::try_from(container)?;
                for (key, data) in uns.into_iter() {
                    elem.add_data(&key, data)?;
                }
                self.uns.insert(elem.extract().unwrap());
            },
        }
        Ok(())
    }

    pub fn new<P: AsRef<Path>>(filename: P, n_obs: usize, n_vars: usize) -> Result<Self> {
        let file = hdf5::File::create(filename)?;
        let n_obs = Arc::new(Mutex::new(n_obs));
        let n_vars = Arc::new(Mutex::new(n_vars));
        let obsm = {
            let container = file.create_group("obsm")?;
            InnerAxisArrays::new(container, Axis::Row, n_obs.clone())
        };
        let obsp = {
            let container = file.create_group("obsp")?;
            InnerAxisArrays::new(container, Axis::Both, n_obs.clone())
        };
        let varm = {
            let container = file.create_group("varm")?;
            InnerAxisArrays::new(container, Axis::Row, n_vars.clone())
        };
        let varp = {
            let container = file.create_group("varp")?;
            InnerAxisArrays::new(container, Axis::Both, n_vars.clone())
        };
        let uns = {
            let container = file.create_group("uns")?;
            ElemCollection::try_from(container)?
        };
        Ok(Self { file, n_obs, n_vars, x: Slot::empty(), uns: uns,
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

    pub fn subset(&self, obs_idx: Option<&[usize]>, var_idx: Option<&[usize]>) -> Result<()> {
        if !self.x.is_empty() { self.x.subset(obs_idx, var_idx)?; }
        if let Some(i) = obs_idx {
            self.obs.subset_rows(i)?;
            self.obsm.subset(i)?;
            self.obsp.subset(i)?;
            *self.n_obs.lock() = i.len();
        }
        if let Some(j) = var_idx {
            self.var.subset_rows(j)?;
            self.varm.subset(j)?;
            self.varp.subset(j)?;
            *self.n_vars.lock() = j.len();
        }
        Ok(())
    }
}

pub struct StackedAnnData {
    anndatas: IndexMap<String, AnnData>,
    accum: Arc<Mutex<AccumLength>>,
    n_obs: Arc<Mutex<usize>>,
    n_vars: Arc<Mutex<usize>>,
    x: StackedMatrixElem,
    pub obs: StackedDataFrame,
    obsm: StackedAxisArrays,
}

impl std::fmt::Display for StackedAnnData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stacked AnnData objects:")?;
        write!(f, "\n    obs: '{}'", self.obs.column_names.iter().join("', '"))?;
        write!(f, "\n    obsm: '{}'", self.obsm.data.keys().join("', '"))?;
        Ok(())
    }
}

impl StackedAnnData {
    fn new(adatas: IndexMap<String, AnnData>, check: bool) -> Result<Self> {
        if check {
            if !adatas.values().map(|x| x.var_names()).all_equal() { return Err(anyhow!("var names mismatch")); }
        }

        let accum_: AccumLength = adatas.values().map(|x| x.n_obs()).collect();
        let n_obs = Arc::new(Mutex::new(accum_.size()));
        let accum = Arc::new(Mutex::new(accum_));
        let n_vars = adatas.values().next().unwrap().n_vars.clone();

        let x = StackedMatrixElem::new(
            adatas.values().map(|x| x.get_x().clone()).collect(),
            n_obs.clone(),
            n_vars.clone(),
            accum.clone(),
        )?;

        let obs = if adatas.values().any(|x| x.obs.is_empty()) {
            StackedDataFrame::new(Vec::new())
        } else {
            StackedDataFrame::new(adatas.values().map(|x| x.obs.clone()).collect())
        };

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

    pub fn get_x(&self) -> &StackedMatrixElem { &self.x }
    pub fn get_obsm(&self) -> &StackedAxisArrays { &self.obsm }

    pub fn len(&self) -> usize { self.anndatas.len() }

    pub fn keys(&self) -> indexmap::map::Keys<'_, String, AnnData> { self.anndatas.keys() }

    pub fn values(&self) -> indexmap::map::Values<'_, String, AnnData> { self.anndatas.values() }

    pub fn iter(&self) -> indexmap::map::Iter<'_, String, AnnData> { self.anndatas.iter() }

    /// Write a part of stacked AnnData objects to disk, return the key and
    /// file name (without parent paths)
    fn write<P>(&self, obs_idx_: Option<&[usize]>, var_idx: Option<&[usize]>, dir: P) -> Result<HashMap<String, String>>
    where
        P: AsRef<Path> + std::marker::Sync,
    {
        let obs_idx = match obs_idx_ {
            Some(i) => self.accum.lock().normalize_indices(i),
            None => HashMap::new(),
        };
        self.anndatas.iter().enumerate().map(|(i, (k, data))| {
            let file = dir.as_ref().join(k.to_string() + ".h5ad");
            match obs_idx.get(&i) {
                None => data.write(Some(&[]), var_idx, file.clone()),
                Some(idx) => data.write(Some(idx.as_slice()), var_idx, file.clone()),
            }?;
            Ok((k.clone(), file.file_name().unwrap().to_str().unwrap().to_string()))
        }).collect()
    }

    /// Subsetting an AnnDataSet will not rearrange the data between
    /// AnnData objects.
    /// TODO: return rearraged indices
    pub fn subset(&self, obs_idx: Option<&[usize]>, var_idx: Option<&[usize]>) -> Result<()> {
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
        self.anndatas.values().enumerate().try_for_each(|(i, data)| match obs_idx_.get(&i) {
            None => data.subset(Some(&[]), var_idx),
            Some(idx) => data.subset(Some(idx.as_slice()), var_idx),
        })?;
        *accum_len = self.anndatas.values().map(|x| x.n_obs()).collect();
        Ok(())
    }
}

pub struct AnnDataSet {
    annotation: AnnData,
    anndatas: Slot<StackedAnnData>,
}

impl std::fmt::Display for AnnDataSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f, "AnnDataSet object with n_obs x n_vars = {} x {} backed at '{}'",
            self.n_obs(), self.n_vars(), self.annotation.filename(),
        )?;
        if let Some((n, keys)) = self.anndatas.lock().as_ref().map(|x| (x.len(), x.keys().join("', '"))) {
            if n > 0 { write!(f, "\ncontains {} AnnData objects with keys: '{}'", n, keys)?; }
        }
        if let Some(obs) = self.annotation.obs.get_column_names() {
            if !obs.is_empty() { write!(f, "\n    obs: '{}'", obs.into_iter().join("', '"))?; }
        }
        if let Some(var) = self.annotation.var.get_column_names() {
            if !var.is_empty() { write!(f, "\n    var: '{}'", var.into_iter().join("', '"))?; }
        }
        if let Some(keys) = self.annotation.uns.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() { write!(f, "\n    uns: '{}'", keys)?; }
        }
        if let Some(keys) = self.annotation.obsm.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() { write!(f, "\n    obsm: '{}'", keys)?; }
        }
        if let Some(keys) = self.annotation.obsp.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() { write!(f, "\n    obsp: '{}'", keys)?; }
        }
        if let Some(keys) = self.annotation.varm.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() { write!(f, "\n    varm: '{}'", keys)?; }
        }
        if let Some(keys) = self.annotation.varp.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() { write!(f, "\n    varp: '{}'", keys)?; }
        }
        Ok(())
    }
}

macro_rules! def_accessor {
    ($get_type:ty, $set_type:ty, { $($field:ident),* }) => {
        paste! {
            $(
                pub fn [<get_ $field>](&self) -> &$get_type {
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
    let df = ann.read_uns_item("AnnDataSet")?
        .context("key 'AnnDataSet' is not present")?.downcast::<DataFrame>()
        .map_err(|_| anyhow!("cannot downcast to DataFrame"))?;
    let keys = df.column("keys").unwrap();
    let filenames = df.column("file_path").unwrap().utf8()
        .unwrap().into_iter().collect::<Option<Vec<_>>>().unwrap();
    let new_files: Vec<_> = keys.utf8().unwrap().into_iter().zip(filenames)
        .map(|(k, v)| new_locations.get(k.unwrap()).map_or(
            v.to_string(), |x| x.clone()
        )).collect();
    let data = DataFrame::new(vec![
        keys.clone(),
        Series::new("file_path", new_files),
    ]).unwrap();
    ann.get_uns().add_data("AnnDataSet", data)?;
    Ok(())
}

impl AnnDataSet {
    pub fn new<P: AsRef<Path>>(anndatas: IndexMap<String, AnnData>, filename: P, add_key: &str) -> Result<Self> {
        // Compute n_obs and n_vars
        let n_obs = anndatas.values().map(|x| x.n_obs()).sum();
        let n_vars = anndatas.values().next().map(|x| x.n_vars()).unwrap_or(0);

        let annotation = AnnData::new(filename, n_obs, n_vars)?;

        // Set UNS. UNS includes children anndata locations.
        {
            let (keys, filenames): (Vec<_>, Vec<_>) = anndatas.iter().map(|(k, v)| (k.clone(), v.filename())).unzip();
            let data = DataFrame::new(vec![Series::new("keys", keys), Series::new("file_path", filenames)])?;
            annotation.get_uns().add_data("AnnDataSet", data)?;
        }

        // Set OBS.
        {
            let obs_names: DataFrameIndex = anndatas.values().flat_map(|x| x.obs_names()).collect();
            if obs_names.is_empty() {
                annotation.set_obs_names((0..n_obs).into_iter().map(|x| x.to_string()).collect())?;
            } else {
                annotation.set_obs_names(obs_names)?;
            }
            let keys = Series::new(
                add_key,
                anndatas.iter().map(|(k, v)| vec![k.clone(); v.n_obs()])
                .flatten().collect::<Series>(),
            );
            annotation.set_obs(Some(DataFrame::new(vec![keys])?))?;
        }

        // Set VAR.
        {
            let adata = anndatas.values().next().unwrap();
            if !adata.var_names().is_empty() {
                annotation.set_var_names(adata.var_names().into_iter().collect())?;
                annotation.set_var(Some(adata.get_var().read()?))?;
            }
        }
        let stacked = StackedAnnData::new(anndatas, true)?;
        Ok(Self { annotation, anndatas: Slot::new(stacked), })
    }

    /// Get the reference to the concatenated AnnData objects.
    pub fn get_inner_adatas(&self) -> &Slot<StackedAnnData> { &self.anndatas }

    pub fn get_obs(&self) -> &DataFrameElem { &self.annotation.obs }
    pub fn get_var(&self) -> &DataFrameElem { &self.annotation.var }

    pub fn get_x(&self) -> StackedMatrixElem { self.anndatas.inner().x.clone() }

    def_accessor!(
        AxisArrays,
        Option<HashMap<String, Box<dyn MatrixData>>>,
        { obsm, obsp, varm, varp }
    );

    pub fn get_uns(&self) -> &ElemCollection { &self.annotation.uns }
    pub fn set_uns(&self, data: Option<HashMap<String, Box<dyn Data>>>) -> Result<()> {
        self.annotation.set_uns(data)
    }

    pub fn read(file: File, adata_files_: Option<HashMap<String, String>>, check: bool) -> Result<Self> {
        let annotation = AnnData::read(file)?;
        let filename = annotation.filename();
        let file_path = Path::new(&filename).read_link()
            .unwrap_or(Path::new(&filename).to_path_buf());
        let df = annotation.read_uns_item("AnnDataSet")?
            .context("key 'AnnDataSet' is not present")?.downcast::<DataFrame>()
            .map_err(|_| anyhow!("cannot downcast to DataFrame"))?;
        let keys = df.column("keys").unwrap().utf8().unwrap()
            .into_iter().collect::<Option<Vec<_>>>().unwrap();
        let filenames = df.column("file_path").unwrap().utf8()
            .unwrap().into_iter().collect::<Option<Vec<_>>>().unwrap();
        let adata_files = adata_files_.unwrap_or(HashMap::new());
        let anndatas = keys.into_iter().zip(filenames).map(|(k, v)| {
            let path = Path::new(adata_files.get(k).map_or(v, |x| &*x));
            let fl = if path.is_absolute() {
                File::open(path)
            } else {
                File::open(file_path.parent().unwrap_or(Path::new("./")).join(path))
            }?;
            Ok((k.to_string(), AnnData::read(fl)?))
        }).collect::<Result<_>>()?;

        if !adata_files.is_empty() { update_anndata_locations(&annotation, adata_files)?; }
        Ok(Self { annotation, anndatas: Slot::new(StackedAnnData::new(anndatas, check)?) })
    }

    /// Subsetting an AnnDataSet will not rearrange the data between
    /// AnnData objects.
    pub fn subset(&self, obs_idx: Option<&[usize]>, var_idx: Option<&[usize]>) -> Result<()> {
        ensure!(!self.anndatas.is_empty(), "anndatas is empty");
        match self.anndatas.inner().0.deref() {
            None => self.annotation.subset(obs_idx, var_idx)?,
            Some(ann) => {
                let obs_idx_ = obs_idx.map(|x|
                    ann.accum.lock().sort_index_to_buckets(x)
                );
                let i = obs_idx_.as_ref().map(|x| x.as_slice());
                self.annotation.subset(i, var_idx)?;
                ann.subset(i, var_idx)?;
            }
        }
        Ok(())
    }

    pub fn write<P: AsRef<Path>>(&self, obs_idx: Option<&[usize]>, var_idx: Option<&[usize]>, dir: P) -> Result<()>
    {
        let file = dir.as_ref().join("_dataset.h5ads");
        let anndata_dir = dir.as_ref().join("anndatas");
        std::fs::create_dir_all(anndata_dir.clone())?;
        match self.anndatas.inner().0.deref() {
            None => self.annotation.write(obs_idx, var_idx, file)?,
            Some(ann) => {
                let obs_idx_ = obs_idx.map(|x|
                    ann.accum.lock().sort_index_to_buckets(x)
                );
                let i = obs_idx_.as_ref().map(|x| x.as_slice());
                let adata = self.annotation.copy(i, var_idx, file)?;

                let mut filenames = ann.write(i, var_idx, anndata_dir.clone())?;
                let parent_dir = if anndata_dir.is_absolute() {
                    anndata_dir
                } else {
                    Path::new("anndatas").to_path_buf()
                };
                filenames.values_mut().for_each(|fl|
                    *fl = parent_dir.join(fl.as_str()).to_str().unwrap().to_string()
                );
                update_anndata_locations(&adata, filenames)?;

                adata.close()?;
            }
        }
        Ok(())
    }

    /// Copy and save the AnnDataSet to a new directory.
    /// This will copy all children AnnData files.
    pub fn copy<P: AsRef<Path>>(&self, obs_idx: Option<&[usize]>, var_idx: Option<&[usize]>, dir: P) -> Result<Self> {
        let file = dir.as_ref().join("_dataset.h5ads");
        self.write(obs_idx, var_idx, dir)?;
        AnnDataSet::read(File::open_rw(file)?, None, false)
    }

    /// Get reference to the inner AnnData annotation without the `.X` field.
    pub fn as_adata(&self) -> &AnnData { &self.annotation }

    pub fn to_adata<P: AsRef<Path>>(&self, obs_idx: Option<&[usize]>, var_idx: Option<&[usize]>, out: P) -> Result<AnnData> {
        self.annotation.copy(obs_idx, var_idx, out)
    }

    /// Convert AnnDataSet to AnnData object
    pub fn into_adata(self) -> Result<AnnData> {
        if let Some(anndatas) = self.anndatas.extract() {
            for ann in anndatas.anndatas.into_values() { ann.close()?; }
        }
        Ok(self.annotation)
    }

    pub fn close(self) -> Result<()> {
        self.annotation.close()?;
        if let Some(anndatas) = self.anndatas.extract() {
            for ann in anndatas.anndatas.into_values() { ann.close()?; }
        }
        Ok(())
    }
}

pub trait AnnDataOp {
    /// Reading/writing the 'X' element.
    fn read_x(&self) -> Result<Option<Box<dyn MatrixData>>>;
    fn set_x<D: MatrixData>(&self, data_: Option<D>) -> Result<()>;

    /// Return the number of observations (rows).
    fn n_obs(&self) -> usize;
    /// Return the number of variables (columns).
    fn n_vars(&self) -> usize;

    /// Return the names of observations.
    fn obs_names(&self) -> Vec<String>;
    /// Return the names of variables.
    fn var_names(&self) -> Vec<String>;

    /// Chagne the names of observations.
    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()>;
    /// Chagne the names of variables.
    fn set_var_names(&self, index: DataFrameIndex) -> Result<()>;

    fn obs_ix(&self, names: &[String]) -> Result<Vec<usize>>;
    fn var_ix(&self, names: &[String]) -> Result<Vec<usize>>;

    fn read_obs(&self) -> Result<DataFrame>;
    fn read_var(&self) -> Result<DataFrame>;

    /// Change the observation annotations. If `obs == None`, the `obs` will be
    /// removed.
    fn set_obs(&self, obs: Option<DataFrame>) -> Result<()>;
    /// Change the variable annotations. If `var == None`, the `var` will be
    /// removed.
    fn set_var(&self, var: Option<DataFrame>) -> Result<()>;

    fn uns_keys(&self) -> Vec<String>;
    fn obsm_keys(&self) -> Vec<String>;
    fn obsp_keys(&self) -> Vec<String>;
    fn varm_keys(&self) -> Vec<String>;
    fn varp_keys(&self) -> Vec<String>;

    fn read_uns_item(&self, key: &str) -> Result<Option<Box<dyn Data>>>;
    fn read_obsm_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>>;
    fn read_obsp_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>>;
    fn read_varm_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>>;
    fn read_varp_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>>;

    fn add_uns_item<D: Data>(&self, key: &str, data: D) -> Result<()>;
    fn add_obsm_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()>;
    fn add_obsp_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()>;
    fn add_varm_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()>;
    fn add_varp_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()>;
}

impl AnnDataOp for AnnData {
    fn read_x(&self) -> Result<Option<Box<dyn MatrixData>>> {
        let x = self.get_x();
        if x.is_empty() { Ok(None) } else { x.read(None, None).map(Option::Some) }
    }
    fn set_x<D: MatrixData>(&self, data_: Option<D>) -> Result<()> {
        match data_ {
            Some(data) => {
                self.set_n_obs(data.nrows());
                self.set_n_vars(data.ncols());
                if !self.x.is_empty() { self.file.unlink("X")?; }
                self.x.insert(InnerMatrixElem::try_from(data.write(&self.file, "X")?)?);
            },
            None => if !self.x.is_empty() {
                self.file.unlink("X")?;
                self.x.drop();
            },
        }
        Ok(())
    }

    fn n_obs(&self) -> usize { *self.n_obs.lock() }
    fn n_vars(&self) -> usize { *self.n_vars.lock() }

    fn obs_names(&self) -> Vec<String> { 
        if self.obs.is_empty() { Vec::new() } else { self.obs.inner().index.names.clone() }
    }
        
    fn var_names(&self) -> Vec<String> {
        if self.var.is_empty() { Vec::new() } else { self.var.inner().index.names.clone() }
    }

    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()> {
        self.set_n_obs(index.len());
        if self.obs.is_empty() {
            let df = InnerDataFrameElem::new(&self.file, "obs", index, &DataFrame::empty())?;
            self.obs.insert(df);
        } else {
            self.obs.set_index(index)?;
        }
        Ok(())
    }

    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> { 
        self.set_n_vars(index.len());
        if self.var.is_empty() {
            let df = InnerDataFrameElem::new(&self.file, "var", index, &DataFrame::empty())?;
            self.var.insert(df);
        } else {
            self.var.set_index(index)?;
        }
        Ok(())
    }

    fn obs_ix(&self, names: &[String]) -> Result<Vec<usize>> {
        let inner = self.obs.inner();
        names.iter().map(|i| inner.index.get(i).context(format!("'{}' does not exist in obs_names", i))).collect()
    }

    fn var_ix(&self, names: &[String]) -> Result<Vec<usize>> {
        let inner = self.var.inner();
        names.iter().map(|i| inner.index.get(i).context(format!("'{}' does not exist in obs_names", i))).collect()
    }

    fn read_obs(&self) -> Result<DataFrame> { self.get_obs().read() }
    fn read_var(&self) -> Result<DataFrame> { self.get_var().read() }
    fn set_obs(&self, obs_: Option<DataFrame>) -> Result<()> {
        match obs_ {
            None => if !self.obs.is_empty() {
                self.file.unlink("obs")?;
                self.obs.drop();
            },
            Some(obs) => {
                let nrows = obs.nrows();
                self.set_n_obs(nrows);
                if self.obs.is_empty() {
                    self.obs.insert(InnerDataFrameElem::new(&self.file, "obs", DataFrameIndex::from(nrows), &obs)?);
                } else {
                    self.obs.update(obs)?;
                }
            },
        }
        Ok(())
    }

    fn set_var(&self, var_: Option<DataFrame>) -> Result<()> {
        match var_ {
            None => if !self.var.is_empty() {
                self.file.unlink("var")?;
                self.var.drop();
            },
            Some(var) => {
                let nrows = var.nrows();
                self.set_n_vars(nrows);
                if self.var.is_empty() {
                    let index = (0..nrows).into_iter().map(|x| x.to_string()).collect();
                    let df = InnerDataFrameElem::new(&self.file, "var", index, &var)?;
                    self.var.insert(df);
                } else {
                    self.var.update(var)?;
                }
            },
        }
        Ok(())
    }

    fn uns_keys(&self) -> Vec<String> {
        self.get_uns().lock().as_ref().map(|x| x.keys().map(|x| x.to_string()).collect()).unwrap_or(Vec::new())
    }
    fn obsm_keys(&self) -> Vec<String> {
        self.get_obsm().lock().as_ref().map(|x| x.keys().map(|x| x.to_string()).collect()).unwrap_or(Vec::new())
    }
    fn obsp_keys(&self) -> Vec<String> {
        self.get_obsp().lock().as_ref().map(|x| x.keys().map(|x| x.to_string()).collect()).unwrap_or(Vec::new())
    }
    fn varm_keys(&self) -> Vec<String> {
        self.get_varm().lock().as_ref().map(|x| x.keys().map(|x| x.to_string()).collect()).unwrap_or(Vec::new())
    }
    fn varp_keys(&self) -> Vec<String> {
        self.get_varp().lock().as_ref().map(|x| x.keys().map(|x| x.to_string()).collect()).unwrap_or(Vec::new())
    }

    fn read_uns_item(&self, key: &str) -> Result<Option<Box<dyn Data>>> {
        self.get_uns().lock().as_mut().and_then(|x| x.get_mut(key)).map(|x| x.read()).transpose()
    }
    fn read_obsm_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> {
        self.get_obsm().lock().as_mut().and_then(|x| x.get_mut(key)).map(|x| x.read(None, None)).transpose()
    }
    fn read_obsp_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> {
        self.get_obsp().lock().as_mut().and_then(|x| x.get_mut(key)).map(|x| x.read(None, None)).transpose()
    }
    fn read_varm_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> {
        self.get_varm().lock().as_mut().and_then(|x| x.get_mut(key)).map(|x| x.read(None, None)).transpose()
    }
    fn read_varp_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> {
        self.get_varp().lock().as_mut().and_then(|x| x.get_mut(key)).map(|x| x.read(None, None)).transpose()
    }

    fn add_uns_item<D: Data>(&self, key: &str, data: D) -> Result<()> { self.get_uns().add_data(key, data) }
    fn add_obsm_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()> { self.get_obsm().add_data(key, data) }
    fn add_obsp_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()> { self.get_obsp().add_data(key, data) }
    fn add_varm_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()> { self.get_varm().add_data(key, data) }
    fn add_varp_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()> { self.get_varp().add_data(key, data) }
}

impl AnnDataOp for AnnDataSet {
    fn read_x(&self) -> Result<Option<Box<dyn MatrixData>>> {
        let adatas = &self.anndatas;
        if adatas.is_empty() { Ok(None) } else { adatas.inner().x.read(None, None).map(Option::Some) }
    }
    fn set_x<D: MatrixData>(&self, _: Option<D>) -> Result<()> { bail!("cannot set X in AnnDataSet") }

    fn n_obs(&self) -> usize { *self.anndatas.inner().n_obs.lock() }
    fn n_vars(&self) -> usize { *self.anndatas.inner().n_vars.lock() }

    fn obs_ix(&self, names: &[String]) -> Result<Vec<usize>> { self.annotation.obs_ix(names) }
    fn var_ix(&self, names: &[String]) -> Result<Vec<usize>> { self.annotation.var_ix(names) }
    fn obs_names(&self) -> Vec<String> { self.annotation.obs_names() }
    fn var_names(&self) -> Vec<String> { self.annotation.var_names() }
    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()> { self.annotation.set_obs_names(index) }
    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> { self.annotation.set_var_names(index) }

    fn read_obs(&self) -> Result<DataFrame> { self.annotation.read_obs() }
    fn read_var(&self) -> Result<DataFrame> { self.annotation.read_var() }
    fn set_obs(&self, obs: Option<DataFrame>) -> Result<()> { self.annotation.set_obs(obs) }
    fn set_var(&self, var: Option<DataFrame>) -> Result<()> { self.annotation.set_var(var) }

    fn uns_keys(&self) -> Vec<String> { self.annotation.uns_keys() }
    fn obsm_keys(&self) -> Vec<String> { self.annotation.obsm_keys() }
    fn obsp_keys(&self) -> Vec<String> { self.annotation.obsp_keys() }
    fn varm_keys(&self) -> Vec<String> { self.annotation.varm_keys() }
    fn varp_keys(&self) -> Vec<String> { self.annotation.varp_keys() }

    fn read_uns_item(&self, key: &str) -> Result<Option<Box<dyn Data>>> { self.annotation.read_uns_item(key) }
    fn read_obsm_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> { self.annotation.read_obsm_item(key) }
    fn read_obsp_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> { self.annotation.read_obsp_item(key) }
    fn read_varm_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> { self.annotation.read_varm_item(key) }
    fn read_varp_item(&self, key: &str) -> Result<Option<Box<dyn MatrixData>>> { self.annotation.read_varp_item(key) }

    fn add_uns_item<D: Data>(&self, key: &str, data: D) -> Result<()> { self.annotation.add_uns_item(key, data) }
    fn add_obsm_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()> { self.annotation.add_obsm_item(key, data) }
    fn add_obsp_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()> { self.annotation.add_obsp_item(key, data) }
    fn add_varm_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()> { self.annotation.add_varm_item(key, data) }
    fn add_varp_item<D: MatrixData>(&self, key: &str, data: D) -> Result<()> { self.annotation.add_varp_item(key, data) }
}