use crate::{
    data::*,
    element::{*, collection::InnerElemCollection},
    element::{
        base::{InnerDataFrameElem},
        collection::{AxisArrays, InnerAxisArrays},
    },
    backend::Backend,
};

use anyhow::{anyhow, bail, Context, Result};
use indexmap::map::IndexMap;
use itertools::Itertools;
use parking_lot::Mutex;
use paste::paste;
use polars::prelude::{DataFrame, NamedFrom, Series};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::{collections::HashMap, ops::Deref, path::{Path, PathBuf}, sync::Arc};

pub struct AnnData<B: Backend> {
    pub(crate) file: B::File,
    pub(crate) n_obs: Arc<Mutex<usize>>,
    pub(crate) n_vars: Arc<Mutex<usize>>,
    pub(crate) x: ArrayElem<B>,
    pub(crate) obs: DataFrameElem<B>,
    pub(crate) obsm: AxisArrays<B>,
    pub(crate) obsp: AxisArrays<B>,
    pub(crate) var: DataFrameElem<B>,
    pub(crate) varm: AxisArrays<B>,
    pub(crate) varp: AxisArrays<B>,
    pub(crate) uns: ElemCollection<B>,
}

/*
impl<B: Backend> std::fmt::Display for AnnData<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AnnData object with n_obs x n_vars = {} x {} backed at '{}'",
            self.n_obs(),
            self.n_vars(),
            self.filename(),
        )?;
        if let Some(obs) = self.obs.get_column_names() {
            if !obs.is_empty() {
                write!(f, "\n    obs: '{}'", obs.into_iter().join("', '"))?;
            }
        }
        if let Some(var) = self.var.get_column_names() {
            if !var.is_empty() {
                write!(f, "\n    var: '{}'", var.into_iter().join("', '"))?;
            }
        }
        if let Some(keys) = self.uns.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() {
                write!(f, "\n    uns: '{}'", keys)?;
            }
        }
        if let Some(keys) = self.obsm.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() {
                write!(f, "\n    obsm: '{}'", keys)?;
            }
        }
        if let Some(keys) = self.obsp.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() {
                write!(f, "\n    obsp: '{}'", keys)?;
            }
        }
        if let Some(keys) = self.varm.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() {
                write!(f, "\n    varm: '{}'", keys)?;
            }
        }
        if let Some(keys) = self.varp.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() {
                write!(f, "\n    varp: '{}'", keys)?;
            }
        }
        Ok(())
    }
}
*/

/*
macro_rules! anndata_setter {
    ($(($field:ident, $ty:ty, $n:ident)),*) => {
        paste! {
            $(
            pub fn [<set_ $field>](
                &self,
                data_: Option<HashMap<String, Box<dyn ArrayData>>>,
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
*/

impl<B: Backend> AnnData<B> {
    pub fn get_x(&self) -> &ArrayElem<B> {
        &self.x
    }
    pub fn get_obs(&self) -> &DataFrameElem<B> {
        &self.obs
    }
    pub fn get_obsm(&self) -> &AxisArrays<B> {
        &self.obsm
    }
    pub fn get_obsp(&self) -> &AxisArrays<B> {
        &self.obsp
    }
    pub fn get_var(&self) -> &DataFrameElem<B> {
        &self.var
    }
    pub fn get_varm(&self) -> &AxisArrays<B> {
        &self.varm
    }
    pub fn get_varp(&self) -> &AxisArrays<B> {
        &self.varp
    }
    pub fn get_uns(&self) -> &ElemCollection<B> {
        &self.uns
    }

    pub fn set_n_obs(&self, n: usize) {
        let mut n_obs = self.n_obs.lock();
        if *n_obs != n {
            let obs_is_none= self.obs.is_empty()
                && self.x.is_empty()
                && (self.obsm.is_empty() || self.obsm.inner().is_empty())
                && (self.obsp.is_empty() || self.obsp.inner().is_empty());
            if obs_is_none {
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
            let var_is_none= self.var.is_empty()
                && self.x.is_empty()
                && (self.varm.is_empty() || self.varm.inner().is_empty())
                && (self.varp.is_empty() || self.varp.inner().is_empty());
            if var_is_none {
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

    pub fn new<P: AsRef<Path>>(filename: P, n_obs: usize, n_vars: usize) -> Result<Self> {
        let file = B::create(filename)?;
        let n_obs = Arc::new(Mutex::new(n_obs));
        let n_vars = Arc::new(Mutex::new(n_vars));
        Ok(Self {
            file,
            n_obs,
            n_vars,
            x: Slot::empty(),
            uns: ElemCollection::empty(),
            obs: Slot::empty(),
            obsm: AxisArrays::empty(),
            obsp: AxisArrays::empty(),
            var: Slot::empty(),
            varm: AxisArrays::empty(),
            varp: AxisArrays::empty(),
        })
    }

    pub fn filename(&self) -> PathBuf{
        B::filename(&self.file)
    }

    pub fn close(self) -> Result<()> {
        macro_rules! close {
            ($($name:ident),*) => {
                $(
                self.$name.lock().as_ref().map(|x| x.values().for_each(|x| x.drop()));
                self.$name.drop();
                )*
            };
        }
        self.x.drop();
        self.obs.drop();
        self.var.drop();
        close!(obsm, obsp, varm, varp, uns);
        B::close(self.file)
    }

    pub fn subset<S, E>(&self, selection: S) -> Result<()>
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
    {
        self.x.lock().as_mut().map(|x| x.subset(&selection)).transpose()?;

        selection.as_ref().get(0).map(|i| {
            self.obs.lock().as_mut().map(|x| x.subset_rows(i)).transpose()?;
            self.obsm.lock().as_ref().map(|obsm| obsm.subset(i)).transpose()?;
            self.obsp.lock().as_ref().map(|obsp| obsp.subset(i)).transpose()?;
            let mut n_obs = self.n_obs.lock();
            *n_obs = i.as_ref().output_len(*n_obs);
            Ok::<(), anyhow::Error>(())
        }).transpose()?;

        selection.as_ref().get(1).map(|i| {
            self.var.lock().as_mut().map(|x| x.subset_rows(i)).transpose()?;
            self.varm.lock().as_ref().map(|varm| varm.subset(i)).transpose()?;
            self.varp.lock().as_ref().map(|varp| varp.subset(i)).transpose()?;
            let mut n_vars = self.n_vars.lock();
            *n_vars = i.as_ref().output_len(*n_vars);
            Ok::<(), anyhow::Error>(())
        }).transpose()?;
        Ok(())
    }
}

/*
pub struct StackedAnnData {
    anndatas: IndexMap<String, AnnData>,
    index: Arc<Mutex<VecVecIndex>>,
    n_obs: Arc<Mutex<usize>>,
    n_vars: Arc<Mutex<usize>>,
    x: StackedArrayElem,
    pub obs: StackedDataFrame,
    obsm: StackedAxisArrays,
}

impl std::fmt::Display for StackedAnnData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stacked AnnData objects:")?;
        write!(
            f,
            "\n    obs: '{}'",
            self.obs.column_names.iter().join("', '")
        )?;
        write!(f, "\n    obsm: '{}'", self.obsm.data.keys().join("', '"))?;
        Ok(())
    }
}

impl StackedAnnData {
    fn new(adatas: IndexMap<String, AnnData>) -> Result<Option<Self>> {
        if adatas.is_empty() {
            return Ok(None);
        }
        if let Some((_, first)) = adatas.first() {
            let lock = first.var.lock();
            let var_names: Option<&Vec<String>> = lock.as_ref().map(|x| &x.index.names);
            if !adatas
                .par_values()
                .skip(1)
                .all(|x| x.var.lock().as_ref().map(|x| &x.index.names).eq(&var_names))
            {
                bail!("var names mismatch");
            }
        }

        let index_: VecVecIndex = adatas.values().map(|x| x.n_obs()).collect();
        let n_obs = Arc::new(Mutex::new(index_.len()));
        let index = Arc::new(Mutex::new(index_));
        let n_vars = adatas.values().next().unwrap().n_vars.clone();

        let x = StackedArrayElem::new(
            adatas.values().map(|x| x.get_x().clone()).collect(),
            n_obs.clone(),
            n_vars.clone(),
            index.clone(),
        )?;

        let obs = if adatas.values().any(|x| x.obs.is_empty()) {
            StackedDataFrame::new(Vec::new())
        } else {
            StackedDataFrame::new(adatas.values().map(|x| x.obs.clone()).collect())
        };

        let obsm = if adatas.values().any(|x| x.obsm.is_empty()) {
            Ok(StackedAxisArrays {
                axis: Axis::Row,
                data: HashMap::new(),
            })
        } else {
            let obsm_guard: Vec<_> = adatas.values().map(|x| x.obsm.inner()).collect();
            StackedAxisArrays::new(
                obsm_guard.iter().map(|x| x.deref()).collect(),
                &n_obs,
                &n_vars,
                &index,
            )
        }?;

        Ok(Some(Self {
            anndatas: adatas,
            x,
            obs,
            obsm,
            index,
            n_obs,
            n_vars,
        }))
    }

    pub fn get_x(&self) -> &StackedArrayElem {
        &self.x
    }
    pub fn get_obsm(&self) -> &StackedAxisArrays {
        &self.obsm
    }

    pub fn len(&self) -> usize {
        self.anndatas.len()
    }

    pub fn keys(&self) -> indexmap::map::Keys<'_, String, AnnData> {
        self.anndatas.keys()
    }

    pub fn values(&self) -> indexmap::map::Values<'_, String, AnnData> {
        self.anndatas.values()
    }

    pub fn iter(&self) -> indexmap::map::Iter<'_, String, AnnData> {
        self.anndatas.iter()
    }

    /// Write a part of stacked AnnData objects to disk, return the key and
    /// file name (without parent paths)
    fn write<P>(
        &self,
        obs_indices: Option<&[usize]>,
        var_indices: Option<&[usize]>,
        dir: P,
    ) -> Result<(IndexMap<String, String>, Option<Vec<usize>>)>
    where
        P: AsRef<Path> + std::marker::Sync,
    {
        let index = self.index.lock();
        let obs_outer2inner = obs_indices.map(|x| index.ix_group_by_outer(x.iter()));
        let (files, ori_idx): (IndexMap<_, _>, Vec<_>) = self
            .anndatas
            .par_iter()
            .enumerate()
            .flat_map(|(i, (k, data))| {
                let file = dir.as_ref().join(k.to_string() + ".h5ad");
                let filename = (
                    k.clone(),
                    file.file_name().unwrap().to_str().unwrap().to_string(),
                );

                if let Some(get_inner_indices) = obs_outer2inner.as_ref() {
                    get_inner_indices.get(&i).map(|(indices, ori_idx)| {
                        data.write(Some(indices.as_slice()), var_indices, file.clone())
                            .unwrap();
                        (filename, Some(ori_idx.clone()))
                    })
                } else {
                    data.write(None, var_indices, file.clone()).unwrap();
                    Some((filename, None))
                }
            })
            .unzip();
        Ok((
            files,
            ori_idx
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .map(|x| x.into_iter().flatten().collect()),
        ))
    }

    /// Subsetting an AnnDataSet will not rearrange the data between
    /// AnnData objects.
    pub fn subset(
        &self,
        obs_indices: Option<&[usize]>,
        var_indices: Option<&[usize]>,
    ) -> Result<Option<Vec<usize>>> {
        let index = self.index.lock();

        let obs_outer2inner = obs_indices.map(|x| {
            *self.n_obs.lock() = x.len();
            index.ix_group_by_outer(x.iter())
        });
        if let Some(j) = var_indices {
            *self.n_vars.lock() = j.len();
        }

        let ori_idx: Vec<_> = self
            .anndatas
            .par_values()
            .enumerate()
            .map(|(i, data)| {
                if let Some(get_inner_indices) = obs_outer2inner.as_ref() {
                    if let Some((indices, ori_idx)) = get_inner_indices.get(&i) {
                        data.subset(Some(indices.as_slice()), var_indices).unwrap();
                        Some(ori_idx.clone())
                    } else {
                        data.subset(Some(&[]), var_indices).unwrap();
                        Some(Vec::new())
                    }
                } else {
                    data.subset(None, var_indices).unwrap();
                    None
                }
            })
            .collect();
        Ok(ori_idx
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .map(|x| x.into_iter().flatten().collect()))
    }
}

pub struct AnnDataSet {
    annotation: AnnData,
    anndatas: Slot<StackedAnnData>,
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
        if let Some((n, keys)) = self
            .anndatas
            .lock()
            .as_ref()
            .map(|x| (x.len(), x.keys().join("', '")))
        {
            if n > 0 {
                write!(f, "\ncontains {} AnnData objects with keys: '{}'", n, keys)?;
            }
        }
        if let Some(obs) = self.annotation.obs.get_column_names() {
            if !obs.is_empty() {
                write!(f, "\n    obs: '{}'", obs.into_iter().join("', '"))?;
            }
        }
        if let Some(var) = self.annotation.var.get_column_names() {
            if !var.is_empty() {
                write!(f, "\n    var: '{}'", var.into_iter().join("', '"))?;
            }
        }
        if let Some(keys) = self
            .annotation
            .uns
            .lock()
            .as_ref()
            .map(|x| x.keys().join("', '"))
        {
            if !keys.is_empty() {
                write!(f, "\n    uns: '{}'", keys)?;
            }
        }
        if let Some(keys) = self
            .annotation
            .obsm
            .lock()
            .as_ref()
            .map(|x| x.keys().join("', '"))
        {
            if !keys.is_empty() {
                write!(f, "\n    obsm: '{}'", keys)?;
            }
        }
        if let Some(keys) = self
            .annotation
            .obsp
            .lock()
            .as_ref()
            .map(|x| x.keys().join("', '"))
        {
            if !keys.is_empty() {
                write!(f, "\n    obsp: '{}'", keys)?;
            }
        }
        if let Some(keys) = self
            .annotation
            .varm
            .lock()
            .as_ref()
            .map(|x| x.keys().join("', '"))
        {
            if !keys.is_empty() {
                write!(f, "\n    varm: '{}'", keys)?;
            }
        }
        if let Some(keys) = self
            .annotation
            .varp
            .lock()
            .as_ref()
            .map(|x| x.keys().join("', '"))
        {
            if !keys.is_empty() {
                write!(f, "\n    varp: '{}'", keys)?;
            }
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
    let df = ann
        .read_uns_item("AnnDataSet")?
        .context("key 'AnnDataSet' is not present")?
        .downcast::<DataFrame>()
        .map_err(|_| anyhow!("cannot downcast to DataFrame"))?;
    let keys = df.column("keys").unwrap();
    let filenames = df
        .column("file_path")
        .unwrap()
        .utf8()
        .unwrap()
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .unwrap();
    let new_files: Vec<_> = keys
        .utf8()
        .unwrap()
        .into_iter()
        .zip(filenames)
        .map(|(k, v)| {
            new_locations
                .get(k.unwrap())
                .map_or(v.to_string(), |x| x.clone())
        })
        .collect();
    let data = DataFrame::new(vec![keys.clone(), Series::new("file_path", new_files)]).unwrap();
    ann.get_uns().add_data("AnnDataSet", data)?;
    Ok(())
}

impl AnnDataSet {
    pub fn new<P: AsRef<Path>>(
        anndatas: IndexMap<String, AnnData>,
        filename: P,
        add_key: &str,
    ) -> Result<Self> {
        // Compute n_obs and n_vars
        let n_obs = anndatas.values().map(|x| x.n_obs()).sum();
        let n_vars = anndatas.values().next().map(|x| x.n_vars()).unwrap_or(0);

        let annotation = AnnData::new(filename, n_obs, n_vars)?;

        // Set UNS. UNS includes children anndata locations.
        {
            let (keys, filenames): (Vec<_>, Vec<_>) = anndatas
                .iter()
                .map(|(k, v)| (k.clone(), v.filename()))
                .unzip();
            let data = DataFrame::new(vec![
                Series::new("keys", keys),
                Series::new("file_path", filenames),
            ])?;
            annotation.get_uns().add_data("AnnDataSet", data)?;
        }

        // Set OBS.
        {
            let obs_names: DataFrameIndex = anndatas.values().flat_map(|x| x.obs_names()).collect();
            if obs_names.is_empty() {
                annotation
                    .set_obs_names((0..n_obs).into_iter().map(|x| x.to_string()).collect())?;
            } else {
                annotation.set_obs_names(obs_names)?;
            }
            let keys = Series::new(
                add_key,
                anndatas
                    .iter()
                    .map(|(k, v)| vec![k.clone(); v.n_obs()])
                    .flatten()
                    .collect::<Series>(),
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
        let anndatas = match StackedAnnData::new(anndatas)? {
            Some(ann) => Slot::new(ann),
            _ => Slot::empty(),
        };
        Ok(Self {
            annotation,
            anndatas,
        })
    }

    /// Get the reference to the concatenated AnnData objects.
    pub fn get_inner_adatas(&self) -> &Slot<StackedAnnData> {
        &self.anndatas
    }

    pub fn get_obs(&self) -> &DataFrameElem {
        &self.annotation.obs
    }
    pub fn get_var(&self) -> &DataFrameElem {
        &self.annotation.var
    }

    pub fn get_x(&self) -> StackedArrayElem {
        self.anndatas.inner().x.clone()
    }

    def_accessor!(
        AxisArrays,
        Option<HashMap<String, Box<dyn ArrayData>>>,
        { obsm, obsp, varm, varp }
    );

    pub fn get_uns(&self) -> &ElemCollection {
        &self.annotation.uns
    }
    pub fn set_uns(&self, data: Option<HashMap<String, Box<dyn Data>>>) -> Result<()> {
        self.annotation.set_uns(data)
    }

    pub fn read(file: File, adata_files_: Option<HashMap<String, String>>) -> Result<Self> {
        let annotation = AnnData::read(file)?;
        let filename = annotation.filename();
        let file_path = Path::new(&filename)
            .read_link()
            .unwrap_or(Path::new(&filename).to_path_buf());
        let df = annotation
            .read_uns_item("AnnDataSet")?
            .context("key 'AnnDataSet' is not present")?
            .downcast::<DataFrame>()
            .map_err(|_| anyhow!("cannot downcast to DataFrame"))?;
        let keys = df
            .column("keys")
            .unwrap()
            .utf8()
            .unwrap()
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .unwrap();
        let filenames = df
            .column("file_path")
            .unwrap()
            .utf8()
            .unwrap()
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .unwrap();
        let adata_files = adata_files_.unwrap_or(HashMap::new());
        let anndatas = keys
            .into_par_iter()
            .zip(filenames)
            .map(|(k, v)| {
                let path = Path::new(adata_files.get(k).map_or(v, |x| &*x));
                let fl = if path.is_absolute() {
                    File::open(path)
                } else {
                    File::open(file_path.parent().unwrap_or(Path::new("./")).join(path))
                }?;
                Ok((k.to_string(), AnnData::read(fl)?))
            })
            .collect::<Result<_>>()?;

        if !adata_files.is_empty() {
            update_anndata_locations(&annotation, adata_files)?;
        }
        let anndatas = match StackedAnnData::new(anndatas)? {
            Some(ann) => Slot::new(ann),
            _ => Slot::empty(),
        };
        Ok(Self {
            annotation,
            anndatas,
        })
    }

    /// Subsetting an AnnDataSet will not rearrange the data between
    /// AnnData objects.
    pub fn subset(
        &self,
        obs_idx: Option<&[usize]>,
        var_idx: Option<&[usize]>,
    ) -> Result<Option<Vec<usize>>> {
        match self.anndatas.inner().0.deref() {
            None => {
                self.annotation.subset(obs_idx, var_idx)?;
                Ok(None)
            }
            Some(ann) => {
                let obs_idx_order = ann.subset(obs_idx, var_idx)?;
                if let Some(order) = obs_idx_order.as_ref() {
                    let new_idx = obs_idx.map(|x| order.iter().map(|i| x[*i]).collect::<Vec<_>>());
                    self.annotation
                        .subset(new_idx.as_ref().map(|x| x.as_slice()), var_idx)?;
                } else {
                    self.annotation.subset(obs_idx, var_idx)?;
                };
                Ok(obs_idx_order)
            }
        }
    }

    pub fn write<P: AsRef<Path>>(
        &self,
        obs_idx: Option<&[usize]>,
        var_idx: Option<&[usize]>,
        dir: P,
    ) -> Result<Option<Vec<usize>>> {
        let file = dir.as_ref().join("_dataset.h5ads");
        let anndata_dir = dir.as_ref().join("anndatas");
        std::fs::create_dir_all(anndata_dir.clone())?;
        match self.anndatas.inner().0.deref() {
            None => {
                self.annotation.write(obs_idx, var_idx, file)?;
                Ok(None)
            }
            Some(ann) => {
                let (files, obs_idx_order) = ann.write(obs_idx, var_idx, anndata_dir.clone())?;
                let adata = if let Some(order) = obs_idx_order.as_ref() {
                    let new_idx = obs_idx.map(|x| order.iter().map(|i| x[*i]).collect::<Vec<_>>());
                    self.annotation
                        .copy(new_idx.as_ref().map(|x| x.as_slice()), var_idx, file)?
                } else {
                    self.annotation.copy(obs_idx, var_idx, file)?
                };
                let parent_dir = if anndata_dir.is_absolute() {
                    anndata_dir
                } else {
                    Path::new("anndatas").to_path_buf()
                };

                let (keys, filenames): (Vec<_>, Vec<_>) = files
                    .into_iter()
                    .map(|(k, v)| (k, parent_dir.join(v.as_str()).to_str().unwrap().to_string()))
                    .unzip();
                let file_loc = DataFrame::new(vec![
                    Series::new("keys", keys),
                    Series::new("file_path", filenames),
                ])?;
                adata.add_uns_item("AnnDataSet", file_loc)?;
                adata.close()?;
                Ok(obs_idx_order)
            }
        }
    }

    /// Copy and save the AnnDataSet to a new directory.
    /// This will copy all children AnnData files.
    pub fn copy<P: AsRef<Path>>(
        &self,
        obs_idx: Option<&[usize]>,
        var_idx: Option<&[usize]>,
        dir: P,
    ) -> Result<(Self, Option<Vec<usize>>)> {
        let file = dir.as_ref().join("_dataset.h5ads");
        let ori_idx = self.write(obs_idx, var_idx, dir)?;
        let data = AnnDataSet::read(File::open_rw(file)?, None)?;
        Ok((data, ori_idx))
    }

    /// Get reference to the inner AnnData annotation without the `.X` field.
    pub fn as_adata(&self) -> &AnnData {
        &self.annotation
    }

    pub fn to_adata<P: AsRef<Path>>(
        &self,
        obs_idx: Option<&[usize]>,
        var_idx: Option<&[usize]>,
        out: P,
    ) -> Result<AnnData> {
        self.annotation.copy(obs_idx, var_idx, out)
    }

    /// Convert AnnDataSet to AnnData object
    pub fn into_adata(self) -> Result<AnnData> {
        if let Some(anndatas) = self.anndatas.extract() {
            for ann in anndatas.anndatas.into_values() {
                ann.close()?;
            }
        }
        Ok(self.annotation)
    }

    pub fn close(self) -> Result<()> {
        self.annotation.close()?;
        if let Some(anndatas) = self.anndatas.extract() {
            for ann in anndatas.anndatas.into_values() {
                ann.close()?;
            }
        }
        Ok(())
    }
}
*/

pub trait AnnDataOp {
    /// Reading/writing the 'X' element.
    fn read_x(&self) -> Result<Option<ArrayData>>;
    fn set_x<D: WriteData + Into<ArrayData> + ArrayOp>(&self, data_: Option<D>) -> Result<()>;

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

    fn read_uns_item(&self, key: &str) -> Result<Option<Data>>;
    fn read_obsm_item(&self, key: &str) -> Result<Option<ArrayData>>;
    fn read_obsp_item(&self, key: &str) -> Result<Option<ArrayData>>;
    fn read_varm_item(&self, key: &str) -> Result<Option<ArrayData>>;
    fn read_varp_item(&self, key: &str) -> Result<Option<ArrayData>>;

    fn add_uns_item<D: WriteData + Into<Data>>(&self, key: &str, data: D) -> Result<()>;
    fn add_obsm_item<D: WriteArrayData + ArrayOp + Into<ArrayData>>(&self, key: &str, data: D) -> Result<()>;
    fn add_obsp_item<D: WriteArrayData + ArrayOp + Into<ArrayData>>(&self, key: &str, data: D) -> Result<()>;
    fn add_varm_item<D: WriteArrayData + ArrayOp + Into<ArrayData>>(&self, key: &str, data: D) -> Result<()>;
    fn add_varp_item<D: WriteArrayData + ArrayOp + Into<ArrayData>>(&self, key: &str, data: D) -> Result<()>;
}

impl<B: Backend> AnnDataOp for AnnData<B> {
    fn read_x(&self) -> Result<Option<ArrayData>> {
        let x = self.get_x();
        if x.is_empty() {
            Ok(None)
        } else {
            x.inner().data().map(Option::Some)
        }
    }
    fn set_x<D: WriteData + Into<ArrayData> + ArrayOp>(&self, data_: Option<D>) -> Result<()> {
        match data_ {
            Some(data) => {
                let shape = data.shape();
                self.set_n_obs(shape[0]);
                self.set_n_vars(shape[1]);
                if !self.x.is_empty() {
                    self.x.inner().save(data)?;
                } else {
                    let new_elem = ArrayElem::try_from(data.write::<B>(&self.file, "X")?)?;
                    self.x.swap(&new_elem);
                }
            }
            None => {
                if !self.x.is_empty() {
                    B::delete(&self.file, "X")?;
                    self.x.drop();
                }
            }
        }
        Ok(())
    }

    fn n_obs(&self) -> usize {
        *self.n_obs.lock()
    }
    fn n_vars(&self) -> usize {
        *self.n_vars.lock()
    }

    fn obs_names(&self) -> Vec<String> {
        self.obs.lock().as_ref().map_or(Vec::new(), |obs| obs.index.names.clone())
    }

    fn var_names(&self) -> Vec<String> {
        self.var.lock().as_ref().map_or(Vec::new(), |var| var.index.names.clone())
    }

    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()> {
        self.set_n_obs(index.len());
        if self.obs.is_empty() {
            let df = InnerDataFrameElem::new(self.file.deref(), "obs", index, &DataFrame::empty())?;
            self.obs.insert(df);
        } else {
            self.obs.inner().set_index(index)?;
        }
        Ok(())
    }

    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> {
        self.set_n_vars(index.len());
        if self.var.is_empty() {
            let df = InnerDataFrameElem::new(self.file.deref(), "var", index, &DataFrame::empty())?;
            self.var.insert(df);
        } else {
            self.var.inner().set_index(index)?;
        }
        Ok(())
    }

    fn obs_ix(&self, names: &[String]) -> Result<Vec<usize>> {
        let inner = self.obs.inner();
        names
            .iter()
            .map(|i| {
                inner
                    .index
                    .get(i)
                    .context(format!("'{}' does not exist in obs_names", i))
            })
            .collect()
    }

    fn var_ix(&self, names: &[String]) -> Result<Vec<usize>> {
        let inner = self.var.inner();
        names
            .iter()
            .map(|i| {
                inner
                    .index
                    .get(i)
                    .context(format!("'{}' does not exist in obs_names", i))
            })
            .collect()
    }

    fn read_obs(&self) -> Result<DataFrame> {
        self.get_obs().lock().as_mut().map_or(Ok(DataFrame::empty()), |x| x.data())
    }
    fn read_var(&self) -> Result<DataFrame> {
        self.get_var().lock().as_mut().map_or(Ok(DataFrame::empty()), |x| x.data())
    }
    fn set_obs(&self, obs_: Option<DataFrame>) -> Result<()> {
        if let Some(obs) = obs_ {
            let nrows = obs.height();
            if nrows != 0 { self.set_n_obs(nrows); }
            if self.obs.is_empty() {
                self.obs.insert(InnerDataFrameElem::new(
                    self.file.deref(),
                    "obs",
                    DataFrameIndex::from(nrows),
                    &obs,
                )?);
            } else {
                self.obs.inner().save(obs)?;
            }
        } else {
            if !self.obs.is_empty() {
                B::delete(&self.file, "obs")?;
                self.obs.drop();
            }
        }
        Ok(())
    }

    fn set_var(&self, var_: Option<DataFrame>) -> Result<()> {
        if let Some(var) = var_ {
            let nrows = var.height();
            if nrows != 0 { self.set_n_vars(nrows); }
            if self.var.is_empty() {
                self.var.insert(InnerDataFrameElem::new(
                    self.file.deref(),
                    "var",
                    DataFrameIndex::from(nrows),
                    &var,
                )?);
            } else {
                self.var.inner().save(var)?;
            }
        } else {
            if !self.var.is_empty() {
                B::delete(&self.file, "var")?;
                self.var.drop();
            }
        }
        Ok(())
    }

    fn uns_keys(&self) -> Vec<String> {
        self.get_uns()
            .lock()
            .as_ref()
            .map(|x| x.keys().map(|x| x.to_string()).collect())
            .unwrap_or(Vec::new())
    }
    fn obsm_keys(&self) -> Vec<String> {
        self.get_obsm()
            .lock()
            .as_ref()
            .map(|x| x.keys().map(|x| x.to_string()).collect())
            .unwrap_or(Vec::new())
    }
    fn obsp_keys(&self) -> Vec<String> {
        self.get_obsp()
            .lock()
            .as_ref()
            .map(|x| x.keys().map(|x| x.to_string()).collect())
            .unwrap_or(Vec::new())
    }
    fn varm_keys(&self) -> Vec<String> {
        self.get_varm()
            .lock()
            .as_ref()
            .map(|x| x.keys().map(|x| x.to_string()).collect())
            .unwrap_or(Vec::new())
    }
    fn varp_keys(&self) -> Vec<String> {
        self.get_varp()
            .lock()
            .as_ref()
            .map(|x| x.keys().map(|x| x.to_string()).collect())
            .unwrap_or(Vec::new())
    }

    fn read_uns_item(&self, key: &str) -> Result<Option<Data>> {
        self.get_uns()
            .lock()
            .as_mut()
            .and_then(|x| x.get_mut(key))
            .map(|x| x.inner().data())
            .transpose()
    }
    fn read_obsm_item(&self, key: &str) -> Result<Option<ArrayData>> {
        self.get_obsm()
            .lock()
            .as_mut()
            .and_then(|x| x.get_mut(key))
            .map(|x| x.inner().data())
            .transpose()
    }
    fn read_obsp_item(&self, key: &str) -> Result<Option<ArrayData>> {
        self.get_obsp()
            .lock()
            .as_mut()
            .and_then(|x| x.get_mut(key))
            .map(|x| x.inner().data())
            .transpose()
    }
    fn read_varm_item(&self, key: &str) -> Result<Option<ArrayData>> {
        self.get_varm()
            .lock()
            .as_mut()
            .and_then(|x| x.get_mut(key))
            .map(|x| x.inner().data())
            .transpose()
    }
    fn read_varp_item(&self, key: &str) -> Result<Option<ArrayData>> {
        self.get_varp()
            .lock()
            .as_mut()
            .and_then(|x| x.get_mut(key))
            .map(|x| x.inner().data())
            .transpose()
    }

    fn add_uns_item<D: WriteData + Into<Data>>(&self, key: &str, data: D) -> Result<()> {
        if self.get_uns().is_empty() {
            let collection = ElemCollection::new(B::create_group(&self.file, "uns")?)?;
            self.uns.swap(&collection);
        }
        self.get_uns().inner().add_data(key, data)
    }
    fn add_obsm_item<D: WriteArrayData + ArrayOp + Into<ArrayData>>(&self, key: &str, data: D) -> Result<()> {
        if self.get_obsm().is_empty() {
            let group = B::create_group(&self.file, "obsm")?;
            let arrays = AxisArrays::new(group, Axis::Row, self.n_obs.clone())?;
            self.obsm.swap(&arrays);
        }
        self.get_obsm().inner().add_data(key, data)
    }
    fn add_obsp_item<D: WriteArrayData + ArrayOp + Into<ArrayData>>(&self, key: &str, data: D) -> Result<()> {
        if self.get_obsp().is_empty() {
            let group = B::create_group(&self.file, "obsp")?;
            let arrays = AxisArrays::new(group, Axis::RowColumn, self.n_obs.clone())?;
            self.obsp.swap(&arrays);
        }
        self.get_obsp().inner().add_data(key, data)
    }
    fn add_varm_item<D: WriteArrayData + ArrayOp + Into<ArrayData>>(&self, key: &str, data: D) -> Result<()> {
        if self.get_varm().is_empty() {
            let group = B::create_group(&self.file, "varm")?;
            let arrays = AxisArrays::new(group, Axis::Row, self.n_vars.clone())?;
            self.varm.swap(&arrays);
        }
        self.varm.inner().add_data(key, data)
    }
    fn add_varp_item<D: WriteArrayData + ArrayOp + Into<ArrayData>>(&self, key: &str, data: D) -> Result<()> {
        if self.get_varp().is_empty() {
            let group = B::create_group(&self.file, "varp")?;
            let arrays = AxisArrays::new(group, Axis::RowColumn, self.n_vars.clone())?;
            self.varp.swap(&arrays);
        }
        self.varp.inner().add_data(key, data)
    }
}

/*
impl AnnDataOp for AnnDataSet {
    fn read_x(&self) -> Result<Option<Box<dyn ArrayData>>> {
        let adatas = &self.anndatas;
        if adatas.is_empty() {
            Ok(None)
        } else {
            adatas.inner().x.read(None, None).map(Option::Some)
        }
    }
    fn set_x<D: ArrayData>(&self, _: Option<D>) -> Result<()> {
        bail!("cannot set X in AnnDataSet")
    }

    fn n_obs(&self) -> usize {
        *self.anndatas.inner().n_obs.lock()
    }
    fn n_vars(&self) -> usize {
        *self.anndatas.inner().n_vars.lock()
    }

    fn obs_ix(&self, names: &[String]) -> Result<Vec<usize>> {
        self.annotation.obs_ix(names)
    }
    fn var_ix(&self, names: &[String]) -> Result<Vec<usize>> {
        self.annotation.var_ix(names)
    }
    fn obs_names(&self) -> Vec<String> {
        self.annotation.obs_names()
    }
    fn var_names(&self) -> Vec<String> {
        self.annotation.var_names()
    }
    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()> {
        self.annotation.set_obs_names(index)
    }
    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> {
        self.annotation.set_var_names(index)
    }

    fn read_obs(&self) -> Result<DataFrame> {
        self.annotation.read_obs()
    }
    fn read_var(&self) -> Result<DataFrame> {
        self.annotation.read_var()
    }
    fn set_obs(&self, obs: Option<DataFrame>) -> Result<()> {
        self.annotation.set_obs(obs)
    }
    fn set_var(&self, var: Option<DataFrame>) -> Result<()> {
        self.annotation.set_var(var)
    }

    fn uns_keys(&self) -> Vec<String> {
        self.annotation.uns_keys()
    }
    fn obsm_keys(&self) -> Vec<String> {
        self.annotation.obsm_keys()
    }
    fn obsp_keys(&self) -> Vec<String> {
        self.annotation.obsp_keys()
    }
    fn varm_keys(&self) -> Vec<String> {
        self.annotation.varm_keys()
    }
    fn varp_keys(&self) -> Vec<String> {
        self.annotation.varp_keys()
    }

    fn read_uns_item(&self, key: &str) -> Result<Option<Box<dyn Data>>> {
        self.annotation.read_uns_item(key)
    }
    fn read_obsm_item(&self, key: &str) -> Result<Option<Box<dyn ArrayData>>> {
        self.annotation.read_obsm_item(key)
    }
    fn read_obsp_item(&self, key: &str) -> Result<Option<Box<dyn ArrayData>>> {
        self.annotation.read_obsp_item(key)
    }
    fn read_varm_item(&self, key: &str) -> Result<Option<Box<dyn ArrayData>>> {
        self.annotation.read_varm_item(key)
    }
    fn read_varp_item(&self, key: &str) -> Result<Option<Box<dyn ArrayData>>> {
        self.annotation.read_varp_item(key)
    }

    fn add_uns_item<D: Data>(&self, key: &str, data: D) -> Result<()> {
        self.annotation.add_uns_item(key, data)
    }
    fn add_obsm_item<D: ArrayData>(&self, key: &str, data: D) -> Result<()> {
        self.annotation.add_obsm_item(key, data)
    }
    fn add_obsp_item<D: ArrayData>(&self, key: &str, data: D) -> Result<()> {
        self.annotation.add_obsp_item(key, data)
    }
    fn add_varm_item<D: ArrayData>(&self, key: &str, data: D) -> Result<()> {
        self.annotation.add_varm_item(key, data)
    }
    fn add_varp_item<D: ArrayData>(&self, key: &str, data: D) -> Result<()> {
        self.annotation.add_varp_item(key, data)
    }
}
*/