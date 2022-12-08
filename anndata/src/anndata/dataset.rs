use crate::element::base::VecVecIndex;
use crate::{
    anndata::AnnData,
    backend::{Backend, FileOp, GroupOp},
    data::*,
    element::{
        base::InnerDataFrameElem,
        collection::{AxisArrays, InnerAxisArrays},
    },
    element::{collection::InnerElemCollection, *},
    traits::AnnDataOp,
};

use anyhow::{bail, ensure, Context, Result};
use indexmap::map::IndexMap;
use itertools::Itertools;
use parking_lot::Mutex;
use polars::prelude::{DataFrame, NamedFrom, Series};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::{
    collections::HashMap,
    ops::Deref,
    path::{Path, PathBuf},
    sync::Arc,
};

pub struct AnnDataSet<B: Backend> {
    annotation: AnnData<B>,
    anndatas: StackedAnnData<B>,
}

impl<B: Backend> std::fmt::Display for AnnDataSet<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AnnDataSet object with n_obs x n_vars = {} x {} backed at '{}'",
            self.n_obs(),
            self.n_vars(),
            self.annotation.filename().display(),
        )?;
        if self.anndatas.len() > 0 {
            write!(
                f,
                "\ncontains {} AnnData objects with keys: '{}'",
                self.anndatas.len(),
                self.anndatas.keys().join("', '")
            )?;
        }
        if let Some(obs) = self
            .annotation
            .obs
            .lock()
            .as_ref()
            .map(|x| x.get_column_names())
        {
            if !obs.is_empty() {
                write!(f, "\n    obs: '{}'", obs.into_iter().join("', '"))?;
            }
        }
        if let Some(var) = self
            .annotation
            .var
            .lock()
            .as_ref()
            .map(|x| x.get_column_names())
        {
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

impl<B: Backend> AnnDataSet<B> {
    pub fn new<'a, T, S, P>(
        data: T,
        filename: P,
        add_key: &str,
    ) -> Result<Self>
    where
        T: IntoIterator<Item = (S, AnnData<B>)>,
        S: ToString,
        P: AsRef<Path>,
    {
        let anndatas = StackedAnnData::new(data)?;
        let n_obs = anndatas.n_obs;
        let n_vars = anndatas.n_vars;

        let annotation = AnnData::new(filename, n_obs, n_vars)?;
        { // Set UNS. UNS includes children anndata locations.
            let (keys, filenames): (Vec<_>, Vec<_>) = anndatas
                .iter()
                .map(|(k, v)| (k.clone(), v.filename().display().to_string()))
                .unzip();
            let data = DataFrame::new(vec![
                Series::new("keys", keys),
                Series::new("file_path", filenames),
            ])?;
            annotation.add_uns("AnnDataSet", data)?;
        }
        { // Set OBS.
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
        { // Set VAR.
            let adata = anndatas.values().next().unwrap();
            if !adata.var_names().is_empty() {
                annotation.set_var_names(adata.var_names().into_iter().collect())?;
            }
        }
        Ok(Self {
            annotation,
            anndatas,
        })
    }

    pub fn get_x(&self) -> &StackedArrayElem<B> {
        &self.anndatas.x
    }

    /*
    pub fn to_adata<P: AsRef<Path>>(
        &self,
        obs_idx: Option<&[usize]>,
        var_idx: Option<&[usize]>,
        out: P,
    ) -> Result<AnnData> {
        self.annotation.copy(obs_idx, var_idx, out)
    }
    */

    /// Convert AnnDataSet to AnnData object
    pub fn into_adata(self) -> Result<AnnData<B>> {
        for ann in self.anndatas.elems.into_values() {
            ann.close()?;
        }
        Ok(self.annotation)
    }

    pub fn close(self) -> Result<()> {
        self.annotation.close()?;
        for ann in self.anndatas.elems.into_values() {
            ann.close()?;
        }
        Ok(())
    }
}

/*
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


*/

impl<B: Backend> AnnDataOp for AnnDataSet<B> {
    fn read_x<D>(&self) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        Ok(Some(self.anndatas.x.data()?))
    }

    fn read_x_slice<D, S>(&self, select: S) -> Result<Option<D>>
    where
        D: ReadArrayData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        Ok(Some(self.anndatas.x.select(select)?))
    }

    fn set_x<D: WriteData + Into<ArrayData> + HasShape>(&self, data_: D) -> Result<()> {
        bail!("cannot set X in AnnDataSet")
    }

    fn set_x_from_iter<I: Iterator<Item = D>, D: WriteArrayData>(&self, iter: I) -> Result<()> {
        bail!("cannot set X in AnnDataSet")
    }

    fn del_x(&self) -> Result<()> {
        bail!("cannot delete X in AnnDataSet")
    }

    fn n_obs(&self) -> usize {
        self.anndatas.n_obs
    }
    fn n_vars(&self) -> usize {
        self.anndatas.n_vars
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

    fn fetch_uns<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<Data> + TryFrom<Data> + Clone,
        <D as TryFrom<Data>>::Error: Into<anyhow::Error>,
    {
        self.annotation.fetch_uns(key)
    }

    fn fetch_obsm<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.annotation.fetch_obsm(key)
    }

    fn fetch_obsp<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.annotation.fetch_obsp(key)
    }

    fn fetch_varm<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.annotation.fetch_varm(key)
    }

    fn fetch_varp<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.annotation.fetch_varp(key)
    }

    fn add_uns<D: WriteData + Into<Data>>(&self, key: &str, data: D) -> Result<()> {
        self.annotation.add_uns(key, data)
    }

    fn add_obsm<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()> {
        self.annotation.add_obsm(key, data)
    }

    fn add_obsp<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()> {
        self.annotation.add_obsp(key, data)
    }

    fn add_varm<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()> {
        self.annotation.add_varm(key, data)
    }

    fn add_varp<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()> {
        self.annotation.add_varp(key, data)
    }
}

pub struct StackedAnnData<B: Backend> {
    elems: IndexMap<String, AnnData<B>>,
    n_obs: usize,
    n_vars: usize,
    x: StackedArrayElem<B>,
    pub obs: StackedDataFrame<B>,
    obsm: StackedAxisArrays<B>,
}

impl<B: Backend> std::fmt::Display for StackedAnnData<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stacked AnnData objects:")?;
        write!(
            f,
            "\n    obs: '{}'",
            self.obs.get_column_names().iter().join("', '")
        )?;
        write!(f, "\n    obsm: '{}'", self.obsm.keys().join("', '"))?;
        Ok(())
    }
}

impl<B: Backend> StackedAnnData<B> {
    fn new<'a, T, S>(iter: T) -> Result<Self>
    where
        T: IntoIterator<Item = (S, AnnData<B>)>,
        S: ToString,
    {
        let adatas: IndexMap<String, AnnData<B>> = iter.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        ensure!(!adatas.is_empty(), "no AnnData objects to stack");

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

        let x = StackedArrayElem::new(adatas.values().map(|x| x.get_x().clone()).collect())?;

        let obs = if adatas.values().any(|x| x.obs.is_empty()) {
            StackedDataFrame::new(Vec::new())
        } else {
            StackedDataFrame::new(adatas.values().map(|x| x.obs.clone()).collect())
        }?;

        let obsm = {
            let arrays: Vec<AxisArrays<_>> = adatas.values().map(|x| x.obsm.clone()).collect();
            StackedAxisArrays::new(Axis::Row, arrays)?
        };

        let index_: VecVecIndex = adatas.values().map(|x| x.n_obs()).collect();
        let n_obs = Arc::new(Mutex::new(index_.len()));
        let index = Arc::new(Mutex::new(index_));

        Ok(Self {
            n_obs: adatas.values().map(|x| x.n_obs()).sum(),
            n_vars: adatas.values().next().unwrap().n_vars(),
            elems: adatas,
            x,
            obs,
            obsm,
        })
    }

    pub fn get_x(&self) -> &StackedArrayElem<B> {
        &self.x
    }
    pub fn get_obsm(&self) -> &StackedAxisArrays<B> {
        &self.obsm
    }

    pub fn len(&self) -> usize {
        self.elems.len()
    }

    pub fn keys(&self) -> indexmap::map::Keys<'_, String, AnnData<B>> {
        self.elems.keys()
    }

    pub fn values(&self) -> indexmap::map::Values<'_, String, AnnData<B>> {
        self.elems.values()
    }

    pub fn iter(&self) -> indexmap::map::Iter<'_, String, AnnData<B>> {
        self.elems.iter()
    }

    /*
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
    */
}
