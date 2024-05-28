use crate::{
    traits::{AnnDataOp, ElemCollectionOp},
    anndata::AnnData,
    backend::Backend,
    container::{Slot, Dim, Axis, AxisArrays, StackedArrayElem, StackedAxisArrays, StackedDataFrame, ElemCollection},
    data::*,
    data::index::VecVecIndex,
};

use anyhow::{anyhow, bail, ensure, Context, Result};
use indexmap::map::IndexMap;
use itertools::Itertools;
use polars::prelude::{DataFrame, NamedFrom, Series};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use std::{collections::{HashMap, HashSet}, path::{Path, PathBuf}};

pub struct AnnDataSet<B: Backend> {
    annotation: AnnData<B>,
    anndatas: Slot<StackedAnnData<B>>,
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
        let adatas = self.anndatas.inner();
        if adatas.len() > 0 {
            write!(
                f,
                "\ncontains {} AnnData objects with keys: '{}'",
                adatas.len(),
                adatas.keys().join("', '")
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
    pub fn adatas(&self) -> &Slot<StackedAnnData<B>> {
        &self.anndatas
    }

    pub fn get_anno(&self) -> &AnnData<B> {
        &self.annotation
    }

    pub fn new<'a, T, S, P>(data: T, filename: P, add_key: &str) -> Result<Self>
    where
        T: IntoIterator<Item = (S, AnnData<B>)>,
        S: ToString,
        P: AsRef<Path>,
    {
        let anndatas = StackedAnnData::new(data)?;
        let n_obs = anndatas.n_obs;
        let n_vars = anndatas.n_vars;

        let mut annotation = AnnData::new(filename)?;
        annotation.n_obs = Dim::new(n_obs);
        annotation.n_vars = Dim::new(n_vars);
        { // Set UNS. UNS includes children anndata locations and shared elements.
            let (keys, filenames): (Vec<_>, Vec<_>) = anndatas
                .iter()
                .map(|(k, v)| (k.clone(), v.filename().display().to_string()))
                .unzip();
            let data = DataFrame::new(vec![
                Series::new("keys", keys),
                Series::new("file_path", filenames),
            ])?;
            annotation.uns().add("AnnDataSet", data)?;

            // Add shared uns elements.
            let shared_keys: HashSet<String> = anndatas
                .values()
                .map(|x| x.uns().keys().into_iter().collect::<HashSet<_>>())
                .reduce(|a, b| a.intersection(&b).cloned().collect())
                .unwrap_or(HashSet::new());
            for key in shared_keys {
                if anndatas.values().map(|x| x.uns().get_item::<Data>(&key).unwrap().unwrap()).all_equal() {
                    annotation.uns().add(&key, anndatas.values().next().unwrap().uns().get_item::<Data>(&key)?.unwrap())?;
                }
            }
        }
        { // Set OBS.
            let obs_names: DataFrameIndex = anndatas.values().flat_map(|x| x.obs_names().into_iter()).collect();
            if !obs_names.is_empty() && obs_names.len() == n_obs {
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
            annotation.set_obs(DataFrame::new(vec![keys])?)?;
        }
        { // Set VAR.
            let adata = anndatas.values().next().unwrap();
            let var_names = adata.var_names();
            if !var_names.is_empty() {
                annotation.set_var_names(var_names)?;
            }
        }
        Ok(Self {
            annotation,
            anndatas: Slot::new(anndatas),
        })
    }

    pub fn open<P: AsRef<Path>>(
        file: B::Store,
        adata_files_update: Option<Result<HashMap<String, P>, P>>
    ) -> Result<Self> {
        let annotation: AnnData<B> = AnnData::open(file)?;
        let file_path = annotation
            .filename()
            .read_link()
            .unwrap_or(annotation.filename())
            .to_path_buf();

        let adata_files = match adata_files_update {
            None => update_anndata_locations_by_map(&annotation, HashMap::<String, P>::new())?,
            Some(Ok(adata_files)) => update_anndata_locations_by_map(&annotation, adata_files)?,
            Some(Err(dir)) => update_anndata_location_dir(&annotation, dir)?,
        };

        let anndatas: Vec<(String, AnnData<B>)> = adata_files
            .into_iter()
            .map(|(k, path)| {
                let fl = if path.is_absolute() {
                    B::open(path)
                } else {
                    B::open(file_path.parent().unwrap_or(Path::new("./")).join(path))
                }?;
                Ok((k, AnnData::open(fl)?))
            })
            .collect::<Result<_>>()?;

        Ok(Self {
            annotation,
            anndatas: Slot::new(StackedAnnData::new(anndatas.into_iter())?),
        })
    }

    /// AnnDataSet will not move data across underlying AnnData objects. So the
    /// orders of rows in the resultant AnnDataSet object may not be consistent
    /// with the input `obs_indices`. This function will return a vector that can
    /// be used to reorder the `obs_indices` to match the final order of rows in
    /// the AnnDataSet.
    pub fn write_select<O: Backend, S: AsRef<[SelectInfoElem]>, P: AsRef<Path>>(
        &self,
        selection: S,
        dir: P,
    ) -> Result<Option<Vec<usize>>> {
        selection.as_ref()[0].bound_check(self.n_obs())
            .map_err(|e| anyhow!("AnnDataSet obs {}", e))?;
        selection.as_ref()[1].bound_check(self.n_vars())
            .map_err(|e| anyhow!("AnnDataSet var {}", e))?;

        let file = dir.as_ref().join("_dataset.h5ads");
        let anndata_dir = dir.as_ref().join("anndatas");
        std::fs::create_dir_all(&anndata_dir)?;

        let (files, obs_idx_order) =
            self.anndatas.inner()
                .write_select::<O, _, _>(&selection, &anndata_dir, ".h5ad")?;

        if let Some(order) = obs_idx_order.as_ref() {
            let idx = BoundedSelectInfoElem::new(&selection.as_ref()[0], self.n_obs()).to_vec();
            let new_idx = order.iter().map(|i| idx[*i]).collect::<SelectInfoElem>();
            self.annotation
                .write_select::<O, _, _>([new_idx, selection.as_ref()[1].clone()], &file)?;
        } else {
            self.annotation.write_select::<O, _, _>(selection, &file)?;
        };

        let adata: AnnData<O> = AnnData::open(O::open_rw(&file)?)?;

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
        adata.uns().add("AnnDataSet", file_loc)?;
        adata.close()?;
        Ok(obs_idx_order)
    }

    /// Convert AnnDataSet to AnnData object
    pub fn to_adata<O: Backend, P: AsRef<Path>>(&self, out: P, copy_x: bool) -> Result<AnnData<O>> {
        self.annotation.write::<O, _>(&out)?;
        let adata = AnnData::open(O::open_rw(&out)?)?;
        if copy_x {
            adata
                .set_x_from_iter::<_, ArrayData>(self.anndatas.inner().x.chunked(500).map(|x| x.0))?;
        }
        Ok(adata)
    }

    pub fn to_adata_select<O, P, S>(&self, select: S, out: P, copy_x: bool) -> Result<AnnData<O>>
    where
        O: Backend,
        P: AsRef<Path>,
        S: AsRef<[SelectInfoElem]>,
    {
        self.annotation.write_select::<O, _, _>(&select, &out)?;
        let adata = AnnData::open(O::open_rw(&out)?)?;
        if copy_x {
            let x: ArrayData = self.anndatas.inner().x.select(select.as_ref())?.unwrap();
            adata.set_x(x)?;
        }
        Ok(adata)
    }

    /// Convert AnnDataSet to AnnData object
    pub fn into_adata(self, copy_x: bool) -> Result<AnnData<B>> {
        if copy_x {
            self.annotation
                .set_x_from_iter::<_, ArrayData>(self.anndatas.inner().x.chunked(500).map(|x| x.0))?;
        }
        for ann in self.anndatas.extract().unwrap().elems.into_values() {
            ann.close()?;
        }
        Ok(self.annotation)
    }

    pub fn close(self) -> Result<()> {
        self.annotation.close()?;
        for ann in self.anndatas.extract().unwrap().elems.into_values() {
            ann.close()?;
        }
        Ok(())
    }
}

/// Update the locations of AnnData files.
fn update_anndata_locations_by_map<B: Backend, P: AsRef<Path>>(
    ann: &AnnData<B>,
    new_locations: HashMap<String, P>,
) -> Result<Vec<(String, PathBuf)>> {
    let df: DataFrame = ann
        .uns().get_item("AnnDataSet")?
        .context("key 'AnnDataSet' is not present")?;
    let keys = df.column("keys").unwrap();
    let filenames = df
        .column("file_path")?
        .str()?
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .unwrap();
    let new_files: Vec<_> = keys
        .str()?
        .into_iter()
        .zip(filenames)
        .map(|(k, v)| {
            let k = k.unwrap();
            let name = new_locations
                .get(k)
                .map_or(PathBuf::from(v), |x| x.as_ref().to_path_buf());
            (k.to_string(), name)
        })
        .collect();
    let data = DataFrame::new(
        vec![keys.clone(),
        Series::new("file_path", new_files.iter().map(|x| x.1.to_str().unwrap().to_string()).collect::<Vec<_>>())]
    ).unwrap();
    if !new_locations.is_empty() {
        ann.uns().add("AnnDataSet", data)?;
    }
    Ok(new_files)
}

fn update_anndata_location_dir<B: Backend, P: AsRef<Path>>(
    ann: &AnnData<B>,
    dir: P,
) -> Result<Vec<(String, PathBuf)>> {
    let df: DataFrame = ann
        .uns().get_item("AnnDataSet")?
        .context("key 'AnnDataSet' is not present")?;
    let keys = df.column("keys").unwrap();
    let file_map: HashMap<String, PathBuf> = std::fs::read_dir(dir)?.map(|x| x.map(|entry|
        (entry.file_name().into_string().unwrap(), entry.path())
    )).collect::<Result<_, std::io::Error>>()?;
    let filenames = df
        .column("file_path")?
        .str()?
        .into_iter()
        .map(Option::unwrap)
        .collect::<Vec<_>>();

    let new_files: Vec<_> = keys
        .str()?
        .into_iter()
        .zip(filenames)
        .map(|(k, filename)| {
            let path = PathBuf::from(filename);
            let name = path.file_name().unwrap().to_str().unwrap();
            (k.unwrap().to_string(), file_map.get(name).map_or(path, |x| std::fs::canonicalize(x).unwrap()))
        })
        .collect();
    let data = DataFrame::new(
        vec![keys.clone(),
        Series::new("file_path", new_files.iter().map(|x| x.1.to_str().unwrap().to_string()).collect::<Vec<_>>())]
    ).unwrap();
    ann.uns().add("AnnDataSet", data)?;
    Ok(new_files)
}


impl<B: Backend> AnnDataOp for AnnDataSet<B> {
    type X = StackedArrayElem<B>;
    type AxisArraysRef<'a> = &'a AxisArrays<B>;
    type ElemCollectionRef<'a> = &'a ElemCollection<B>;

    fn x(&self) -> Self::X {
        self.anndatas.inner().x.clone()
    }

    fn set_x_from_iter<I: Iterator<Item = D>, D: ArrayChunk>(&self, _iter: I) -> Result<()> {
        bail!("cannot set X in AnnDataSet")
    }

    fn set_x<D: WriteData + Into<ArrayData> + HasShape>(&self, _: D) -> Result<()> {
        bail!("cannot set X in AnnDataSet")
    }

    fn del_x(&self) -> Result<()> {
        bail!("cannot delete X in AnnDataSet")
    }

    fn n_obs(&self) -> usize {
        self.anndatas.inner().n_obs
    }
    fn n_vars(&self) -> usize {
        self.anndatas.inner().n_vars
    }
    fn set_n_obs(&self, n: usize) -> Result<()> {
        self.annotation.set_n_obs(n)
    }
    fn set_n_vars(&self, n: usize) -> Result<()> {
        self.annotation.set_n_vars(n)
    }

    fn obs_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>> {
        self.annotation.obs_ix(names)
    }
    fn var_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>> {
        self.annotation.var_ix(names)
    }
    fn obs_names(&self) -> DataFrameIndex {
        self.annotation.obs_names()
    }
    fn var_names(&self) -> DataFrameIndex {
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
    fn set_obs(&self, obs: DataFrame) -> Result<()> {
        self.annotation.set_obs(obs)
    }
    fn set_var(&self, var: DataFrame) -> Result<()> {
        self.annotation.set_var(var)
    }
    fn del_obs(&self) -> Result<()> {
        self.annotation.del_obs()
    }
    fn del_var(&self) -> Result<()> {
        self.annotation.del_var()
    }

    fn uns(&self) -> Self::ElemCollectionRef<'_> {
        self.annotation.uns()
    }
    fn obsm(&self) -> Self::AxisArraysRef<'_> {
        self.annotation.obsm()
    }
    fn obsp(&self) -> Self::AxisArraysRef<'_> {
        self.annotation.obsp()
    }
    fn varm(&self) -> Self::AxisArraysRef<'_> {
        self.annotation.varm()
    }
    fn varp(&self) -> Self::AxisArraysRef<'_> {
        self.annotation.varp()
    }
    fn layers(&self) -> Self::AxisArraysRef<'_> {
        self.annotation.layers()
    }

    fn set_uns<I: Iterator<Item = (String, Data)>>(&self, data: I) -> Result<()> {
        self.annotation.set_uns(data)
    }
    fn set_obsm<I: Iterator<Item = (String, ArrayData)>>(&self, data: I) -> Result<()> {
        self.annotation.set_obsm(data)
    }
    fn set_obsp<I: Iterator<Item = (String, ArrayData)>>(&self, data: I) -> Result<()> {
        self.annotation.set_obsp(data)
    }
    fn set_varm<I: Iterator<Item = (String, ArrayData)>>(&self, data: I) -> Result<()> {
        self.annotation.set_varm(data)
    }
    fn set_varp<I: Iterator<Item = (String, ArrayData)>>(&self, data: I) -> Result<()> {
        self.annotation.set_varp(data)
    }

    fn del_uns(&self) -> Result<()> {
        self.annotation.del_uns()
    }
    fn del_obsm(&self) -> Result<()> {
        self.annotation.del_obsm()
    }
    fn del_obsp(&self) -> Result<()> {
        self.annotation.del_obsp()
    }
    fn del_varm(&self) -> Result<()> {
        self.annotation.del_varm()
    }
    fn del_varp(&self) -> Result<()> {
        self.annotation.del_varp()
    }
    fn del_layers(&self) -> Result<()> {
        self.annotation.del_layers()
    }
}

pub struct StackedAnnData<B: Backend> {
    index: VecVecIndex,
    elems: IndexMap<String, AnnData<B>>,
    n_obs: usize,
    n_vars: usize,
    x: StackedArrayElem<B>,
    obs: StackedDataFrame<B>,
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
        let adatas: IndexMap<String, AnnData<B>> =
            iter.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        ensure!(!adatas.is_empty(), "no AnnData objects to stack");

        if let Some((_, first)) = adatas.first() {
            let lock = first.var.lock();
            let var_names = lock.as_ref().map(|x| &x.index);
            if !adatas
                .par_values()
                .skip(1)
                .all(|x| x.var.lock().as_ref().map(|x| &x.index).eq(&var_names))
            {
                bail!("var names mismatch");
            }
        }

        let x = StackedArrayElem::new(adatas.values().map(|x| x.get_x().clone()).collect())?;

        let obs = if adatas.values().any(|x| x.obs.is_none()) {
            StackedDataFrame::new(Vec::new())
        } else {
            StackedDataFrame::new(adatas.values().map(|x| x.obs.clone()).collect())
        }?;

        let obsm = {
            let arrays: Vec<AxisArrays<_>> = adatas.values().map(|x| x.obsm.clone()).collect();
            StackedAxisArrays::new(Axis::Row, arrays)?
        };

        Ok(Self {
            index: adatas.values().map(|x| x.n_obs()).collect(),
            n_obs: adatas.values().map(|x| x.n_obs()).sum(),
            n_vars: adatas.values().next().unwrap().n_vars(),
            elems: adatas,
            x,
            obs,
            obsm,
        })
    }

    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    pub fn get_x(&self) -> &StackedArrayElem<B> {
        &self.x
    }

    pub fn get_obs(&self) -> &StackedDataFrame<B> {
        &self.obs
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

    /// Write a part of stacked AnnData objects to disk, return the key and
    /// file name (without parent paths)
    pub fn write_select<O, S, P>(
        &self,
        selection: S,
        dir: P,
        suffix: &str,
    ) -> Result<(IndexMap<String, String>, Option<Vec<usize>>)>
    where
        O: Backend,
        S: AsRef<[SelectInfoElem]>,
        P: AsRef<Path> + std::marker::Sync,
    {
        let slice = selection.as_ref();
        ensure!(slice.len() == 2, "selection must be 2D");

        let (slices, mapping) = self.index.split_select(&slice[0]);

        let files: Result<_> = self
            .elems
            .iter()
            .enumerate()
            .map(|(i, (k, adata))| {
                let name = k.to_owned() + suffix;
                let file = dir.as_ref().join(&name);
                let select = if let Some(s) = slices.get(&i) {
                    [s.clone(), slice[1].clone()]
                } else {
                    [Vec::new().into(), slice[1].clone()]
                };
                adata.write_select::<O, _, _>(select, file)?;
                Ok((k.clone(), name))
            })
            .collect();
        Ok((files?, mapping))
    }
}