use crate::{
    traits::AnnDataOp,
    data::*,
    element::{*, collection::InnerElemCollection},
    element::{
        base::{InnerDataFrameElem},
        collection::{AxisArrays, InnerAxisArrays},
    },
    backend::{Backend, GroupOp, FileOp},
};

use anyhow::{ensure, bail, Context, Result};
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
        self.file.filename()
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
        self.file.close()
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
            *n_obs = BoundedSelectInfoElem::new(i.as_ref(), *n_obs).unwrap().len();
            Ok::<(), anyhow::Error>(())
        }).transpose()?;

        selection.as_ref().get(1).map(|i| {
            self.var.lock().as_mut().map(|x| x.subset_rows(i)).transpose()?;
            self.varm.lock().as_ref().map(|varm| varm.subset(i)).transpose()?;
            self.varp.lock().as_ref().map(|varp| varp.subset(i)).transpose()?;
            let mut n_vars = self.n_vars.lock();
            *n_vars = BoundedSelectInfoElem::new(i.as_ref(), *n_vars).unwrap().len();
            Ok::<(), anyhow::Error>(())
        }).transpose()?;
        Ok(())
    }
}

impl<B: Backend> AnnDataOp for AnnData<B> {
    fn read_x<D>(&self) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let x = self.get_x();
        if x.is_empty() {
            Ok(None)
        } else {
            x.inner().data().map(Option::Some)
        }
    }

    fn read_x_slice<D, S>(&self, select: S) -> Result<Option<D>>
    where
        D: ReadArrayData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let x = self.get_x();
        if x.is_empty() {
            Ok(None)
        } else {
            x.inner().select(select).map(Option::Some)
        }
    }

    fn set_x<D: WriteData + Into<ArrayData> + HasShape>(&self, data: D) -> Result<()> {
        let shape = data.shape();
        ensure!(shape.ndim() >= 2, "X must be a N dimensional array, where N >= 2");
        self.set_n_obs(shape[0]);
        self.set_n_vars(shape[1]);
        if !self.x.is_empty() {
            self.x.inner().save(data)?;
        } else {
            let new_elem = ArrayElem::try_from(data.write(&self.file, "X")?)?;
            self.x.swap(&new_elem);
        }
        Ok(())
    }
    fn del_x(&self) -> Result<()> {
        if !self.x.is_empty() {
            self.file.delete("X")?;
            self.x.drop();
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
            let df = InnerDataFrameElem::new(&self.file, "obs", index, &DataFrame::empty())?;
            self.obs.insert(df);
        } else {
            self.obs.inner().set_index(index)?;
        }
        Ok(())
    }

    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> {
        self.set_n_vars(index.len());
        if self.var.is_empty() {
            let df = InnerDataFrameElem::new(&self.file, "var", index, &DataFrame::empty())?;
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
                    &self.file,
                    "obs",
                    DataFrameIndex::from(nrows),
                    &obs,
                )?);
            } else {
                self.obs.inner().save(obs)?;
            }
        } else {
            if !self.obs.is_empty() {
                self.file.delete("obs")?;
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
                    &self.file,
                    "var",
                    DataFrameIndex::from(nrows),
                    &var,
                )?);
            } else {
                self.var.inner().save(var)?;
            }
        } else {
            if !self.var.is_empty() {
                self.file.delete("var")?;
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
            let collection = ElemCollection::new(self.file.create_group("uns")?)?;
            self.uns.swap(&collection);
        }
        self.get_uns().inner().add_data(key, data)
    }
    fn add_obsm_item<D: WriteArrayData + HasShape + Into<ArrayData>>(&self, key: &str, data: D) -> Result<()> {
        if self.get_obsm().is_empty() {
            let group = self.file.create_group("obsm")?;
            let arrays = AxisArrays::new(group, Axis::Row, self.n_obs.clone())?;
            self.obsm.swap(&arrays);
        }
        self.get_obsm().inner().add_data(key, data)
    }
    fn add_obsp_item<D: WriteArrayData + HasShape + Into<ArrayData>>(&self, key: &str, data: D) -> Result<()> {
        if self.get_obsp().is_empty() {
            let group = self.file.create_group("obsp")?;
            let arrays = AxisArrays::new(group, Axis::RowColumn, self.n_obs.clone())?;
            self.obsp.swap(&arrays);
        }
        self.get_obsp().inner().add_data(key, data)
    }
    fn add_varm_item<D: WriteArrayData + HasShape + Into<ArrayData>>(&self, key: &str, data: D) -> Result<()> {
        if self.get_varm().is_empty() {
            let group = self.file.create_group("varm")?;
            let arrays = AxisArrays::new(group, Axis::Row, self.n_vars.clone())?;
            self.varm.swap(&arrays);
        }
        self.varm.inner().add_data(key, data)
    }
    fn add_varp_item<D: WriteArrayData + HasShape + Into<ArrayData>>(&self, key: &str, data: D) -> Result<()> {
        if self.get_varp().is_empty() {
            let group = self.file.create_group("varp")?;
            let arrays = AxisArrays::new(group, Axis::RowColumn, self.n_vars.clone())?;
            self.varp.swap(&arrays);
        }
        self.varp.inner().add_data(key, data)
    }
}