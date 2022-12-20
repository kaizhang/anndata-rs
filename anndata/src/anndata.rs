mod dataset;

pub use dataset::{AnnDataSet, StackedAnnData};
use smallvec::SmallVec;

use crate::{
    backend::{Backend, DataContainer, FileOp, GroupOp},
    container::{
        ArrayElem, Axis, AxisArrays, ChunkedArrayElem, DataFrameElem, ElemCollection,
        InnerDataFrameElem, Slot,
    },
    data::*,
    traits::{AnnDataIterator, AnnDataOp},
};

use anyhow::{bail, ensure, Context, Result};
use itertools::Itertools;
use parking_lot::{Mutex, MutexGuard};
use polars::prelude::DataFrame;
use std::{
    ops::Deref,
    path::{Path, PathBuf},
    sync::Arc,
};

pub struct AnnData<B: Backend> {
    file: B::File,
    // Put n_obs in a mutex to allow concurrent access to different slots
    // because modifying n_obs requires modifying slots will also modify n_obs.
    // Operations that modify n_obs must acquire a lock until the end of the operation.
    n_obs: Arc<Mutex<usize>>,
    n_vars: Arc<Mutex<usize>>,
    x: ArrayElem<B>,
    obs: DataFrameElem<B>,
    obsm: AxisArrays<B>,
    obsp: AxisArrays<B>,
    var: DataFrameElem<B>,
    varm: AxisArrays<B>,
    varp: AxisArrays<B>,
    uns: ElemCollection<B>,
}

impl<B: Backend> std::fmt::Display for AnnData<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AnnData object with n_obs x n_vars = {} x {} backed at '{}'",
            self.n_obs(),
            self.n_vars(),
            self.filename().to_str().unwrap().to_string(),
        )?;
        if let Some(obs) = self.obs.lock().as_ref().map(|x| x.get_column_names()) {
            if !obs.is_empty() {
                write!(f, "\n    obs: '{}'", obs.into_iter().join("', '"))?;
            }
        }
        if let Some(var) = self.var.lock().as_ref().map(|x| x.get_column_names()) {
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

impl<B: Backend> AnnData<B> {
    pub(crate) fn lock(&self) -> AnnDataLock<'_> {
        AnnDataLock {
            n_obs: self.lock_obs(),
            n_vars: self.lock_vars(),
        }
    }
    pub(crate) fn lock_obs(&self) -> Lock<'_> {
        Lock {
            guard: self.n_obs.lock(),
            is_empty: self.x.is_empty()
                && self.obs.is_empty()
                && (self.obsm.is_empty() || self.obsm.inner().is_empty())
                && (self.obsp.is_empty() || self.obsp.inner().is_empty()),
        }
    }
    pub(crate) fn lock_vars(&self) -> Lock<'_> {
        Lock {
            guard: self.n_vars.lock(),
            is_empty: self.x.is_empty()
                && self.var.is_empty()
                && (self.varm.is_empty() || self.varm.inner().is_empty())
                && (self.varp.is_empty() || self.varp.inner().is_empty()),
        }
    }

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

    /// Open an existing AnnData.
    pub fn open(file: B::File) -> Result<Self> {
        let n_obs = Arc::new(Mutex::new(0));
        let n_vars = Arc::new(Mutex::new(0));

        // Read X
        let x = if file.exists("X")? {
            let x = ArrayElem::try_from(DataContainer::open(&file, "X")?)?;
            *n_obs.lock() = x.inner().shape()[0];
            *n_vars.lock() = x.inner().shape()[1];
            x
        } else {
            Slot::empty()
        };

        // Read obs
        let obs = if file.exists("obs")? {
            let obs = DataFrameElem::try_from(DataContainer::open(&file, "obs")?)?;
            let n_records = obs.inner().height();
            let n = *n_obs.lock();
            if n == 0 {
                *n_obs.lock() = n_records;
            } else {
                assert!(
                    n == n_records,
                    "Inconsistent number of observations: {} (X) != {} (obs)",
                    n,
                    n_records,
                );
            }
            obs
        } else {
            Slot::empty()
        };

        // Read var
        let var = if file.exists("var")? {
            let var = DataFrameElem::try_from(DataContainer::open(&file, "var")?)?;
            let n_records = obs.inner().height();
            let n = *n_vars.lock();
            if n == 0 {
                *n_vars.lock() = n_records;
            } else {
                assert!(
                    n == n_records,
                    "Inconsistent number of variables: {} (X) != {} (var)",
                    n,
                    n_records,
                );
            }
            var
        } else {
            Slot::empty()
        };

        let obsm = match file.open_group("obsm").or(file.create_group("obsm")) {
            Ok(group) => AxisArrays::new(group, Axis::Row, n_obs.clone())?,
            _ => AxisArrays::empty(),
        };

        let obsp = match file.open_group("obsp").or(file.create_group("obsp")) {
            Ok(group) => AxisArrays::new(group, Axis::RowColumn, n_obs.clone())?,
            _ => AxisArrays::empty(),
        };

        let varm = match file.open_group("varm").or(file.create_group("varm")) {
            Ok(group) => AxisArrays::new(group, Axis::Row, n_vars.clone())?,
            _ => AxisArrays::empty(),
        };

        let varp = match file.open_group("varp").or(file.create_group("varp")) {
            Ok(group) => AxisArrays::new(group, Axis::RowColumn, n_vars.clone())?,
            _ => AxisArrays::empty(),
        };

        let uns = match file.open_group("uns").or(file.create_group("uns")) {
            Ok(group) => ElemCollection::new(group)?,
            _ => ElemCollection::empty(),
        };

        Ok(Self {
            file,
            n_obs,
            n_vars,
            x,
            obs,
            obsm,
            obsp,
            var,
            varm,
            varp,
            uns,
        })
    }

    pub fn new<P: AsRef<Path>>(filename: P, n_obs: usize, n_vars: usize) -> Result<Self> {
        let file = B::create(filename)?;
        let n_obs = Arc::new(Mutex::new(n_obs));
        let n_vars = Arc::new(Mutex::new(n_vars));
        Ok(Self {
            x: Slot::empty(),
            obs: Slot::empty(),
            var: Slot::empty(),
            obsm: AxisArrays::new(file.create_group("obsm")?, Axis::Row, n_obs.clone())?,
            obsp: AxisArrays::new(file.create_group("obsp")?, Axis::RowColumn, n_obs.clone())?,
            varm: AxisArrays::new(file.create_group("varm")?, Axis::Row, n_vars.clone())?,
            varp: AxisArrays::new(file.create_group("varp")?, Axis::RowColumn, n_vars.clone())?,
            uns: ElemCollection::new(file.create_group("uns")?)?,
            file,
            n_obs,
            n_vars,
        })
    }

    pub fn write<O: Backend, P: AsRef<Path>>(&self, filename: P) -> Result<()> {
        let file = O::create(filename)?;
        self.get_x()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file, "X"))
            .transpose()?;
        self.get_obs()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file, "obs"))
            .transpose()?;
        self.get_var()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file, "var"))
            .transpose()?;
        self.get_obsm()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file, "obsm"))
            .transpose()?;
        self.get_obsp()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file, "obsp"))
            .transpose()?;
        self.get_varm()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file, "varm"))
            .transpose()?;
        self.get_varp()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file, "varp"))
            .transpose()?;
        self.get_uns()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file, "uns"))
            .transpose()?;
        file.close()?;
        Ok(())
    }

    pub fn write_select<O, S, P>(&self, selection: S, filename: P) -> Result<()>
    where
        O: Backend,
        S: AsRef<[SelectInfoElem]>,
        P: AsRef<Path>,
    {
        let slice: SmallVec<[_; 3]> = selection.as_ref().iter().collect();
        let _lock = self.lock();
        let file = O::create(filename)?;
        self.get_x()
            .lock()
            .as_mut()
            .map(|x| x.export_select::<O, _>(slice.as_slice(), &file, "X"))
            .transpose()?;

        /*
        self.get_obs()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file, "obs"))
            .transpose()?;
        self.get_var()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file, "var"))
            .transpose()?;
        self.get_obsm()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file.create_group("obsm")?))
            .transpose()?;
        self.get_obsp()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file.create_group("obsp")?))
            .transpose()?;
        self.get_varm()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file.create_group("varm")?))
            .transpose()?;
        self.get_varp()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file.create_group("varp")?))
            .transpose()?;
        self.get_uns()
            .lock()
            .as_mut()
            .map(|x| x.export::<O>(&file.create_group("uns")?))
            .transpose()?;
        */
        file.close()?;
        Ok(())
    }

    pub fn filename(&self) -> PathBuf {
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

    pub fn subset<S>(&self, selection: S) -> Result<()>
    where
        S: AsRef<[SelectInfoElem]>,
    {
        let mut lock = self.lock();
        let slice = selection.as_ref();
        ensure!(
            slice.len() == 2,
            format!("subset only supports 2D selections, got {}", slice.len())
        );
        let obs_ix = &slice[0];
        let var_ix = &slice[1];

        self.x
            .lock()
            .as_mut()
            .map(|x| x.subset(slice))
            .transpose()?;

        self.obs
            .lock()
            .as_mut()
            .map(|x| x.subset_axis(0, obs_ix))
            .transpose()?;
        self.obsm
            .lock()
            .as_ref()
            .map(|obsm| obsm.subset(obs_ix))
            .transpose()?;
        self.obsp
            .lock()
            .as_ref()
            .map(|obsp| obsp.subset(obs_ix))
            .transpose()?;
        lock.n_obs
            .set(BoundedSelectInfoElem::new(obs_ix, *lock.n_obs).len());

        self.var
            .lock()
            .as_mut()
            .map(|x| x.subset_axis(0, var_ix))
            .transpose()?;
        self.varm
            .lock()
            .as_ref()
            .map(|varm| varm.subset(var_ix))
            .transpose()?;
        self.varp
            .lock()
            .as_ref()
            .map(|varp| varp.subset(var_ix))
            .transpose()?;
        lock.n_vars
            .set(BoundedSelectInfoElem::new(var_ix, *lock.n_vars).len());

        Ok(())
    }
}

pub(crate) struct AnnDataLock<'a> {
    pub(crate) n_obs: Lock<'a>,
    pub(crate) n_vars: Lock<'a>,
}

pub(crate) struct Lock<'a> {
    guard: MutexGuard<'a, usize>,
    is_empty: bool,
}

impl Deref for Lock<'_> {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl Lock<'_> {
    pub fn set(&mut self, n: usize) {
        *self.guard = n;
    }

    pub fn try_set(&mut self, n: usize) -> Result<()> {
        if *self.guard != n {
            if self.is_empty {
                *self.guard = n;
            } else {
                bail!("cannot change n_obs or n_vars");
            }
        }
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
            x.inner().select(select.as_ref()).map(Option::Some)
        }
    }

    fn set_x<D: WriteArrayData + Into<ArrayData> + HasShape>(&self, data: D) -> Result<()> {
        let mut lock = self.lock();
        let shape = data.shape();
        ensure!(
            shape.ndim() >= 2,
            "X must be a N dimensional array, where N >= 2"
        );
        lock.n_obs.try_set(shape[0])?;
        lock.n_vars.try_set(shape[1])?;
        if !self.x.is_empty() {
            self.x.inner().save(data)?;
        } else {
            let new_elem = ArrayElem::try_from(data.write(&self.file, "X")?)?;
            self.x.swap(&new_elem);
        }
        Ok(())
    }

    fn del_x(&self) -> Result<()> {
        self.x.clear()
    }

    fn n_obs(&self) -> usize {
        *self.n_obs.lock()
    }
    fn n_vars(&self) -> usize {
        *self.n_vars.lock()
    }

    fn obs_names(&self) -> Vec<String> {
        self.obs
            .lock()
            .as_ref()
            .map_or(Vec::new(), |obs| obs.index.names.clone())
    }

    fn var_names(&self) -> Vec<String> {
        self.var
            .lock()
            .as_ref()
            .map_or(Vec::new(), |var| var.index.names.clone())
    }

    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()> {
        let mut obs_lock = self.lock_obs();
        obs_lock.try_set(index.len())?;
        if self.obs.is_empty() {
            let df = InnerDataFrameElem::new(&self.file, "obs", index, &DataFrame::empty())?;
            self.obs.insert(df);
        } else {
            self.obs.inner().set_index(index)?;
        }
        Ok(())
    }

    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> {
        let mut var_lock = self.lock_vars();
        var_lock.try_set(index.len())?;
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
        self.get_obs()
            .lock()
            .as_mut()
            .map_or(Ok(DataFrame::empty()), |x| x.data().map(Clone::clone))
    }
    fn read_var(&self) -> Result<DataFrame> {
        self.get_var()
            .lock()
            .as_mut()
            .map_or(Ok(DataFrame::empty()), |x| x.data().map(Clone::clone))
    }
    fn set_obs(&self, obs: DataFrame) -> Result<()> {
        let nrows = obs.height();
        if nrows != 0 {
            let mut obs_lock = self.lock_obs();
            obs_lock.try_set(nrows)?;
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
        }
        Ok(())
    }

    fn set_var(&self, var: DataFrame) -> Result<()> {
        let nrows = var.height();
        if nrows != 0 {
            let mut vars_lock = self.lock_vars();
            vars_lock.try_set(nrows)?;
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
        }
        Ok(())
    }

    fn del_obs(&self) -> Result<()> {
        self.get_obs().clear()
    }

    fn del_var(&self) -> Result<()> {
        self.get_var().clear()
    }

    fn set_uns<I: Iterator<Item = (String, Data)>>(&self, mut data: I) -> Result<()> {
        self.del_uns()?;
        let uns: ElemCollection<B> = ElemCollection::new(self.file.create_group("uns")?)?;
        data.try_for_each(|(k, v)| uns.inner().add_data(&k, v))?;
        self.get_uns().swap(&uns);
        Ok(())
    }

    fn set_obsm<I: Iterator<Item = (String, ArrayData)>>(&self, mut data: I) -> Result<()> {
        self.del_obsm()?;
        let obsm = AxisArrays::new(
            self.file.create_group("obsm")?,
            Axis::Row,
            self.n_obs.clone(),
        )?;
        data.try_for_each(|(k, v)| obsm.inner().add_data(&k, v))?;
        self.get_obsm().swap(&obsm);
        Ok(())
    }

    fn set_obsp<I: Iterator<Item = (String, ArrayData)>>(&self, mut data: I) -> Result<()> {
        self.del_obsp()?;
        let obsp = AxisArrays::new(
            self.file.create_group("obsp")?,
            Axis::RowColumn,
            self.n_obs.clone(),
        )?;
        data.try_for_each(|(k, v)| obsp.inner().add_data(&k, v))?;
        self.get_obsp().swap(&obsp);
        Ok(())
    }

    fn set_varm<I: Iterator<Item = (String, ArrayData)>>(&self, mut data: I) -> Result<()> {
        self.del_varm()?;
        let varm = AxisArrays::new(
            self.file.create_group("varm")?,
            Axis::Row,
            self.n_vars.clone(),
        )?;
        data.try_for_each(|(k, v)| varm.inner().add_data(&k, v))?;
        self.get_varm().swap(&varm);
        Ok(())
    }

    fn set_varp<I: Iterator<Item = (String, ArrayData)>>(&self, mut data: I) -> Result<()> {
        self.del_varp()?;
        let varp = AxisArrays::new(
            self.file.create_group("varp")?,
            Axis::RowColumn,
            self.n_vars.clone(),
        )?;
        data.try_for_each(|(k, v)| varp.inner().add_data(&k, v))?;
        self.get_varp().swap(&varp);
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

    fn fetch_uns<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<Data> + TryFrom<Data> + Clone,
        <D as TryFrom<Data>>::Error: Into<anyhow::Error>,
    {
        self.get_uns()
            .lock()
            .as_mut()
            .and_then(|x| x.get_mut(key))
            .map(|x| x.inner().data())
            .transpose()
    }
    fn fetch_obsm<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get_obsm()
            .lock()
            .as_mut()
            .and_then(|x| x.get_mut(key))
            .map(|x| x.inner().data())
            .transpose()
    }
    fn fetch_obsp<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get_obsp()
            .lock()
            .as_mut()
            .and_then(|x| x.get_mut(key))
            .map(|x| x.inner().data())
            .transpose()
    }
    fn fetch_varm<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get_varm()
            .lock()
            .as_mut()
            .and_then(|x| x.get_mut(key))
            .map(|x| x.inner().data())
            .transpose()
    }
    fn fetch_varp<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get_varp()
            .lock()
            .as_mut()
            .and_then(|x| x.get_mut(key))
            .map(|x| x.inner().data())
            .transpose()
    }

    fn add_uns<D: WriteData + Into<Data>>(&self, key: &str, data: D) -> Result<()> {
        if self.get_uns().is_empty() {
            let collection = ElemCollection::new(self.file.create_group("uns")?)?;
            self.uns.swap(&collection);
        }
        self.get_uns().inner().add_data(key, data)
    }
    fn add_obsm<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()> {
        if self.get_obsm().is_empty() {
            let group = self.file.create_group("obsm")?;
            let arrays = AxisArrays::new(group, Axis::Row, self.n_obs.clone())?;
            self.obsm.swap(&arrays);
        }
        self.get_obsm().inner().add_data(key, data)
    }
    fn add_obsp<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()> {
        if self.get_obsp().is_empty() {
            let group = self.file.create_group("obsp")?;
            let arrays = AxisArrays::new(group, Axis::RowColumn, self.n_obs.clone())?;
            self.obsp.swap(&arrays);
        }
        self.get_obsp().inner().add_data(key, data)
    }
    fn add_varm<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()> {
        if self.get_varm().is_empty() {
            let group = self.file.create_group("varm")?;
            let arrays = AxisArrays::new(group, Axis::Row, self.n_vars.clone())?;
            self.varm.swap(&arrays);
        }
        self.varm.inner().add_data(key, data)
    }
    fn add_varp<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()> {
        if self.get_varp().is_empty() {
            let group = self.file.create_group("varp")?;
            let arrays = AxisArrays::new(group, Axis::RowColumn, self.n_vars.clone())?;
            self.varp.swap(&arrays);
        }
        self.varp.inner().add_data(key, data)
    }

    fn del_uns(&self) -> Result<()> {
        self.get_uns().clear()
    }
    fn del_obsm(&self) -> Result<()> {
        self.get_obsm().clear()
    }
    fn del_obsp(&self) -> Result<()> {
        self.get_obsp().clear()
    }
    fn del_varm(&self) -> Result<()> {
        self.get_varm().clear()
    }
    fn del_varp(&self) -> Result<()> {
        self.get_varp().clear()
    }
}

impl<B: Backend> AnnDataIterator for AnnData<B> {
    type ArrayIter<'a, T> = ChunkedArrayElem<B, T>
    where
        B: 'a,
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

    fn read_x_iter<'a, T>(&'a self, chunk_size: usize) -> Self::ArrayIter<'a, T>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get_x().chunked(chunk_size)
    }

    /// Set the 'X' element from an iterator. Note that the original data will be
    /// lost if an error occurs during the writing.
    fn set_x_from_iter<I: Iterator<Item = D>, D: WriteArrayData>(&self, iter: I) -> Result<()> {
        let mut lock = self.lock();
        self.del_x()?;
        let new_elem =
            ArrayElem::try_from(WriteArrayData::write_from_iter(iter, &self.file, "X")?)?;
        let shape = new_elem.inner().shape().clone();

        match lock
            .n_obs
            .try_set(shape[0])
            .and(lock.n_vars.try_set(shape[1]))
        {
            Ok(_) => {
                self.x.swap(&new_elem);
                Ok(())
            }
            Err(e) => {
                new_elem.clear()?;
                Err(e)
            }
        }
    }

    fn fetch_obsm_iter<'a, T>(
        &'a self,
        key: &str,
        chunk_size: usize,
    ) -> Result<Self::ArrayIter<'a, T>>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        todo!()
    }

    fn add_obsm_from_iter<I, D>(&self, key: &str, data: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: WriteArrayData,
    {
        todo!()
    }
}
