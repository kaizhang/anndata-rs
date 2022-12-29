mod dataset;

pub use dataset::{AnnDataSet, StackedAnnData};
use smallvec::SmallVec;

use crate::{
    backend::{Backend, DataContainer, FileOp, GroupOp},
    container::{
        Dim, ArrayElem, Axis, AxisArrays, ChunkedArrayElem, DataFrameElem, ElemCollection,
        InnerDataFrameElem, Slot,
    },
    data::*,
    traits::{AnnDataIterator, AnnDataOp, AxisArraysOp, ElemCollectionOp},
};

use anyhow::{ensure, Context, Result};
use itertools::Itertools;
use polars::prelude::DataFrame;
use std::path::{Path, PathBuf};

pub struct AnnData<B: Backend> {
    file: B::File,
    // Put n_obs in a Slot to allow concurrent access to different slots
    // because modifying n_obs requires modifying slots will also modify n_obs.
    // Operations that modify n_obs must acquire a lock until the end of the operation.
    pub(crate) n_obs: Dim,
    pub(crate) n_vars: Dim,
    x: ArrayElem<B>,
    obs: DataFrameElem<B>,
    obsm: AxisArrays<B>,
    obsp: AxisArrays<B>,
    var: DataFrameElem<B>,
    varm: AxisArrays<B>,
    varp: AxisArrays<B>,
    uns: ElemCollection<B>,
}

impl<B: Backend> std::fmt::Debug for AnnData<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
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
    pub(crate) fn obs_is_empty(&self) -> bool {
        self.x.is_empty()
            && self.obs.is_empty()
            && self.obsm.is_empty()
            && self.obsp.is_empty()
    }

    pub(crate) fn var_is_empty(&self) -> bool {
        self.x.is_empty()
            && self.var.is_empty()
            && self.varm.is_empty()
            && self.varp.is_empty()
    }

    pub fn get_x(&self) -> &ArrayElem<B> {
        &self.x
    }
    pub fn get_obs(&self) -> &DataFrameElem<B> {
        &self.obs
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
        let n_obs = Dim::empty();
        let n_vars = Dim::empty();

        // Read X
        let x = if file.exists("X")? {
            let x = ArrayElem::try_from(DataContainer::open(&file, "X")?)?;
            n_obs.try_set(x.inner().shape()[0])?;
            n_vars.try_set(x.inner().shape()[1])?;
            x
        } else {
            Slot::empty()
        };

        // Read obs
        let obs = if file.exists("obs")? {
            let obs = DataFrameElem::try_from(DataContainer::open(&file, "obs")?)?;
            n_obs.try_set(obs.inner().height())?;
            obs
        } else {
            Slot::empty()
        };

        // Read var
        let var = if file.exists("var")? {
            let var = DataFrameElem::try_from(DataContainer::open(&file, "var")?)?;
            n_vars.try_set(var.inner().height())?;
            var
        } else {
            Slot::empty()
        };

        let obsm = match file.open_group("obsm").or(file.create_group("obsm")) {
            Ok(group) => AxisArrays::new(group, Axis::Row, &n_obs)?,
            _ => AxisArrays::empty(),
        };

        let obsp = match file.open_group("obsp").or(file.create_group("obsp")) {
            Ok(group) => AxisArrays::new(group, Axis::RowColumn, &n_obs)?,
            _ => AxisArrays::empty(),
        };

        let varm = match file.open_group("varm").or(file.create_group("varm")) {
            Ok(group) => AxisArrays::new(group, Axis::Row, &n_vars)?,
            _ => AxisArrays::empty(),
        };

        let varp = match file.open_group("varp").or(file.create_group("varp")) {
            Ok(group) => AxisArrays::new(group, Axis::RowColumn, &n_vars)?,
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

    pub fn new<P: AsRef<Path>>(filename: P) -> Result<Self> {
        let file = B::create(filename)?;
        let n_obs = Dim::empty();
        let n_vars = Dim::empty();
        Ok(Self {
            x: Slot::empty(),
            obs: Slot::empty(),
            var: Slot::empty(),
            obsm: AxisArrays::new(file.create_group("obsm")?, Axis::Row, &n_obs)?,
            obsp: AxisArrays::new(file.create_group("obsp")?, Axis::RowColumn, &n_obs)?,
            varm: AxisArrays::new(file.create_group("varm")?, Axis::Row, &n_vars)?,
            varp: AxisArrays::new(file.create_group("varp")?, Axis::RowColumn, &n_vars)?,
            uns: ElemCollection::new(file.create_group("uns")?)?,
            file,
            n_obs,
            n_vars,
        })
    }

    pub fn write<O: Backend, P: AsRef<Path>>(&self, filename: P) -> Result<()> {
        let file = O::create(filename)?;
        let _obs_lock = self.n_obs.lock();
        let _vars_lock = self.n_vars.lock();
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
        self.obsm()
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
        let file = O::create(filename)?;
        let _obs_lock = self.n_obs.lock();
        let _vars_lock = self.n_vars.lock();
        self.get_x()
            .lock()
            .as_mut()
            .map(|x| x.export_select::<O, _>(slice.as_slice(), &file, "X"))
            .transpose()?;

        self.get_obs()
            .lock()
            .as_mut()
            .map(|x| x.export_axis(0, slice[0], &file, "obs"))
            .transpose()?;
        self.get_var()
            .lock()
            .as_mut()
            .map(|x| x.export_axis(0, slice[1], &file, "var"))
            .transpose()?;
        self.get_uns()
            .lock()
            .as_mut()
            .map(|x| x.export(&file, "uns"))
            .transpose()?;
        self.obsm()
            .lock()
            .as_mut()
            .map(|x| x.export_select(slice[0], &file, "obsm"))
            .transpose()?;
        self.get_obsp()
            .lock()
            .as_mut()
            .map(|x| x.export_select(slice[0], &file, "obsp"))
            .transpose()?;
        self.get_varm()
            .lock()
            .as_mut()
            .map(|x| x.export_select(slice[1], &file, "varm"))
            .transpose()?;
        self.get_varp()
            .lock()
            .as_mut()
            .map(|x| x.export_select(slice[1], &file, "varp"))
            .transpose()?;
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
        let mut obs_lock = self.n_obs.lock();
        let mut vars_lock = self.n_vars.lock();
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
            .as_mut()
            .map(|obsm| obsm.subset(obs_ix))
            .transpose()?;
        self.obsp
            .lock()
            .as_mut()
            .map(|obsp| obsp.subset(obs_ix))
            .transpose()?;

        self.var
            .lock()
            .as_mut()
            .map(|x| x.subset_axis(0, var_ix))
            .transpose()?;
        self.varm
            .lock()
            .as_mut()
            .map(|varm| varm.subset(var_ix))
            .transpose()?;
        self.varp
            .lock()
            .as_mut()
            .map(|varp| varp.subset(var_ix))
            .transpose()?;

        if !obs_lock.is_empty() {
            obs_lock.set(BoundedSelectInfoElem::new(obs_ix, obs_lock.get()).len());
        }
        if !vars_lock.is_empty() {
            vars_lock.set(BoundedSelectInfoElem::new(var_ix, vars_lock.get()).len());
        }

        Ok(())
    }
}

impl<B: Backend> AnnDataOp for AnnData<B> {
    type AxisArraysRef<'a> = &'a AxisArrays<B>;
    type ElemCollectionRef<'a> = &'a ElemCollection<B>;

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
        let shape = data.shape();
        ensure!(
            shape.ndim() >= 2,
            "X must be a N dimensional array, where N >= 2"
        );
        self.n_obs.try_set(shape[0])?;
        self.n_vars.try_set(shape[1])?;

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
        self.n_obs.get()
    }
    fn n_vars(&self) -> usize {
        self.n_vars.get()
    }

    fn obs_names(&self) -> DataFrameIndex {
        self.obs
            .lock()
            .as_ref()
            .map_or(DataFrameIndex::new(), |obs| obs.index.clone())
    }

    fn var_names(&self) -> DataFrameIndex {
        self.var
            .lock()
            .as_ref()
            .map_or(DataFrameIndex::new(), |var| var.index.clone())
    }

    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()> {
        self.n_obs.try_set(index.len())?;
        if self.obs.is_empty() {
            let df = InnerDataFrameElem::new(&self.file, "obs", index, &DataFrame::empty())?;
            self.obs.insert(df);
        } else {
            self.obs.inner().set_index(index)?;
        }
        Ok(())
    }

    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> {
        self.n_vars.try_set(index.len())?;
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
    // TODO: empty dataframe should be allowed
    fn set_obs(&self, obs: DataFrame) -> Result<()> {
        let nrows = obs.height();
        if nrows != 0 {
            self.n_obs.try_set(nrows)?;
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
            self.n_vars.try_set(nrows)?;
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

    fn uns(&self) -> Self::ElemCollectionRef<'_> {
        if self.uns.is_empty() {
            let elems = self.file.create_group("uns").and_then(|g| ElemCollection::new(g));
            if let Ok(uns) = elems {
                self.uns().swap(&uns);
            }
        }
        &self.uns
    }
    fn obsm(&self) -> Self::AxisArraysRef<'_> {
        if self.obsm.is_empty() {
            let arrays = self.file.create_group("obsm")
                .and_then(|g| AxisArrays::new(g, Axis::Row, &self.n_obs));
            if let Ok(obsm) = arrays {
                self.obsm().swap(&obsm);
            }
        }
        &self.obsm
    }
    fn obsp(&self) -> Self::AxisArraysRef<'_> {
        if self.obsp.is_empty() {
            let arrays = self.file.create_group("obsp")
                .and_then(|g| AxisArrays::new(g, Axis::RowColumn, &self.n_obs));
            if let Ok(obsp) = arrays {
                self.obsp().swap(&obsp);
            }
        }
        &self.obsp
    }
    fn varm(&self) -> Self::AxisArraysRef<'_> {
        if self.varm.is_empty() {
            let arrays = self.file.create_group("varm")
                .and_then(|g| AxisArrays::new(g, Axis::Row, &self.n_vars));
            if let Ok(varm) = arrays {
                self.varm().swap(&varm);
            }
        }
        &self.varm
    }
    fn varp(&self) -> Self::AxisArraysRef<'_> {
        if self.varp.is_empty() {
            let arrays = self.file.create_group("varp")
                .and_then(|g| AxisArrays::new(g, Axis::RowColumn, &self.n_vars));
            if let Ok(varp) = arrays {
                self.varp().swap(&varp);
            }
        }
        &self.varp
    }

    fn del_uns(&self) -> Result<()> {
        self.get_uns().clear()
    }
    fn del_obsm(&self) -> Result<()> {
        self.obsm().clear()
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
    fn set_x_from_iter<I: Iterator<Item = D>, D: ArrayChunk>(&self, iter: I) -> Result<()> {
        let mut obs_lock = self.n_obs.lock();
        let mut vars_lock = self.n_vars.lock();
        self.del_x()?;
        let new_elem =
            ArrayElem::try_from(ArrayChunk::write_by_chunk(iter, &self.file, "X")?)?;
        let shape = new_elem.inner().shape().clone();

        match obs_lock
            .try_set(shape[0])
            .and(vars_lock.try_set(shape[1]))
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
}

impl<B: Backend> AxisArraysOp for &AxisArrays<B> {
    type ArrayIter<'a, T> = ChunkedArrayElem<B, T>
    where
        B: 'a,
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
        Self: 'a;

    fn keys(&self) -> Vec<String> {
        self.inner().keys().cloned().collect()
    }

    fn get<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.lock()
            .as_mut()
            .and_then(|x| x.get_mut(key))
            .map(|x| x.inner().data())
            .transpose()
    }

    fn get_slice<D, S>(&self, key: &str, slice: S) -> Result<Option<D>>
    where
        D: ReadArrayData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>
    {
        self.lock()
            .as_mut()
            .and_then(|x| x.get_mut(key))
            .map(|x| x.inner().select(slice.as_ref()))
            .transpose()
    }

    fn get_iter<'a, T>(
        &'a self,
        key: &str,
        chunk_size: usize,
    ) -> Result<Self::ArrayIter<'a, T>>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        Ok(self.inner().get(key).unwrap().chunked(chunk_size))
    }

    fn add<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()>
    {
        self.inner().add_data(key, data)
    }

    fn add_iter<I, D>(&self, key: &str, data: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk,
    {
        self.inner().add_data_from_iter(key, data)
    }
}

impl<B: Backend> ElemCollectionOp for &ElemCollection<B> {
    fn keys(&self) -> Vec<String> {
        self.inner().keys().cloned().collect()
    }

    fn get<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<Data> + TryFrom<Data> + Clone,
        <D as TryFrom<Data>>::Error: Into<anyhow::Error>,
    {
        self.lock()
            .as_mut()
            .and_then(|x| x.get_mut(key))
            .map(|x| x.inner().data())
            .transpose()
    }

    fn add<D: WriteData + Into<Data>>(&self, key: &str, data: D) -> Result<()> {
        self.inner().add_data(key, data)
    }
}