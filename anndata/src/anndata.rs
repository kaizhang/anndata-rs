mod dataset;

pub use dataset::{AnnDataSet, StackedAnnData};
use smallvec::SmallVec;

use crate::{
    backend::{Backend, DataContainer, StoreOp, GroupOp},
    container::{
        Dim, ArrayElem, Axis, AxisArrays, DataFrameElem, ElemCollection,
        InnerDataFrameElem, Slot,
    },
    data::*,
    traits::AnnDataOp,
};

use anyhow::{anyhow, ensure, Context, Result};
use itertools::Itertools;
use polars::prelude::DataFrame;
use std::path::{Path, PathBuf};

pub struct AnnData<B: Backend> {
    file: B::Store,
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
    layers: AxisArrays<B>,
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
        if let Some(keys) = self.layers.lock().as_ref().map(|x| x.keys().join("', '")) {
            if !keys.is_empty() {
                write!(f, "\n    layers: '{}'", keys)?;
            }
        }
        Ok(())
    }
}

fn new_obsm<B: Backend>(group: B::Group, n_obs: &Dim) -> Result<AxisArrays<B>> {
    AxisArrays::new(group, Axis::Row, n_obs, None)
}

fn new_obsp<B: Backend>(group: B::Group, n_obs: &Dim) -> Result<AxisArrays<B>> {
    AxisArrays::new(group, Axis::Pairwise, n_obs, None)
}

fn new_varm<B: Backend>(group: B::Group, n_vars: &Dim) -> Result<AxisArrays<B>> {
    AxisArrays::new(group, Axis::Row, n_vars, None)
}

fn new_varp<B: Backend>(group: B::Group, n_vars: &Dim) -> Result<AxisArrays<B>> {
    AxisArrays::new(group, Axis::Pairwise, n_vars, None)
}

fn new_layers<B: Backend>(group: B::Group, n_obs: &Dim, n_vars: &Dim) -> Result<AxisArrays<B>> {
    AxisArrays::new(group, Axis::RowColumn, n_obs, Some(n_vars))
}

impl<B: Backend> AnnData<B> {
    pub fn get_x(&self) -> &ArrayElem<B> {
        &self.x
    }
    pub fn get_obs(&self) -> &DataFrameElem<B> {
        &self.obs
    }
    pub fn get_var(&self) -> &DataFrameElem<B> {
        &self.var
    }

    /// Open an existing AnnData.
    pub fn open(file: B::Store) -> Result<Self> {
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
            Ok(group) => new_obsm(group, &n_obs)?,
            _ => AxisArrays::empty(),
        };

        let obsp = match file.open_group("obsp").or(file.create_group("obsp")) {
            Ok(group) => new_obsp(group, &n_obs)?,
            _ => AxisArrays::empty(),
        };

        let varm = match file.open_group("varm").or(file.create_group("varm")) {
            Ok(group) => new_varm(group, &n_vars)?,
            _ => AxisArrays::empty(),
        };

        let varp = match file.open_group("varp").or(file.create_group("varp")) {
            Ok(group) => new_varp(group, &n_vars)?,
            _ => AxisArrays::empty(),
        };

        let uns = match file.open_group("uns").or(file.create_group("uns")) {
            Ok(group) => ElemCollection::new(group)?,
            _ => ElemCollection::empty(),
        };

        let layers = match file.open_group("layers").or(file.create_group("layers")) {
            Ok(group) => new_layers(group, &n_obs, &n_vars)?,
            _ => AxisArrays::empty(),
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
            layers,
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
            obsm: new_obsm(file.create_group("obsm")?, &n_obs)?,
            obsp: new_obsp(file.create_group("obsp")?, &n_obs)?,
            varm: new_varm(file.create_group("varm")?, &n_vars)?,
            varp: new_varp(file.create_group("varp")?, &n_vars)?,
            uns: ElemCollection::new(file.create_group("uns")?)?,
            layers: new_layers(file.create_group("layers")?, &n_obs, &n_vars)?,
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
        self.obsp()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file, "obsp"))
            .transpose()?;
        self.varm()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file, "varm"))
            .transpose()?;
        self.varp()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file, "varp"))
            .transpose()?;
        self.uns()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file, "uns"))
            .transpose()?;
        self.layers()
            .lock()
            .as_mut()
            .map(|x| x.export::<O, _>(&file, "layers"))
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
        selection.as_ref()[0].bound_check(self.n_obs())
            .map_err(|e| anyhow!("AnnData obs {}", e))?;
        selection.as_ref()[1].bound_check(self.n_vars())
            .map_err(|e| anyhow!("AnnData var {}", e))?;
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
        self.uns()
            .lock()
            .as_mut()
            .map(|x| x.export(&file, "uns"))
            .transpose()?;
        self.obsm()
            .lock()
            .as_mut()
            .map(|x| x.export_select(&[slice[0]], &file, "obsm"))
            .transpose()?;
        self.obsp()
            .lock()
            .as_mut()
            .map(|x| x.export_select(&[slice[0]], &file, "obsp"))
            .transpose()?;
        self.varm()
            .lock()
            .as_mut()
            .map(|x| x.export_select(&[slice[1]], &file, "varm"))
            .transpose()?;
        self.varp()
            .lock()
            .as_mut()
            .map(|x| x.export_select(&[slice[1]], &file, "varp"))
            .transpose()?;
        self.layers()
            .lock()
            .as_mut()
            .map(|x| x.export_select(slice.as_slice(), &file, "layers"))
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
            .map(|obsm| obsm.subset(&[obs_ix]))
            .transpose()?;
        self.obsp
            .lock()
            .as_mut()
            .map(|obsp| obsp.subset(&[obs_ix]))
            .transpose()?;

        self.var
            .lock()
            .as_mut()
            .map(|x| x.subset_axis(0, var_ix))
            .transpose()?;
        self.varm
            .lock()
            .as_mut()
            .map(|varm| varm.subset(&[var_ix]))
            .transpose()?;
        self.varp
            .lock()
            .as_mut()
            .map(|varp| varp.subset(&[var_ix]))
            .transpose()?;

        self.layers
            .lock()
            .as_mut()
            .map(|layers| layers.subset(&[obs_ix, var_ix]))
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
    type X = ArrayElem<B>;
    type AxisArraysRef<'a> = &'a AxisArrays<B>;
    type ElemCollectionRef<'a> = &'a ElemCollection<B>;

    fn x(&self) -> Self::X {
        self.x.clone()
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
            .map_or(DataFrameIndex::empty(), |obs| obs.index.clone())
    }

    fn var_names(&self) -> DataFrameIndex {
        self.var
            .lock()
            .as_ref()
            .map_or(DataFrameIndex::empty(), |var| var.index.clone())
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

    fn obs_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>>
    {
        let inner = self.obs.inner();
        names
            .into_iter()
            .map(|i| {
                inner
                    .index
                    .get_index(i)
                    .context(format!("'{}' does not exist in obs_names", i))
            })
            .collect()
    }

    fn var_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>> {
        let inner = self.var.inner();
        names
            .into_iter()
            .map(|i| {
                inner
                    .index
                    .get_index(i)
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
                self.uns.swap(&uns);
            }
        }
        &self.uns
    }
    fn obsm(&self) -> Self::AxisArraysRef<'_> {
        if self.obsm.is_empty() {
            let arrays = self.file.create_group("obsm")
                .and_then(|g| new_obsm(g, &self.n_obs));
            if let Ok(obsm) = arrays {
                self.obsm.swap(&obsm);
            }
        }
        &self.obsm
    }
    fn obsp(&self) -> Self::AxisArraysRef<'_> {
        if self.obsp.is_empty() {
            let arrays = self.file.create_group("obsp")
                .and_then(|g| new_obsp(g, &self.n_obs));
            if let Ok(obsp) = arrays {
                self.obsp.swap(&obsp);
            }
        }
        &self.obsp
    }
    fn varm(&self) -> Self::AxisArraysRef<'_> {
        if self.varm.is_empty() {
            let arrays = self.file.create_group("varm")
                .and_then(|g| new_varm(g, &self.n_vars));
            if let Ok(varm) = arrays {
                self.varm.swap(&varm);
            }
        }
        &self.varm
    }
    fn varp(&self) -> Self::AxisArraysRef<'_> {
        if self.varp.is_empty() {
            let arrays = self.file.create_group("varp")
                .and_then(|g| new_varp(g, &self.n_vars));
            if let Ok(varp) = arrays {
                self.varp.swap(&varp);
            }
        }
        &self.varp
    }
    fn layers(&self) -> Self::AxisArraysRef<'_> {
        if self.layers.is_empty() {
            let arrays = self.file.create_group("layers")
                .and_then(|g| new_layers(g, &self.n_obs, &self.n_vars));
            if let Ok(layers) = arrays {
                self.layers.swap(&layers);
            }
        }
        &self.layers
    }

    fn del_uns(&self) -> Result<()> {
        self.uns.clear()
    }
    fn del_obsm(&self) -> Result<()> {
        self.obsm.clear()
    }
    fn del_obsp(&self) -> Result<()> {
        self.obsp.clear()
    }
    fn del_varm(&self) -> Result<()> {
        self.varm.clear()
    }
    fn del_varp(&self) -> Result<()> {
        self.varp.clear()
    }
    fn del_layers(&self) -> Result<()> {
        self.layers.clear()
    }
}