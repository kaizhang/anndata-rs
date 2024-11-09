mod dataset;

pub use dataset::{AnnDataSet, StackedAnnData};
use smallvec::SmallVec;

use crate::{
    backend::{Backend, DataContainer, GroupOp, StoreOp},
    container::{ArrayElem, Axis, AxisArrays, DataFrameElem, Dim, ElemCollection, Slot},
    data::*,
    traits::AnnDataOp,
};

use anyhow::{anyhow, ensure, Result};
use itertools::Itertools;
use std::path::{Path, PathBuf};

/// Represents an annotated data object backed by a specified backend.
pub struct AnnData<B: Backend> {
    /// The file storage backend.
    pub(crate) file: B::Store,
    /// Number of observations (rows).
    pub(crate) n_obs: Dim,
    /// Number of variables (columns).
    pub(crate) n_vars: Dim,
    /// Data matrix.
    pub(crate) x: ArrayElem<B>,
    /// Observations metadata.
    pub(crate) obs: DataFrameElem<B>,
    /// Observation multi-dimensional annotation.
    pub(crate) obsm: AxisArrays<B>,
    /// Observation pairwise annotation.
    pub(crate) obsp: AxisArrays<B>,
    /// Variables metadata.
    pub(crate) var: DataFrameElem<B>,
    /// Variable multi-dimensional annotation.
    pub(crate) varm: AxisArrays<B>,
    /// Variable pairwise annotation.
    pub(crate) varp: AxisArrays<B>,
    /// Unstructured annotation.
    pub(crate) uns: ElemCollection<B>,
    /// Layers of data.
    pub(crate) layers: AxisArrays<B>,
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

// Helper function to create a new observation matrix (obsm)
pub(crate) fn new_obsm<B: Backend>(group: B::Group, n_obs: &Dim) -> Result<AxisArrays<B>> {
    AxisArrays::new(group, Axis::Row, n_obs, None)
}

// Helper function to create a new pairwise observation matrix (obsp)
pub(crate) fn new_obsp<B: Backend>(group: B::Group, n_obs: &Dim) -> Result<AxisArrays<B>> {
    AxisArrays::new(group, Axis::Pairwise, n_obs, None)
}

// Helper function to create a new variable matrix (varm)
pub(crate) fn new_varm<B: Backend>(group: B::Group, n_vars: &Dim) -> Result<AxisArrays<B>> {
    AxisArrays::new(group, Axis::Row, n_vars, None)
}

// Helper function to create a new pairwise variable matrix (varp)
pub(crate) fn new_varp<B: Backend>(group: B::Group, n_vars: &Dim) -> Result<AxisArrays<B>> {
    AxisArrays::new(group, Axis::Pairwise, n_vars, None)
}

// Helper function to create new layers of data
pub(crate) fn new_layers<B: Backend>(group: B::Group, n_obs: &Dim, n_vars: &Dim) -> Result<AxisArrays<B>> {
    AxisArrays::new(group, Axis::RowColumn, n_obs, Some(n_vars))
}

impl<B: Backend> AnnData<B> {
    /// Get the data matrix.
    pub fn get_x(&self) -> &ArrayElem<B> {
        &self.x
    }

    /// Get the observations metadata.
    pub fn get_obs(&self) -> &DataFrameElem<B> {
        &self.obs
    }

    /// Get the variables metadata.
    pub fn get_var(&self) -> &DataFrameElem<B> {
        &self.var
    }

    /// Open an existing AnnData store.
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
            Slot::none()
        };

        // Read obs
        let obs = if file.exists("obs")? {
            let obs = DataFrameElem::try_from(DataContainer::open(&file, "obs")?)?;
            n_obs.try_set(obs.inner().height())?;
            obs
        } else {
            Slot::none()
        };

        // Read var
        let var = if file.exists("var")? {
            let var = DataFrameElem::try_from(DataContainer::open(&file, "var")?)?;
            n_vars.try_set(var.inner().height())?;
            var
        } else {
            Slot::none()
        };

        let obsm = match file.open_group("obsm").or(file.new_group("obsm")) {
            Ok(group) => new_obsm(group, &n_obs)?,
            _ => AxisArrays::empty(),
        };

        let obsp = match file.open_group("obsp").or(file.new_group("obsp")) {
            Ok(group) => new_obsp(group, &n_obs)?,
            _ => AxisArrays::empty(),
        };

        let varm = match file.open_group("varm").or(file.new_group("varm")) {
            Ok(group) => new_varm(group, &n_vars)?,
            _ => AxisArrays::empty(),
        };

        let varp = match file.open_group("varp").or(file.new_group("varp")) {
            Ok(group) => new_varp(group, &n_vars)?,
            _ => AxisArrays::empty(),
        };

        let uns = match file.open_group("uns").or(file.new_group("uns")) {
            Ok(group) => ElemCollection::new(group)?,
            _ => ElemCollection::empty(),
        };

        let layers = match file.open_group("layers").or(file.new_group("layers")) {
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

    /// Create a new AnnData file.
    pub fn new<P: AsRef<Path>>(filename: P) -> Result<Self> {
        let file = B::new(filename)?;
        let n_obs = Dim::empty();
        let n_vars = Dim::empty();
        Ok(Self {
            x: Slot::none(),
            obs: Slot::none(),
            var: Slot::none(),
            obsm: new_obsm(file.new_group("obsm")?, &n_obs)?,
            obsp: new_obsp(file.new_group("obsp")?, &n_obs)?,
            varm: new_varm(file.new_group("varm")?, &n_vars)?,
            varp: new_varp(file.new_group("varp")?, &n_vars)?,
            uns: ElemCollection::new(file.new_group("uns")?)?,
            layers: new_layers(file.new_group("layers")?, &n_obs, &n_vars)?,
            file,
            n_obs,
            n_vars,
        })
    }

    /// Write the AnnData object to a new file.
    pub fn write<O: Backend, P: AsRef<Path>>(&self, filename: P) -> Result<()> {
        let file = O::new(filename)?;
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

    /// Write a subset of the AnnData object to a new file.
    pub fn write_select<O, S, P>(&self, selection: S, filename: P) -> Result<()>
    where
        O: Backend,
        S: AsRef<[SelectInfoElem]>,
        P: AsRef<Path>,
    {
        selection.as_ref()[0]
            .bound_check(self.n_obs())
            .map_err(|e| anyhow!("AnnData obs {}", e))?;
        selection.as_ref()[1]
            .bound_check(self.n_vars())
            .map_err(|e| anyhow!("AnnData var {}", e))?;
        let slice: SmallVec<[_; 3]> = selection.as_ref().iter().collect();
        let file = O::new(filename)?;
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

    /// Get the filename of the AnnData file.
    pub fn filename(&self) -> PathBuf {
        self.file.filename()
    }

    /// Close the AnnData object and release all resources.
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

    /// Subset the AnnData object based on a selection.
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
            obs_lock.set(SelectInfoElemBounds::new(obs_ix, obs_lock.get()).len());
        }
        if !vars_lock.is_empty() {
            vars_lock.set(SelectInfoElemBounds::new(var_ix, vars_lock.get()).len());
        }

        Ok(())
    }
}