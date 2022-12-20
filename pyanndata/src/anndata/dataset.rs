use crate::container::{PyArrayElem, PyAxisArrays, PyDataFrameElem, PyElem, PyElemCollection};
use crate::data::{to_select_elem, to_select_info, PyArrayData, PyData, PyDataFrame};
use crate::anndata::PyAnnData;
use crate::AnnData;

use std::collections::HashMap;
use std::path::PathBuf;
use anndata_hdf5::H5;
use pyo3::prelude::*;
use anndata;
use anndata::{AnnDataOp, ArrayData, Backend, Data, StackedArrayElem};
use anndata::container::Slot;
use anndata::data::{DataFrameIndex, SelectInfoElem};
use anyhow::Result;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

/** Similar to `AnnData`, `AnnDataSet` contains annotations of
    observations `obs` (`obsm`, `obsp`), variables `var` (`varm`, `varp`),
    and unstructured annotations `uns`. Additionally it provides lazy access to
    concatenated component AnnData objects, including `X`, `obs`, `obsm`, `obsp`.

    Parameters
    ----------
    adatas: list[(str, Path)] | list[(str, AnnData)]
        List of key and file name (or backed AnnData object) pairs.
    filename: Path
        File name of the output file containing the AnnDataSet object.
    add_key: str
        The column name in obs to store the keys

    Note
    ----
    AnnDataSet does not copy underlying AnnData objects. It stores the references
    to individual anndata files. If you move the anndata files to a new location,
    remember to update the anndata file locations when opening an AnnDataSet object.

    See Also
    --------
    read_dataset
*/
#[pyclass]
#[pyo3(text_signature = "(*, adatas, filename, add_key /)")]
#[repr(transparent)]
#[derive(Clone)]
pub struct AnnDataSet(Slot<Box<dyn AnnDataSetTrait>>);

impl<B: Backend + 'static> From<anndata::AnnDataSet<B>> for AnnDataSet {
    fn from(adata: anndata::AnnDataSet<B>) -> Self {
        AnnDataSet(Slot::new(Box::new(adata)))
    }
}

#[derive(FromPyObject)]
pub enum AnnDataFile {
    Path(PathBuf),
    Data(AnnData),
}

#[pymethods]
impl AnnDataSet {
    #[new]
    //#[args("*", adatas, filename, add_key = "\"sample\"", backend = None)]
    pub fn new(
        adatas: Vec<(String, AnnDataFile)>,
        filename: PathBuf,
        add_key: &str,
        backend: Option<&str>,
    ) -> Result<Self> {
        let anndatas = adatas
            .into_par_iter()
            .map(|(key, data_file)| {
                let adata = match data_file {
                    AnnDataFile::Data(data) => data,
                    AnnDataFile::Path(path) => AnnData::open(path, "r", backend)?,
                };
                Ok((key, adata))
            })
            .collect::<Result<Vec<_>>>()?;
        match anndatas.get(0).map(|x| x.1.backend()).and(backend).unwrap_or(H5::NAME) {
            H5::NAME => {
                let data = anndatas.into_iter().map(|(key, adata)| {
                    let adata = adata.into_inner::<H5>().unwrap();
                    (key, adata)
                });
                Ok(anndata::AnnDataSet::new(data, filename, add_key)?.into())
            },
            _ => todo!(),
        }
    }

    /// Shape of data matrix (`n_obs`, `n_vars`).
    ///
    /// Returns
    /// -------
    /// tuple[int, int]
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        self.0.inner().shape()
    }

    /// Number of observations.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub fn n_obs(&self) -> usize {
        self.shape().0
    }

    /// Number of variables/features.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub fn n_vars(&self) -> usize {
        self.shape().1
    }

    /// Names of observations.
    ///
    /// Returns
    /// -------
    /// list[str]
    #[getter]
    pub fn obs_names(&self) -> Vec<String> {
        self.0.inner().obs_names()
    }
    #[setter(obs_names)]
    pub fn set_obs_names(&self, names: &PyAny) -> Result<()> {
        self.0.inner().set_obs_names(names)
    }

    /// Names of variables.
    ///
    /// Returns
    /// -------
    /// list[str]
    #[getter]
    pub fn var_names(&self) -> Vec<String> {
        self.0.inner().var_names()
    }
    #[setter(var_names)]
    pub fn set_var_names(&self, names: &PyAny) -> Result<()> {
        self.0.inner().set_var_names(names)
    }

    /// Data matrix of shape n_obs × n_vars.
    ///
    /// Returns
    /// -------
    /// PyArrayElem
    #[getter(X)]
    pub fn get_x(&self) -> Option<PyArrayElem> {
        self.0.inner().get_x()
    }

    /// Observation annotations.
    ///
    /// Returns
    /// -------
    /// PyDataFrameElem
    #[getter(obs)]
    fn get_obs(&self) -> Option<PyDataFrameElem> {
        self.0.inner().get_obs()
    }
    #[setter(obs)]
    fn set_obs(&self, obs: Option<PyDataFrame>) -> Result<()> {
        self.0.inner().set_obs(obs)
    }

    /// Variable annotations.
    ///
    /// Returns
    /// -------
    /// PyDataFrameElem
    #[getter(var)]
    fn get_var(&self) -> Option<PyDataFrameElem> {
        self.0.inner().get_var()
    }
    #[setter(var)]
    fn set_var(&self, var: Option<PyDataFrame>) -> Result<()> {
        self.0.inner().set_var(var)
    }

    /// Unstructured annotation (ordered dictionary).
    ///
    /// Returns
    /// -------
    /// PyElemCollection
    #[getter(uns)]
    pub fn get_uns(&self) -> Option<PyElemCollection> {
        self.0.inner().get_uns()
    }
    #[setter(uns)]
    pub fn set_uns(&self, uns: Option<HashMap<String, PyData>>) -> Result<()> {
        self.0.inner().set_uns(uns)
    }

    #[getter(obsm)]
    pub fn get_obsm(&self) -> Option<PyAxisArrays> {
        self.0.inner().get_obsm()
    }
    #[setter(obsm)]
    pub fn set_obsm(&self, obsm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.inner().set_obsm(obsm)
    }

    #[getter(obsp)]
    pub fn get_obsp(&self) -> Option<PyAxisArrays> {
        self.0.inner().get_obsp()
    }
    #[setter(obsp)]
    pub fn set_obsp(&self, obsp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.inner().set_obsp(obsp)
    }

    #[getter(varm)]
    pub fn get_varm(&self) -> Option<PyAxisArrays> {
        self.0.inner().get_varm()
    }
    #[setter(varm)]
    pub fn set_varm(&self, varm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.inner().set_varm(varm)
    }

    #[getter(varp)]
    pub fn get_varp(&self) -> Option<PyAxisArrays> {
        self.0.inner().get_varp()
    }
    #[setter(varp)]
    pub fn set_varp(&self, varp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        self.0.inner().set_varp(varp)
    }

}


/// Lazily concatenated AnnData objects.
/*
#[pyclass]
#[repr(transparent)]
pub struct StackedAnnData(pub Slot<anndata::StackedAnnData>);
*/

trait AnnDataSetTrait: Send {
    fn shape(&self) -> (usize, usize);
    fn obs_names(&self) -> Vec<String>;
    fn set_obs_names(&self, names: &PyAny) -> Result<()>;
    fn var_names(&self) -> Vec<String>;
    fn set_var_names(&self, names: &PyAny) -> Result<()>;

    fn get_x(&self) -> Option<PyArrayElem>;
    fn get_obs(&self) -> Option<PyDataFrameElem>;
    fn get_var(&self) -> Option<PyDataFrameElem>;
    fn get_uns(&self) -> Option<PyElemCollection>;
    fn get_obsm(&self) -> Option<PyAxisArrays>;
    fn get_obsp(&self) -> Option<PyAxisArrays>;
    fn get_varm(&self) -> Option<PyAxisArrays>;
    fn get_varp(&self) -> Option<PyAxisArrays>;

    fn set_obs(&self, obs: Option<PyDataFrame>) -> Result<()>;
    fn set_var(&self, var: Option<PyDataFrame>) -> Result<()>;
    fn set_uns(&self, uns: Option<HashMap<String, PyData>>) -> Result<()>;
    fn set_obsm(&self, obsm: Option<HashMap<String, PyArrayData>>) -> Result<()>;
    fn set_obsp(&self, obsp: Option<HashMap<String, PyArrayData>>) -> Result<()>;
    fn set_varm(&self, varm: Option<HashMap<String, PyArrayData>>) -> Result<()>;
    fn set_varp(&self, varp: Option<HashMap<String, PyArrayData>>) -> Result<()>;
}

impl<B: Backend + 'static> AnnDataSetTrait for anndata::AnnDataSet<B> {
    fn shape(&self) -> (usize, usize) {
        (self.n_obs(), self.n_vars())
    }

    fn obs_names(&self) -> Vec<String> {
        AnnDataOp::obs_names(self)
    }

    fn set_obs_names(&self, names: &PyAny) -> Result<()> {
        let obs_names: Result<DataFrameIndex> =
            names.iter()?.map(|x| Ok(x?.extract::<String>()?)).collect();
        AnnDataOp::set_obs_names(self, obs_names?)
    }

    fn var_names(&self) -> Vec<String> {
        AnnDataOp::var_names(self)
    }

    fn set_var_names(&self, names: &PyAny) -> Result<()> {
        let var_names: Result<DataFrameIndex> =
            names.iter()?.map(|x| Ok(x?.extract::<String>()?)).collect();
        AnnDataOp::set_var_names(self, var_names?)
    }

    fn get_x(&self) -> Option<PyArrayElem> {
        Some(self.get_x().clone().into())
    }
    fn get_obs(&self) -> Option<PyDataFrameElem> {
        let obs = self.get_anno().get_obs();
        if obs.is_empty() {
            None
        } else {
            Some(obs.clone().into())
        }
    }
    fn get_var(&self) -> Option<PyDataFrameElem> {
        let var = self.get_anno().get_var();
        if var.is_empty() {
            None
        } else {
            Some(var.clone().into())
        }
    }
    fn get_uns(&self) -> Option<PyElemCollection> {
        let uns = self.get_anno().get_uns();
        if uns.is_empty() {
            None
        } else {
            Some(uns.clone().into())
        }
    }
    fn get_obsm(&self) -> Option<PyAxisArrays> {
        let obsm = self.get_anno().get_obsm();
        if obsm.is_empty() {
            None
        } else {
            Some(obsm.clone().into())
        }
    }
    fn get_obsp(&self) -> Option<PyAxisArrays> {
        let obsp = self.get_anno().get_obsp();
        if obsp.is_empty() {
            None
        } else {
            Some(obsp.clone().into())
        }
    }
    fn get_varm(&self) -> Option<PyAxisArrays> {
        let varm = self.get_anno().get_varm();
        if varm.is_empty() {
            None
        } else {
            Some(varm.clone().into())
        }
    }
    fn get_varp(&self) -> Option<PyAxisArrays> {
        let varp = self.get_anno().get_varp();
        if varp.is_empty() {
            None
        } else {
            Some(varp.clone().into())
        }
    }

    fn set_obs(&self, obs: Option<PyDataFrame>) -> Result<()> {
        if let Some(o) = obs {
            AnnDataOp::set_obs(self, o.into())?;
        } else {
            self.del_obs()?;
        }
        Ok(())
    }
    fn set_var(&self, var: Option<PyDataFrame>) -> Result<()> {
        if let Some(v) = var {
            AnnDataOp::set_var(self, v.into())?;
        } else {
            self.del_var()?;
        }
        Ok(())
    }
    fn set_uns(&self, uns: Option<HashMap<String, PyData>>) -> Result<()> {
        if let Some(u) = uns {
            AnnDataOp::set_uns(self, u.into_iter().map(|(k, v)| (k, v.into())))?;
        } else {
            self.del_uns()?;
        }
        Ok(())
    }
    fn set_obsm(&self, obsm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        if let Some(o) = obsm {
            AnnDataOp::set_obsm(self, o.into_iter().map(|(k, v)| (k, v.into())))?;
        } else {
            self.del_obsm()?;
        }
        Ok(())
    }
    fn set_obsp(&self, obsp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        if let Some(o) = obsp {
            AnnDataOp::set_obsp(self, o.into_iter().map(|(k, v)| (k, v.into())))?;
        } else {
            self.del_obsp()?;
        }
        Ok(())
    }
    fn set_varm(&self, varm: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        if let Some(v) = varm {
            AnnDataOp::set_varm(self, v.into_iter().map(|(k, v)| (k, v.into())))?;
        } else {
            self.del_varm()?;
        }
        Ok(())
    }
    fn set_varp(&self, varp: Option<HashMap<String, PyArrayData>>) -> Result<()> {
        if let Some(v) = varp {
            AnnDataOp::set_varp(self, v.into_iter().map(|(k, v)| (k, v.into())))?;
        } else {
            self.del_varp()?;
        }
        Ok(())
    }
}