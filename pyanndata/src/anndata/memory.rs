use crate::data::{isinstance_of_pyanndata, isinstance_of_polars, PyArrayData, PyData, PyDataFrame};

use std::ops::Deref;
use std::fmt::Debug;
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3::exceptions::PyTypeError;
use pyo3::types::IntoPyDict;
use anndata::{self, ArrayOp};
use anndata::{AnnDataIterator, AnnDataOp, ArrayData, Data, ReadArrayData, WriteData, ReadData, Backend, WriteArrayData, HasShape};
use anndata::data::{DataFrameIndex, SelectInfoElem, concat_array_data};
use anyhow::{Result, bail};

pub struct PyAnnData<'py>(&'py PyAny);

impl<'py> Deref for PyAnnData<'py> {
    type Target = PyAny;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'py> FromPyObject<'py> for PyAnnData<'py> {
    fn extract(obj: &'py PyAny) -> PyResult<Self> {
        Python::with_gil(|py| {
            if isinstance_of_pyanndata(py, obj)? {
                Ok(PyAnnData(obj))
            } else {
                Err(PyTypeError::new_err("Not a Python AnnData object"))
            }
        })
    }
}

impl ToPyObject for PyAnnData<'_> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.0.into_py(py)
    }
}


impl IntoPy<PyObject> for PyAnnData<'_> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.0.into_py(py)
    }
}

impl<'py> PyAnnData<'py> {
    pub fn new(py: Python<'py>) -> PyResult<Self> {
        PyModule::import(py, "anndata")?
            .call_method0("AnnData")?
            .extract()
    }

    pub fn from_anndata<B: Backend>(py: Python<'py>, inner: &anndata::AnnData<B>) -> Result<Self> {
        let adata = PyAnnData::new(py)?;
        {
            // Set X
            adata.set_n_obs(inner.n_obs())?;
            adata.set_n_vars(inner.n_vars())?;
            if let Some(x) = inner.read_x::<ArrayData>()? {
                adata.set_x(x)?;
            }
        }
        {
            // Set obs and var
            adata.set_obs_names(inner.obs_names().into())?;
            adata.set_var_names(inner.var_names().into())?;
            adata.set_obs(inner.read_obs()?)?;
            adata.set_var(inner.read_var()?)?;
        }
        {
            // Set uns
            inner
                .uns_keys()
                .into_iter()
                .try_for_each(|k| adata.add_uns(&k, inner.fetch_uns::<Data>(&k)?.unwrap()))?;
        }
        {
            // Set obsm
            inner
                .obsm_keys()
                .into_iter()
                .try_for_each(|k| adata.add_obsm(&k, inner.fetch_obsm::<ArrayData>(&k)?.unwrap()))?;
        }
        {
            // Set obsp
            inner
                .obsp_keys()
                .into_iter()
                .try_for_each(|k| adata.add_obsp(&k, inner.fetch_obsp::<ArrayData>(&k)?.unwrap()))?;
        }
        {
            // Set varm
            inner
                .varm_keys()
                .into_iter()
                .try_for_each(|k| adata.add_varm(&k, inner.fetch_varm::<ArrayData>(&k)?.unwrap()))?;
        }
        {
            // Set varp
            inner
                .varp_keys()
                .into_iter()
                .try_for_each(|k| adata.add_varp(&k, inner.fetch_varp::<ArrayData>(&k)?.unwrap()))?;
        }
        Ok(adata)
    }

    fn get_item(&'py self, slot: &str, key: &str) -> Result<Option<&'py PyAny>>
    {
        Ok(self
            .getattr(slot)?
            .call_method1("__getitem__", (key,))
            .ok())
    }

    fn set_item<T: IntoPy<PyObject>>(&'py self, slot: &str, key: &str, data: T) -> Result<()> {
        let py = self.py();
        let d = data.into_py(py);
        let new_d = if isinstance_of_polars(py, d.as_ref(py))? {
            d.call_method0(py, "to_pandas")?
        } else {
            d
        };
        self.getattr(slot)?
            .call_method1("__setitem__", (key, new_d))?;
        Ok(())
    }

    fn get_keys(&self, slot: &str) -> Result<Vec<String>> {
        Ok(self.getattr(slot)?.call_method0("keys")?.extract()?)
    }

    pub(crate) fn set_n_obs(&self, n_obs: usize) -> Result<()> {
        let n = self.n_obs();
        if n == n_obs {
            Ok(())
        } else if n == 0 {
            self.0.setattr("_n_obs", n_obs)?;
            Ok(())
        } else {
            bail!("cannot set n_obs unless n_obs == 0")
        }
    }

    pub(crate) fn set_n_vars(&self, n_vars: usize) -> Result<()> {
        let n = self.n_vars();
        if n == n_vars {
            Ok(())
        } else if n == 0 {
            self.0.setattr("_n_vars", n_vars)?;
            Ok(())
        } else {
            bail!("cannot set n_vars unless n_vars == 0")
        }
    }
}

impl<'py> AnnDataOp for PyAnnData<'py> {
    fn read_x<D>(&self) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let x = self.getattr("X")?;
        if x.is_none() {
            Ok(None)
        } else {
            let data: ArrayData = x.extract::<PyArrayData>()?.into();
            D::try_from(data).map(Some).map_err(Into::into)
        }
    }

    fn read_x_slice<D, S>(&self, select: S) -> Result<Option<D>>
    where
        D: ReadArrayData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>
    {
        todo!()
    }

    fn set_x<D: WriteArrayData + Into<ArrayData> + HasShape>(&self, data: D) -> Result<()> {
        let py = self.py();
        let shape = data.shape();
        self.set_n_obs(shape[0])?;
        self.set_n_vars(shape[1])?;
        let ob: ArrayData = data.into();
        self.setattr("X", PyArrayData::from(ob).into_py(py))?;
        Ok(())
    }

    fn del_x(&self) -> Result<()> {
        self.setattr("X", None::<PyObject>)?;
        Ok(())
    }

    fn n_obs(&self) -> usize {
        self.0.getattr("n_obs").unwrap().extract().unwrap()
    }
    fn n_vars(&self) -> usize {
        self.0.getattr("n_vars").unwrap().extract().unwrap()
    }

    fn obs_names(&self) -> DataFrameIndex {
        self.0.getattr("obs_names").unwrap().extract::<Vec<String>>().unwrap().into()
    }
    fn var_names(&self) -> DataFrameIndex {
        self.0.getattr("var_names").unwrap().extract::<Vec<String>>().unwrap().into()
    }

    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()> {
        if self.getattr("obs")?.getattr("empty")?.is_true()? {
            let py = self.py();
            let df = py.import("pandas")?.call_method(
                "DataFrame",
                (),
                Some(&[("index", index.names)].into_py_dict(py)),
            )?;
            self.setattr("obs", df)?;
        } else {
            self.setattr("obs_names", index.names)?;
        }
        Ok(())
    }
    fn set_var_names(&self, index: DataFrameIndex) -> Result<()> {
        if self.getattr("var")?.getattr("empty")?.is_true()? {
            let py = self.py();
            let df = py.import("pandas")?.call_method(
                "DataFrame",
                (),
                Some(&[("index", index.names)].into_py_dict(py)),
            )?;
            self.setattr("var", df)?;
        } else {
            self.setattr("var_names", index.names)?;
        }
        Ok(())
    }

    fn obs_ix(&self, _names: &[String]) -> Result<Vec<usize>> {
        todo!()
    }
    fn var_ix(&self, _names: &[String]) -> Result<Vec<usize>> {
        todo!()
    }

    fn read_obs(&self) -> Result<DataFrame> {
        let py = self.py();
        let df: PyDataFrame = py
            .import("polars")?
            .call_method1("from_pandas", (self.0.getattr("obs")?,))?
            .extract()?;
        Ok(df.into())
    }
    fn read_var(&self) -> Result<DataFrame> {
        let py = self.py();
        let df: PyDataFrame = py
            .import("polars")?
            .call_method1("from_pandas", (self.0.getattr("var")?,))?
            .extract()?;
        Ok(df.into())
    }

    fn set_obs(&self, obs: DataFrame) -> Result<()> {
        let py = self.py();
        let index = self.getattr("obs")?.getattr("index")?;
        let df = if obs.is_empty() {
            py.import("pandas")?
                .call_method1("DataFrame", (py.None(), index))?
                .into_py(py)
        } else {
            PyDataFrame::from(obs).into_py(py)
                .call_method0(py, "to_pandas")?
                .call_method1(py, "set_index", (index,))?
        };
        self.setattr("obs", df)?;
        Ok(())
    }

    fn set_var(&self, var: DataFrame) -> Result<()> {
        let py = self.py();
        let index = self.getattr("var")?.getattr("index")?;
        let df = if var.is_empty() {
            py.import("pandas")?
                .call_method1("DataFrame", (py.None(), index))?
                .into_py(py)
        } else {
            PyDataFrame::from(var).into_py(py)
                .call_method0(py, "to_pandas")?
                .call_method1(py, "set_index", (index,))?
        };
        self.setattr("var", df)?;
        Ok(())
    }

    fn del_obs(&self) -> Result<()> {
        self.0.setattr("obs", None::<PyObject>)?;
        Ok(())
    }

    fn del_var(&self) -> Result<()> {
        self.0.setattr("var", None::<PyObject>)?;
        Ok(())
    }

    fn uns_keys(&self) -> Vec<String> {
        self.get_keys("uns").unwrap()
    }
    fn obsm_keys(&self) -> Vec<String> {
        self.get_keys("obsm").unwrap()
    }
    fn obsp_keys(&self) -> Vec<String> {
        self.get_keys("obsp").unwrap()
    }
    fn varm_keys(&self) -> Vec<String> {
        self.get_keys("varm").unwrap()
    }
    fn varp_keys(&self) -> Vec<String> {
        self.get_keys("varp").unwrap()
    }

    fn fetch_uns<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<Data> + TryFrom<Data> + Clone,
        <D as TryFrom<Data>>::Error: Into<anyhow::Error>,
    {
        self.get_item("uns", key)?.map(|x| {
            let data: Data = x.extract::<PyData>()?.into();
            data.try_into().map_err(Into::into)
        }).transpose()
    }

    fn fetch_obsm<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get_item("obsm", key)?.map(|x| {
            let data: ArrayData = x.extract::<PyArrayData>()?.into();
            data.try_into().map_err(Into::into)
        }).transpose()
    }

    fn fetch_obsp<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get_item("obsp", key)?.map(|x| {
            let data: ArrayData = x.extract::<PyArrayData>()?.into();
            data.try_into().map_err(Into::into)
        }).transpose()
    }

    fn fetch_varm<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get_item("varm", key)?.map(|x| {
            let data: ArrayData = x.extract::<PyArrayData>()?.into();
            data.try_into().map_err(Into::into)
        }).transpose()
    }

    fn fetch_varp<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get_item("varp", key)?.map(|x| {
            let data: ArrayData = x.extract::<PyArrayData>()?.into();
            data.try_into().map_err(Into::into)
        }).transpose()
    }

    fn add_uns<D: WriteData + Into<Data>>(&self, key: &str, data: D) -> Result<()> {
        self.set_item("uns", key, PyData::from(data.into()))
    }
    fn add_obsm<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()>
    {
        self.set_n_obs(data.shape()[0])?;
        self.set_item("obsm", key, PyArrayData::from(data.into()))
    }

    fn add_obsp<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()>
    {
        self.set_n_obs(data.shape()[0])?;
        self.set_item("obsp", key, PyArrayData::from(data.into()))
    }

    fn add_varm<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()>
    {
        self.set_n_vars(data.shape()[0])?;
        self.set_item("varm", key, PyArrayData::from(data.into()))
    }

    fn add_varp<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()>
    {
        self.set_n_vars(data.shape()[0])?;
        self.set_item("varp", key, PyArrayData::from(data.into()))
    }

    fn del_uns(&self) -> Result<()> {
        self.0.setattr("uns", pyo3::types::PyDict::new(self.py()))?;
        Ok(())
    }

    fn del_obsm(&self) -> Result<()> {
        self.0.setattr("obsm", None::<PyObject>)?;
        Ok(())
    }

    fn del_obsp(&self) -> Result<()> {
        self.0.setattr("obsp", None::<PyObject>)?;
        Ok(())
    }

    fn del_varm(&self) -> Result<()> {
        self.0.setattr("varm", None::<PyObject>)?;
        Ok(())
    }

    fn del_varp(&self) -> Result<()> {
        self.0.setattr("varp", None::<PyObject>)?;
        Ok(())
    }
}

impl<'py> AnnDataIterator for PyAnnData<'py> {
    type ArrayIter<'a, T> = PyArrayIterator<T>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
        Self: 'a;

    fn read_x_iter<'a, T>(&'a self, chunk_size: usize) -> Self::ArrayIter<'a, T>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        PyArrayIterator::new(
            self.getattr("X").unwrap().extract().unwrap(),
            chunk_size
        ).unwrap()
    }

    fn set_x_from_iter<I, D>(&self, iter: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: Into<ArrayData>,
    {
        let array = concat_array_data(iter.map(|x| x.into()))?;
        let shape = array.shape();
        self.set_n_obs(shape[0])?;
        self.set_n_vars(shape[1])?;
        self.setattr("X", PyArrayData::from(array).into_py(self.py()))?;
        Ok(())
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
        let array = self.getattr("obsm")?
            .call_method1("__getitem__", (key,))?.extract::<PyArrayData>()?;
        Ok(PyArrayIterator::new(array, chunk_size)?)
    }

    fn add_obsm_from_iter<I, D>(&self, key: &str, data: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: Into<ArrayData>,
    { 
        let array = concat_array_data(data.map(|x| x.into()))?;
        let shape = array.shape();
        self.set_n_obs(shape[0])?;
        self.getattr("obsm")?
            .call_method1("__setitem__", (key, PyArrayData::from(array).into_py(self.py())))?;
        Ok(())
    }
}

pub struct PyArrayIterator<T> {
    array: PyArrayData,
    chunk_size: usize,
    total_rows: usize,
    current_row: usize,
    phantom: std::marker::PhantomData<T>,
}

impl<T> PyArrayIterator<T> {
    pub(crate) fn new(array: PyArrayData, chunk_size: usize) -> PyResult<Self> {
        let total_rows = array.shape()[0];
        Ok(Self {
            array,
            chunk_size,
            total_rows,
            current_row: 0,
            phantom: std::marker::PhantomData,
        })
    }
}

impl<T> Iterator for PyArrayIterator<T>
where
    T: TryFrom<ArrayData>,
{
    type Item = (T, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.total_rows {
            None
        } else {
            let i = self.current_row;
            let j = std::cmp::min(self.total_rows, self.current_row + self.chunk_size);
            self.current_row = j;
            let slice = SelectInfoElem::from(i..j);
            let data = self.array.select_axis(0, slice);
            Some((T::try_from(data).ok().unwrap(), i, j))
        }
    }
}

impl<T> ExactSizeIterator for PyArrayIterator<T>
where
    T: TryFrom<ArrayData>,
{
    fn len(&self) -> usize {
        let n = self.total_rows / self.chunk_size;
        if self.total_rows % self.chunk_size == 0 {
            n
        } else {
            n + 1
        }
    }
}