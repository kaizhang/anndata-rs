use crate::data::{isinstance_of_pyanndata, isinstance_of_polars, PyArrayData, PyData, PyDataFrame};

use std::ops::Deref;
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3::exceptions::PyTypeError;
use pyo3::types::IntoPyDict;
use anndata::{self, ArrayOp, ElemCollectionOp, ArrayElemOp};
use anndata::{AnnDataOp, AxisArraysOp, ArrayData, Data, ReadArrayData, ReadData, Backend, WriteArrayData, HasShape};
use anndata::data::{DataFrameIndex, SelectInfoElem, concat_array_data, ArrayChunk, Shape};
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
            if let Some(x) = inner.x().get::<ArrayData>()? {
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
                .uns().keys()
                .into_iter()
                .try_for_each(|k| adata.uns().add(&k, inner.uns().get_item::<Data>(&k)?.unwrap()))?;
        }
        {
            // Set obsm
            inner
                .obsm().keys()
                .into_iter()
                .try_for_each(|k| adata.obsm().add(&k, inner.obsm().get_item::<ArrayData>(&k)?.unwrap()))?;
        }
        {
            // Set obsp
            inner
                .obsp().keys()
                .into_iter()
                .try_for_each(|k| adata.obsp().add(&k, inner.obsp().get_item::<ArrayData>(&k)?.unwrap()))?;
        }
        {
            // Set varm
            inner
                .varm().keys()
                .into_iter()
                .try_for_each(|k| adata.varm().add(&k, inner.varm().get_item::<ArrayData>(&k)?.unwrap()))?;
        }
        {
            // Set varp
            inner
                .varp().keys()
                .into_iter()
                .try_for_each(|k| adata.varp().add(&k, inner.varp().get_item::<ArrayData>(&k)?.unwrap()))?;
        }
        Ok(adata)
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
    type X = ArrayElem<'py>;
    type ElemCollectionRef<'a> = ElemCollection<'a> where Self: 'a;
    type AxisArraysRef<'a> = AxisArrays<'a> where Self: 'a;

    fn x(&self) -> Self::X {
        ArrayElem(self.0.getattr("X").unwrap())
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

    fn obs_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>> {todo!()}
    fn var_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>> {todo!()}

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

    fn uns(&self) -> Self::ElemCollectionRef<'_> {
        ElemCollection(self.getattr("uns").unwrap())
    }
    fn obsm(&self) -> Self::AxisArraysRef<'_> {
        AxisArrays {
            arrays: self.getattr("obsm").unwrap(),
            adata: self,
        }
    }
    fn obsp(&self) -> Self::AxisArraysRef<'_> {
        AxisArrays {
            arrays: self.getattr("obsp").unwrap(),
            adata: self,
        }
    }
    fn varm(&self) -> Self::AxisArraysRef<'_> {
        AxisArrays {
            arrays: self.getattr("varm").unwrap(),
            adata: self,
        }
    }
    fn varp(&self) -> Self::AxisArraysRef<'_> {
        AxisArrays {
            arrays: self.getattr("varp").unwrap(),
            adata: self,
        }
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

pub struct ElemCollection<'a>(&'a PyAny);

impl ElemCollectionOp for ElemCollection<'_> {
    fn keys(&self) -> Vec<String> {
        self.0.call_method0("keys").unwrap().extract().unwrap()
    }

    fn get_item<D>(&self, key: &str) -> Result<Option<D>>
        where
            D: Into<Data> + TryFrom<Data> + Clone,
            <D as TryFrom<Data>>::Error: Into<anyhow::Error>
    {
        self.0.call_method1("__getitem__", (key,)).ok().map(|x| {
            let data: Data = x.extract::<PyData>()?.into();
            data.try_into().map_err(Into::into)
        }).transpose()
    }

    fn add<D: Into<Data>>(
            &self,
            key: &str,
            data: D,
        ) -> Result<()>
    {
        let py = self.0.py();
        let d = PyData::from(data.into()).into_py(py);
        let new_d = if isinstance_of_polars(py, d.as_ref(py))? {
            d.call_method0(py, "to_pandas")?
        } else {
            d
        };
        self.0.call_method1("__setitem__", (key, new_d))?;
        Ok(())
    }

    fn remove(&self, key: &str) -> Result<()> {
        self.0.call_method1("__delitem__", (key,))?;
        Ok(())
    }
}

pub struct AxisArrays<'a> {
    arrays: &'a PyAny,
    adata: &'a PyAnnData<'a>,
}

impl<'py> AxisArraysOp for AxisArrays<'py> {
    type ArrayElem = ArrayElem<'py>;

    fn keys(&self) -> Vec<String> {
        self.arrays.call_method0("keys").unwrap().extract().unwrap()
    }

    fn get(&self, key: &str) -> Option<Self::ArrayElem> {
        self.arrays.call_method1("__getitem__", (key,)).ok().map(ArrayElem)
    }

    fn add<D: HasShape + Into<ArrayData>>(
            &self,
            key: &str,
            data: D,
        ) -> Result<()>
    {
        let py = self.arrays.py();
        let shape = data.shape();
        self.adata.set_n_obs(shape[0])?;
        let d = PyArrayData::from(data.into()).into_py(py);
        let new_d = if isinstance_of_polars(py, d.as_ref(py))? {
            d.call_method0(py, "to_pandas")?
        } else {
            d
        };
        self.arrays.call_method1("__setitem__", (key, new_d))?;
        Ok(())
    }

    fn add_iter<I, D>(&self, key: &str, data: I) -> Result<()>
        where
            I: Iterator<Item = D>,
            D: ArrayChunk + Into<ArrayData>,
    {
        let py = self.arrays.py();
        let array = ArrayChunk::concat(data)?;
        let shape = array.shape();
        self.adata.set_n_obs(shape[0])?;
        self.arrays
            .call_method1("__setitem__", (key, PyArrayData::from(array.into()).into_py(py)))?;
        Ok(())
    }

    fn remove(&self, key: &str) -> Result<()> {
        self.arrays.call_method1("__delitem__", (key,))?;
        Ok(())
    }

}

pub struct ArrayElem<'a>(&'a PyAny);

impl ArrayElemOp for ArrayElem<'_> {
    type ArrayIter<T> = PyArrayIterator<T>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

    fn shape(&self) -> Option<Shape> {
        let shape: Vec<usize> = self.0.getattr("shape").unwrap().extract().unwrap();
        Some(shape.into())
    }

    fn get<D>(&self) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let data: Option<ArrayData> = self.0.extract::<Option<PyArrayData>>()?.map(Into::into);
        data.map(|x| x.try_into().map_err(Into::into)).transpose()
    }

    fn slice<D, S>(&self, slice: S) -> Result<Option<D>>
    where
        D: ReadArrayData + Into<ArrayData> + TryFrom<ArrayData> + ArrayOp + Clone,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>
    {
        self.get::<D>().map(|mat| mat.map(|x| x.select(slice.as_ref())))
    }

    fn iter<T>(
        &self,
        chunk_size: usize,
    ) -> Self::ArrayIter<T>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        let array = self.0.extract::<PyArrayData>().unwrap();
        PyArrayIterator::new(array, chunk_size).unwrap()
    }
}