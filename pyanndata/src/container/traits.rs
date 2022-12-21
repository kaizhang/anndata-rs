use std::ops::Deref;

use crate::data::{
    is_none_slice, to_select_info, IntoPython, PyArrayData, PyData, PyDataFrame, PySeries,
};

use anndata::backend::DataType;
use anndata::data::SelectInfoElem;
use anndata::{
    ArrayData, ArrayElem, AxisArrays, Backend, Data,
    DataFrameElem, Elem, ElemCollection, StackedArrayElem, StackedDataFrame,
};
use anndata::container::{ChunkedArrayElem, StackedChunkedArrayElem};
use anyhow::{bail, Context, Result};
use pyo3::prelude::*;
use rand::Rng;
use rand::SeedableRng;

use super::{PyArrayElem, PyElem, PyChunkedArray};

/// Trait for `Elem` to abtract over different backends.
pub trait ElemTrait: Send {
    fn enable_cache(&self);
    fn disable_cache(&self);
    fn is_scalar(&self) -> bool;
    fn get<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> Result<PyData>;
    fn show(&self) -> String;
}

impl<B: Backend> ElemTrait for Elem<B> {
    fn enable_cache(&self) {
        self.lock().as_mut().map(|x| x.enable_cache());
    }

    fn disable_cache(&self) {
        self.lock().as_mut().map(|x| x.disable_cache());
    }

    fn is_scalar(&self) -> bool {
        match self.inner().dtype() {
            DataType::Scalar(_) => true,
            _ => false,
        }
    }

    fn get<'py>(&self, py: Python<'py>, slice: &'py PyAny) -> Result<PyData> {
        if is_none_slice(py, slice)? {
            Ok(self.inner().data::<Data>()?.into())
        } else {
            bail!("Please use None slice to retrieve data.")
        }
    }

    fn show(&self) -> String {
        format!("{}", self)
    }
}

pub trait ArrayElemTrait: Send {
    fn enable_cache(&self);
    fn disable_cache(&self);
    fn show(&self) -> String;
    fn get<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> Result<PyArrayData>;
    fn shape(&self) -> Vec<usize>;
    fn chunk(
        &self,
        size: usize,
        replace: bool,
        seed: u64,
    ) -> Result<ArrayData>;
    fn chunked(&self, chunk_size: usize) -> PyChunkedArray;
}

impl<B: Backend + 'static> ArrayElemTrait for ArrayElem<B> {
    fn enable_cache(&self) {
        self.lock().as_mut().map(|x| x.enable_cache());
    }

    fn disable_cache(&self) {
        self.lock().as_mut().map(|x| x.disable_cache());
    }

    fn get<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> Result<PyArrayData> {
        let slice = to_select_info(subscript, self.inner().shape())?;
        self.inner()
            .select::<ArrayData, _>(slice.as_ref())
            .map(|x| x.into())
    }

    fn show(&self) -> String {
        format!("{}", self)
    }

    fn shape(&self) -> Vec<usize> {
        self.inner().shape().as_ref().to_vec()
    }

    fn chunk(
        &self,
        size: usize,
        replace: bool,
        seed: u64,
    ) -> Result<ArrayData> {
        let length = self.shape()[0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let idx: Vec<usize> = if replace {
            std::iter::repeat_with(|| rng.gen_range(0..length))
                .take(size)
                .collect()
        } else {
            rand::seq::index::sample(&mut rng, length, size).into_vec()
        };
        self.inner().select_axis::<ArrayData, _>(0, &SelectInfoElem::from(idx))
    }

    fn chunked(&self, chunk_size: usize) -> PyChunkedArray {
        self.chunked::<ArrayData>(chunk_size).into()
    }
}

impl<B: Backend + 'static> ArrayElemTrait for StackedArrayElem<B> {
    fn enable_cache(&self) {
        self.deref().enable_cache();
    }

    fn disable_cache(&self) {
        self.deref().disable_cache();
    }

    fn get<'py>(&self, py: Python<'py>, subscript: &'py PyAny) -> Result<PyArrayData> {
        let slice = to_select_info(subscript, self.deref().shape())?;
        self.select::<ArrayData, _>(slice.as_ref())
            .map(|x| x.into())
    }

    fn show(&self) -> String {
        format!("{}", self)
    }

    fn shape(&self) -> Vec<usize> {
        self.deref().shape().as_ref().to_vec()
    }

    fn chunk(
        &self,
        size: usize,
        replace: bool,
        seed: u64,
    ) -> Result<ArrayData> {
        let length = self.shape()[0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let idx: Vec<usize> = if replace {
            std::iter::repeat_with(|| rng.gen_range(0..length))
                .take(size)
                .collect()
        } else {
            rand::seq::index::sample(&mut rng, length, size).into_vec()
        };
        self.select_axis::<ArrayData, _>(0, &SelectInfoElem::from(idx))
    }

    fn chunked(&self, chunk_size: usize) -> PyChunkedArray {
        self.chunked::<ArrayData>(chunk_size).into()
    }
}

pub trait DataFrameElemTrait: Send {
    fn get(&self, subscript: &PyAny) -> Result<PyObject>;
    fn set<'py>(&self, py: Python<'py>, key: &str, data: PySeries) -> Result<()>;
    fn contains(&self, key: &str) -> bool;
    fn show(&self) -> String;
}

impl<B: Backend> DataFrameElemTrait for DataFrameElem<B> {
    fn get(&self, subscript: &PyAny) -> Result<PyObject> {
        let py = subscript.py();
        if let Ok(key) = subscript.extract::<&str>() {
            Ok(self.inner().column(key)?.into_python(py)?)
        } else {
            let width = self.inner().width();
            let height = self.inner().height();
            let shape = [width, height].as_slice().into();
            let slice = to_select_info(subscript, &shape)?;
            let df = self.inner().select(slice.as_ref())?;
            Ok(PyDataFrame::from(df).into_py(py))
        }
    }

    fn set<'py>(&self, py: Python<'py>, key: &str, data: PySeries) -> Result<()> {
        todo!()
    }

    fn contains(&self, key: &str) -> bool {
        self.lock()
            .as_ref()
            .map(|x| x.get_column_names().contains(key))
            .unwrap_or(false)
    }

    fn show(&self) -> String {
        format!("{}", self)
    }
}

impl<B: Backend> DataFrameElemTrait for StackedDataFrame<B> {
    fn get(&self, subscript: &PyAny) -> Result<PyObject> {
        let py = subscript.py();
        if let Ok(key) = subscript.extract::<&str>() {
            Ok(self.column(key)?.into_python(py)?)
        } else {
            let width = self.width();
            let height = self.height();
            let shape = [width, height].as_slice().into();
            let slice = to_select_info(subscript, &shape)?;
            let df = self.select(slice.as_ref())?;
            Ok(PyDataFrame::from(df).into_py(py))
        }
    }

    fn set<'py>(&self, py: Python<'py>, key: &str, data: PySeries) -> Result<()> {
        todo!()
    }

    fn contains(&self, key: &str) -> bool {
        self.get_column_names().contains(key)
    }

    fn show(&self) -> String {
        format!("{}", self)
    }
}

pub trait AxisArrayTrait: Send {
    fn keys(&self) -> Vec<String>;
    fn contains(&self, key: &str) -> bool;
    fn get(&self, key: &str) -> Result<PyArrayData>;
    fn el(&self, key: &str) -> Result<PyArrayElem>;
    fn set(&self, key: &str, data: PyArrayData) -> Result<()>;
    fn show(&self) -> String;
}

impl<B: Backend + 'static> AxisArrayTrait for AxisArrays<B> {
    fn keys(&self) -> Vec<String> {
        self.inner().keys().map(|x| x.to_string()).collect()
    }

    fn contains(&self, key: &str) -> bool {
        self.inner().contains_key(key)
    }

    fn get(&self, key: &str) -> Result<PyArrayData> {
        Ok(self
            .inner()
            .get(key)
            .context(format!("No such key: {}", key))?
            .inner()
            .data::<ArrayData>()?
            .into())
    }

    fn el(&self, key: &str) -> Result<PyArrayElem> {
        Ok(self
            .inner()
            .get(key)
            .context(format!("No such key: {}", key))?
            .clone()
            .into())
    }

    fn set(&self, key: &str, data: PyArrayData) -> Result<()> {
        self.inner().add_data::<ArrayData>(key, data.into())
    }

    fn show(&self) -> String {
        format!("{}", self)
    }
}

pub trait ElemCollectionTrait: Send {
    fn keys(&self) -> Vec<String>;
    fn contains(&self, key: &str) -> bool;
    fn get(&self, key: &str) -> Result<PyData>;
    fn el(&self, key: &str) -> Result<PyElem>;
    fn set(&self, key: &str, data: PyData) -> Result<()>;
    fn show(&self) -> String;
}

impl<B: Backend + 'static> ElemCollectionTrait for ElemCollection<B> {
    fn keys(&self) -> Vec<String> {
        self.inner().keys().map(|x| x.to_string()).collect()
    }

    fn contains(&self, key: &str) -> bool {
        self.inner().contains_key(key)
    }

    fn get(&self, key: &str) -> Result<PyData> {
        Ok(self
            .inner()
            .get(key)
            .context(format!("No such key: {}", key))?
            .inner()
            .data::<Data>()?
            .into())
    }

    fn el(&self, key: &str) -> Result<PyElem> {
        Ok(self
            .inner()
            .get(key)
            .context(format!("No such key: {}", key))?
            .clone()
            .into())
    }

    fn set(&self, key: &str, data: PyData) -> Result<()> {
        self.inner().add_data::<Data>(key, data.into())
    }

    fn show(&self) -> String {
        format!("{}", self)
    }
}

pub trait ChunkedArrayTrait: ExactSizeIterator<Item = (ArrayData, usize, usize)> + Send {}

impl<B: Backend> ChunkedArrayTrait for ChunkedArrayElem<B, ArrayData> {}
impl<B: Backend> ChunkedArrayTrait for StackedChunkedArrayElem<B, ArrayData> {}