use crate::{
    data::*,
    utils::hdf5::{
        create_str_attr, read_str_attr, read_str_vec_attr, read_str_vec,
        create_dataset,
    },
};

use std::boxed::Box;
use hdf5::Group; 
use anyhow::{Result, bail};
use ndarray::Array1;
use polars::{frame::DataFrame, series::Series};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::{Mutex, MutexGuard};
use std::ops::{Deref, DerefMut};
use indexmap::set::IndexSet;

pub struct Slot<T>(pub(crate) Arc<Mutex<Option<T>>>);

impl<T> Clone for Slot<T> {
    fn clone(&self) -> Self { Slot(self.0.clone()) }
}

impl<T> std::fmt::Display for Slot<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            write!(f, "Empty or closed slot")
        } else {
            write!(f, "{}", self.inner().deref())
        }
    }
}

impl<T> Slot<T> {
    pub fn new(x: T) -> Self { Slot(Arc::new(Mutex::new(Some(x)))) }

    pub fn empty() -> Self { Slot(Arc::new(Mutex::new(None))) }

    pub fn is_empty(&self) -> bool { self.0.lock().is_none() }

    pub fn inner(&self) -> Inner<'_, T> { Inner(self.0.lock()) }

    /// Insert data to the slot, and return the old data.
    pub fn insert(&self, data: T) -> Option<T> {
        std::mem::replace(self.0.lock().deref_mut(), Some(data))
    }

    /// Extract the data from the slot. The slot becomes empty after this operation.
    pub fn extract(&self) -> Option<T> {
        std::mem::replace(self.0.lock().deref_mut(), None)
    } 

    /// Remove the data from the slot.
    pub fn drop(&self) { let _ = self.extract(); } 
}

pub struct Inner<'a, T>(pub MutexGuard<'a, Option<T>>);

// TODO: change to result type
impl<T> Deref for Inner<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &self.0.deref() {
            None => panic!("accessing an empty slot"),
            Some(x) => x,
        }
    }
}

impl<T> DerefMut for Inner<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self.0.deref_mut() {
            None => panic!("accessing an empty slot"),
            Some(ref mut x) => x,
        }
    }
}


pub struct RawElem<T: ?Sized> {
    pub dtype: DataType,
    pub(crate) cache_enabled: bool,
    pub(crate) container: DataContainer,
    pub(crate) element: Option<Box<T>>,
}

impl<T> std::fmt::Display for RawElem<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} element, cache_enabled: {}, cached: {}",
            self.dtype,
            if self.cache_enabled { "yes" } else { "no" },
            if self.element.is_some() { "yes" } else { "no" },
        )
    }
}

impl std::fmt::Display for RawElem<dyn DataIO> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} element, cache_enabled: {}, cached: {}",
            self.dtype,
            if self.cache_enabled { "yes" } else { "no" },
            if self.element.is_some() { "yes" } else { "no" },
        )
    }
}

impl<T> RawElem<T>
where
    T: DataIO + Clone,
{
    pub fn read(&mut self) -> Result<T> { 
        match &self.element {
            Some(data) => Ok((*data.as_ref()).clone()),
            None => {
                let data: T = ReadData::read(&self.container)?;
                if self.cache_enabled {
                    self.element = Some(Box::new(data.clone()));
                }
                Ok(data)
            },
        }
    }

    pub fn write(&self, location: &Group, name: &str) -> Result<()> {
        match &self.element {
            Some(data) => data.write(location, name)?,
            None => T::read(&self.container)?.write(location, name)?,
        };
        Ok(())
    }

    pub fn enable_cache(&mut self) {
        self.cache_enabled = true;
    }

    pub fn disable_cache(&mut self) {
        if self.element.is_some() { self.element = None; }
        self.cache_enabled = false;
    }
}

impl RawElem<dyn DataIO>
{
    pub fn new(container: DataContainer) -> Result<Self> {
        let dtype = container.get_encoding_type()?;
        Ok(Self { dtype, cache_enabled: false, element: None, container })
    }

    pub fn read(&mut self) -> Result<Box<dyn DataIO>> {
        match &self.element {
            Some(data) => Ok(dyn_clone::clone_box(data.as_ref())),
            None => {
                let data = <Box<dyn DataIO>>::read(&self.container)?;
                if self.cache_enabled {
                    self.element = Some(dyn_clone::clone_box(data.as_ref()));
                }
                Ok(data)
            }
        }
    }

    pub fn write(&self, location: &Group, name: &str) -> Result<()> {
        match &self.element {
            Some(data) => data.write(location, name)?,
            None => <Box<dyn DataIO>>::read(&self.container)?.write(location, name)?,
        };
        Ok(())
    }

    pub fn enable_cache(&mut self) {
        self.cache_enabled = true;
    }

    pub fn disable_cache(&mut self) {
        if self.element.is_some() { self.element = None; }
        self.cache_enabled = false;
    }

    pub fn update<D: DataIO>(&mut self, data: &D) -> Result<()> {
        self.container = data.update(&self.container)?;
        self.element = None;
        Ok(())
    }
}

pub struct RawMatrixElem<T: ?Sized> {
    nrows: usize,
    ncols: usize,
    pub inner: RawElem<T>,
}

impl<T> std::fmt::Display for RawMatrixElem<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} element, cache_enabled: {}, cached: {}",
            self.inner.dtype,
            if self.inner.cache_enabled { "yes" } else { "no" },
            if self.inner.element.is_some() { "yes" } else { "no" },
        )
    }
}

impl std::fmt::Display for RawMatrixElem<dyn DataPartialIO> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} element, cache_enabled: {}, cached: {}",
            self.inner.dtype,
            if self.inner.cache_enabled { "yes" } else { "no" },
            if self.inner.element.is_some() { "yes" } else { "no" },
        )
    }
}

impl<T> RawMatrixElem<T>
where
    T: DataPartialIO + Clone,
{
    pub fn dtype(&self) -> DataType { self.inner.dtype.clone() }

    pub fn nrows(&self) -> usize { self.nrows }
    pub fn ncols(&self) -> usize { self.ncols }

    pub fn new_elem(container: DataContainer) -> Result<Self> {
        let dtype = container.get_encoding_type()?;
        let nrows = <Box<dyn DataPartialIO>>::get_nrows(&container);
        let ncols = <Box<dyn DataPartialIO>>::get_ncols(&container);
        let inner = RawElem { dtype, cache_enabled: false, element: None, container };
        Ok(Self { nrows, ncols, inner })
    }
    
    pub fn enable_cache(&mut self) { self.inner.enable_cache() }
    
    pub fn disable_cache(&mut self) { self.inner.disable_cache() }

    pub fn read_rows(&self, idx: &[usize]) -> T {
        match &self.inner.element {
            Some(data) => data.get_rows(idx),
            None => MatrixIO::read_rows(&self.inner.container, idx),
        }
    }

    pub fn read_columns(&self, idx: &[usize]) -> T {
        match &self.inner.element {
            Some(data) => data.get_columns(idx),
            None => MatrixIO::read_columns(&self.inner.container, idx),
        }
    }

    pub fn read_partial(&self, ridx: &[usize], cidx: &[usize]) -> T {
        match &self.inner.element {
            Some(data) => data.subset(ridx, cidx),
            None => MatrixIO::read_partial(&self.inner.container, ridx, cidx),
        }
    }

    pub fn read(&mut self) -> Result<T> { self.inner.read() }

    pub fn write(&self, location: &Group, name: &str) -> Result<()> {
        self.inner.write(location, name)
    }

    pub fn write_rows(&self, idx: &[usize], location: &Group, name: &str) -> Result<()> {
        self.read_rows(idx).write(location, name)?;
        Ok(())
    }

    pub fn write_columns(&self, idx: &[usize], location: &Group, name: &str) -> Result<()> {
        self.read_columns(idx).write(location, name)?;
        Ok(())
    }

    pub fn write_partial(&self, ridx: &[usize], cidx: &[usize], location: &Group, name: &str) -> Result<()> {
        self.read_partial(ridx, cidx).write(location, name)?;
        Ok(())
    }

    pub fn subset_rows(&mut self, idx: &[usize]) -> Result<()> {
        for i in idx {
            if *i >= self.nrows {
                panic!("index out of bound")
            }
        }
        let data = self.read_rows(idx);
        self.inner.container = data.update(&self.inner.container)?;
        if self.inner.element.is_some() {
            self.inner.element = Some(Box::new(data));
        }
        self.nrows = idx.len();
        Ok(())
    }

    pub fn subset_cols(&mut self, idx: &[usize]) -> Result<()> {
        for i in idx {
            if *i >= self.ncols {
                panic!("index out of bound")
            }
        }
        let data = self.read_columns(idx);
        self.inner.container = data.update(&self.inner.container)?;
        if self.inner.element.is_some() {
            self.inner.element = Some(Box::new(data));
        }
        self.ncols = idx.len();
        Ok(())
    }

    pub fn subset(&mut self, ridx: &[usize], cidx: &[usize]) -> Result<()> {
        for i in ridx {
            if *i >= self.nrows {
                panic!("row index out of bound")
            }
        }
        for j in cidx {
            if *j >= self.ncols {
                panic!("column index out of bound")
            }
        }
        let data = self.read_partial(ridx, cidx);
        self.inner.container = data.update(&self.inner.container)?;
        if self.inner.element.is_some() {
            self.inner.element = Some(Box::new(data));
        }
        self.nrows = ridx.len();
        self.ncols = cidx.len();
        Ok(())
    }

    pub fn update(&mut self, data: &T) -> Result<()> {
        self.nrows = data.nrows();
        self.ncols = data.ncols();
        self.inner.container = data.update(&self.inner.container)?;
        self.inner.element = None;
        Ok(())
    }
}

impl RawMatrixElem<dyn DataPartialIO>
{
    pub fn new(container: DataContainer) -> Result<Self> {
        let dtype = container.get_encoding_type()?;
        let nrows = <Box<dyn DataPartialIO>>::get_nrows(&container);
        let ncols = <Box<dyn DataPartialIO>>::get_ncols(&container);
        let inner = RawElem { dtype, cache_enabled: false, element: None, container };
        Ok(Self { nrows, ncols, inner })
    }

    pub fn enable_cache(&mut self) { self.inner.cache_enabled = true; }

    pub fn disable_cache(&mut self) {
        if self.inner.element.is_some() { self.inner.element = None; }
        self.inner.cache_enabled = false;
    }

    pub fn nrows(&self) -> usize { self.nrows }
    pub fn ncols(&self) -> usize { self.ncols }

    pub fn read_rows(&self, idx: &[usize]) -> Result<Box<dyn DataPartialIO>> {
        Ok(read_dyn_data_subset(&self.inner.container, Some(idx), None)?)
    }

    pub fn read_dyn_row_slice(&self, slice: std::ops::Range<usize>) -> Result<Box<dyn DataPartialIO>> {
        Ok(<Box<dyn DataPartialIO>>::read_row_slice(&self.inner.container, slice)?)
    }

    pub fn read_columns(&self, idx: &[usize]) -> Result<Box<dyn DataPartialIO>> {
        Ok(read_dyn_data_subset(&self.inner.container, None, Some(idx))?)
    }

    pub fn read_partial(&self, ridx: &[usize], cidx: &[usize]) -> Result<Box<dyn DataPartialIO>> {
        Ok(read_dyn_data_subset(&self.inner.container, Some(ridx), Some(cidx))?)
    }

    pub fn read(&mut self) -> Result<Box<dyn DataPartialIO>> {
        match &self.inner.element {
            Some(data) => Ok(dyn_clone::clone_box(data.as_ref())),
            None => {
                let data = read_dyn_data_subset(&self.inner.container, None, None)?;
                if self.inner.cache_enabled {
                    self.inner.element = Some(dyn_clone::clone_box(data.as_ref()));
                }
                Ok(data)
            },
        }
    }

    pub fn write(&self, location: &Group, name: &str) -> Result<()> {
        match &self.inner.element {
            Some(data) => data.write(location, name)?,
            None => read_dyn_data_subset(&self.inner.container, None, None)?
                .write(location, name)?,
        };
        Ok(())
    }

    pub fn write_rows(&self, idx: &[usize], location: &Group, name: &str) -> Result<()> {
        self.read_rows(idx)?.write(location, name)?;
        Ok(())
    }

    pub fn write_columns(&self, idx: &[usize], location: &Group, name: &str) -> Result<()> {
        self.read_columns(idx)?.write(location, name)?;
        Ok(())
    }

    pub fn write_partial(&self, ridx: &[usize], cidx: &[usize], location: &Group, name: &str) -> Result<()> {
        self.read_partial(ridx, cidx)?.write(location, name)?;
        Ok(())
    }

    pub fn update<D: DataPartialIO>(&mut self, data: &D) -> Result<()> {
        self.nrows = data.nrows();
        self.ncols = data.ncols();
        self.inner.container = data.update(&self.inner.container)?;
        self.inner.element = None;
        Ok(())
    }

    pub fn subset_rows(&mut self, idx: &[usize]) -> Result<()> {
        for i in idx {
            if *i >= self.nrows {
                panic!("index out of bound")
            }
        }
        let data = self.read_rows(idx)?;
        self.inner.container = data.update(&self.inner.container)?;
        if self.inner.element.is_some() {
            self.inner.element = Some(data);
        }
        self.nrows = idx.len();
        Ok(())
    }

    pub fn subset_cols(&mut self, idx: &[usize]) -> Result<()> {
        for i in idx {
            if *i >= self.ncols {
                panic!("index out of bound")
            }
        }
        let data = self.read_columns(idx)?;
        self.inner.container = data.update(&self.inner.container)?;
        if self.inner.element.is_some() {
            self.inner.element = Some(data);
        }
        self.ncols = idx.len();
        Ok(())
    }

    pub fn subset(&mut self, ridx: &[usize], cidx: &[usize]) -> Result<()> {
        for i in ridx {
            if *i >= self.nrows {
                panic!("row index out of bound")
            }
        }
        for j in cidx {
            if *j >= self.ncols {
                panic!("column index out of bound")
            }
        }
        let data = self.read_partial(ridx, cidx)?;
        self.inner.container = data.update(&self.inner.container)?;
        if self.inner.element.is_some() {
            self.inner.element = Some(data);
        }
        self.nrows = ridx.len();
        self.ncols = cidx.len();
        Ok(())
    }
}

pub struct DataFrameIndex {
    pub index_name: String,
    pub names: Vec<String>,
    pub index_map: HashMap<String, usize>,
}

impl DataFrameIndex {
    pub fn len(&self) -> usize { self.names.len() }

    pub fn get(&self, k: &String) -> Option<usize> { self.index_map.get(k).map(|x| *x) }

    pub fn is_empty(&self) -> bool { self.names.is_empty() }

    fn write(&self, container: &DataContainer) -> Result<()> {
        let group = container.get_group_ref()?;
        create_str_attr(group, "_index", &self.index_name)?;

        if group.link_exists(&self.index_name) { group.unlink(&self.index_name)?; }

        let names: Array1<hdf5::types::VarLenUnicode> =
            self.names.iter().map(|x| x.parse().unwrap()).collect();
        create_dataset(group, &self.index_name, &names)?;
        Ok(())
    }
}

impl From<Vec<String>> for DataFrameIndex {
    fn from(names: Vec<String>) -> Self {
        let index_map = names.clone().into_iter().enumerate().map(|(a, b)| (b, a)).collect();
        Self { index_name: "index".to_owned(), names, index_map }
    }
}

impl FromIterator<String> for DataFrameIndex {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = String>,
    {
        let names: Vec<_> = iter.into_iter().collect();
        let index_map = names.clone().into_iter().enumerate().map(|(a, b)| (b, a)).collect();
        Self { index_name: "index".to_owned(), names, index_map }
    }
}

impl<'a> FromIterator<&'a str> for DataFrameIndex {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a str>,
    {
        let names: Vec<_> = iter.into_iter().map(|x| x.to_owned()).collect();
        let index_map = names.clone().into_iter().enumerate().map(|(a, b)| (b, a)).collect();
        Self { index_name: "index".to_owned(), names, index_map }
    }
}

pub struct InnerDataFrameElem {
    element: Option<DataFrame>,
    container: DataContainer,
    column_names: IndexSet<String>,
    pub index: DataFrameIndex,
}

impl InnerDataFrameElem {
    pub fn new(location: &Group, name: &str, index: DataFrameIndex, df: &DataFrame) -> Result<Self> {
        if df.height() == 0 || index.len() == df.height() {
            let container = df.write(location, name)?;
            index.write(&container)?;
            let column_names = df.get_column_names_owned().into_iter().collect();
            Ok(Self { element: None, container, column_names, index })
        } else {
            bail!("cannot create dataframe element as lengths of index and dataframe differ")
        }
    }
}

impl std::fmt::Display for InnerDataFrameElem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dataframe element")
    }
}

pub type DataFrameElem = Slot<InnerDataFrameElem>;

impl TryFrom<DataContainer> for DataFrameElem {
    type Error = anyhow::Error;

    fn try_from(container: DataContainer) -> Result<Self> {
        match container.get_encoding_type()? {
            DataType::DataFrame => {
                let grp = container.get_group_ref()?;
                let index_name = read_str_attr(grp, "_index")?;
                let mut index: DataFrameIndex = read_str_vec(&grp.dataset(&index_name)?)?.into();
                index.index_name = index_name;
                let column_names = read_str_vec_attr(grp, "column-order")?.into_iter().collect();
                let df = InnerDataFrameElem { element: None, container, column_names, index };
                Ok(Slot::new(df))
            },
            ty => bail!("Expecting a dataframe but found: {}", ty),
        }
    }
}

impl DataFrameElem {
    pub fn nrows(&self) -> usize { self.inner().index.len() }

    pub fn with_data_ref<F, O>(&self, f: F) -> Result<O>
    where
        F: Fn(Option<(&DataFrameIndex, &DataFrame)>) -> O,
    {
        if self.is_empty() {
            Ok(f(None))
        } else {
            let mut inner = self.inner();
            if let None = inner.element {
                inner.element = Some(DataFrame::read(&inner.container)?);
            }
            Ok(f(Some(
                (&inner.index, inner.element.as_ref().unwrap())
            )))
        }
    }

    pub fn with_data_mut_ref<F, O>(&self, mut f: F) -> Result<O>
    where
        F: FnMut(Option<(&DataFrameIndex, &DataFrame)>) -> O,
    {
        if self.is_empty() {
            Ok(f(None))
        } else {
            let mut inner = self.inner();
            if let None = inner.element {
                inner.element = Some(DataFrame::read(&inner.container)?);
            }
            Ok(f(Some(
                (&inner.index, inner.element.as_ref().unwrap())
            )))
        }
    }

    pub fn read(&self) -> Result<DataFrame> {
        self.with_data_ref(|x| x.map(|(_, df)| df.clone()).unwrap_or(DataFrame::empty()))
    }

    pub fn write(&self, location: &Group, name: &str) -> Result<()> {
        self.with_data_ref(|x| {
            if let Some((index, df)) = x {
                let container = df.write(location, name)?;
                index.write(&container)?;
            }
            Ok(())
        })?
    }

    pub fn write_rows(&self, idx: &[usize], location: &Group, name: &str) -> Result<()> {
        let subset = self.with_data_ref(|x| x.map(|(index, df)| {
            let new_df = df.take_iter(idx.into_iter().map(|x| *x))?;
            let new_index: DataFrameIndex = idx.into_iter().map(|i| index.names[*i].clone()).collect();
            Ok::<_, anyhow::Error>((new_index, new_df))
        }))?.transpose()?;

        if let Some((index, df)) = subset {
            let container = df.write(location, name)?;
            index.write(&container)?;
        }
        Ok(())
    }

    pub fn update(&self, data: &DataFrame) -> Result<()> {
        let mut inner = self.inner();
        if inner.index.len() == data.height() {
            inner.container = data.update(&inner.container)?;
            inner.element = None;
            inner.column_names = data.get_column_names_owned().into_iter().collect();
            Ok(())
        } else {
            bail!("cannot update dataframe as lengths differ")
        }
    }

    pub fn column(&self, name: &str) -> Result<Series> {
        self.with_data_ref(|x| x.unwrap().1.column(name).unwrap().clone())
    }

    pub fn get_column_names(&self) -> Option<IndexSet<String>> {
        if self.is_empty() { None } else { Some(self.inner().column_names.clone()) }
    }

    pub fn set_index(&self, index: DataFrameIndex) -> Result<()> {
        let mut inner = self.inner();
        let container = &inner.container;
        if inner.index.len() == index.len() {
            index.write(container)?;
            inner.index = index;
            Ok(())
        } else {
            bail!("cannot change the index as the lengths differ");
        }
    }

    pub fn subset_rows(&self, idx: &[usize]) -> Result<()> {
        let subset = self.with_data_ref(|x| x.map(|(index, df)| {
            let new_df = df.take_iter(idx.into_iter().map(|x| *x))?;
            let new_index: DataFrameIndex = idx.into_iter().map(|i| index.names[*i].clone()).collect();
            Ok::<_, anyhow::Error>((new_index, new_df))
        }))?.transpose()?;

        if let Some((index, df)) = subset {
            let mut inner = self.inner();
            let container = df.update(&inner.container)?;
            inner.column_names = df.get_column_names_owned().into_iter().collect();
            inner.element = None;
            index.write(&container)?;
            inner.index = index;
            inner.container = container;
        }
        Ok(())
    }
}