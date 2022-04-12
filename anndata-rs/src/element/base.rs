use crate::anndata_trait::*;

use std::boxed::Box;
use hdf5::{Result, Group}; 

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
                let data = read_dyn_data(&self.container)?;
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
            None => read_dyn_data(&self.container)?.write(location, name)?,
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

    pub fn update(&mut self, data: &Box<dyn DataIO>) -> Result<()> {
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
        let nrows = get_nrows(&container);
        let ncols = get_ncols(&container);
        let inner = RawElem { dtype, cache_enabled: false, element: None, container };
        Ok(Self { nrows, ncols, inner })
    }
    
    pub fn enable_cache(&mut self) { self.inner.enable_cache() }
    
    pub fn disable_cache(&mut self) { self.inner.disable_cache() }

    pub fn read_rows(&self, idx: &[usize]) -> T {
        match &self.inner.element {
            Some(data) => data.get_rows(idx),
            None => ReadRows::read_rows(&self.inner.container, idx),
        }
    }

    pub fn read_columns(&self, idx: &[usize]) -> T {
        match &self.inner.element {
            Some(data) => data.get_columns(idx),
            None => ReadCols::read_columns(&self.inner.container, idx),
        }
    }

    pub fn read_partial(&self, ridx: &[usize], cidx: &[usize]) -> T {
        match &self.inner.element {
            Some(data) => data.subset(ridx, cidx),
            None => ReadPartial::read_partial(&self.inner.container, ridx, cidx),
        }
    }

    pub fn read(&mut self) -> Result<T> { self.inner.read() }

    pub fn write(&self, location: &Group, name: &str) -> Result<()> {
        self.inner.write(location, name)
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
        let nrows = get_nrows(&container);
        let ncols = get_ncols(&container);
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
        read_dyn_data_subset(&self.inner.container, Some(idx), None)
    }

    pub fn read_dyn_row_slice(&self, slice: std::ops::Range<usize>) -> Result<Box<dyn DataPartialIO>> {
        read_dyn_row_slice(&self.inner.container, slice)
    }

    pub fn read_columns(&self, idx: &[usize]) -> Result<Box<dyn DataPartialIO>> {
        read_dyn_data_subset(&self.inner.container, None, Some(idx))
    }

    pub fn read_partial(&self, ridx: &[usize], cidx: &[usize]) -> Result<Box<dyn DataPartialIO>> {
        read_dyn_data_subset(&self.inner.container, Some(ridx), Some(cidx))
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

    pub fn update(&mut self, data: &Box<dyn DataPartialIO>) -> Result<()> {
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

    pub fn downcast<T>(&self) -> &RawMatrixElem<T>
    where
        T: DataPartialIO,
    {
        if self.inner.dtype == T::dtype() {
            unsafe { &*(self as *const RawMatrixElem<dyn DataPartialIO> as *const RawMatrixElem<T>) }
        } else {
            panic!(
                "implementation error, cannot convert {:?} to {:?}",
                self.inner.dtype,
                T::dtype(),
            )
        }
    }
}