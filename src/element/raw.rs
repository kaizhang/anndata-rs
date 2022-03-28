use crate::anndata_trait::*;

use std::boxed::Box;
use hdf5::{Result, Group}; 

pub struct RawElem<T: ?Sized> {
    pub dtype: DataType,
    pub(crate) container: DataContainer,
    pub(crate) element: Option<Box<T>>,
}

impl<T> RawElem<T>
where
    T: DataIO,
{
    pub fn read_data(&self) -> T { ReadData::read(&self.container).unwrap() }
}

impl<T> AsRef<RawElem<T>> for RawElem<dyn DataIO>
where
    T: DataIO,
{
    fn as_ref(&self) -> &RawElem<T> {
        if self.dtype == T::dtype() {
            unsafe { &*(self as *const RawElem<dyn DataIO> as *const RawElem<T>) }
        } else {
            panic!(
                "implementation error, cannot convert {:?} to {:?}",
                self.dtype,
                T::dtype(),
            )
        }
    }
}


pub struct RawMatrixElem<T: ?Sized> {
    pub obs_indices: Option<Vec<usize>>,
    pub var_indices: Option<Vec<usize>>,
    pub nrows: usize,
    pub ncols: usize,
    pub inner: RawElem<T>,
}

impl<T> RawMatrixElem<T>
where
    T: DataPartialIO + Clone,
{
    pub fn dtype(&self) -> DataType { self.inner.dtype.clone() }

    pub fn new_elem(container: DataContainer) -> Result<Self> {
        let dtype = container.get_encoding_type().unwrap();
        let nrows = get_nrows(&container);
        let ncols = get_ncols(&container);
        let inner = RawElem { dtype, element: None, container };
        Ok(Self { obs_indices: None, var_indices: None, nrows, ncols, inner })
    }

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

    pub fn read_elem(&self) -> T {
        match &self.inner.element {
            Some(data) => *data.clone(),
            None => ReadData::read(&self.inner.container).unwrap(),
        }
    }

    pub fn write_elem(&self, location: &Group, name: &str) -> Result<()> {
        match &self.inner.element {
            Some(data) => data.write(location, name)?,
            None => self.read_elem().write(location, name)?,
        };
        Ok(())
    }

    pub fn subset_rows(&mut self, idx: &[usize]) {
        for i in idx {
            if *i >= self.nrows {
                panic!("index out of bound")
            }
        }
        let data = self.read_rows(idx);
        self.inner.container = data.update(&self.inner.container).unwrap();
        if self.inner.element.is_some() {
            self.inner.element = Some(Box::new(data));
        }
        self.nrows = idx.len();
    }

    pub fn subset_cols(&mut self, idx: &[usize]) {
        for i in idx {
            if *i >= self.ncols {
                panic!("index out of bound")
            }
        }
        let data = self.read_columns(idx);
        self.inner.container = data.update(&self.inner.container).unwrap();
        if self.inner.element.is_some() {
            self.inner.element = Some(Box::new(data));
        }
        self.ncols = idx.len();
    }

    pub fn subset(&mut self, ridx: &[usize], cidx: &[usize]) {
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
        self.inner.container = data.update(&self.inner.container).unwrap();
        if self.inner.element.is_some() {
            self.inner.element = Some(Box::new(data));
        }
        self.nrows = ridx.len();
        self.ncols = cidx.len();
    }
}

// NOTE: this requires `element` is the last field, as trait object contains a vtable
// at the end: https://docs.rs/vptr/latest/vptr/index.html.
impl<T> AsRef<RawMatrixElem<T>> for RawMatrixElem<dyn DataPartialIO>
where
    T: DataPartialIO,
{
    fn as_ref(&self) -> &RawMatrixElem<T> {
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

impl RawMatrixElem<dyn DataPartialIO>
{
    pub fn new(container: DataContainer) -> Result<Self> {
        let dtype = container.get_encoding_type().unwrap();
        let nrows = get_nrows(&container);
        let ncols = get_ncols(&container);
        let inner = RawElem { dtype, element: None, container };
        Ok(Self { obs_indices: None, var_indices: None, nrows, ncols, inner })
    }

    pub fn read_rows(&self, idx: &[usize]) -> Box<dyn DataPartialIO> {
        read_dyn_data_subset(&self.inner.container, Some(idx), None).unwrap()
    }

    pub fn read_columns(&self, idx: &[usize]) -> Box<dyn DataPartialIO> {
        read_dyn_data_subset(&self.inner.container, None, Some(idx)).unwrap()
    }

    pub fn read_partial(&self, ridx: &[usize], cidx: &[usize]) -> Box<dyn DataPartialIO> {
        read_dyn_data_subset(&self.inner.container, Some(ridx), Some(cidx)).unwrap()
    }

    pub fn read_elem(&self) -> Box<dyn DataPartialIO> {
        match &self.inner.element {
            Some(data) => dyn_clone::clone_box(data.as_ref()),
            None => read_dyn_data_subset(&self.inner.container, None, None).unwrap(),
        }
    }

    pub fn write_elem(&self, location: &Group, name: &str) -> Result<()> {
        match &self.inner.element {
            Some(data) => data.write(location, name)?,
            None => self.read_elem().write(location, name)?,
        };
        Ok(())
    }

    pub fn subset_rows(&mut self, idx: &[usize]) {
        for i in idx {
            if *i >= self.nrows {
                panic!("index out of bound")
            }
        }
        let data = self.read_rows(idx);
        self.inner.container = data.update(&self.inner.container).unwrap();
        if self.inner.element.is_some() {
            self.inner.element = Some(data);
        }
        self.nrows = idx.len();
    }

    pub fn subset_cols(&mut self, idx: &[usize]) {
        for i in idx {
            if *i >= self.ncols {
                panic!("index out of bound")
            }
        }
        let data = self.read_columns(idx);
        self.inner.container = data.update(&self.inner.container).unwrap();
        if self.inner.element.is_some() {
            self.inner.element = Some(data);
        }
        self.ncols = idx.len();
    }

    pub fn subset(&mut self, ridx: &[usize], cidx: &[usize]) {
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
        self.inner.container = data.update(&self.inner.container).unwrap();
        if self.inner.element.is_some() {
            self.inner.element = Some(data);
        }
        self.nrows = ridx.len();
        self.ncols = cidx.len();
    }
}