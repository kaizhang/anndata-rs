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
    pub fn read_data(&self) -> T { DataIO::read(&self.container).unwrap() }
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
    T: DataSubset2D,
{
    pub fn dtype(&self) -> DataType { self.inner.dtype.clone() }

    pub fn read_data(&self) -> T {
        match self.obs_indices.as_ref() {
            None => match self.var_indices.as_ref() {
                None => DataIO::read(&self.inner.container).unwrap(),
                Some(cidx) => DataSubsetCol::read_columns(
                    &self.inner.container, cidx
                ),
            },
            Some(ridx) => match self.var_indices.as_ref() {
                None => DataSubsetRow::read_rows(&self.inner.container, ridx),
                Some(cidx) => DataSubset2D::read_partial(
                    &self.inner.container, ridx, cidx,
                ),
            }
        }
    }
}

// NOTE: this requires `element` is the last field, as trait object contains a vtable
// at the end: https://docs.rs/vptr/latest/vptr/index.html.
impl<T> AsRef<RawMatrixElem<T>> for RawMatrixElem<dyn DataSubset2D>
where
    T: DataSubset2D,
{
    fn as_ref(&self) -> &RawMatrixElem<T> {
        if self.inner.dtype == T::dtype() {
            unsafe { &*(self as *const RawMatrixElem<dyn DataSubset2D> as *const RawMatrixElem<T>) }
        } else {
            panic!(
                "implementation error, cannot convert {:?} to {:?}",
                self.inner.dtype,
                T::dtype(),
            )
        }
    }
}

impl RawMatrixElem<dyn DataSubset2D>
{
    pub fn new(container: DataContainer) -> Result<Self> {
        let dtype = container.get_encoding_type().unwrap();
        let nrows = get_nrows(&container);
        let ncols = get_ncols(&container);
        let inner = RawElem { dtype, element: None, container };
        Ok(Self { obs_indices: None, var_indices: None, nrows, ncols, inner })
    }

    pub fn read_elem(&self) -> Result<Box<dyn DataSubset2D>> {
        match &self.inner.element {
            Some(data) => Ok(dyn_clone::clone_box(data.as_ref())),
            None => read_dyn_data_subset(
                &self.inner.container,
                self.obs_indices.as_ref().map(Vec::as_slice),
                self.var_indices.as_ref().map(Vec::as_slice),
            ),
        }
    }

    pub fn write_elem(&self, location: &Group, name: &str) -> Result<()> {
        match &self.inner.element {
            Some(data) => data.as_ref().write(location, name)?,
            None => read_dyn_data_subset(
                &self.inner.container,
                self.obs_indices.as_ref().map(Vec::as_slice),
                self.var_indices.as_ref().map(Vec::as_slice),
            )?.as_ref().write(location, name)?,
        };
        Ok(())
    }

    // TODO: fix subsetting
    pub fn subset_rows(&self, idx: &[usize]) -> Self {
        for i in idx {
            if *i >= self.nrows {
                panic!("index out of bound")
            }
        }

        let inner = RawElem {
            dtype: self.inner.dtype.clone(),
            container: self.inner.container.clone(),
            element: None,
        };
        Self {
            obs_indices: Some(idx.iter().map(|x| *x).collect()),
            var_indices: self.var_indices.clone(),
            nrows: self.nrows,
            ncols: self.ncols,
            inner,
        }
    }

    pub fn subset_cols(&self, idx: &[usize]) -> Self {
        let inner = RawElem {
            dtype: self.inner.dtype.clone(),
            container: self.inner.container.clone(),
            element: None,
        };
        Self {
            obs_indices: self.obs_indices.clone(),
            var_indices: Some(idx.iter().map(|x| *x).collect()),
            nrows: self.nrows,
            ncols: self.ncols,
            inner,
        }
    }

    pub fn subset(&self, ridx: &[usize], cidx: &[usize]) -> Self {
        let inner = RawElem {
            dtype: self.inner.dtype.clone(),
            container: self.inner.container.clone(),
            element: None,
        };
        Self {
            obs_indices: Some(ridx.iter().map(|x| *x).collect()),
            var_indices: Some(cidx.iter().map(|x| *x).collect()),
            nrows: self.nrows,
            ncols: self.ncols,
            inner,
        }
    }
}