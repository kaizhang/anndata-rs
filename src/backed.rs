use crate::anndata_trait::*;
use crate::base::*;

use std::boxed::Box;
use hdf5::{Result, Group}; 
use ndarray::{ArrayD};
use nalgebra_sparse::csr::CsrMatrix;
use std::sync::Arc;

pub type AnnData = AnnDataBase<
    Elem2dView,
    Elem2dView,
    Elem2dView,
>;

pub type Elem2dView = Arc<RawElemView<dyn DataSubset2D>>;

impl BoxedData for Elem2dView {
    fn new(container: DataContainer) -> Result<Self> {
        let dtype = container.get_encoding_type().unwrap();
        let inner = RawElem { dtype, element: None, container };
        let elem = RawElemView { obs_indices: None, var_indices: None, inner };
        Ok(Arc::new(elem))
    }

    fn write(&self, location: &Group, name: &str) -> Result<()> {
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
}

// TODO: fix subsetting
impl BoxedDataSubRow for Elem2dView {
    fn subset_rows(&self, idx: &[usize]) -> Self {
        let inner = RawElem {
            dtype: self.inner.dtype.clone(),
            container: self.inner.container.clone(),
            element: None,
        };
        let elem = RawElemView {
            obs_indices: Some(idx.iter().map(|x| *x).collect()),
            var_indices: self.var_indices.clone(),
            inner,
        };
        Arc::new(elem)
    }
}

impl BoxedDataSubCol for Elem2dView {
    fn subset_cols(&self, idx: &[usize]) -> Self {
        let mut data = self.clone();
        let raw = Arc::get_mut(&mut data).unwrap();
        raw.var_indices = Some(idx.iter().map(|x| *x).collect());
        raw.inner.element = None;
        data
    }
}

impl BoxedDataSub2D for Elem2dView {
    fn subset(&self, ridx: &[usize], cidx: &[usize]) -> Self {
        let inner = RawElem {
            dtype: self.inner.dtype.clone(),
            container: self.inner.container.clone(),
            element: None,
        };
        let elem = RawElemView {
            obs_indices: Some(ridx.iter().map(|x| *x).collect()),
            var_indices: Some(cidx.iter().map(|x| *x).collect()),
            inner,
        };
        Arc::new(elem)
    }
}

////////////////////////////////////////////////////////////////////////////////

pub struct RawElem<T: ?Sized> {
    pub dtype: DataType,
    container: DataContainer,
    element: Option<Box<T>>,
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

impl RawElem<dyn DataIO>
{
    pub fn csr_f32(&self) -> &RawElem<CsrMatrix<f32>> { self.as_ref() }
    pub fn csr_f64(&self) -> &RawElem<CsrMatrix<f64>> { self.as_ref() }
    pub fn csr_u32(&self) -> &RawElem<CsrMatrix<u32>> { self.as_ref() }
    pub fn csr_u64(&self) -> &RawElem<CsrMatrix<u64>> { self.as_ref() }
    pub fn arr_f32(&self) -> &RawElem<ArrayD<f32>> { self.as_ref() }
    pub fn arr_f64(&self) -> &RawElem<ArrayD<f64>> { self.as_ref() }
    pub fn arr_u32(&self) -> &RawElem<ArrayD<u32>> { self.as_ref() }
    pub fn arr_u64(&self) -> &RawElem<ArrayD<u64>> { self.as_ref() }
}

impl BoxedData for RawElem<dyn DataIO> {
    fn new(container: DataContainer) -> Result<Self> {
        let dtype = container.get_encoding_type().unwrap();
        Ok(Self { dtype, element: None, container })
    }

    fn write(&self, location: &Group, name: &str) -> Result<()> {
        match &self.element {
            None => read_dyn_data(&self.container)?.as_ref().write(location, name)?,
            Some(data) => data.as_ref().write(location, name)?,
        };
        Ok(())
    }
}

pub struct RawElemView<T: ?Sized> {
    pub obs_indices: Option<Vec<usize>>,
    pub var_indices: Option<Vec<usize>>,
    pub inner: RawElem<T>,
}

impl<T> RawElemView<T>
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
impl<T> AsRef<RawElemView<T>> for RawElemView<dyn DataSubset2D>
where
    T: DataSubset2D,
{
    fn as_ref(&self) -> &RawElemView<T> {
        if self.inner.dtype == T::dtype() {
            unsafe { &*(self as *const RawElemView<dyn DataSubset2D> as *const RawElemView<T>) }
        } else {
            panic!(
                "implementation error, cannot convert {:?} to {:?}",
                self.inner.dtype,
                T::dtype(),
            )
        }
    }
}

impl RawElemView<dyn DataSubset2D>
{
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

    pub fn csr_f32(&self) -> &RawElemView<CsrMatrix<f32>> { self.as_ref() }
    pub fn csr_f64(&self) -> &RawElemView<CsrMatrix<f64>> { self.as_ref() }
    pub fn csr_u32(&self) -> &RawElemView<CsrMatrix<u32>> { self.as_ref() }
    pub fn csr_u64(&self) -> &RawElemView<CsrMatrix<u64>> { self.as_ref() }
    pub fn arr_f32(&self) -> &RawElemView<ArrayD<f32>> { self.as_ref() }
    pub fn arr_f64(&self) -> &RawElemView<ArrayD<f64>> { self.as_ref() }
    pub fn arr_u32(&self) -> &RawElemView<ArrayD<u32>> { self.as_ref() }
    pub fn arr_u64(&self) -> &RawElemView<ArrayD<u64>> { self.as_ref() }
}
