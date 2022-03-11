use crate::anndata_trait::*;
use crate::base::*;

use std::boxed::Box;
use hdf5::{Result, Group}; 
use ndarray::{ArrayD};
use nalgebra_sparse::csr::CsrMatrix;

pub type AnnData = AnnDataBase<
    ElemView<dyn DataSubset2D>,
    ElemView<dyn DataSubset2D>,
    ElemView<dyn DataSubset2D>
>;

pub struct Elem<T: ?Sized> {
    pub dtype: DataType,
    container: Box<dyn DataContainer>,
    element: Option<Box<T>>,
}

impl<T> Elem<T>
where
    T: DataIO,
{
    pub fn read_data(&self) -> T { DataIO::read(&self.container).unwrap() }
}

impl<T> AsRef<Elem<T>> for Elem<dyn DataIO>
where
    T: DataIO,
{
    fn as_ref(&self) -> &Elem<T> {
        if self.dtype == T::dtype() {
            unsafe { &*(self as *const Elem<dyn DataIO> as *const Elem<T>) }
        } else {
            panic!(
                "implementation error, cannot convert {:?} to {:?}",
                self.dtype,
                T::dtype(),
            )
        }
    }
}

impl Elem<dyn DataIO>
{
    pub fn csr_f32(&self) -> &Elem<CsrMatrix<f32>> { self.as_ref() }
    pub fn csr_f64(&self) -> &Elem<CsrMatrix<f64>> { self.as_ref() }
    pub fn csr_u32(&self) -> &Elem<CsrMatrix<u32>> { self.as_ref() }
    pub fn csr_u64(&self) -> &Elem<CsrMatrix<u64>> { self.as_ref() }
    pub fn arr_f32(&self) -> &Elem<ArrayD<f32>> { self.as_ref() }
    pub fn arr_f64(&self) -> &Elem<ArrayD<f64>> { self.as_ref() }
    pub fn arr_u32(&self) -> &Elem<ArrayD<u32>> { self.as_ref() }
    pub fn arr_u64(&self) -> &Elem<ArrayD<u64>> { self.as_ref() }
}

impl BoxedData for Elem<dyn DataIO> {
    fn new(container: Box<dyn DataContainer>) -> Result<Self> {
        let dtype = container.get_encoding_type().expect(container.container_type());
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

pub struct ElemView<T: ?Sized> {
    obs_indices: Option<Vec<usize>>,
    var_indices: Option<Vec<usize>>,
    inner: Elem<T>,
}

impl<T> ElemView<T>
where
    T: DataSubset2D,
{
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
impl<T> AsRef<ElemView<T>> for ElemView<dyn DataSubset2D>
where
    T: DataSubset2D,
{
    fn as_ref(&self) -> &ElemView<T> {
        if self.inner.dtype == T::dtype() {
            unsafe { &*(self as *const ElemView<dyn DataSubset2D> as *const ElemView<T>) }
        } else {
            panic!(
                "implementation error, cannot convert {:?} to {:?}",
                self.inner.dtype,
                T::dtype(),
            )
        }
    }
}

impl BoxedData for ElemView<dyn DataSubset2D> {
    fn new(container: Box<dyn DataContainer>) -> Result<Self> {
        let dtype = container.get_encoding_type().expect(container.container_type());
        let inner = Elem { dtype, element: None, container };
        Ok(Self { obs_indices: None, var_indices: None, inner })
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

impl ElemView<dyn DataSubset2D>
{
    pub fn csr_f32(&self) -> &ElemView<CsrMatrix<f32>> { self.as_ref() }
    pub fn csr_f64(&self) -> &ElemView<CsrMatrix<f64>> { self.as_ref() }
    pub fn csr_u32(&self) -> &ElemView<CsrMatrix<u32>> { self.as_ref() }
    pub fn csr_u64(&self) -> &ElemView<CsrMatrix<u64>> { self.as_ref() }
    pub fn arr_f32(&self) -> &ElemView<ArrayD<f32>> { self.as_ref() }
    pub fn arr_f64(&self) -> &ElemView<ArrayD<f64>> { self.as_ref() }
    pub fn arr_u32(&self) -> &ElemView<ArrayD<u32>> { self.as_ref() }
    pub fn arr_u64(&self) -> &ElemView<ArrayD<u64>> { self.as_ref() }
}

// TODO: fix subsetting
impl BoxedDataSubRow for ElemView<dyn DataSubset2D> {
    fn subset_rows(mut self, idx: &[usize]) -> Self {
        self.obs_indices = Some(idx.iter().map(|x| *x).collect());
        if let Some(x) = self.inner.element {
            self.inner.element = None;
        }
        self
    }
}

impl BoxedDataSubCol for ElemView<dyn DataSubset2D> {
    fn subset_cols(mut self, idx: &[usize]) -> Self {
        self.var_indices = Some(idx.iter().map(|x| *x).collect());
        if let Some(x) = self.inner.element {
            self.inner.element = None;
        }
        self
    }
}

impl BoxedDataSub2D for ElemView<dyn DataSubset2D> {
    fn subset(mut self, ridx: &[usize], cidx: &[usize]) -> Self {
        self.obs_indices = Some(ridx.iter().map(|x| *x).collect());
        self.var_indices = Some(cidx.iter().map(|x| *x).collect());
        if let Some(x) = self.inner.element {
            self.inner.element = None;
        }
        self
    }
}