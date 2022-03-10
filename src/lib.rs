pub mod utils;
pub mod anndata_trait;

use anndata_trait::*;
use crate::utils::{read_str_attr};

use std::ops::Range;
use std::boxed::Box;
use hdf5::{File, Result, Group}; 
use ndarray::{Array2};
use nalgebra_sparse::csr::CsrMatrix;
use std::collections::HashMap;

pub struct Elem<T: ?Sized> {
    pub dtype: DataType,
    container: Box<dyn ContainerType>,
    element: Option<Box<T>>,
}

pub struct ElemView<T: ?Sized> {
    obs_indices: Option<Vec<usize>>,
    var_indices: Option<Vec<usize>>,
    inner: Elem<T>,
}

// NOTE: this requires `element` is the last field, as trait object contains a vtable
// at the end: https://docs.rs/vptr/latest/vptr/index.html.
impl<T> AsRef<ElemView<T>> for ElemView<dyn AnnDataSubset>
where
    T: AnnDataType,
{
    fn as_ref(&self) -> &ElemView<T> {
        if self.inner.dtype == T::dtype() {
            unsafe { &*(self as *const ElemView<dyn AnnDataSubset> as *const ElemView<T>) }
        } else {
            panic!(
                "implementation error, cannot convert {:?} to {:?}",
                self.inner.dtype,
                T::dtype(),
            )
        }
    }
}

impl<T> AsRef<Elem<T>> for Elem<dyn AnnDataType>
where
    T: AnnDataType,
{
    fn as_ref(&self) -> &Elem<T> {
        if self.dtype == T::dtype() {
            unsafe { &*(self as *const Elem<dyn AnnDataType> as *const Elem<T>) }
        } else {
            panic!(
                "implementation error, cannot convert {:?} to {:?}",
                self.dtype,
                T::dtype(),
            )
        }
    }
}

impl Elem<dyn AnnDataType>
{
    pub fn csr_f32(&self) -> &Elem<CsrMatrix<f32>> { self.as_ref() }
    pub fn csr_f64(&self) -> &Elem<CsrMatrix<f64>> { self.as_ref() }
    pub fn csr_u32(&self) -> &Elem<CsrMatrix<u32>> { self.as_ref() }
    pub fn csr_u64(&self) -> &Elem<CsrMatrix<u64>> { self.as_ref() }
    pub fn arr2_f32(&self) -> &Elem<Array2<f32>> { self.as_ref() }
    pub fn arr2_f64(&self) -> &Elem<Array2<f64>> { self.as_ref() }
    pub fn arr2_u32(&self) -> &Elem<Array2<u32>> { self.as_ref() }
    pub fn arr2_u64(&self) -> &Elem<Array2<u64>> { self.as_ref() }
}

impl ElemView<dyn AnnDataSubset>
{
    pub fn csr_f32(&self) -> &ElemView<CsrMatrix<f32>> { self.as_ref() }
    pub fn csr_f64(&self) -> &ElemView<CsrMatrix<f64>> { self.as_ref() }
    pub fn csr_u32(&self) -> &ElemView<CsrMatrix<u32>> { self.as_ref() }
    pub fn csr_u64(&self) -> &ElemView<CsrMatrix<u64>> { self.as_ref() }
    pub fn arr2_f32(&self) -> &ElemView<Array2<f32>> { self.as_ref() }
    pub fn arr2_f64(&self) -> &ElemView<Array2<f64>> { self.as_ref() }
    pub fn arr2_u32(&self) -> &ElemView<Array2<u32>> { self.as_ref() }
    pub fn arr2_u64(&self) -> &ElemView<Array2<u64>> { self.as_ref() }
}

pub trait Subsettable {
    fn subset(self, ridx: Option<&[usize]>, cidx: Option<&[usize]>) -> Result<Self> where Self: Sized;
}

// TODO: fix subsetting
impl Subsettable for ElemView<dyn AnnDataSubset> {
    fn subset(mut self, ridx: Option<&[usize]>, cidx: Option<&[usize]>) -> Result<Self>
    where Self: Sized,
    {
        if let Some(obs_i) = ridx {
            self.obs_indices = Some(obs_i.iter().map(|x| *x).collect());
        }
        if let Some(var_i) = cidx {
            self.var_indices = Some(var_i.iter().map(|x| *x).collect());
        }
        if let Some(x) = self.inner.element {
            self.inner.element = None;
        }
        Ok(self)
    }
}

/*
impl Elem<dyn AnnDataType> {
    pub fn read_elem(&self) -> Result<Box<dyn AnnDataType>> {
        let err = Err(hdf5::Error::Internal("not".to_string()));
        let container = &self.container;
        match self.dtype {
            DataType::CsrMatrix(Integer(_)) => {
                let mat: CsrMatrix<i64> = AnnDataType::read(container)?;
                Ok(Box::new(mat))
            },
            DataType::CsrMatrix(Unsigned(_)) => {
                let mat: CsrMatrix<u64> = AnnDataType::read(container)?;
                Ok(Box::new(mat))
            },
            DataType::CsrMatrix(Float(_)) => {
                let mat: CsrMatrix<f64> = AnnDataType::read(container)?;
                Ok(Box::new(mat))
            },
            _ => err?,
        }
    }

    pub fn load(&mut self) -> Result<()> {
        if let None = self.element {
            self.element = Some(self.read_elem()?);
        }
        Ok(())
    }

    pub fn typed<'a, T>(&'a self) ->  TypedElem<'a, T>
    where
        T: AnnDataType,
    {
        if self.dtype == T::dtype() {
            TypedElem { elem: self, phantom: PhantomData }
        } else {
            panic!(
                "implementation error, cannot convert {:?} to {:?}",
                self.dtype,
                T::dtype(),
            )
        }
    }


}
*/

impl<T> Elem<T>
where
    T: AnnDataType,
{
    pub fn read_elem(&self) -> T { AnnDataType::read(&self.container).unwrap() }
}

impl<T> ElemView<T>
where
    T: AnnDataSubset,
{
    pub fn read_elem(&self) -> T {
        AnnDataSubset::read_partial(
            &self.inner.container,
            self.obs_indices.as_ref().map(Vec::as_slice),
            self.var_indices.as_ref().map(Vec::as_slice),
        )
    }
}

pub fn read_dataframe_index(group: &Group) -> Result<Elem<dyn AnnDataType>> {
    let index_name = read_str_attr(group, "_index")?;
    let elem = Box::new(group.dataset(&index_name)?);
    Elem::new(elem)
}

pub type AnnData = AnnDataBase<
    ElemView<dyn AnnDataSubset>, ElemView<dyn AnnDataSubset>, ElemView<dyn AnnDataSubset>
>;

pub struct AnnDataBase<X, O, V> {
    file: File,
    pub x: X,
    //var_names: Elem,
    //obs_names: Elem,
    pub obsm: HashMap<String, O>,
    varm: Vec<V>,
}

pub trait Data {
    fn new(container: Box<dyn ContainerType>) -> Result<Self> where Self: Sized;
}

impl Data for Elem<dyn AnnDataType> {
    fn new(container: Box<dyn ContainerType>) -> Result<Self> {
        let dtype = container.get_encoding_type().expect(container.container_type());
        Ok(Self { dtype, element: None, container })
    }
}

impl Data for ElemView<dyn AnnDataSubset> {
    fn new(container: Box<dyn ContainerType>) -> Result<Self> {
        let dtype = container.get_encoding_type().expect(container.container_type());
        let inner = Elem { dtype, element: None, container };
        Ok(Self { obs_indices: None, var_indices: None, inner })
    }
}

impl<X, O, V> AnnDataBase<X, O, V> {
    pub fn read(path: &str) -> Result<Self>
    where
        X: Data,
        O: Data,
        V: Data,
    {
        let file = File::open(path)?;
        let x = Data::new(Box::new(file.group("X")?))?;
        let get_name = |x: String| std::path::Path::new(&x).file_name()
            .unwrap().to_str().unwrap().to_string();
        //let obs_names = read_dataframe_index(&file.group("obs")?)?;
        //let var_names = read_dataframe_index(&file.group("var")?)?;
        let obsm =
            file.group("obsm")?.groups()?.into_iter().map(|x|
                (get_name(x.name()), Data::new(Box::new(x)).unwrap())
            ).chain(file.group("obsm")?.datasets()?.into_iter().map(|x|
                (get_name(x.name()), Data::new(Box::new(x)).unwrap())
            )).collect();
        Ok(Self { file, x, obsm, varm: Vec::new() })
    }

    pub fn obs_subset(mut self, idx: &[usize]) -> Self
    where
        X: Subsettable,
        O: Subsettable,
    {
        AnnDataBase {
            file: self.file,
            x: self.x.subset(Some(idx), None).unwrap(),
            obsm: self.obsm.drain()
                .map(|(k, v)| (k, v.subset(Some(idx), None).unwrap()))
                .collect(),
            varm: self.varm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let adata: AnnData = AnnData::read("data.h5ad").unwrap();
        println!(
            "{:?}",
            adata.obsm.get("X_spectral").unwrap().arr2_f64().read_elem(),
        );

        let adata_subset = adata.obs_subset(&[1,2,3]);
        println!(
            "{:?}",
            adata_subset.obsm.get("X_spectral").unwrap().arr2_f64().read_elem(),
        );

        //assert_eq!(beds, expected);
    }
}