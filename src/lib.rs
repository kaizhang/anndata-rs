pub mod utils;
pub mod anndata_trait;

use anndata_trait::*;
use crate::utils::{create_str_attr, read_str_attr, COMPRESSION};

use std::boxed::Box;
use std::sync::Arc;
use std::any::Any;
use hdf5::{File, H5Type, Result, Group, Dataset, types::VarLenUnicode};
use hdf5::types::TypeDescriptor::*;
use ndarray::{arr1, Array1, Array, Dimension};
use itertools::Itertools;
use std::marker::PhantomData;
use sprs::CsMat;

pub struct Elem {
    pub dtype: DataType,
    element: Option<Box<dyn AnnDataType>>,
    container: Box<dyn ContainerType>,
}

impl Elem {
    pub fn new(container: Box<dyn ContainerType>) -> Result<Self> {
        let dtype = container.get_encoding_type().expect(container.container_type());
        Ok(Self { dtype, element: None, container })
    }

    pub fn read_elem(&self) -> Result<Box<dyn AnnDataType>> {
        let err = Err(hdf5::Error::Internal("not".to_string()));
        let container = &self.container;
        match self.dtype {
            DataType::CsMatrix(Integer(_)) => {
                let mat: CsMat<i64> = AnnDataType::read(container)?;
                Ok(Box::new(mat))
            },
            DataType::CsMatrix(Unsigned(_)) => {
                let mat: CsMat<u64> = AnnDataType::read(container)?;
                Ok(Box::new(mat))
            },
            DataType::CsMatrix(Float(_)) => {
                let mat: CsMat<f64> = AnnDataType::read(container)?;
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

    pub fn csr_f32<'a>(&'a self) -> TypedElem<'a, CsMat<f32>> { self.typed() }
    pub fn csr_f64<'a>(&'a self) -> TypedElem<'a, CsMat<f64>> { self.typed() }
    pub fn csr_u32<'a>(&'a self) -> TypedElem<'a, CsMat<u32>> { self.typed() }
    pub fn csr_u64<'a>(&'a self) -> TypedElem<'a, CsMat<u64>> { self.typed() }

}

pub struct TypedElem<'a, T> {
    elem: &'a Elem,
    phantom: PhantomData<T>,
}

impl<'a, T> TypedElem<'a, T>
where
    T: AnnDataType,
{
    pub fn read_elem(&self) -> Result<T> {
        AnnDataType::read(&self.elem.container)
    }
}

//type Obsm = Box<dyn RowSlice<View=Box<dyn Any>> + AnnLoader<Data=Box<dyn Any>>>;
//type Varm = Box<dyn VarmItem<View = Box<dyn Any>, Data = Box<dyn Any>>>;

pub struct AnnData {
    file: File,
    pub x: Elem,
    //var_names: Elem,
    //obs_names: Elem,
    obsm: Vec<Elem>,
    varm: Vec<Elem>,
}

impl AnnData {
    pub fn read(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let x = Elem::new(Box::new(file.group("X")?))?;
        //let obs_names = read_dataframe_index(&file.group("obs")?)?;
        //let var_names = read_dataframe_index(&file.group("var")?)?;
        let mut obsm: Vec<Elem> = Vec::new();
        //file.group("obsm")?.groups()?.into_iter().for_each(|x| {
        //    obsm.push(Box::new(AnnDataType::read(&x).unwrap()));
        //});
        Ok(AnnData { file, x, obsm, varm: Vec::new() })
    }

    //pub fn x(&self) -> Box<csr::CsrMatrix<T>> {
    //    todo!()
    //}

    //pub fn var_names(&mut self) -> &[String] { self.var_names.get().0.as_slice() }

    //pub fn obs_names(&mut self) -> &[String] { self.obs_names.get().0.as_slice() }
}

pub fn read_dataframe_index(group: &Group) -> Result<Elem> {
    let index_name = read_str_attr(group, "_index")?;
    let elem = Box::new(group.dataset(&index_name)?);
    Elem::new(elem)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let adata: AnnData = AnnData::read("data.h5ad").unwrap();
        let mat: CsMat<f32> = adata.x.typed().read_elem().unwrap();
        println!("{:?}", mat.nnz());

        //assert_eq!(beds, expected);
    }
}