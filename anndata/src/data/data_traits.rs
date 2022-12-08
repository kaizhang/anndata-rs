use crate::backend::{Backend, DataContainer, GroupOp, LocationOp};
use crate::data::{
    array::slice::{SelectInfoElem, Shape},
    scalar::DynScalar,
};

use anyhow::Result;
use smallvec::SmallVec;

/// Read data from a backend
pub trait ReadData {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self>
    where
        Self: Sized;
}

/// Write data to a backend
pub trait WriteData {
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>>;
    fn overwrite<B: Backend>(&self, container: DataContainer<B>) -> Result<DataContainer<B>> {
        let file = container.file()?;
        let path = container.path();
        let group = file.open_group(path.parent().unwrap().to_str().unwrap())?;
        let name = path.file_name().unwrap().to_str().unwrap();
        group.delete(name)?;
        self.write(&group, name)
    }
}

/// Anything that has a shape.
pub trait HasShape {
    fn shape(&self) -> Shape;
}

/// Things that may have a shape.
pub trait MayHaveShape {
    fn shape_maybe(&self) -> SmallVec<[Option<usize>; 3]>;
}

impl<T: HasShape> MayHaveShape for T {
    fn shape_maybe(&self) -> SmallVec<[Option<usize>; 3]> {
        self.shape().as_ref().iter().map(|x| Some(*x)).collect()
    }
}

pub trait ArrayOp: HasShape {
    fn get(&self, index: &[usize]) -> Option<DynScalar>;
    fn select<S, E>(&self, info: S) -> Self
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>;
}

pub trait ReadArrayData: ReadData {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape>;

    fn read_select<B, S, E>(container: &DataContainer<B>, info: S) -> Result<Self>
    where
        B: Backend,
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
        Self: Sized;
}

pub trait WriteArrayData: WriteData {
    fn write_from_iter<B, G, I>(iter: I, group: &G, name: &str) -> Result<DataContainer<B>>
    where
        B: Backend,
        G: GroupOp<Backend = B>,
        I: Iterator<Item = Self>,
    {
        todo!()
    }
}