use crate::backend::{Backend, DataContainer, GroupOp, LocationOp, DataType};
use crate::data::{
    array::slice::{SelectInfoElem, Shape},
    scalar::DynScalar,
};

use anyhow::Result;

/// Read data from a backend
pub trait ReadData {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self>
    where
        Self: Sized;
}

/// Write data to a backend
pub trait WriteData {
    fn data_type(&self) -> DataType;
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

impl<T> WriteData for &T
where
    T: WriteData,
{
    fn data_type(&self) -> DataType {
        (*self).data_type()
    }
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
            (*self).write(location, name)
    }
}

/// Anything that has a shape.
pub trait HasShape {
    fn shape(&self) -> Shape;
}

impl<T> HasShape for &T
where
    T: HasShape,
{
    fn shape(&self) -> Shape {
        (*self).shape()
    }
}

pub trait ArrayOp: HasShape {
    fn get(&self, index: &[usize]) -> Option<DynScalar>;

    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>;

    fn select_axis<S>(&self, axis: usize, slice: S) -> Self
    where
        S: AsRef<SelectInfoElem>,
        Self: Sized,
    {
        let full = SelectInfoElem::full();
        let selection = slice.as_ref().set_axis(axis, self.shape().ndim(), &full);
        self.select(selection.as_slice())
    }
}

pub trait ReadArrayData: ReadData {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape>;

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
        Self: Sized;

    fn read_axis<B, S>(container: &DataContainer<B>, axis: usize, slice: S) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
        Self: Sized,
    {
        let ndim = Self::get_shape(container)?.ndim();
        let full = SelectInfoElem::full();
        let selection = slice.as_ref().set_axis(axis, ndim, &full);
        Self::read_select(container, selection.as_slice())
    }
}

pub trait WriteArrayData: WriteData {}