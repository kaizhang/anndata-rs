use crate::{
    backend::{Backend, DataContainer, iter_containers},
    data::*,
    element::base::*,
};

use either::Either;
use anyhow::{anyhow, ensure, Result};
use itertools::Itertools;
use parking_lot::Mutex;
use std::{
    collections::{HashMap, HashSet},
    ops::{Deref, DerefMut},
    sync::Arc,
};

pub struct InnerElemCollection<B: Backend> {
    container: B::Group,
    data: HashMap<String, Elem<B>>,
}

impl<B: Backend> Deref for InnerElemCollection<B> {
    type Target = HashMap<String, Elem<B>>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<B: Backend> DerefMut for InnerElemCollection<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<B: Backend> std::fmt::Display for InnerElemCollection<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let keys = self
            .keys()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Dict with keys: {}", keys)
    }
}

impl<B> InnerElemCollection<B>
where
    B: Backend,
{
    pub fn export(&self, location: &B::Group) -> Result<()> {
        for (key, val) in self.iter() {
            val.inner().export(location, key)?;
        }
        Ok(())
    }

    pub fn add_data<D: WriteData + Into<Data>>(&mut self, key: &str, data: D) -> Result<()> {
        match self.get_mut(key) {
            None => {
                let container = data.write(&self.container, key)?;
                self.insert(key.to_string(), container.try_into()?);
            }
            Some(elem) => elem.inner().save(data)?,
        }
        Ok(())
    }
}

pub struct ElemCollection<B: Backend>(Slot<InnerElemCollection<B>>);

impl<B: Backend> Deref for ElemCollection<B> {
    type Target = Slot<InnerElemCollection<B>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B: Backend> DerefMut for ElemCollection<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<B: Backend> ElemCollection<B> {
    pub fn empty() -> Self {
        Self(Slot::empty())
    }

    pub fn new(container: B::Group) -> Result<Self> {
        let data: Result<HashMap<_, _>> = iter_containers(&container)
            .map(|(k, v)| Ok((k, Elem::try_from(v)?)))
            .collect();
        let collection = InnerElemCollection {
            container,
            data: data?,
        };
        Ok(Self(Slot::new(collection)))
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum Axis {
    Row,
    RowColumn,
}

pub struct InnerAxisArrays<B: Backend> {
    pub axis: Axis,
    pub(crate) container: B::Group,
    pub(crate) size: Arc<Mutex<usize>>, // shared global reference
    data: HashMap<String, ArrayElem<B>>,
}

impl<B: Backend> Deref for InnerAxisArrays<B> {
    type Target = HashMap<String, ArrayElem<B>>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<B: Backend> DerefMut for InnerAxisArrays<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<B: Backend> std::fmt::Display for InnerAxisArrays<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ty = match self.axis {
            Axis::Row => "row",
            Axis::RowColumn => "square",
        };
        let keys = self
            .keys()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "AxisArrays ({}) with keys: {}", ty, keys)
    }
}

impl<B: Backend> InnerAxisArrays<B> {
    pub fn size(&self) -> usize {
        *self.size.lock()
    }

    pub fn add_data<D: WriteArrayData + ArrayOp + Into<ArrayData>>(
        &mut self,
        key: &str,
        data: D,
    ) -> Result<()> {
        { // Check if the data is compatible with the current size
            let shape = data.shape();
            let mut size= self.size.lock();

            match self.axis {
                Axis::Row => {
                    ensure!(*size == 0 || *size == shape[0], "Row arrays must have same length");
                    *size = shape[0];
                }
                Axis::RowColumn => {
                    ensure!(shape[0] == shape[1], "Square arrays must be square");
                    ensure!(*size == 0 || *size == shape[0], "Square arrays must have same length");
                    *size = shape[0];
                }
            }
        }
        match self.get_mut(key) {
            None => {
                let container = data.write(&self.container, key)?;
                let elem = container.try_into()?;
                self.insert(key.to_string(), elem);
            }
            Some(elem) => elem.inner().save(data)?,
        }
        Ok(())
    }

    pub fn export(&self, location: &B::Group) -> Result<()> {
        for (key, val) in self.iter() {
            val.inner().export(location, key)?;
        }
        Ok(())
    }

    pub fn export_select<S>(&self, selection: S, location: &B::Group) -> Result<()>
    where
        S: AsRef<SelectInfoElem>,
    {
        if selection.as_ref().is_full_slice() {
            self.export(location)
        } else {
            match self.axis {
                Axis::Row => {
                    let s = vec![selection.as_ref()];
                    self
                        .iter()
                        .try_for_each(|(k, x)| x.inner().export_select(&s, location, k))
                },
                Axis::RowColumn => {
                    let s = vec![selection.as_ref(), selection.as_ref()];
                    self
                        .iter()
                        .try_for_each(|(k, x)| x.inner().export_select(&s, location, k))
                },
            }
        }
    }

    pub fn subset<S: AsRef<SelectInfoElem>>(&self, selection: S) -> Result<()> {
        match self.axis {
            Axis::Row => {
                let s = vec![selection.as_ref()];
                self.values().try_for_each(|x| x.inner().subset(&s))?;
            },
            Axis::RowColumn => {
                let s = vec![selection.as_ref(), selection.as_ref()];
                self.values().try_for_each(|x| x.inner().subset(&s))?;
            }
        }
        Ok(())
    }
}

pub struct AxisArrays<B: Backend>(Slot<InnerAxisArrays<B>>);

impl<B: Backend> Deref for AxisArrays<B> {
    type Target = Slot<InnerAxisArrays<B>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B: Backend> DerefMut for AxisArrays<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<B: Backend> AxisArrays<B> {
    pub fn empty() -> Self {
        Self(Slot::empty())
    }

    pub fn new(group: B::Group, axis: Axis, size_: Arc<Mutex<usize>>) -> Result<Self> {
        let data: HashMap<_, _> = iter_containers::<B>(&group)
            .map(|(k, v)| (k, ArrayElem::try_from(v).unwrap()))
            .collect();
        { // Check if the data is compatible with the current size
            let mut size= size_.lock();
            for (_, v) in data.iter() {
                let v_lock = v.inner();
                let shape = v_lock.shape();
                match &axis {
                    Axis::Row => {
                        ensure!(*size == 0 || *size == shape[0], "Row arrays must have same length");
                        *size = shape[0];
                    }
                    Axis::RowColumn => {
                        ensure!(shape[0] == shape[1], "Square arrays must be square");
                        ensure!(*size == 0 || *size == shape[0], "Square arrays must have same length");
                        *size = shape[0];
                    }
                }
            }
        }
        let arrays = InnerAxisArrays{
            container: group,
            size: size_,
            axis,
            data,
        };
        Ok(Self(Slot::new(arrays)))
    }
}




/*
#[derive(Clone)]
pub struct StackedAxisArrays {
    pub axis: Axis,
    pub data: HashMap<String, StackedMatrixElem>,
}

impl std::fmt::Display for StackedAxisArrays {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ty = match self.axis {
            Axis::Row => "row",
            Axis::Column => "column",
            Axis::Both => "square",
        };
        let keys = self
            .data
            .keys()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Stacked AxisArrays ({}) with keys: {}", ty, keys)
    }
}

impl StackedAxisArrays {
    pub(crate) fn new(
        arrays: Vec<&InnerAxisArrays>,
        nrows: &Arc<Mutex<usize>>,
        ncols: &Arc<Mutex<usize>>,
        accum: &Arc<Mutex<VecVecIndex>>,
    ) -> Result<Self> {
        if arrays.is_empty() {
            return Err(anyhow!("input is empty"));
        }
        if !arrays.iter().map(|x| x.axis).all_equal() {
            return Err(anyhow!("arrays must have same axis"));
        }
        let keys = intersections(
            arrays
                .iter()
                .map(|x| x.keys().map(Clone::clone).collect())
                .collect(),
        );
        let data = keys
            .into_iter()
            .map(|k| {
                let elems = arrays.iter().map(|x| x.get(&k).unwrap().clone()).collect();
                Ok((
                    k,
                    StackedMatrixElem::new(elems, nrows.clone(), ncols.clone(), accum.clone())?,
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Self {
            axis: arrays[0].axis,
            data,
        })
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
}

fn intersections(mut sets: Vec<HashSet<String>>) -> HashSet<String> {
    {
        let (intersection, others) = sets.split_at_mut(1);
        let intersection = &mut intersection[0];
        for other in others {
            intersection.retain(|e| other.contains(e));
        }
    }
    sets[0].clone()
}
*/
