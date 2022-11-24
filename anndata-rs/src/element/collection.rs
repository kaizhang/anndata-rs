use crate::{data::*, element::base::*};

use anyhow::{anyhow, ensure, Result};
use hdf5::Group;
use itertools::Itertools;
use parking_lot::Mutex;
use std::{
    collections::{HashMap, HashSet},
    ops::{Deref, DerefMut},
    sync::Arc,
};

pub struct InnerElemCollection {
    container: Group,
    data: HashMap<String, Elem>,
}

impl Deref for InnerElemCollection {
    type Target = HashMap<String, Elem>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for InnerElemCollection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl std::fmt::Display for InnerElemCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let keys = self
            .keys()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Dict with keys: {}", keys)
    }
}

impl TryFrom<Group> for InnerElemCollection {
    type Error = anyhow::Error;

    fn try_from(container: Group) -> Result<Self> {
        let data: Result<HashMap<_, _>> = get_all_data(&container)
            .map(|(k, v)| Ok((k, Elem::try_from(v)?)))
            .collect();
        Ok(Self {
            container,
            data: data?,
        })
    }
}

pub type ElemCollection = Slot<InnerElemCollection>;

impl TryFrom<Group> for ElemCollection {
    type Error = anyhow::Error;

    fn try_from(container: Group) -> Result<Self> {
        Ok(Slot::new(InnerElemCollection::try_from(container)?))
    }
}

impl ElemCollection {
    pub fn add_data<D: Data>(&self, key: &str, data: D) -> Result<()> {
        ensure!(
            !self.is_empty(),
            "cannot add data to an empty ElemCollection"
        );
        let mut inner = self.inner();
        match inner.get_mut(key) {
            None => {
                let container = data.write(&inner.container, key)?;
                inner.insert(key.to_string(), Elem::try_from(container)?);
            }
            Some(elem) => elem.update(data)?,
        }
        Ok(())
    }

    pub fn write(&self, location: &Group) -> Result<()> {
        if !self.is_empty() {
            let inner = self.inner();
            for (key, val) in inner.iter() {
                val.write(location, key)?;
            }
        }
        Ok(())
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum Axis {
    Row,
    Column,
    Both,
}

pub struct InnerAxisArrays {
    pub axis: Axis,
    pub(crate) container: Group,
    pub(crate) size: Arc<Mutex<usize>>, // shared global reference
    data: HashMap<String, MatrixElem>,
}

impl Deref for InnerAxisArrays {
    type Target = HashMap<String, MatrixElem>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for InnerAxisArrays {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl std::fmt::Display for InnerAxisArrays {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ty = match self.axis {
            Axis::Row => "row",
            Axis::Column => "column",
            Axis::Both => "square",
        };
        let keys = self
            .keys()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "AxisArrays ({}) with keys: {}", ty, keys)
    }
}

macro_rules! check_dims {
    ($size:expr, $data:expr, $axis: expr) => {
        match $axis {
            Axis::Column => {
                if $data.ncols() == $size {
                    None
                } else if $size == 0 {
                    Some($data.ncols())
                } else {
                    panic!(
                        "inconsistent size, found: {}, expecting: {}",
                        $data.ncols(),
                        $size
                    );
                }
            }
            Axis::Row => {
                if $data.nrows() == $size {
                    None
                } else if $size == 0 {
                    Some($data.nrows())
                } else {
                    panic!(
                        "inconsistent size, found: {}, expecting: {}",
                        $data.nrows(),
                        $size
                    );
                }
            }
            Axis::Both => {
                if $data.nrows() != $data.ncols() {
                    panic!(
                        "not a square matrix: nrow = {}, ncol = {}",
                        $data.nrows(),
                        $data.ncols()
                    );
                } else if $data.ncols() == $size {
                    None
                } else if $size == 0 {
                    Some($data.ncols())
                } else {
                    panic!(
                        "inconsistent size, found: {}, expecting: {}",
                        $data.nrows(),
                        $size
                    );
                }
            }
        }
    };
}

impl InnerAxisArrays {
    pub(crate) fn new(group: Group, axis: Axis, size: Arc<Mutex<usize>>) -> Self {
        let data: HashMap<_, _> = get_all_data(&group)
            .map(|(k, v)| (k, MatrixElem::try_from(v).unwrap()))
            .collect();
        {
            let mut size_guard = size.lock();
            let mut n = *size_guard;
            for (_, v) in data.iter() {
                if let Some(s) = check_dims!(n, v, axis) {
                    n = s;
                }
            }
            *size_guard = n;
        }
        Self {
            container: group,
            size,
            axis,
            data,
        }
    }
}

pub type AxisArrays = Slot<InnerAxisArrays>;

impl AxisArrays {
    pub fn size(&self) -> Option<usize> {
        self.lock().as_ref().map(|x| *x.size.lock())
    }

    pub fn add_data<D: MatrixData>(&self, key: &str, data: D) -> Result<()> {
        ensure!(!self.is_empty(), "cannot add data to a closed AxisArrays");
        let mut inner = self.inner();
        {
            let mut size_guard = inner.size.lock();
            let mut n = *size_guard;
            if let Some(s) = check_dims!(n, data, inner.axis) {
                n = s;
            }
            *size_guard = n;
        }
        match inner.get_mut(key) {
            None => {
                let container = data.write(&inner.container, key)?;
                let elem = MatrixElem::try_from(container)?;
                inner.insert(key.to_string(), elem);
            }
            Some(elem) => elem.update(data)?,
        }
        Ok(())
    }

    pub fn subset(&self, idx: &[usize]) -> Result<()> {
        if !self.is_empty() {
            let inner = self.inner();
            match inner.axis {
                Axis::Row => inner.values().try_for_each(|x| x.subset(Some(idx), None)),
                Axis::Column => inner.values().try_for_each(|x| x.subset(None, Some(idx))),
                Axis::Both => inner
                    .values()
                    .try_for_each(|x| x.subset(Some(idx), Some(idx))),
            }?;
        }
        Ok(())
    }

    pub fn write(&self, idx_: Option<&[usize]>, location: &Group) -> Result<()> {
        if !self.is_empty() {
            let inner = self.inner();
            match idx_ {
                None => inner
                    .iter()
                    .try_for_each(|(k, x)| x.write(None, None, location, k)),
                Some(idx) => match inner.axis {
                    Axis::Row => inner
                        .iter()
                        .try_for_each(|(k, x)| x.write(Some(idx), None, location, k)),
                    Axis::Column => inner
                        .iter()
                        .try_for_each(|(k, x)| x.write(None, Some(idx), location, k)),
                    Axis::Both => inner
                        .iter()
                        .try_for_each(|(k, x)| x.write(Some(idx), Some(idx), location, k)),
                },
            }?;
        }
        Ok(())
    }
}

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
