use crate::{
    anndata_trait::*,
    element::*,
};

use std::sync::Arc;
use parking_lot::Mutex;
use std::collections::HashMap;
use hdf5::Group; 
use anyhow::{anyhow, Result};
use std::collections::HashSet;
use std::ops::{Deref, DerefMut};
use itertools::Itertools;

pub struct ElemCollection {
    container: Group,
    data: HashMap<String, RawElem<dyn DataIO>>,
}

impl Deref for ElemCollection {
    type Target = HashMap<String, RawElem<dyn DataIO>>;

    fn deref(&self) -> &Self::Target { &self.data }
}

impl DerefMut for ElemCollection {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.data }
}

impl ElemCollection {
    pub(crate) fn new(container: Group) -> Self {
        let data = get_all_data(&container).map(|(k, v)|
            (k, RawElem::new(v).unwrap())
        ).collect();
        Self { container, data }
    }

    pub fn add_data(&mut self, key: &str, data: &Box<dyn DataIO>) -> Result<()> {
        match self.data.get_mut(key) {
            None => {
                let container = data.write(&self.container, key)?;
                let elem = RawElem::new(container)?;
                self.data.insert(key.to_string(), elem);
            }
            Some(elem) => elem.update(data)?,
        }
        Ok(())
    }

    pub fn write(&self, location: &Group) -> Result<()> {
        for (key, val) in self.data.iter() {
            val.write(location, key)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for ElemCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let keys = self.data.keys().map(|x| x.to_string())
            .collect::<Vec<_>>().join(", ");
        write!(f, "Dict with keys: {}", keys)
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum Axis {
    Row,
    Column,
    Both,
}

pub struct AxisArrays {
    pub axis: Axis,
    pub(crate) container: Group,
    pub(crate) size: Arc<Mutex<usize>>,   // shared global reference
    data: HashMap<String, MatrixElem>,
}

impl Deref for AxisArrays {
    type Target = HashMap<String, MatrixElem>;

    fn deref(&self) -> &Self::Target { &self.data }
}

impl DerefMut for AxisArrays {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.data }
}

macro_rules! check_dims {
    ($size:expr, $data:expr, $axis: expr) => {
        match $axis {
            Axis::Column => if $data.ncols() == $size {
                None
            } else if $size == 0 {
                Some($data.ncols())
            } else {
                panic!("inconsistent size: {}, {}", $data.ncols(), $size);
            },
            Axis::Row => if $data.nrows() == $size {
                None
            } else if $size == 0 {
                Some($data.nrows())
            } else {
                panic!("inconsistent size: {}, {}", $data.nrows(), $size);
            },
            Axis::Both => if $data.nrows() != $data.ncols() {
                panic!("not a square matrix: nrow = {}, ncol = {}", $data.nrows(), $data.ncols());
            } else if $data.ncols() == $size {
                None
            } else if $size == 0 {
                Some($data.ncols())
            } else {
                panic!("inconsistent size: {}, {}", $data.nrows(), $size);
            }
        }
    }
}

impl std::fmt::Display for AxisArrays {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ty = match self.axis {
            Axis::Row => "row",
            Axis::Column => "column",
            Axis::Both => "square",
        };
        let keys = self.keys().map(|x| x.to_string())
            .collect::<Vec<_>>().join(", ");
        write!(f, "AxisArrays ({}) with keys: {}", ty, keys)
    }
}

impl AxisArrays {
    pub(crate) fn new(group: Group, axis: Axis, size: Arc<Mutex<usize>>) -> Self {
        let data: HashMap<_, _> = get_all_data(&group).map(|(k, v)|
            (k, MatrixElem::new_elem(v).unwrap())
        ).collect();
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
        Self { container: group, size, axis, data }
    }
    
    pub fn size(&self) -> usize { *self.size.lock() }

    pub fn add_data(&mut self, key: &str, data: &Box<dyn DataPartialIO>) -> Result<()> {
        {
            let mut size_guard = self.size.lock();
            let mut n = *size_guard;
            if let Some(s) = check_dims!(n, data, self.axis) {
                n = s;
            }
            *size_guard = n;
        }
        match self.get_mut(key) {
            None => {
                let container = data.write(&self.container, key)?;
                let elem = MatrixElem::new_elem(container)?;
                self.data.insert(key.to_string(), elem);
            }
            Some(elem) => elem.inner().update(data)?,
        }
        Ok(())
    }
    
    pub(crate) fn subset(&mut self, idx: &[usize]) {
        match self.axis {
            Axis::Row => self.data.values_mut().for_each(|x| x.inner().subset_rows(idx).unwrap()),
            Axis::Column => self.data.values_mut().for_each(|x| x.inner().subset_cols(idx).unwrap()),
            Axis::Both => self.data.values_mut().for_each(|x| x.inner().subset(idx, idx).unwrap()),
        }
    }

    pub fn write(&self, location: &Group) -> Result<()> {
        for (key, val) in self.data.iter() {
            val.write(location, key)?;
        }
        Ok(())
    }

    pub fn write_subset(&self, idx: &[usize], location: &Group) -> Result<()> {
        match self.axis {
            Axis::Row => {
                self.data.iter().for_each(|(k, x)| x.inner().write_rows(idx, location, k).unwrap());
            },
            Axis::Column => {
                self.data.iter().for_each(|(k, x)| x.inner().write_columns(idx, location, k).unwrap());
            },
            Axis::Both => {
                self.data.iter().for_each(|(k, x)| x.inner().write_partial(idx, idx, location, k).unwrap());
            },
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct StackedAxisArrays {
    pub axis: Axis,
    pub data: HashMap<String, Stacked<MatrixElem>>,
}

impl std::fmt::Display for StackedAxisArrays {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ty = match self.axis {
            Axis::Row => "row",
            Axis::Column => "column",
            Axis::Both => "square",
        };
        let keys = self.data.keys().map(|x| x.to_string())
            .collect::<Vec<_>>().join(", ");
        write!(f, "Stacked AxisArrays ({}) with keys: {}", ty, keys)
    }
}

impl StackedAxisArrays {
    pub(crate) fn new(
        arrays: Vec<&AxisArrays>,
        nrows: &Arc<Mutex<usize>>,
        ncols: &Arc<Mutex<usize>>,
        accum: &Arc<Mutex<AccumLength>>,
    ) -> Result<Self> {
        if arrays.is_empty() {
            return Err(anyhow!("input is empty"));
        }
        if !arrays.iter().map(|x| x.axis).all_equal() {
            return Err(anyhow!("arrays must have same axis"));
        }
        let keys = intersections(arrays.iter()
            .map(|x| x.keys().map(Clone::clone).collect()).collect()
        );
        let data = keys.into_iter().map(|k| {
            let elems = arrays.iter()
                .map(|x| x.get(&k).unwrap().clone()).collect();
            Ok((
                k, Stacked::new(elems, nrows.clone(), ncols.clone(), accum.clone())?
            ))
        }).collect::<Result<HashMap<_, _>>>()?;
        Ok(Self { axis: arrays[0].axis, data })
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