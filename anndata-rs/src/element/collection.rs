use crate::{
    anndata_trait::*,
    element::{Stacked, Elem, MatrixElem, ElemTrait},
};

use std::sync::Arc;
use parking_lot::Mutex;
use std::collections::HashMap;
use hdf5::{Result, Group}; 
use itertools::Itertools;
use std::collections::HashSet;

#[derive(Clone)]
pub struct ElemCollection {
    pub(crate) container: Group,
    pub data: Arc<Mutex<HashMap<String, Elem>>>,
}

impl ElemCollection {
    pub fn new(container: Group) -> Self {
        let data = Arc::new(Mutex::new(get_all_data(&container).map(|(k, v)|
            (k, Elem::new(v).unwrap())
        ).collect()));
        Self { container, data }
    }

    pub fn is_empty(&self) -> bool {
        self.data.lock().is_empty()
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.data.lock().contains_key(key)
    }

    pub fn insert(&self, key: &str, data: &Box<dyn DataIO>) -> Result<()> {
        let mut data_guard = self.data.lock();
        match data_guard.get(key) {
            None => {
                let container = data.write(&self.container, key)?;
                let elem = Elem::new(container)?;
                data_guard.insert(key.to_string(), elem);
            }
            Some(elem) => elem.update(data),
        }
        Ok(())
    }

    pub fn write(&self, location: &Group) -> Result<()> {
        for (key, val) in self.data.lock().iter() {
            val.write(location, key)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for ElemCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let keys = self.data.lock().keys().map(|x| x.to_string())
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

#[derive(Clone)]
pub struct AxisArrays {
    pub container: Group,
    pub size: Arc<Mutex<usize>>,   // shared global reference
    pub axis: Axis,
    pub data: Arc<Mutex<HashMap<String, MatrixElem>>>,
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
                panic!("not a square matrix");
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
        let keys = self.data.lock().keys().map(|x| x.to_string())
            .collect::<Vec<_>>().join(", ");
        write!(f, "AxisArrays ({}) with keys: {}", ty, keys)
    }
}

impl AxisArrays {
    pub fn new(
        group: Group,
        axis: Axis,
        size: Arc<Mutex<usize>>,
    ) -> Self {
        let data: HashMap<String, MatrixElem> = get_all_data(&group).map(|(k, v)|
            (k, MatrixElem::new(v).unwrap())
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
        Self { container: group, size, axis, data: Arc::new(Mutex::new(data))}
    }
    
    pub fn size(&self) -> usize { *self.size.lock() }

    pub fn is_empty(&self) -> bool {
        self.data.lock().is_empty()
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.data.lock().contains_key(key)
    }

    pub fn insert(&self, key: &str, data: &Box<dyn DataPartialIO>) -> Result<()> {
        let mut size_guard = self.size.lock();
        let mut n = *size_guard;
        if let Some(s) = check_dims!(n, data, self.axis) {
            n = s;
        }
        let mut data_guard = self.data.lock();
        match data_guard.get(key) {
            None => {
                let container = data.write(&self.container, key)?;
                let elem = MatrixElem::new(container)?;
                data_guard.insert(key.to_string(), elem);
            }
            Some(elem) => elem.update(data),
        }
        *size_guard = n;
        Ok(())
    }
    
    pub(crate) fn subset(&self, idx: &[usize]) {
        match self.axis {
            Axis::Row => self.data.lock().values_mut().for_each(|x| x.subset_rows(idx)),
            Axis::Column => self.data.lock().values_mut().for_each(|x| x.subset_cols(idx)),
            Axis::Both => self.data.lock().values_mut().for_each(|x| x.subset(idx, idx)),
        }
    }

    pub fn write(&self, location: &Group) -> Result<()> {
        for (key, val) in self.data.lock().iter() {
            val.write(location, key)?;
        }
        Ok(())
    }
}

fn get_all_data(group: &Group) -> impl Iterator<Item=(String, DataContainer)> {
    let get_name = |x: String| std::path::Path::new(&x).file_name()
        .unwrap().to_str().unwrap().to_string();
    group.groups().unwrap().into_iter().map(move |x|
        (get_name(x.name()), DataContainer::H5Group(x))
    ).chain(group.datasets().unwrap().into_iter().map(move |x|
        (get_name(x.name()), DataContainer::H5Dataset(x))
    ))
}

#[derive(Clone)]
pub struct StackedAxisArrays {
    pub axis: Axis,
    pub data: Arc<Mutex<HashMap<String, Stacked<MatrixElem>>>>,
}

impl std::fmt::Display for StackedAxisArrays {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ty = match self.axis {
            Axis::Row => "row",
            Axis::Column => "column",
            Axis::Both => "square",
        };
        let keys = self.data.lock().keys().map(|x| x.to_string())
            .collect::<Vec<_>>().join(", ");
        write!(f, "Stacked AxisArrays ({}) with keys: {}", ty, keys)
    }
}

impl StackedAxisArrays {
    pub fn new(arrays: &Vec<AxisArrays>) -> Result<Self> {
        if arrays.is_empty() {
            return Err(hdf5::Error::from("input is empty"));
        }
        if !arrays.iter().map(|x| x.axis).all_equal() {
            return Err(hdf5::Error::from("arrays must have same axis"));
        }
        let keys = intersections(arrays.iter()
            .map(|x| x.data.lock().keys().map(Clone::clone).collect()).collect()
        );
        let data = keys.into_iter().map(|k| {
            let elems = arrays.iter()
                .map(|x| x.data.lock().get(&k).unwrap().clone()).collect();
            Ok((k, Stacked::new(elems)?))
        }).collect::<Result<HashMap<_, _>>>()?;
        Ok(Self { axis: arrays[0].axis, data: Arc::new(Mutex::new(data)) })
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.data.lock().contains_key(key)
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