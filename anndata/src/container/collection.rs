use crate::{
    backend::{iter_containers, Backend, GroupOp, LocationOp},
    container::base::*,
    data::*, ElemCollectionOp, AxisArraysOp,
};

use anyhow::{bail, ensure, Result};
use itertools::Itertools;
use parking_lot::{MutexGuard, Mutex};
use smallvec::{smallvec, SmallVec};
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    ops::{Deref, DerefMut},
    sync::Arc,
};

pub struct InnerElemCollection<B: Backend> {
    container: B::Group,
    data: HashMap<String, Elem<B>>,
}

impl<B: Backend> std::fmt::Debug for InnerElemCollection<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
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

impl<B: Backend> InnerElemCollection<B> {
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

    pub fn remove_data(&mut self, key: &str) -> Result<()> {
        self.remove(key).map(|x| x.clear()).transpose()?;
        Ok(())
    }

    pub fn export<O: Backend, G: GroupOp<Backend = O>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<()> {
        let group = location.create_group(name)?;
        for (key, val) in self.iter() {
            val.inner().export::<O, _>(&group, key)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ElemCollection<B: Backend>(Slot<InnerElemCollection<B>>);

impl<B: Backend> Clone for ElemCollection<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<B: Backend> Deref for ElemCollection<B> {
    type Target = Slot<InnerElemCollection<B>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B: Backend> Display for ElemCollection<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<B: Backend> DerefMut for ElemCollection<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<B: Backend> ElemCollectionOp for &ElemCollection<B> {
    fn keys(&self) -> Vec<String> {
        self.inner().keys().cloned().collect()
    }

    fn get_item<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<Data> + TryFrom<Data> + Clone,
        <D as TryFrom<Data>>::Error: Into<anyhow::Error>,
    {
        self.lock()
            .as_mut()
            .and_then(|x| x.get_mut(key))
            .map(|x| x.inner().data())
            .transpose()
    }

    fn add<D: WriteData + Into<Data>>(&self, key: &str, data: D) -> Result<()> {
        self.inner().add_data(key, data)
    }

    fn remove(&self, key: &str) -> Result<()> {
        self.inner().remove_data(key)
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

    pub fn clear(&self) -> Result<()> {
        self.0
            .lock()
            .as_ref()
            .map(|x| {
                let g = &x.container;
                g.file()?.delete(&g.path().to_string_lossy())
            })
            .transpose()?;
        self.0.drop();
        Ok(())
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Axis {
    Row,
    RowColumn,
}

/// Nullable dimension. None means that the dimension is not set.
/// Dimension can be set only once.
#[derive(Debug, Clone)]
pub struct Dim(Arc<Mutex<Option<usize>>>);

impl Dim {
    pub fn empty() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }

    pub fn new(n: usize) -> Self {
        Self(Arc::new(Mutex::new(Some(n))))
    }

    pub fn lock(&self) -> DimLock<'_> {
        DimLock(self.0.lock())
    }

    pub fn try_lock(&self) -> Option<DimLock<'_>> {
        self.0.try_lock().map(DimLock)
    }

    pub fn get(&self) -> usize {
        self.lock().get()
    }

    pub fn try_set(&self, n: usize) -> Result<()> {
        self.lock().try_set(n)
    }
}

impl Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lock = self.lock().0;
        if lock.is_none() {
            write!(f, "None")?;
        } else {
            write!(f, "{}", lock.unwrap())?;
        }
        Ok(())
    }
}

pub struct DimLock<'a>(MutexGuard<'a, Option<usize>>);

impl DimLock<'_> {
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    pub fn get(&self) -> usize {
        self.0.unwrap_or(0)
    }

    pub fn try_set(&mut self, n: usize) -> Result<()> {
        if self.0.is_some() && self.0.unwrap() != n {
            bail!("dimension cannot be changed from {} to {}", self.0.unwrap(), n);
        } else {
            *self.0 = Some(n);
        }
        Ok(())
    }

    pub(crate) fn set(&mut self, n: usize) {
        *self.0 = Some(n);
    }
}

pub struct InnerAxisArrays<B: Backend> {
    pub axis: Axis,
    pub(crate) container: B::Group,
    pub(crate) size: Dim,
    data: HashMap<String, ArrayElem<B>>,
}

impl<B: Backend> std::fmt::Debug for InnerAxisArrays<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
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
        self.size.get()
    }

    pub fn add_data<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &mut self,
        key: &str,
        data: D,
    ) -> Result<()> {
        // Check if the data is compatible with the current size
        let shape = data.shape();
        if let Axis::RowColumn = self.axis {
            ensure!(shape[0] == shape[1], "expecting a square array, but receive a {:?} array", shape);
        }
        self.size.try_set(shape[0])?;

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

    pub fn add_data_from_iter<I, D>(&mut self, key: &str, data: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk,
    {
        if let Some(elem) = self.get(key) {
            elem.clear()?;
        }
        let elem = ArrayElem::try_from(ArrayChunk::write_by_chunk(data, &self.container, key)?)?;

        let shape = { elem.inner().shape().clone() };
        if self.axis == Axis::RowColumn && shape[0] != shape[1] {
            elem.clear()?;
            bail!("expecting a square array, but receive a {:?} array", shape)
        } else if let Err(e) = self.size.try_set(shape[0]) {
            elem.clear()?;
            bail!(e)
        } else {
            self.insert(key.to_string(), elem);
            Ok(())
        }
    }

    pub fn remove_data(&mut self, key: &str) -> Result<()> {
        self.remove(key).map(|x| x.clear()).transpose()?;
        Ok(())
    }

    pub fn export<O: Backend, G: GroupOp<Backend = O>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<()> {
        let group = location.create_group(name)?;
        for (key, val) in self.iter() {
            val.inner().export::<O, _>(&group, key)?;
        }
        Ok(())
    }

    pub fn export_select<O, G>(
        &self,
        selection: &SelectInfoElem,
        location: &G,
        name: &str,
    ) -> Result<()>
    where
        O: Backend,
        G: GroupOp<Backend = O>,
    {
        if selection.is_full() {
            self.export::<O, _>(location, name)
        } else {
            let group = location.create_group(name)?;
            match self.axis {
                Axis::Row => self
                    .iter()
                    .try_for_each(|(k, x)| x.inner().export_axis::<O, _>(0, selection, &group, k)),
                Axis::RowColumn => {
                    let s = vec![selection, selection];
                    self.iter().try_for_each(|(k, x)| {
                        x.inner().export_select::<O, _>(s.as_ref(), &group, k)
                    })
                }
            }
        }
    }

    pub(crate) fn subset<S: AsRef<SelectInfoElem>>(&mut self, selection: S) -> Result<()> {
        match self.axis {
            Axis::Row => {
                self.values()
                    .try_for_each(|x| x.inner().subset_axis(0, &selection))?;
            }
            Axis::RowColumn => {
                self.values().try_for_each(|x| {
                    let full = SelectInfoElem::full();
                    let mut slice: SmallVec<[_; 3]> = smallvec![&full; x.inner().shape().ndim()];
                    slice[0] = selection.as_ref();
                    slice[1] = selection.as_ref();
                    x.inner().subset(slice.as_slice())
                })?;
            }
        }

        if let Some(mut lock) = self.size.try_lock() {
            lock.set(BoundedSelectInfoElem::new(selection.as_ref(), lock.get()).len());
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct AxisArrays<B: Backend>(Slot<InnerAxisArrays<B>>);

impl<B: Backend> std::fmt::Display for AxisArrays<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<B: Backend> Clone for AxisArrays<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

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

    pub fn new(group: B::Group, axis: Axis, size: &Dim) -> Result<Self> {
        let data: HashMap<_, _> = iter_containers::<B>(&group)
            .map(|(k, v)| (k, ArrayElem::try_from(v).unwrap()))
            .collect();
        let sizes = data.iter().map(|(_, v)| {
            let v_lock = v.inner();
            let shape = v_lock.shape();
            match &axis {
                Axis::Row => Ok(shape[0]),
                Axis::RowColumn => {
                    ensure!(shape[0] == shape[1], "Square arrays must be square");
                    Ok(shape[0])
                },
            }
        }).collect::<Result<Vec<_>>>()?;
        if sizes.iter().all_equal() {
            if let Some(n) = sizes.get(0) {
                size.try_set(*n)?;
            }
            let arrays = InnerAxisArrays {
                container: group,
                size: size.clone(),
                axis,
                data,
            };
            Ok(Self(Slot::new(arrays)))
        } else {
            bail!("Arrays must be the same size")
        }
    }

    pub fn clear(&self) -> Result<()> {
        self.0
            .lock()
            .as_ref()
            .map(|x| {
                let g = &x.container;
                g.file()?.delete(&g.path().to_string_lossy())
            })
            .transpose()?;
        self.0.drop();
        Ok(())
    }
}

impl<B: Backend> AxisArraysOp for &AxisArrays<B> {
    type ArrayElem = ArrayElem<B>;

    fn keys(&self) -> Vec<String> {
        self.inner().keys().cloned().collect()
    }

    fn get(&self, key: &str) -> Option<Self::ArrayElem> {
        self.lock().as_ref().and_then(|x| x.get(key).cloned())
    }

    fn add<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()>
    {
        self.inner().add_data(key, data)
    }

    fn add_iter<I, D>(&self, key: &str, data: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk,
    {
        self.inner().add_data_from_iter(key, data)
    }

    fn remove(&self, key: &str) -> Result<()> {
        self.inner().remove_data(key)
    }
}

/// Stacked axis arrays, providing Read-only access to the data.
pub struct StackedAxisArrays<B: Backend> {
    axis: Axis,
    data: Arc<HashMap<String, StackedArrayElem<B>>>,
}

impl<B: Backend> Clone for StackedAxisArrays<B> {
    fn clone(&self) -> Self {
        Self {
            axis: self.axis,
            data: self.data.clone(),
        }
    }
}

impl<B: Backend> Deref for StackedAxisArrays<B> {
    type Target = HashMap<String, StackedArrayElem<B>>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<B: Backend> std::fmt::Display for StackedAxisArrays<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ty = match self.axis {
            Axis::Row => "row",
            Axis::RowColumn => "square",
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

impl<B: Backend> AxisArraysOp for &StackedAxisArrays<B> {
    type ArrayElem = StackedArrayElem<B>;

    fn keys(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }

    fn get(&self, key: &str) -> Option<Self::ArrayElem> {
        self.data.get(key).cloned()
    }

    fn add<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()>
    {
        todo!()
    }

    fn add_iter<I, D>(&self, key: &str, data: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk,
    {
        todo!()
    }

    fn remove(&self, key: &str) -> Result<()> {
        todo!()
    }
}

impl<B: Backend> StackedAxisArrays<B> {
    pub fn empty(axis: Axis) -> Self {
        Self {
            axis,
            data: Arc::new(HashMap::new()),
        }
    }

    pub(crate) fn new(axis: Axis, arrays: Vec<AxisArrays<B>>) -> Result<Self> {
        if arrays.iter().any(|x| x.is_empty()) {
            return Ok(Self::empty(axis));
        }

        ensure!(
            arrays.iter().all(|x| x.inner().axis == axis),
            "Axis mismatch"
        );

        let shared_keys: HashSet<String> = arrays
            .iter()
            .map(|x| x.inner().keys().cloned().collect::<HashSet<_>>())
            .reduce(|a, b| a.intersection(&b).cloned().collect())
            .unwrap_or(HashSet::new());

        let data = shared_keys
            .into_iter()
            .map(|k| {
                let elems = arrays
                    .iter()
                    .map(|x| x.inner().get(&k).unwrap().clone())
                    .collect();
                Ok((k, StackedArrayElem::new(elems)?))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Self {
            axis: axis,
            data: Arc::new(data),
        })
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
}
