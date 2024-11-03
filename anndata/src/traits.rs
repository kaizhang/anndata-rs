use crate::data::*;

use anyhow::Result;
use polars::prelude::DataFrame;
use smallvec::SmallVec;

/// Trait defining operations on an AnnData container.
pub trait AnnDataOp {
    type X: ArrayElemOp;
    type AxisArraysRef<'a>: AxisArraysOp where Self: 'a;
    type ElemCollectionRef<'a>: ElemCollectionOp where Self: 'a;

    /// Returns the 'X' element.
    fn x(&self) -> Self::X;

    /// Sets the 'X' element from an iterator.
    /// Note: The original data will be lost if an error occurs during the writing.
    fn set_x_from_iter<I, D>(&self, iter: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk + Into<ArrayData>;

    /// Sets the 'X' element.
    fn set_x<D: WriteArrayData + Into<ArrayData> + HasShape>(&self, data: D) -> Result<()>;

    /// Deletes the 'X' element.
    fn del_x(&self) -> Result<()>;

    /// Returns the number of observations (rows).
    fn n_obs(&self) -> usize;
    /// Returns the number of variables (columns).
    fn n_vars(&self) -> usize;

    /// Sets the number of observations.
    fn set_n_obs(&self, n: usize) -> Result<()>;
    fn set_n_vars(&self, n: usize) -> Result<()>;

    /// Returns the names of observations.
    fn obs_names(&self) -> DataFrameIndex;
    /// Returns the names of variables.
    fn var_names(&self) -> DataFrameIndex;

    /// Changes the names of observations.
    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()>;
    /// Changes the names of variables.
    fn set_var_names(&self, index: DataFrameIndex) -> Result<()>;

    /// Returns the indices of specified observations.
    fn obs_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>>;
    /// Returns the indices of specified variables.
    fn var_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>>;

    /// Reads the observation annotations.
    fn read_obs(&self) -> Result<DataFrame>;
    /// Reads the variable annotations.
    fn read_var(&self) -> Result<DataFrame>;

    /// Changes the observation annotations.
    fn set_obs(&self, obs: DataFrame) -> Result<()>;

    /// Changes the variable annotations.
    fn set_var(&self, var: DataFrame) -> Result<()>;

    /// Deletes the observation annotations.
    fn del_obs(&self) -> Result<()>;

    /// Deletes the variable annotations.
    fn del_var(&self) -> Result<()>;

    /// Returns a reference to the unstructured data.
    fn uns(&self) -> Self::ElemCollectionRef<'_>;
    /// Returns a reference to the observation matrix.
    fn obsm(&self) -> Self::AxisArraysRef<'_>;
    /// Returns a reference to the observation pairwise data.
    fn obsp(&self) -> Self::AxisArraysRef<'_>;
    /// Returns a reference to the variable matrix.
    fn varm(&self) -> Self::AxisArraysRef<'_>;
    /// Returns a reference to the variable pairwise data.
    fn varp(&self) -> Self::AxisArraysRef<'_>;
    /// Returns a reference to the layers.
    fn layers(&self) -> Self::AxisArraysRef<'_>;

    /// Sets the unstructured data.
    fn set_uns<I: Iterator<Item = (String, Data)>>(&self, mut data: I) -> Result<()> {
        self.del_uns()?;
        let uns = self.uns();
        data.try_for_each(|(k, v)| uns.add(&k, v))
    }
    /// Sets the observation matrix.
    fn set_obsm<I: Iterator<Item = (String, ArrayData)>>(&self, mut data: I) -> Result<()> {
        self.del_obsm()?;
        let obsm = self.obsm();
        data.try_for_each(|(k, v)| obsm.add(&k, v))
    }
    /// Sets the observation pairwise data.
    fn set_obsp<I: Iterator<Item = (String, ArrayData)>>(&self, mut data: I) -> Result<()> {
        self.del_obsp()?;
        let obsp = self.obsp();
        data.try_for_each(|(k, v)| obsp.add(&k, v))
    }
    /// Sets the variable matrix.
    fn set_varm<I: Iterator<Item = (String, ArrayData)>>(&self, mut data: I) -> Result<()> {
        self.del_varm()?;
        let varm = self.varm();
        data.try_for_each(|(k, v)| varm.add(&k, v))
    }
    /// Sets the variable pairwise data.
    fn set_varp<I: Iterator<Item = (String, ArrayData)>>(&self, mut data: I) -> Result<()> {
        self.del_varp()?;
        let varp = self.varp();
        data.try_for_each(|(k, v)| varp.add(&k, v))
    }
    /// Sets the layers.
    fn set_layers<I: Iterator<Item = (String, ArrayData)>>(&self, mut data: I) -> Result<()> {
        self.del_layers()?;
        let layers = self.layers();
        data.try_for_each(|(k, v)| layers.add(&k, v))
    }

    /// Deletes the unstructured data.
    fn del_uns(&self) -> Result<()>;
    /// Deletes the observation matrix.
    fn del_obsm(&self) -> Result<()>;
    /// Deletes the observation pairwise data.
    fn del_obsp(&self) -> Result<()>;
    /// Deletes the variable matrix.
    fn del_varm(&self) -> Result<()>;
    /// Deletes the variable pairwise data.
    fn del_varp(&self) -> Result<()>;
    /// Deletes the layers.
    fn del_layers(&self) -> Result<()>;
}

/// Trait for operations on element collections.
pub trait ElemCollectionOp {
    /// Returns the keys of the collection.
    fn keys(&self) -> Vec<String>;

    /// Gets an item from the collection by key.
    fn get_item<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<Data> + TryFrom<Data> + Clone,
        <D as TryFrom<Data>>::Error: Into<anyhow::Error>;

    /// Adds an item to the collection.
    fn add<D: WriteData + Into<Data>>(&self, key: &str, data: D) -> Result<()>;

    /// Removes an item from the collection by key.
    fn remove(&self, key: &str) -> Result<()>;
}

/// Trait for accessing arrays with multiple axes.
pub trait AxisArraysOp {
    type ArrayElem: ArrayElemOp;

    /// Returns the keys of the axis arrays.
    fn keys(&self) -> Vec<String>;

    /// Gets an ArrayElem object by key, without reading the data.
    fn get(&self, key: &str) -> Option<Self::ArrayElem>;

    /// Gets the array data by key.
    fn get_item<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get(key).and_then(|x| x.get().transpose()).transpose()
            .map_err(|e| e.context(format!("key: {}", key)))
    }

    /// Gets a slice of the array data by key.
    fn get_item_slice<D, S>(&self, key: &str, slice: S) -> Result<Option<D>>
    where
        D: ReadArrayData + Into<ArrayData> + TryFrom<ArrayData> + ArrayOp + Clone,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get(key).and_then(|x| x.slice(slice).transpose()).transpose()
    }

    /// Returns an iterator over the data.
    fn get_item_iter<T>(&self, key: &str, chunk_size: usize
    ) -> Option<<Self::ArrayElem as ArrayElemOp>::ArrayIter<T>>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get(key).map(|x| x.iter(chunk_size))
    }

    /// Adds array data by key.
    fn add<D: WriteArrayData + HasShape + Into<ArrayData>>(&self, key: &str, data: D) -> Result<()>;

    /// Adds array data from an iterator by key.
    fn add_iter<I, D>(&self, key: &str, data: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk + Into<ArrayData>;

    /// Removes data by key.
    fn remove(&self, key: &str) -> Result<()>;
}

/// Trait for operations on array elements.
pub trait ArrayElemOp {
    type ArrayIter<T>: ExactSizeIterator<Item = (T, usize, usize)>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

    /// Returns the shape of the array.
    fn shape(&self) -> Option<Shape>;

    /// Gets the data.
    fn get<D>(&self) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

    /// Gets a slice of the data.
    fn slice<D, S>(&self, slice: S) -> Result<Option<D>>
    where
        D: ReadArrayData + Into<ArrayData> + TryFrom<ArrayData> + ArrayOp + Clone,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

    /// Gets a slice of the data along an axis.
    fn slice_axis<D, S>(&self, axis: usize, slice: S) -> Result<Option<D>>
    where
        D: ReadArrayData + Into<ArrayData> + TryFrom<ArrayData> + ArrayOp + Clone,
        S: AsRef<SelectInfoElem>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.shape().and_then(|shape| {
            let full = SelectInfoElem::full();
            let slice: SmallVec<[SelectInfoElem; 3]> = slice
                .as_ref()
                .set_axis(axis, shape.ndim(), &full).into_iter().cloned().collect();
            self.slice(slice.as_slice()).transpose()
        }).transpose()
    }

    /// Returns an iterator over the data.
    fn iter<T>(
        &self,
        chunk_size: usize,
    ) -> Self::ArrayIter<T>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;
}