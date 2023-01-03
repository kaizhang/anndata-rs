use crate::data::*;

use anyhow::Result;
use polars::prelude::DataFrame;
use smallvec::SmallVec;

/// AnnData container operations.
pub trait AnnDataOp {
    type X: ArrayElemOp;
    type AxisArraysRef<'a>: AxisArraysOp where Self: 'a;
    type ElemCollectionRef<'a>: ElemCollectionOp where Self: 'a;

    fn x(&self) -> Self::X;

    /// Set the 'X' element from an iterator. Note that the original data will be
    /// lost if an error occurs during the writing.
    fn set_x_from_iter<I, D>(&self, iter: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk + Into<ArrayData>;

    fn set_x<D: WriteArrayData + Into<ArrayData> + HasShape>(&self, data: D) -> Result<()>;

    /// Delete the 'X' element.
    fn del_x(&self) -> Result<()>;

    /// Return the number of observations (rows).
    fn n_obs(&self) -> usize;
    /// Return the number of variables (columns).
    fn n_vars(&self) -> usize;

    /// Return the names of observations.
    fn obs_names(&self) -> DataFrameIndex;
    /// Return the names of variables.
    fn var_names(&self) -> DataFrameIndex;

    /// Chagne the names of observations.
    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()>;
    /// Chagne the names of variables.
    fn set_var_names(&self, index: DataFrameIndex) -> Result<()>;

    fn obs_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>>;
    fn var_ix<'a, I: IntoIterator<Item = &'a str>>(&self, names: I) -> Result<Vec<usize>>;

    fn read_obs(&self) -> Result<DataFrame>;
    fn read_var(&self) -> Result<DataFrame>;

    /// Change the observation annotations.
    fn set_obs(&self, obs: DataFrame) -> Result<()>;

    /// Change the variable annotations.
    fn set_var(&self, var: DataFrame) -> Result<()>;

    /// Delete the observation annotations.
    fn del_obs(&self) -> Result<()>;

    /// Delete the variable annotations.
    fn del_var(&self) -> Result<()>;

    fn uns(&self) -> Self::ElemCollectionRef<'_>;
    fn obsm(&self) -> Self::AxisArraysRef<'_>;
    fn obsp(&self) -> Self::AxisArraysRef<'_>;
    fn varm(&self) -> Self::AxisArraysRef<'_>;
    fn varp(&self) -> Self::AxisArraysRef<'_>;

    fn set_uns<I: Iterator<Item = (String, Data)>>(&self, mut data: I) -> Result<()> {
        self.del_uns()?;
        let uns = self.uns();
        data.try_for_each(|(k, v)| uns.add(&k, v))
    }
    fn set_obsm<I: Iterator<Item = (String, ArrayData)>>(&self, mut data: I) -> Result<()> {
        self.del_obsm()?;
        let obsm = self.obsm();
        data.try_for_each(|(k, v)| obsm.add(&k, v))
    }
    fn set_obsp<I: Iterator<Item = (String, ArrayData)>>(&self, mut data: I) -> Result<()> {
        self.del_obsp()?;
        let obsp = self.obsp();
        data.try_for_each(|(k, v)| obsp.add(&k, v))
    }
    fn set_varm<I: Iterator<Item = (String, ArrayData)>>(&self, mut data: I) -> Result<()> {
        self.del_varm()?;
        let varm = self.varm();
        data.try_for_each(|(k, v)| varm.add(&k, v))
    }
    fn set_varp<I: Iterator<Item = (String, ArrayData)>>(&self, mut data: I) -> Result<()> {
        self.del_varp()?;
        let varp = self.varp();
        data.try_for_each(|(k, v)| varp.add(&k, v))
    }

    fn del_uns(&self) -> Result<()>;
    fn del_obsm(&self) -> Result<()>;
    fn del_obsp(&self) -> Result<()>;
    fn del_varm(&self) -> Result<()>;
    fn del_varp(&self) -> Result<()>;
}

pub trait ElemCollectionOp {
    fn keys(&self) -> Vec<String>;

    fn get_item<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<Data> + TryFrom<Data> + Clone,
        <D as TryFrom<Data>>::Error: Into<anyhow::Error>;

    fn add<D: WriteData + Into<Data>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()>;

    fn remove(&self, key: &str) -> Result<()>;
}

/// A trait for accessing arrays with multiple axes.
pub trait AxisArraysOp {
    type ArrayElem: ArrayElemOp;

    fn keys(&self) -> Vec<String>;

    fn get(&self, key: &str) -> Option<Self::ArrayElem>;

    fn get_item<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get(key).and_then(|x| x.get().transpose()).transpose()
    }

    fn get_item_slice<D, S>(&self, key: &str, slice: S) -> Result<Option<D>>
    where
        D: ReadArrayData + Into<ArrayData> + TryFrom<ArrayData> + ArrayOp + Clone,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get(key).and_then(|x| x.slice(slice).transpose()).transpose()
    }

    fn get_item_iter<T>(&self, key: &str, chunk_size: usize
    ) -> Option<<Self::ArrayElem as ArrayElemOp>::ArrayIter<T>>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
    {
        self.get(key).map(|x| x.iter(chunk_size))
    }

    fn add<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()>;

    fn add_iter<I, D>(&self, key: &str, data: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: ArrayChunk + Into<ArrayData>;

    /// Remove data by key.
    fn remove(&self, key: &str) -> Result<()>;
}

pub trait ArrayElemOp {
    type ArrayIter<T>: ExactSizeIterator<Item = (T, usize, usize)>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

    fn shape(&self) -> Option<Shape>;

    fn get<D>(&self) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

    fn slice<D, S>(&self, slice: S) -> Result<Option<D>>
    where
        D: ReadArrayData + Into<ArrayData> + TryFrom<ArrayData> + ArrayOp + Clone,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

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

    fn iter<T>(
        &self,
        chunk_size: usize,
    ) -> Self::ArrayIter<T>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;
}