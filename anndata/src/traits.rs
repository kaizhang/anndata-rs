use crate::data::*;

use anyhow::Result;
use polars::prelude::DataFrame;

pub trait AnnDataOp {
    /// Reading/writing the 'X' element.
    fn read_x<D>(&self) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

    fn read_x_slice<D, S>(&self, select: S) -> Result<Option<D>>
    where
        D: ReadArrayData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        S: AsRef<[SelectInfoElem]>,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

    fn set_x<D: WriteArrayData + Into<ArrayData> + HasShape>(&self, data: D) -> Result<()>;

    /// Delete the 'X' element.
    fn del_x(&self) -> Result<()>;

    /// Return the number of observations (rows).
    fn n_obs(&self) -> usize;
    /// Return the number of variables (columns).
    fn n_vars(&self) -> usize;

    /// Return the names of observations.
    fn obs_names(&self) -> Vec<String>;
    /// Return the names of variables.
    fn var_names(&self) -> Vec<String>;

    /// Chagne the names of observations.
    fn set_obs_names(&self, index: DataFrameIndex) -> Result<()>;
    /// Chagne the names of variables.
    fn set_var_names(&self, index: DataFrameIndex) -> Result<()>;

    fn obs_ix(&self, names: &[String]) -> Result<Vec<usize>>;
    fn var_ix(&self, names: &[String]) -> Result<Vec<usize>>;

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

    fn uns_keys(&self) -> Vec<String>;
    fn obsm_keys(&self) -> Vec<String>;
    fn obsp_keys(&self) -> Vec<String>;
    fn varm_keys(&self) -> Vec<String>;
    fn varp_keys(&self) -> Vec<String>;

    fn fetch_uns<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<Data> + TryFrom<Data> + Clone,
        <D as TryFrom<Data>>::Error: Into<anyhow::Error>;
    fn fetch_obsm<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;
    fn fetch_obsp<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;
    fn fetch_varm<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;
    fn fetch_varp<D>(&self, key: &str) -> Result<Option<D>>
    where
        D: ReadData + Into<ArrayData> + TryFrom<ArrayData> + Clone,
        <D as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;  
    fn add_uns<D: WriteData + Into<Data>>(&self, key: &str, data: D) -> Result<()>;
    fn add_obsm<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()>;
    fn add_obsp<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()>;
    fn add_varm<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()>;
    fn add_varp<D: WriteArrayData + HasShape + Into<ArrayData>>(
        &self,
        key: &str,
        data: D,
    ) -> Result<()>;
}

pub trait AnnDataIterator: AnnDataOp {
    type ArrayIter<'a, T>: Iterator<Item = (T, usize, usize)> + ExactSizeIterator
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>,
        Self: 'a;

    fn read_x_iter<'a, T>(&'a self, chunk_size: usize) -> Self::ArrayIter<'a, T>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

    /// Set the 'X' element from an iterator. Note that the original data will be
    /// lost if an error occurs during the writing.
    fn set_x_from_iter<I: Iterator<Item = D>, D: WriteArrayData>(&self, iter: I) -> Result<()>;

    fn fetch_obsm_iter<'a, T>(
        &'a self,
        key: &str,
        chunk_size: usize,
    ) -> Result<Self::ArrayIter<'a, T>>
    where
        T: Into<ArrayData> + TryFrom<ArrayData> + ReadArrayData + Clone,
        <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>;

    fn add_obsm_from_iter<I, D>(&self, key: &str, data: I) -> Result<()>
    where
        I: Iterator<Item = D>,
        D: WriteArrayData;
}