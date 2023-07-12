use crate::backend::{Backend, DataContainer, GroupOp, LocationOp, BackendData, ScalarType};
use crate::ArrayOp;
use crate::data::{
    ArrayData,
    array::utils::ExtendableDataset,
};

use anyhow::{bail, Result};
use ndarray::{Array, ArrayView1, ArrayD, RemoveAxis};
use nalgebra_sparse::na::Scalar;
use nalgebra_sparse::{CsrMatrix, CscMatrix};
use super::utils::{vstack_arr, vstack_csr, hstack_csc};
use super::{DynCsrMatrix, DynCscMatrix, DynArray};

pub trait ArrayChunk: ArrayOp {
    fn concat<I: Iterator<Item = Self>>(iter: I) -> Result<Self> where Self: Sized;

    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<Backend = B>;
}

impl ArrayChunk for ArrayData {
    fn concat<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            ArrayData::Array(_) => DynArray::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            ArrayData::CsrMatrix(_) => DynCsrMatrix::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            ArrayData::CscMatrix(_) => DynCscMatrix::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            ArrayData::DataFrame(_) => todo!(),
        }
    }

    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<Backend = B>,
    {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            ArrayData::Array(_) => DynArray::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            ArrayData::CsrMatrix(_) => DynCsrMatrix::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            ArrayData::CscMatrix(_) => DynCscMatrix::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            ArrayData::DataFrame(_) => todo!(),
        }
    }
}

impl ArrayChunk for DynArray {
    fn concat<I: Iterator<Item = Self>>(iter: I) -> Result<Self> where Self: Sized {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            DynArray::U8(_) => ArrayD::<u8>::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::U16(_) => ArrayD::<u16>::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::U32(_) => ArrayD::<u32>::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::U64(_) => ArrayD::<u64>::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::Usize(_) => ArrayD::<usize>::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::I8(_) => ArrayD::<i8>::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::I16(_) => ArrayD::<i16>::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::I32(_) => ArrayD::<i32>::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::I64(_) => ArrayD::<i64>::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::F32(_) => ArrayD::<f32>::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::F64(_) => ArrayD::<f64>::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::Bool(_) => ArrayD::<bool>::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::String(_) => ArrayD::<String>::concat(iter.map(|x| x.try_into().unwrap())).map(|x| x.into()),
            DynArray::Categorical(_) => todo!(),
        }
    }

    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<Backend = B>,
    {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            DynArray::U8(_) => ArrayD::<u8>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynArray::U16(_) => ArrayD::<u16>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynArray::U32(_) => ArrayD::<u32>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynArray::U64(_) => ArrayD::<u64>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynArray::Usize(_) => ArrayD::<usize>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynArray::I8(_) => ArrayD::<i8>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynArray::I16(_) => ArrayD::<i16>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynArray::I32(_) => ArrayD::<i32>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynArray::I64(_) => ArrayD::<i64>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynArray::F32(_) => ArrayD::<f32>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynArray::F64(_) => ArrayD::<f64>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynArray::Bool(_) => ArrayD::<bool>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynArray::String(_) => ArrayD::<String>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynArray::Categorical(_) => todo!(),
        }
    }
}

impl<D: RemoveAxis, T: BackendData> ArrayChunk for Array<T, D> {
    fn concat<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        Ok(iter.reduce(|acc, x| vstack_arr(acc, x)).unwrap())
    }

    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<Backend = B>,
    {
        let mut iter = iter.peekable();
        let chunk_size = if let Some(n) = D::NDIM {
            vec![1000; n].into()
        } else {
            let n = iter.peek().unwrap().ndim();
            vec![1000; n].into()
        };
        let mut data: ExtendableDataset<B, T> = ExtendableDataset::with_capacity(
            location, name, chunk_size,
        )?;

        iter.try_for_each(|x| data.extend(0, x.view()))?;
        let dataset = data.finish()?;
        let encoding_type = if T::DTYPE == ScalarType::String {
            "string-array"
        } else {
            "array"
        };
        let container = DataContainer::<B>::Dataset(dataset);
        container.write_str_attr("encoding-type", encoding_type)?;
        container.write_str_attr("encoding-version", "0.2.0")?;
        Ok(container)
    }
}

impl ArrayChunk for DynCsrMatrix {
    fn concat<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            DynCsrMatrix::U8(_) => Ok(DynCsrMatrix::U8(CsrMatrix::<u8>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::U16(_) => Ok(DynCsrMatrix::U16(CsrMatrix::<u16>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::U32(_) => Ok(DynCsrMatrix::U32(CsrMatrix::<u32>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::U64(_) => Ok(DynCsrMatrix::U64(CsrMatrix::<u64>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::Usize(_) => Ok(DynCsrMatrix::Usize(CsrMatrix::<usize>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::I8(_) => Ok(DynCsrMatrix::I8(CsrMatrix::<i8>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::I16(_) => Ok(DynCsrMatrix::I16(CsrMatrix::<i16>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::I32(_) => Ok(DynCsrMatrix::I32(CsrMatrix::<i32>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::I64(_) => Ok(DynCsrMatrix::I64(CsrMatrix::<i64>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::F32(_) => Ok(DynCsrMatrix::F32(CsrMatrix::<f32>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::F64(_) => Ok(DynCsrMatrix::F64(CsrMatrix::<f64>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::Bool(_) => Ok(DynCsrMatrix::Bool(CsrMatrix::<bool>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCsrMatrix::String(_) => Ok(DynCsrMatrix::String(CsrMatrix::<String>::concat(iter.map(|x| x.try_into().unwrap()))?)),
        }
    }

    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<Backend = B>,
    {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            DynCsrMatrix::U8(_) => CsrMatrix::<u8>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCsrMatrix::U16(_) => CsrMatrix::<u16>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCsrMatrix::U32(_) => CsrMatrix::<u32>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCsrMatrix::U64(_) => CsrMatrix::<u64>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCsrMatrix::Usize(_) => CsrMatrix::<usize>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCsrMatrix::I8(_) => CsrMatrix::<i8>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCsrMatrix::I16(_) => CsrMatrix::<i16>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCsrMatrix::I32(_) => CsrMatrix::<i32>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCsrMatrix::I64(_) => CsrMatrix::<i64>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCsrMatrix::F32(_) => CsrMatrix::<f32>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCsrMatrix::F64(_) => CsrMatrix::<f64>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCsrMatrix::Bool(_) => CsrMatrix::<bool>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCsrMatrix::String(_) => CsrMatrix::<String>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
        }
    }
}


impl<T: BackendData> ArrayChunk for CsrMatrix<T> {
    fn concat<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        Ok(iter.reduce(|acc, x| vstack_csr(acc, x)).unwrap())
    }

    fn write_by_chunk<B, G, I>(mut iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<Backend = B>,
    {
        let group = location.create_group(name)?;
        group.write_str_attr("encoding-type", "csr_matrix")?;
        group.write_str_attr("encoding-version", "0.1.0")?;
        group.write_str_attr("h5sparse_format", "csr")?;

        let mut data: ExtendableDataset<B, T> = ExtendableDataset::with_capacity(
            &group, "data", 1000.into(),
        )?;
        let mut indices: ExtendableDataset<B, i64> = ExtendableDataset::with_capacity(
            &group, "indices", 1000.into(),
        )?;
        let mut indptr: Vec<i64> = Vec::new();
        let mut num_rows = 0;
        let mut num_cols: Option<usize> = None;
        let mut nnz = 0;

        iter.try_for_each(|csr| {
            let c = csr.ncols();
            if num_cols.is_none() {
                num_cols = Some(c);
            }
            if num_cols.unwrap() == c {
                num_rows += csr.nrows();
                let (indptr_, indices_, data_) = csr.csr_data();
                indptr_[..indptr_.len() - 1]
                    .iter()
                    .for_each(|x| indptr.push(i64::try_from(*x).unwrap() + nnz));
                nnz += *indptr_.last().unwrap_or(&0) as i64;
                data.extend(0, ArrayView1::from_shape(data_.len(), data_)?)?;
                indices.extend(0, ArrayView1::from_shape(indices_.len(), indices_)?.mapv(|x| x as i64).view())
            } else {
                bail!("All matrices must have the same number of columns");
            }
        })?;

        indices.finish()?;
        data.finish()?;
        indptr.push(nnz);
        group.create_array_data("indptr", &indptr, Default::default())?;
        group.write_array_attr("shape", &[num_rows, num_cols.unwrap_or(0)])?;
        Ok(DataContainer::Group(group))
    }
}





impl ArrayChunk for DynCscMatrix {
    fn concat<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            DynCscMatrix::U8(_) => Ok(DynCscMatrix::U8(CscMatrix::<u8>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCscMatrix::U16(_) => Ok(DynCscMatrix::U16(CscMatrix::<u16>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCscMatrix::U32(_) => Ok(DynCscMatrix::U32(CscMatrix::<u32>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCscMatrix::U64(_) => Ok(DynCscMatrix::U64(CscMatrix::<u64>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCscMatrix::Usize(_) => Ok(DynCscMatrix::Usize(CscMatrix::<usize>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCscMatrix::I8(_) => Ok(DynCscMatrix::I8(CscMatrix::<i8>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCscMatrix::I16(_) => Ok(DynCscMatrix::I16(CscMatrix::<i16>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCscMatrix::I32(_) => Ok(DynCscMatrix::I32(CscMatrix::<i32>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCscMatrix::I64(_) => Ok(DynCscMatrix::I64(CscMatrix::<i64>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCscMatrix::F32(_) => Ok(DynCscMatrix::F32(CscMatrix::<f32>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCscMatrix::F64(_) => Ok(DynCscMatrix::F64(CscMatrix::<f64>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCscMatrix::Bool(_) => Ok(DynCscMatrix::Bool(CscMatrix::<bool>::concat(iter.map(|x| x.try_into().unwrap()))?)),
            DynCscMatrix::String(_) => Ok(DynCscMatrix::String(CscMatrix::<String>::concat(iter.map(|x| x.try_into().unwrap()))?)),
        }
    }

    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<Backend = B>,
    {
        let mut iter = iter.peekable();
        match iter.peek().unwrap() {
            DynCscMatrix::U8(_) => CscMatrix::<u8>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCscMatrix::U16(_) => CscMatrix::<u16>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCscMatrix::U32(_) => CscMatrix::<u32>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCscMatrix::U64(_) => CscMatrix::<u64>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCscMatrix::Usize(_) => CscMatrix::<usize>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCscMatrix::I8(_) => CscMatrix::<i8>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCscMatrix::I16(_) => CscMatrix::<i16>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCscMatrix::I32(_) => CscMatrix::<i32>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCscMatrix::I64(_) => CscMatrix::<i64>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCscMatrix::F32(_) => CscMatrix::<f32>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCscMatrix::F64(_) => CscMatrix::<f64>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCscMatrix::Bool(_) => CscMatrix::<bool>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
            DynCscMatrix::String(_) => CscMatrix::<String>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name),
        }
    }
}


impl<T: BackendData+Scalar> ArrayChunk for CscMatrix<T> {
    fn concat<I: Iterator<Item = Self>>(iter: I) -> Result<Self> {
        // TODO! more efficent way should be implement
        // Ok(iter.reduce(|acc, x| vstack_csc(acc, x)).unwrap())
        Ok(iter.map(|csc| csc.transpose())
               .reduce(|acc, x| hstack_csc(acc, x))
               .unwrap()
               .transpose())
    }

    fn write_by_chunk<B, G, I>(mut iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<Backend = B>,
    {
        let group = location.create_group(name)?;
        group.write_str_attr("encoding-type", "csc_matrix")?;
        group.write_str_attr("encoding-version", "0.1.0")?;
        group.write_str_attr("h5sparse_format", "csc")?;

        let mut data: ExtendableDataset<B, T> = ExtendableDataset::with_capacity(
            &group, "data", 1000.into(),
        )?;
        let mut indices: ExtendableDataset<B, i64> = ExtendableDataset::with_capacity(
            &group, "indices", 1000.into(),
        )?;
        let mut indptr: Vec<i64> = Vec::new();
        let mut num_cols = 0;
        let mut num_rows: Option<usize> = None;
        let mut nnz = 0;

        iter.try_for_each(|csc| {
            let r = csc.nrows();
            if num_rows.is_none() {
                num_rows = Some(r);
            }
            if num_rows.unwrap() == r {
                num_cols += csc.ncols();
                let (indptr_, indices_, data_) = csc.csc_data();
                indptr_[..indptr_.len() - 1]
                    .iter()
                    .for_each(|x| indptr.push(i64::try_from(*x).unwrap() + nnz));
                nnz += *indptr_.last().unwrap_or(&0) as i64;
                data.extend(0, ArrayView1::from_shape(data_.len(), data_)?)?;
                indices.extend(0, ArrayView1::from_shape(indices_.len(), indices_)?.mapv(|x| x as i64).view())
            } else {
                bail!("All matrices must have the same number of rows");
            }
        })?;

        indices.finish()?;
        data.finish()?;
        indptr.push(nnz);
        group.create_array_data("indptr", &indptr, Default::default())?;
        group.write_array_attr("shape", &[num_rows.unwrap_or(0), num_cols])?;
        Ok(DataContainer::Group(group))
    }
}







