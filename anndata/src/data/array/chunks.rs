use crate::{ArrayElem, Selectable};
use crate::backend::{AttributeOp, Backend, BackendData, DataContainer, GroupOp, ScalarType};
use crate::data::{ArrayData, array::DynArray, array::utils::ExtendableDataset};

use super::{CsrNonCanonical, DynCscMatrix, DynCsrMatrix, DynCsrNonCanonical};
use anyhow::{Context, Result, bail};
use nalgebra_sparse::na::Scalar;
use nalgebra_sparse::{CscMatrix, CsrMatrix};
use ndarray::{Array, Array1, ArrayD, ArrayView1, RemoveAxis};

pub enum MatrixBuilder<B: Backend> {
    CsrMatrix(CsrMatrixBuilder<B>),
    Array(ArrayBuilder<B>),
}

impl<B: Backend> MatrixBuilder<B> {
    pub fn new_dense<G: GroupOp<B>>(location: &G, name: &str, dtype: ScalarType) -> Result<Self> {
        Ok(Self::Array(ArrayBuilder::new(location, name, dtype, 2)?))
    }

    pub fn new_sparse<G: GroupOp<B>>(location: &G, name: &str, dtype: ScalarType) -> Result<Self> {
        Ok(Self::CsrMatrix(CsrMatrixBuilder::new(
            location, name, dtype,
        )?))
    }

    pub fn add(&mut self, csr: ArrayData) -> Result<()> {
        match self {
            Self::CsrMatrix(builder) => builder.add(csr),
            Self::Array(builder) => builder.add(csr),
        }
    }

    pub fn finish(self) -> Result<ArrayElem<B>> {
        let container = match self {
            Self::CsrMatrix(builder) => builder.finish(),
            Self::Array(builder) => builder.finish(),
        }?;
        ArrayElem::try_from(container)
    }
}

pub struct CsrMatrixBuilder<B: Backend> {
    indices: ExtendableDataset<B, i64>,
    data: ArrayBuilder<B>,
    indptr: Vec<i64>,
    num_rows: usize,
    num_cols: Option<usize>,
    nnz: i64,
    group: B::Group,
}

impl<B: Backend> CsrMatrixBuilder<B> {
    fn new<G: GroupOp<B>>(location: &G, name: &str, dtype: ScalarType) -> Result<Self> {
        let mut group = location.new_group(name)?;
        group.new_attr("encoding-type", "csr_matrix")?;
        group.new_attr("encoding-version", "0.1.0")?;
        group.new_attr("h5sparse_format", "csr")?;

        let data = ArrayBuilder::new(&group, "data", dtype, 1)?;
        let indices: ExtendableDataset<B, i64> =
            ExtendableDataset::with_capacity(&group, "indices", 1000.into())?;

        Ok(Self {
            indices,
            data,
            indptr: Vec::new(),
            num_rows: 0,
            num_cols: None,
            nnz: 0,
            group,
        })
    }

    fn add(&mut self, csr: ArrayData) -> Result<()> {
        fn helper<B, T>(builder: &mut CsrMatrixBuilder<B>, csr: CsrMatrix<T>) -> Result<()>
        where
            B: Backend,
            ArrayData: From<Array1<T>>,
        {
            let c = csr.ncols();
            if builder.num_cols.is_none() {
                builder.num_cols = Some(c);
            }
            if builder.num_cols.unwrap() == c {
                builder.num_rows += csr.nrows();
                let (indptr_, indices_, data_) = csr.disassemble();
                indptr_[..indptr_.len() - 1].iter().for_each(|x| {
                    builder
                        .indptr
                        .push(i64::try_from(*x).unwrap() + builder.nnz)
                });
                builder.nnz += *indptr_.last().unwrap_or(&0) as i64;
                builder.data.add(Array::from_vec(data_).into())?;
                builder.indices.extend(
                    0,
                    Array::from_vec(indices_)
                        .mapv(|x| i64::try_from(x).unwrap())
                        .view(),
                )
            } else {
                bail!("All matrices must have the same number of columns");
            }
        }
        match csr {
            ArrayData::CsrMatrix(DynCsrMatrix::U8(mat)) => helper(self, mat),
            ArrayData::CsrMatrix(DynCsrMatrix::U16(mat)) => helper(self, mat),
            ArrayData::CsrMatrix(DynCsrMatrix::U32(mat)) => helper(self, mat),
            ArrayData::CsrMatrix(DynCsrMatrix::U64(mat)) => helper(self, mat),
            ArrayData::CsrMatrix(DynCsrMatrix::I8(mat)) => helper(self, mat),
            ArrayData::CsrMatrix(DynCsrMatrix::I16(mat)) => helper(self, mat),
            ArrayData::CsrMatrix(DynCsrMatrix::I32(mat)) => helper(self, mat),
            ArrayData::CsrMatrix(DynCsrMatrix::I64(mat)) => helper(self, mat),
            ArrayData::CsrMatrix(DynCsrMatrix::F32(mat)) => helper(self, mat),
            ArrayData::CsrMatrix(DynCsrMatrix::F64(mat)) => helper(self, mat),
            ArrayData::CsrMatrix(DynCsrMatrix::Bool(mat)) => helper(self, mat),
            ArrayData::CsrMatrix(DynCsrMatrix::String(mat)) => helper(self, mat),
            _ => bail!("Expected CsrMatrix"),
        }?;
        Ok(())
    }

    fn finish(mut self) -> Result<DataContainer<B>> {
        self.indices.finish()?;
        self.data.finish()?;
        self.indptr.push(self.nnz);
        self.group
            .new_array_dataset("indptr", self.indptr.into(), Default::default())?;
        self.group.new_attr(
            "shape",
            [self.num_rows as u64, self.num_cols.unwrap_or(0) as u64].as_slice(),
        )?;
        Ok(DataContainer::Group(self.group))
    }
}

pub enum ArrayBuilder<B: Backend> {
    U8(ExtendableDataset<B, u8>),
    U16(ExtendableDataset<B, u16>),
    U32(ExtendableDataset<B, u32>),
    U64(ExtendableDataset<B, u64>),
    I8(ExtendableDataset<B, i8>),
    I16(ExtendableDataset<B, i16>),
    I32(ExtendableDataset<B, i32>),
    I64(ExtendableDataset<B, i64>),
    F32(ExtendableDataset<B, f32>),
    F64(ExtendableDataset<B, f64>),
    Bool(ExtendableDataset<B, bool>),
    String(ExtendableDataset<B, String>),
}

impl<B: Backend> ArrayBuilder<B> {
    fn new<G>(location: &G, name: &str, dtype: ScalarType, ndim: usize) -> Result<Self>
    where
        G: GroupOp<B>,
    {
        let chunk_size = vec![1000; ndim].into();
        let data = match dtype {
            ScalarType::U8 => Self::U8(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::U16 => Self::U16(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::U32 => Self::U32(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::U64 => Self::U64(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::I8 => Self::I8(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::I16 => Self::I16(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::I32 => Self::I32(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::I64 => Self::I64(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::F32 => Self::F32(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::F64 => Self::F64(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::Bool => Self::Bool(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
            ScalarType::String => Self::String(ExtendableDataset::with_capacity(
                location, name, chunk_size,
            )?),
        };
        Ok(data)
    }

    fn add(&mut self, array: ArrayData) -> Result<()> {
        match self {
            Self::U8(data) => data.extend(0, ArrayD::<u8>::try_from(array)?.view()),
            Self::U16(data) => data.extend(0, ArrayD::<u16>::try_from(array)?.view()),
            Self::U32(data) => data.extend(0, ArrayD::<u32>::try_from(array)?.view()),
            Self::U64(data) => data.extend(0, ArrayD::<u64>::try_from(array)?.view()),
            Self::I8(data) => data.extend(0, ArrayD::<i8>::try_from(array)?.view()),
            Self::I16(data) => data.extend(0, ArrayD::<i16>::try_from(array)?.view()),
            Self::I32(data) => data.extend(0, ArrayD::<i32>::try_from(array)?.view()),
            Self::I64(data) => data.extend(0, ArrayD::<i64>::try_from(array)?.view()),
            Self::F32(data) => data.extend(0, ArrayD::<f32>::try_from(array)?.view()),
            Self::F64(data) => data.extend(0, ArrayD::<f64>::try_from(array)?.view()),
            Self::Bool(data) => data.extend(0, ArrayD::<bool>::try_from(array)?.view()),
            Self::String(data) => data.extend(0, ArrayD::<String>::try_from(array)?.view()),
        }
    }

    fn finish(self) -> Result<DataContainer<B>> {
        let (dataset, encoding_type) = match self {
            Self::U8(data) => (data.finish()?, "array"),
            Self::U16(data) => (data.finish()?, "array"),
            Self::U32(data) => (data.finish()?, "array"),
            Self::U64(data) => (data.finish()?, "array"),
            Self::I8(data) => (data.finish()?, "array"),
            Self::I16(data) => (data.finish()?, "array"),
            Self::I32(data) => (data.finish()?, "array"),
            Self::I64(data) => (data.finish()?, "array"),
            Self::F32(data) => (data.finish()?, "array"),
            Self::F64(data) => (data.finish()?, "array"),
            Self::Bool(data) => (data.finish()?, "array"),
            Self::String(data) => (data.finish()?, "string-array"),
        };
        let mut container = DataContainer::<B>::Dataset(dataset);
        container.new_attr("encoding-type", encoding_type)?;
        container.new_attr("encoding-version", "0.2.0")?;
        Ok(container)
    }
}

pub trait ArrayChunk: Selectable {
    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>;
}

impl ArrayChunk for ArrayData {
    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let mut iter = iter.peekable();
        match iter.peek().context("input iterator is empty")? {
            ArrayData::Array(_) => {
                DynArray::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            ArrayData::CsrMatrix(_) | ArrayData::CsrNonCanonical(_) => {
                DynCsrNonCanonical::write_by_chunk(
                    iter.map(|x| x.try_into().unwrap()),
                    location,
                    name,
                )
            }
            ArrayData::CscMatrix(_) => {
                DynCscMatrix::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            ArrayData::DataFrame(_) => todo!(),
        }
    }
}

impl ArrayChunk for DynArray {
    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let mut iter = iter.peekable();
        match iter.peek().context("input iterator is empty")? {
            DynArray::U8(_) => {
                ArrayD::<u8>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            DynArray::U16(_) => {
                ArrayD::<u16>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            DynArray::U32(_) => {
                ArrayD::<u32>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            DynArray::U64(_) => {
                ArrayD::<u64>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            DynArray::I8(_) => {
                ArrayD::<i8>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            DynArray::I16(_) => {
                ArrayD::<i16>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            DynArray::I32(_) => {
                ArrayD::<i32>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            DynArray::I64(_) => {
                ArrayD::<i64>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            DynArray::F32(_) => {
                ArrayD::<f32>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            DynArray::F64(_) => {
                ArrayD::<f64>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            DynArray::Bool(_) => {
                ArrayD::<bool>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            DynArray::String(_) => ArrayD::<String>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
        }
    }
}

impl<D: RemoveAxis, T: BackendData> ArrayChunk for Array<T, D> {
    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let mut iter = iter.peekable();
        let chunk_size = if let Some(n) = D::NDIM {
            vec![1000; n].into()
        } else {
            let n = iter.peek().unwrap().ndim();
            vec![1000; n].into()
        };
        let mut data: ExtendableDataset<B, T> =
            ExtendableDataset::with_capacity(location, name, chunk_size)?;

        iter.try_for_each(|x| data.extend(0, x.view()))?;
        let dataset = data.finish()?;
        let encoding_type = if T::DTYPE == ScalarType::String {
            "string-array"
        } else {
            "array"
        };
        let mut container = DataContainer::<B>::Dataset(dataset);
        container.new_attr("encoding-type", encoding_type)?;
        container.new_attr("encoding-version", "0.2.0")?;
        Ok(container)
    }
}

impl ArrayChunk for DynCsrMatrix {
    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let mut iter = iter.peekable();
        match iter.peek().context("input iterator is empty")? {
            DynCsrMatrix::U8(_) => {
                CsrMatrix::<u8>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            DynCsrMatrix::U16(_) => CsrMatrix::<u16>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrMatrix::U32(_) => CsrMatrix::<u32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrMatrix::U64(_) => CsrMatrix::<u64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrMatrix::I8(_) => {
                CsrMatrix::<i8>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            DynCsrMatrix::I16(_) => CsrMatrix::<i16>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrMatrix::I32(_) => CsrMatrix::<i32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrMatrix::I64(_) => CsrMatrix::<i64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrMatrix::F32(_) => CsrMatrix::<f32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrMatrix::F64(_) => CsrMatrix::<f64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrMatrix::Bool(_) => CsrMatrix::<bool>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrMatrix::String(_) => CsrMatrix::<String>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
        }
    }
}

impl<T: BackendData> ArrayChunk for CsrMatrix<T> {
    fn write_by_chunk<B, G, I>(mut iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let mut group = location.new_group(name)?;
        group.new_attr("encoding-type", "csr_matrix")?;
        group.new_attr("encoding-version", "0.1.0")?;
        group.new_attr("h5sparse_format", "csr")?;

        let mut data: ExtendableDataset<B, T> =
            ExtendableDataset::with_capacity(&group, "data", 1000.into())?;
        let mut indices: ExtendableDataset<B, i64> =
            ExtendableDataset::with_capacity(&group, "indices", 1000.into())?;
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
                indices.extend(
                    0,
                    ArrayView1::from_shape(indices_.len(), indices_)?
                        .mapv(|x| i64::try_from(x).unwrap())
                        .view(),
                )
            } else {
                bail!("All matrices must have the same number of columns");
            }
        })?;

        indices.finish()?;
        data.finish()?;
        indptr.push(nnz);
        group.new_array_dataset("indptr", indptr.into(), Default::default())?;
        group.new_attr(
            "shape",
            [num_rows as u64, num_cols.unwrap_or(0) as u64].as_slice(),
        )?;
        Ok(DataContainer::Group(group))
    }
}

impl ArrayChunk for DynCsrNonCanonical {
    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let mut iter = iter.peekable();
        match iter.peek().context("input iterator is empty")? {
            DynCsrNonCanonical::U8(_) => CsrNonCanonical::<u8>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrNonCanonical::U16(_) => CsrNonCanonical::<u16>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrNonCanonical::U32(_) => CsrNonCanonical::<u32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrNonCanonical::U64(_) => CsrNonCanonical::<u64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrNonCanonical::I8(_) => CsrNonCanonical::<i8>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrNonCanonical::I16(_) => CsrNonCanonical::<i16>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrNonCanonical::I32(_) => CsrNonCanonical::<i32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrNonCanonical::I64(_) => CsrNonCanonical::<i64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrNonCanonical::F32(_) => CsrNonCanonical::<f32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrNonCanonical::F64(_) => CsrNonCanonical::<f64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrNonCanonical::Bool(_) => CsrNonCanonical::<bool>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCsrNonCanonical::String(_) => CsrNonCanonical::<String>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
        }
    }
}

impl<T: BackendData> ArrayChunk for CsrNonCanonical<T> {
    fn write_by_chunk<B, G, I>(mut iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let mut group = location.new_group(name)?;
        group.new_attr("encoding-type", "csr_matrix")?;
        group.new_attr("encoding-version", "0.1.0")?;
        group.new_attr("h5sparse_format", "csr")?;

        let mut data: ExtendableDataset<B, T> =
            ExtendableDataset::with_capacity(&group, "data", 1000.into())?;
        let mut indices: ExtendableDataset<B, i64> =
            ExtendableDataset::with_capacity(&group, "indices", 1000.into())?;
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
                indices.extend(
                    0,
                    ArrayView1::from_shape(indices_.len(), indices_)?
                        .mapv(|x| i64::try_from(x).unwrap())
                        .view(),
                )
            } else {
                bail!("All matrices must have the same number of columns");
            }
        })?;

        indices.finish()?;
        data.finish()?;
        indptr.push(nnz);
        group.new_array_dataset("indptr", indptr.into(), Default::default())?;
        group.new_attr(
            "shape",
            [num_rows as u64, num_cols.unwrap_or(0) as u64].as_slice(),
        )?;
        Ok(DataContainer::Group(group))
    }
}

impl ArrayChunk for DynCscMatrix {
    fn write_by_chunk<B, G, I>(iter: I, location: &G, name: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        let mut iter = iter.peekable();
        match iter.peek().context("input iterator is empty")? {
            DynCscMatrix::U8(_) => {
                CscMatrix::<u8>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            DynCscMatrix::U16(_) => CscMatrix::<u16>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCscMatrix::U32(_) => CscMatrix::<u32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCscMatrix::U64(_) => CscMatrix::<u64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCscMatrix::I8(_) => {
                CscMatrix::<i8>::write_by_chunk(iter.map(|x| x.try_into().unwrap()), location, name)
            }
            DynCscMatrix::I16(_) => CscMatrix::<i16>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCscMatrix::I32(_) => CscMatrix::<i32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCscMatrix::I64(_) => CscMatrix::<i64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCscMatrix::F32(_) => CscMatrix::<f32>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCscMatrix::F64(_) => CscMatrix::<f64>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCscMatrix::Bool(_) => CscMatrix::<bool>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
            DynCscMatrix::String(_) => CscMatrix::<String>::write_by_chunk(
                iter.map(|x| x.try_into().unwrap()),
                location,
                name,
            ),
        }
    }
}

impl<T: BackendData + Scalar> ArrayChunk for CscMatrix<T> {
    // TODO! more efficent way should be implement
    // Ok(iter.reduce(|acc, x| vstack_csc(acc, x)).unwrap())
    /*
    Ok(iter.map(|csc| csc.transpose())
           .reduce(|acc, x| hstack_csc(acc, x))
           .unwrap()
           .transpose())
    */

    fn write_by_chunk<B, G, I>(_: I, _: &G, _: &str) -> Result<DataContainer<B>>
    where
        I: Iterator<Item = Self>,
        B: Backend,
        G: GroupOp<B>,
    {
        todo!()
        /*
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
        */
    }
}
