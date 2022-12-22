use crate::backend::*;
use crate::data::{
    array::utils::{cs_major_minor_index, cs_major_index, cs_major_slice, ExtendableDataset},
    data_traits::*,
    scalar::DynScalar,
    slice::{SelectInfoElem, Shape},
    BoundedSelectInfo, BoundedSelectInfoElem,
};

use anyhow::{bail, Result};
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::pattern::SparsityPattern;
use ndarray::{ArrayView1, Ix1};

use super::slice::BoundedSlice;

#[derive(Debug, Clone, PartialEq)]
pub enum DynCsrMatrix {
    I8(CsrMatrix<i8>),
    I16(CsrMatrix<i16>),
    I32(CsrMatrix<i32>),
    I64(CsrMatrix<i64>),
    U8(CsrMatrix<u8>),
    U16(CsrMatrix<u16>),
    U32(CsrMatrix<u32>),
    U64(CsrMatrix<u64>),
    Usize(CsrMatrix<usize>),
    F32(CsrMatrix<f32>),
    F64(CsrMatrix<f64>),
    Bool(CsrMatrix<bool>),
    String(CsrMatrix<String>),
}

macro_rules! impl_into_dyn_csr {
    ($from_type:ty, $to_type:ident) => {
        impl From<CsrMatrix<$from_type>> for DynCsrMatrix {
            fn from(data: CsrMatrix<$from_type>) -> Self {
                DynCsrMatrix::$to_type(data)
            }
        }
        impl TryFrom<DynCsrMatrix> for CsrMatrix<$from_type> {
            type Error = anyhow::Error;
            fn try_from(data: DynCsrMatrix) -> Result<Self> {
                match data {
                    DynCsrMatrix::$to_type(data) => Ok(data),
                    _ => bail!("Cannot convert to CsrMatrix<$from_type>"),
                }
            }
        }
    };
}

impl_into_dyn_csr!(i8, I8);
impl_into_dyn_csr!(i16, I16);
impl_into_dyn_csr!(i32, I32);
impl_into_dyn_csr!(i64, I64);
impl_into_dyn_csr!(u8, U8);
impl_into_dyn_csr!(u16, U16);
impl_into_dyn_csr!(u32, U32);
impl_into_dyn_csr!(u64, U64);
impl_into_dyn_csr!(usize, Usize);
impl_into_dyn_csr!(f32, F32);
impl_into_dyn_csr!(f64, F64);
impl_into_dyn_csr!(bool, Bool);
impl_into_dyn_csr!(String, String);

macro_rules! impl_dyn_csr_matrix {
    ($self:expr, $fun:ident) => {
        match $self {
            DynCsrMatrix::I8(data) => $fun!(data),
            DynCsrMatrix::I16(data) => $fun!(data),
            DynCsrMatrix::I32(data) => $fun!(data),
            DynCsrMatrix::I64(data) => $fun!(data),
            DynCsrMatrix::U8(data) => $fun!(data),
            DynCsrMatrix::U16(data) => $fun!(data),
            DynCsrMatrix::U32(data) => $fun!(data),
            DynCsrMatrix::U64(data) => $fun!(data),
            DynCsrMatrix::Usize(data) => $fun!(data),
            DynCsrMatrix::F32(data) => $fun!(data),
            DynCsrMatrix::F64(data) => $fun!(data),
            DynCsrMatrix::Bool(data) => $fun!(data),
            DynCsrMatrix::String(data) => $fun!(data),
        }
    };
}

impl WriteData for DynCsrMatrix {
    fn data_type(&self) -> DataType {
        match self {
            DynCsrMatrix::I8(csr) => csr.data_type(),
            DynCsrMatrix::I16(csr) => csr.data_type(),
            DynCsrMatrix::I32(csr) => csr.data_type(),
            DynCsrMatrix::I64(csr) => csr.data_type(),
            DynCsrMatrix::U8(csr) => csr.data_type(),
            DynCsrMatrix::U16(csr) => csr.data_type(),
            DynCsrMatrix::U32(csr) => csr.data_type(),
            DynCsrMatrix::U64(csr) => csr.data_type(),
            DynCsrMatrix::Usize(csr) => csr.data_type(),
            DynCsrMatrix::F32(csr) => csr.data_type(),
            DynCsrMatrix::F64(csr) => csr.data_type(),
            DynCsrMatrix::Bool(csr) => csr.data_type(),
            DynCsrMatrix::String(csr) => csr.data_type(),
        }
    }
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        macro_rules! write_data {
            ($data:expr) => {
                $data.write(location, name)
            };
        }
        impl_dyn_csr_matrix!(self, write_data)
    }
}

impl ReadData for DynCsrMatrix {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        match container {
            DataContainer::Group(group) => match group.open_dataset("data")?.dtype()? {
                ScalarType::I8 => CsrMatrix::<i8>::read(container).map(DynCsrMatrix::I8),
                ScalarType::I16 => CsrMatrix::<i16>::read(container).map(DynCsrMatrix::I16),
                ScalarType::I32 => CsrMatrix::<i32>::read(container).map(DynCsrMatrix::I32),
                ScalarType::I64 => CsrMatrix::<i64>::read(container).map(DynCsrMatrix::I64),
                ScalarType::U8 => CsrMatrix::<u8>::read(container).map(DynCsrMatrix::U8),
                ScalarType::U16 => CsrMatrix::<u16>::read(container).map(DynCsrMatrix::U16),
                ScalarType::U32 => CsrMatrix::<u32>::read(container).map(DynCsrMatrix::U32),
                ScalarType::U64 => CsrMatrix::<u64>::read(container).map(DynCsrMatrix::U64),
                ScalarType::Usize => CsrMatrix::<usize>::read(container).map(DynCsrMatrix::Usize),
                ScalarType::F32 => CsrMatrix::<f32>::read(container).map(DynCsrMatrix::F32),
                ScalarType::F64 => CsrMatrix::<f64>::read(container).map(DynCsrMatrix::F64),
                ScalarType::Bool => CsrMatrix::<bool>::read(container).map(DynCsrMatrix::Bool),
                ScalarType::String => {
                    CsrMatrix::<String>::read(container).map(DynCsrMatrix::String)
                }
            },
            _ => bail!("cannot read csr matrix from non-group container"),
        }
    }
}

impl HasShape for DynCsrMatrix {
    fn shape(&self) -> Shape {
        macro_rules! shape {
            ($data:expr) => {
                $data.shape()
            };
        }
        impl_dyn_csr_matrix!(self, shape)
    }
}

impl ArrayOp for DynCsrMatrix {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        macro_rules! get {
            ($data:expr) => {
                $data.get(index)
            };
        }
        impl_dyn_csr_matrix!(self, get)
    }

    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        macro_rules! select {
            ($data:expr) => {
                $data.select(info).into()
            };
        }
        impl_dyn_csr_matrix!(self, select)
    }
}

impl WriteArrayData for DynCsrMatrix {}
impl ReadArrayData for DynCsrMatrix {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .read_arr_attr("shape")?
            .to_vec()
            .into())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        if let DataType::CsrMatrix(ty) = container.encoding_type()? {
            match ty {
                ScalarType::I8 => CsrMatrix::<i8>::read_select(container, info)
                    .map(Into::into),
                ScalarType::I16 => CsrMatrix::<i16>::read_select(container, info)
                    .map(Into::into),
                ScalarType::I32 => CsrMatrix::<i32>::read_select(container, info)
                    .map(Into::into),
                ScalarType::I64 => CsrMatrix::<i64>::read_select(container, info)
                    .map(Into::into),
                ScalarType::U8 => CsrMatrix::<u8>::read_select(container, info)
                    .map(Into::into),
                ScalarType::U16 => CsrMatrix::<u16>::read_select(container, info)
                    .map(Into::into),
                ScalarType::U32 => CsrMatrix::<u32>::read_select(container, info)
                    .map(Into::into),
                ScalarType::U64 => CsrMatrix::<u64>::read_select(container, info) 
                    .map(Into::into),
                ScalarType::Usize => CsrMatrix::<usize>::read_select(container, info)
                    .map(Into::into),
                ScalarType::F32 => CsrMatrix::<f32>::read_select(container, info)
                    .map(Into::into),
                ScalarType::F64 => CsrMatrix::<f64>::read_select(container, info)
                    .map(Into::into),
                ScalarType::Bool => CsrMatrix::<bool>::read_select(container, info)
                    .map(Into::into),
                ScalarType::String => CsrMatrix::<String>::read_select(container, info)
                    .map(Into::into),
            }
        } else {
            bail!("the container does not contain a csr matrix");
        }
    }
}

impl<T> HasShape for CsrMatrix<T> {
    fn shape(&self) -> Shape {
        vec![self.nrows(), self.ncols()].into()
    }
}

impl<T: Clone> ArrayOp for CsrMatrix<T> {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        todo!()
        /*
        ensure!(index.len() == 2, "index must have length 2");
        self.get_entry(index[0], index[1]).map(|x| DynScalar::from(x.into_value()))
        */
    }

    fn select<S>(&self, info: &[S]) -> Self
    where
        S: AsRef<SelectInfoElem>,
    {
        let info = BoundedSelectInfo::new(&info, &self.shape());
        if info.ndim() != 2 {
            panic!("index must have length 2");
        }
        let row_idx = &info.as_ref()[0];
        let col_idx = &info.as_ref()[1];
        let (row_offsets, col_indices, data) = self.csr_data();
        let (new_row_offsets, new_col_indices, new_data) = if col_idx.is_full(info.in_shape()[1]) {
            match row_idx {
                &BoundedSelectInfoElem::Slice(BoundedSlice { step, start, end }) => {
                    if step == 1 {
                        let (offsets, indices, data) =
                            cs_major_slice(start, end, row_offsets, col_indices, data);
                        (
                            offsets,
                            indices.iter().copied().collect(),
                            data.iter().cloned().collect(),
                        )
                    } else if step < 0 {
                        cs_major_index(
                            (start..end).step_by(step.abs() as usize).rev(),
                            row_offsets,
                            col_indices,
                            data,
                        )
                    } else {
                        cs_major_index(
                            (start..end).step_by(step as usize),
                            row_offsets,
                            col_indices,
                            data,
                        )
                    }
                }
                BoundedSelectInfoElem::Index(idx) => {
                    cs_major_index(idx.iter().copied(), row_offsets, col_indices, data)
                }
            }
        } else {
            match row_idx {
                &BoundedSelectInfoElem::Slice(BoundedSlice { start: row_start,end: row_end, step: row_step }) => {
                    if row_step < 0 {
                        match col_idx {
                            &BoundedSelectInfoElem::Slice(col) => {
                                if col.step < 0 {
                                    cs_major_minor_index(
                                        (row_start..row_end).step_by(row_step.abs() as usize).rev(),
                                        (col.start..col.end).step_by(col.step.abs() as usize).rev(),
                                        self.ncols(),
                                        row_offsets,
                                        col_indices,
                                        data,
                                    )
                                } else {
                                    cs_major_minor_index(
                                        (row_start..row_end).step_by(row_step.abs() as usize).rev(),
                                        (col.start..col.end).step_by(col.step as usize),
                                        self.ncols(),
                                        row_offsets,
                                        col_indices,
                                        data,
                                    )
                                }
                            }
                            BoundedSelectInfoElem::Index(idx) => cs_major_minor_index(
                                (row_start..row_end).step_by(row_step.abs() as usize).rev(),
                                idx.iter().copied(),
                                self.ncols(),
                                row_offsets,
                                col_indices,
                                data,
                            ),
                        }
                    } else {
                        match col_idx {
                            &BoundedSelectInfoElem::Slice(col) => {
                                if col.step < 0 {
                                    cs_major_minor_index(
                                        (row_start..row_end).step_by(row_step as usize),
                                        (col.start..col.end).step_by(col.step.abs() as usize).rev(),
                                        self.ncols(),
                                        row_offsets,
                                        col_indices,
                                        data,
                                    )
                                } else {
                                    cs_major_minor_index(
                                        (row_start..row_end).step_by(row_step as usize),
                                        (col.start..col.end).step_by(col.step as usize),
                                        self.ncols(),
                                        row_offsets,
                                        col_indices,
                                        data,
                                    )
                                }
                            }
                            BoundedSelectInfoElem::Index(idx) => cs_major_minor_index(
                                (row_start..row_end).step_by(row_step as usize),
                                idx.iter().copied(),
                                self.ncols(),
                                row_offsets,
                                col_indices,
                                data,
                            ),
                        }
                    }
                }
                BoundedSelectInfoElem::Index(i) => match col_idx {
                    &BoundedSelectInfoElem::Slice(col) => {
                        if col.step < 0 {
                            cs_major_minor_index(
                                i.iter().copied(),
                                (col.start..col.end).step_by(col.step.abs() as usize).rev(),
                                self.ncols(),
                                row_offsets,
                                col_indices,
                                data,
                            )
                        } else {
                            cs_major_minor_index(
                                i.iter().copied(),
                                (col.start..col.end).step_by(col.step as usize),
                                self.ncols(),
                                row_offsets,
                                col_indices,
                                data,
                            )
                        }
                    }
                    BoundedSelectInfoElem::Index(j) => cs_major_minor_index(
                        i.iter().copied(),
                        j.iter().copied(),
                        self.ncols(),
                        row_offsets,
                        col_indices,
                        data,
                    ),
                },
            }
        };
        let out_shape = info.out_shape();
        let pattern = unsafe {
            SparsityPattern::from_offset_and_indices_unchecked(
                out_shape[0],
                out_shape[1],
                new_row_offsets,
                new_col_indices,
            )
        };
        CsrMatrix::try_from_pattern_and_values(pattern, new_data).unwrap()
    }
}


impl<T: BackendData> WriteData for CsrMatrix<T> {
    fn data_type(&self) -> DataType {
        DataType::CsrMatrix(T::DTYPE)
    }
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        let group = location.create_group(name)?;
        let shape = self.shape();

        group.write_str_attr("encoding-type", "csr_matrix")?;
        group.write_str_attr("encoding-version", "0.2.0")?;
        group.write_arr_attr("shape", shape.as_ref())?;

        group.create_array_data("data", &self.values(), Default::default())?;

        let num_cols = shape[1];
        // Use i32 or i64 as indices type in order to be compatible with scipy
        if TryInto::<i32>::try_into(num_cols.saturating_sub(1)).is_ok() {
            let try_convert_indptr: Option<Vec<i32>> = self
                .row_offsets()
                .iter()
                .map(|x| (*x).try_into().ok())
                .collect();
            if let Some(indptr_i32) = try_convert_indptr {
                group.create_array_data("indptr", &indptr_i32, Default::default())?;
                group.create_array_data(
                    "indices",
                    self.col_indices()
                        .iter()
                        .map(|x| (*x) as i32)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    Default::default(),
                )?;
            } else {
                group.create_array_data(
                    "indptr",
                    self.row_offsets()
                        .iter()
                        .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                        .collect::<Vec<_>>()
                        .as_slice(),
                    Default::default(),
                )?;
                group.create_array_data(
                    "indices",
                    self.col_indices()
                        .iter()
                        .map(|x| (*x) as i64)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    Default::default(),
                )?;
            }
        } else if TryInto::<i64>::try_into(num_cols.saturating_sub(1)).is_ok() {
            group.create_array_data(
                "indptr",
                self.row_offsets()
                    .iter()
                    .map(|x| TryInto::<i64>::try_into(*x).unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
                Default::default(),
            )?;
            group.create_array_data(
                "indices",
                self.col_indices()
                    .iter()
                    .map(|x| (*x) as i64)
                    .collect::<Vec<_>>()
                    .as_slice(),
                Default::default(),
            )?;
        } else {
            panic!(
                "The number of columns ({}) is too large to be stored as i64",
                num_cols
            );
        }

        Ok(DataContainer::Group(group))
    }
}

impl<T: BackendData> ReadData for CsrMatrix<T> {
    fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let group = container.as_group()?;
        let shape: Vec<usize> = group.read_arr_attr("shape")?.to_vec();
        let data = group.open_dataset("data")?.read_array()?.to_vec();
        let indices: Vec<usize> = read_array_as_usize::<B>(&group.open_dataset("indices")?)?;
        let indptr: Vec<usize> = read_array_as_usize::<B>(&group.open_dataset("indptr")?)?;
        Ok(CsrMatrix::try_from_csr_data(shape[0], shape[1], indptr, indices, data).unwrap())
    }
}

impl<T: BackendData> ReadArrayData for CsrMatrix<T> {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container
            .as_group()?
            .read_arr_attr("shape")?
            .to_vec()
            .into())
    }

    fn read_select<B, S>(container: &DataContainer<B>, info: &[S]) -> Result<Self>
    where
        B: Backend,
        S: AsRef<SelectInfoElem>,
    {
        if info.as_ref().len() != 2 {
            panic!("index must have length 2");
        }
        Ok(Self::read(container)?.select(info))
    }
}

impl<T: BackendData> WriteArrayData for &CsrMatrix<T> {}
impl<T: BackendData> WriteArrayData for CsrMatrix<T> {}


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

fn read_array_as_usize<B: Backend>(container: &B::Dataset) -> Result<Vec<usize>> {
    match container.dtype()? {
        ScalarType::U8 => Ok(container
            .read_array::<u8, Ix1>()?
            .map(|x| *x as usize)
            .to_vec()),
        ScalarType::U16 => Ok(container
            .read_array::<u16, Ix1>()?
            .map(|x| *x as usize)
            .to_vec()),
        ScalarType::U32 => Ok(container
            .read_array::<u32, Ix1>()?
            .map(|x| *x as usize)
            .to_vec()),
        ScalarType::U64 => Ok(container
            .read_array::<u64, Ix1>()?
            .map(|x| *x as usize)
            .to_vec()),
        ScalarType::Usize => Ok(container.read_array::<usize, Ix1>()?.to_vec()),
        ScalarType::I8 => Ok(container
            .read_array::<i8, Ix1>()?
            .map(|x| usize::try_from(*x).unwrap())
            .to_vec()),
        ScalarType::I16 => Ok(container
            .read_array::<i16, Ix1>()?
            .map(|x| usize::try_from(*x).unwrap())
            .to_vec()),
        ScalarType::I32 => Ok(container
            .read_array::<i32, Ix1>()?
            .map(|x| usize::try_from(*x).unwrap())
            .to_vec()),
        ScalarType::I64 => Ok(container
            .read_array::<i64, Ix1>()?
            .map(|x| usize::try_from(*x).unwrap())
            .to_vec()),
        ty => bail!("cannot cast array type {} to usize", ty),
    }
}

#[cfg(test)]
mod csr_matrix_index_tests {
    use super::*;
    use crate::s;
    use nalgebra::base::DMatrix;
    use nalgebra_sparse::CooMatrix;
    use ndarray::Array;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn csr_select<I1, I2>(
        csr: &CsrMatrix<i64>,
        row_indices: I1,
        col_indices: I2,
    ) -> CsrMatrix<i64>
    where
        I1: Iterator<Item = usize>,
        I2: Iterator<Item = usize>,
    {
        let i = row_indices.collect::<Vec<_>>();
        let j = col_indices.collect::<Vec<_>>();
        let mut dm = DMatrix::<i64>::zeros(csr.nrows(), csr.ncols());
        csr.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
        CsrMatrix::from(&dm.select_rows(&i).select_columns(&j))
    }

    fn csr_select_rows<I>(csr: &CsrMatrix<i64>, row_indices: I) -> CsrMatrix<i64>
    where
        I: Iterator<Item = usize>,
    {
        let i = row_indices.collect::<Vec<_>>();
        let mut dm = DMatrix::<i64>::zeros(csr.nrows(), csr.ncols());
        csr.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
        CsrMatrix::from(&dm.select_rows(&i))
    }

    fn csr_select_cols<I>(csr: &CsrMatrix<i64>, col_indices: I) -> CsrMatrix<i64>
    where
        I: Iterator<Item = usize>,
    {
        let j = col_indices.collect::<Vec<_>>();
        let mut dm = DMatrix::<i64>::zeros(csr.nrows(), csr.ncols());
        csr.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
        CsrMatrix::from(&dm.select_columns(&j))
    }

    #[test]
    fn test_c() {
        let dense = DMatrix::from_row_slice(3, 3, &[1, 0, 3, 2, 0, 1, 0, 0, 4]);
        let csr = CsrMatrix::from(&dense);

        // Column fancy indexing
        let cidx = [1, 2, 0, 1, 1, 2];
        let mut expected = DMatrix::from_row_slice(
            3,
            6,
            &[0, 3, 1, 0, 0, 3, 0, 1, 2, 0, 0, 1, 0, 4, 0, 0, 0, 4],
        );
        let mut expected_csr = CsrMatrix::from(&expected);
        assert_eq!(csr.select(s![.., cidx.as_ref()].as_ref()), expected_csr,);

        expected = DMatrix::from_row_slice(3, 2, &[1, 0, 2, 0, 0, 0]);
        expected_csr = CsrMatrix::from(&expected);
        assert_eq!(csr.select(s![.., 0..2].as_ref()), expected_csr);

        let ridx = [1, 2, 0, 1];
        expected = DMatrix::from_row_slice(
            4,
            6,
            &[
                0, 1, 2, 0, 0, 1, 0, 4, 0, 0, 0, 4, 0, 3, 1, 0, 0, 3, 0, 1, 2, 0, 0, 1,
            ],
        );
        expected_csr = CsrMatrix::from(&expected);
        let (new_offsets, new_indices, new_data) = cs_major_minor_index(
            ridx.into_iter(),
            cidx.into_iter(),
            csr.ncols(),
            csr.row_offsets(),
            csr.col_indices(),
            csr.values(),
        );

        assert_eq!(new_offsets.as_slice(), expected_csr.row_offsets());
        assert_eq!(new_indices.as_slice(), expected_csr.col_indices());
        assert_eq!(new_data.as_slice(), expected_csr.values());
    }

    #[test]
    fn test_csr() {
        let n: usize = 200;
        let m: usize = 200;
        let nnz: usize = 1000;

        let ridx = Array::random(220, Uniform::new(0, n)).to_vec();
        let cidx = Array::random(100, Uniform::new(0, m)).to_vec();

        let row_indices = Array::random(nnz, Uniform::new(0, n)).to_vec();
        let col_indices = Array::random(nnz, Uniform::new(0, m)).to_vec();
        let values = Array::random(nnz, Uniform::new(-10000, 10000)).to_vec();

        let csr_matrix: CsrMatrix<i64> =
            (&CooMatrix::try_from_triplets(n, m, row_indices, col_indices, values).unwrap()).into();

        // Row slice
        assert_eq!(
            csr_matrix.select(s![2..177, ..].as_ref()),
            csr_select_rows(&csr_matrix, 2..177),
        );

        // Row fancy indexing
        assert_eq!(
            csr_matrix.select(s![&ridx, ..].as_ref()),
            csr_select_rows(&csr_matrix, ridx.iter().cloned()),
        );

        // Column slice
        assert_eq!(
            csr_matrix.select(s![.., 77..200].as_ref()),
            csr_select_cols(&csr_matrix, 77..200),
        );

        // Column fancy indexing
        assert_eq!(
            csr_matrix.select(s![.., &cidx].as_ref()),
            csr_select_cols(&csr_matrix, cidx.iter().cloned()),
        );

        // Both
        assert_eq!(
            csr_matrix.select(s![2..49, 0..77].as_ref()),
            csr_select(&csr_matrix, 2..49, 0..77),
        );

        assert_eq!(
            csr_matrix.select(s![2..177, &cidx].as_ref()),
            csr_select(&csr_matrix, 2..177, cidx.iter().cloned()),
        );

        assert_eq!(
            csr_matrix.select(s![&ridx, &cidx].as_ref()),
            csr_select(&csr_matrix, ridx.iter().cloned(), cidx.iter().cloned()),
        );
    }
}
