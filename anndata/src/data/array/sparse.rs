use crate::backend::*;
use crate::data::{
    data_traits::*,
    scalar::DynScalar,
    slice::{Shape, SelectInfoElem},
};

use anyhow::{bail, Result};
use nalgebra_sparse::csr::CsrMatrix;

#[derive(Debug, Clone)]
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

impl WriteData for DynCsrMatrix {
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        match self {
            DynCsrMatrix::I8(data) => data.write(location, name),
            DynCsrMatrix::I16(data) => data.write(location, name),
            DynCsrMatrix::I32(data) => data.write(location, name),
            DynCsrMatrix::I64(data) => data.write(location, name),
            DynCsrMatrix::U8(data) => data.write(location, name),
            DynCsrMatrix::U16(data) => data.write(location, name),
            DynCsrMatrix::U32(data) => data.write(location, name),
            DynCsrMatrix::U64(data) => data.write(location, name),
            DynCsrMatrix::Usize(data) => data.write(location, name),
            DynCsrMatrix::F32(data) => data.write(location, name),
            DynCsrMatrix::F64(data) => data.write(location, name),
            DynCsrMatrix::Bool(data) => data.write(location, name),
            DynCsrMatrix::String(data) => data.write(location, name),
        }
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
                ScalarType::String => CsrMatrix::<String>::read(container).map(DynCsrMatrix::String),
            },
            _ => bail!("cannot read csr matrix from non-group container"),
        }
    }
}

impl HasShape for DynCsrMatrix {
    fn shape(&self) -> Shape {
        match self {
            DynCsrMatrix::I8(m) => m.shape(),
            DynCsrMatrix::I16(m) => m.shape(),
            DynCsrMatrix::I32(m) => m.shape(),
            DynCsrMatrix::I64(m) => m.shape(),
            DynCsrMatrix::U8(m) => m.shape(),
            DynCsrMatrix::U16(m) => m.shape(),
            DynCsrMatrix::U32(m) => m.shape(),
            DynCsrMatrix::U64(m) => m.shape(),
            DynCsrMatrix::Usize(m) => m.shape(),
            DynCsrMatrix::F32(m) => m.shape(),
            DynCsrMatrix::F64(m) => m.shape(),
            DynCsrMatrix::Bool(m) => m.shape(),
            DynCsrMatrix::String(m) => m.shape(),
        }
    }
}

impl ArrayOp for DynCsrMatrix {
    fn get(&self, index: &[usize]) -> Option<DynScalar> {
        todo!()
    }

    fn select<S, E>(&self, info: S) -> Self
    where
        S: AsRef<[E]>,
        E: AsRef<SelectInfoElem>,
    {
        todo!()
    }
}

impl WriteArrayData for DynCsrMatrix {}
impl ReadArrayData for DynCsrMatrix {
    fn get_shape<B: Backend>(container: &DataContainer<B>) -> Result<Shape> {
        Ok(container.as_group()?.read_arr_attr("shape")?.to_vec().into())
    }

    fn read_select<B, S, E>(container: &DataContainer<B>, info: S) -> Result<Self>
        where
            B: Backend,
            S: AsRef<[E]>,
            E: AsRef<SelectInfoElem>,
            Self: Sized {
        todo!()
    }
}

impl<T> HasShape for &CsrMatrix<T> {
    fn shape(&self) -> Shape {
        vec![self.nrows(), self.ncols()].into()
    }
}

impl<T> HasShape for CsrMatrix<T> {
    fn shape(&self) -> Shape {
        (&self).shape()
    }
}

impl<T: BackendData> WriteData for CsrMatrix<T> {
    fn write<B: Backend, G: GroupOp<Backend = B>>(
        &self,
        location: &G,
        name: &str,
    ) -> Result<DataContainer<B>> {
        (&self).write(location, name)
    }
}
 
impl<T: BackendData> WriteData for &CsrMatrix<T> {
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
        let indices: Vec<usize> = group.open_dataset("indices")?.read_array()?.to_vec();
        let indptr: Vec<usize> = group.open_dataset("indptr")?.read_array()?.to_vec();
        Ok(CsrMatrix::try_from_csr_data(shape[0], shape[1], indptr, indices, data).unwrap())
    }
}