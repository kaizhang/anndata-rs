use crate::utils::{create_str_attr, read_str_attr, COMPRESSION};

use ndarray::{Array1, Array, Dimension};
use hdf5::{H5Type, Result, Group, Dataset,
    types::TypeDescriptor,
};
use hdf5::types::TypeDescriptor::*;
use nalgebra_sparse::csr::CsrMatrix;

#[derive(Debug, PartialEq)]
pub enum DataType {
    CsrMatrix(TypeDescriptor),
    CscMatrix(TypeDescriptor),
    Vector(TypeDescriptor),
    Array(TypeDescriptor),
    StringVector,
    Unknown,
}

pub trait DataContainer {
    fn container_type(&self) -> &str;

    fn _encoding_type(&self) -> Result<String>;
}

impl DataContainer for Group {
    fn container_type(&self) -> &str { "group" }
    
    fn _encoding_type(&self) -> Result<String> {
        read_str_attr(self, "encoding-type")
    }
}

impl DataContainer for Dataset {
    fn container_type(&self) -> &str { "dataset" }

    fn _encoding_type(&self) -> Result<String> {
        read_str_attr(self, "encoding-type")
    }
}

impl dyn DataContainer {
    pub fn get_encoding_type(&self) -> Result<DataType> {
        match self._encoding_type().unwrap_or("array".to_string()).as_ref() {
            "array" => {
                let dataset: &Dataset = self.as_ref();
                let ty = dataset.dtype()?.to_descriptor()?;
                Ok(DataType::Array(ty))
            }
            "csr_matrix" => {
                let group: &Group = self.as_ref();
                let ty = group.dataset("data")?.dtype()?.to_descriptor()?;
                Ok(DataType::CsrMatrix(ty))
            },
            "csc_matrix" => {
                let group: &Group = self.as_ref();
                let ty = group.dataset("data")?.dtype()?.to_descriptor()?;
                Ok(DataType::CscMatrix(ty))
            }
            _ => todo!()
        }
    }
}

impl AsRef<Group> for dyn DataContainer {
    fn as_ref(&self) -> &Group {
        if self.container_type() == "group" {
            unsafe { &*(self as *const dyn DataContainer as *const Group) }
        } else {
            panic!(
                "implementation error, cannot get ref Group from {:?}",
                self.container_type(),
            )
        }
    }
}

impl AsRef<Dataset> for dyn DataContainer {
    fn as_ref(&self) -> &Dataset {
        if self.container_type() == "dataset" {
            unsafe { &*(self as *const dyn DataContainer as *const Dataset) }
        } else {
            panic!(
                "implementation error, cannot get ref Dataset from {:?}",
                self.container_type(),
            )
        }
    }
}

pub trait DataIO {
    fn write(&self, location: &Group, name: &str) -> Result<Box<dyn DataContainer>>;

    fn read(container: &Box<dyn DataContainer>) -> Result<Self> where Self: Sized;

    fn version(&self) -> &str;

    fn get_dtype(&self) -> DataType;

    fn dtype() -> DataType where Self: Sized;
}

#[derive(Clone)]
pub struct StrVec(pub Vec<String>);

impl<T> DataIO for CsrMatrix<T>
where
    T: H5Type,
{
    fn write(&self, location: &Group, name: &str) -> Result<Box<dyn DataContainer>>
    {
        let group = location.create_group(name)?;
        create_str_attr(&group, "encoding-type", "csr_matrix")?;
        create_str_attr(&group, "encoding-version", self.version())?;

        group.new_attr_builder()
            .with_data(&[self.nrows(), self.ncols()]).create("shape")?;
        group.new_dataset_builder().deflate(COMPRESSION)
            .with_data(self.values()).create("data")?;

        // TODO: fix index type
        let indices: Array1<i32> = self.col_indices().iter()
            .map(|x| *x as i32).collect(); // scipy compatibility
        group.new_dataset_builder().deflate(COMPRESSION)
            .with_data(&indices).create("indices")?;

        let indptr: Array1<i32> = self.row_offsets().iter()
            .map(|x| *x as i32).collect();  // scipy compatibility
        group.new_dataset_builder().deflate(COMPRESSION)
            .with_data(&indptr).create("indptr")?;

        Ok(Box::new(group))
    }

    fn read(container: &Box<dyn DataContainer>) -> Result<Self> where Self: Sized {
        let dataset: &Group = container.as_ref().as_ref();
        let shape: Vec<usize> = dataset.attr("shape")?.read_raw()?;
        let data = dataset.dataset("data")?.read_raw()?;
        let indices: Vec<usize> = dataset.dataset("indices")?.read_raw()?;
        let indptr: Vec<usize> = dataset.dataset("indptr")?.read_raw()?;

        match dataset._encoding_type()?.as_str() {
            "csr_matrix" => Ok(CsrMatrix::try_from_csr_data(
                shape[0],
                shape[1],
                indptr,
                indices,
                data,
            ).unwrap()),
            _ => Err(hdf5::Error::from("not a csr matrix!")),
        }
    }

    fn get_dtype(&self) -> DataType { DataType::CsrMatrix(T::type_descriptor()) }
    fn dtype() -> DataType { DataType::CsrMatrix(T::type_descriptor()) }
    fn version(&self) -> &str { "0.1.0" }
}

impl<T, D> DataIO for Array<T, D>
where
    T: H5Type,
    D: Dimension,
{
    fn write(&self, location: &Group, name: &str) -> Result<Box<dyn DataContainer>>
    {
        let dataset = location.new_dataset_builder().deflate(COMPRESSION)
            .with_data(self).create(name)?;

        create_str_attr(&*dataset, "encoding-type", "array")?;
        create_str_attr(&*dataset, "encoding-version", self.version())?;
        Ok(Box::new(dataset))
    }

    fn read(container: &Box<dyn DataContainer>) -> Result<Self> where Self: Sized {
        let dataset: &Dataset = container.as_ref().as_ref();
        dataset.read()
    }

    fn get_dtype(&self) -> DataType { DataType::Array(T::type_descriptor()) }
    fn dtype() -> DataType { DataType::Array(T::type_descriptor()) }
    fn version(&self) -> &str { "0.2.0" }
}

pub fn read_dyn_data(container: &Box<dyn DataContainer>) -> Result<Box<dyn DataIO>> {
    let err = Err(hdf5::Error::Internal("not".to_string()));
    match container.as_ref().get_encoding_type()? {
        DataType::CsrMatrix(Integer(_)) => {
            let mat: CsrMatrix<i64> = DataIO::read(container)?;
            Ok(Box::new(mat))
        },
        DataType::CsrMatrix(Unsigned(_)) => {
            let mat: CsrMatrix<u64> = DataIO::read(container)?;
            Ok(Box::new(mat))
        },
        DataType::CsrMatrix(Float(_)) => {
            let mat: CsrMatrix<f64> = DataIO::read(container)?;
            Ok(Box::new(mat))
        },
        _ => err?,
    }
}


/*
impl<D: H5Type + Clone> AnnDataType for Vec<D> {
    type Container = Dataset;

    fn write(&self, location: &Group, name: &str) -> Result<Self::Container>
    {
        let dataset = location.new_dataset_builder().deflate(COMPRESSION)
            .with_data(&arr1(self)).create(name)?;
        create_str_attr(&*dataset, "encoding-type", "array")?;
        create_str_attr(&*dataset, "encoding-version", self.version())?;
        Ok(dataset)
    }

    fn read(dataset: &Self::Container) -> Result<Self> {
        let data: Array1<D> = dataset.read_1d()?;
        Ok(data.to_vec())
    }

    fn update(&self, container: &Self::Container) -> Result<()> {
        container.resize(self.len())?;
        container.write(&arr1(self))
    }

    fn dtype(&self) -> DataType {
        DataType::Vector(D::type_descriptor())
    }

    fn version(&self) -> &str { "0.1.0" }
}

impl AnnDataType for StrVec {
    type Container = Dataset;

    fn write(&self, location: &Group, name: &str) -> Result<Self::Container>
    {
        let data: Array1<VarLenUnicode> = self.0.iter()
            .map(|x| x.parse::<VarLenUnicode>().unwrap()).collect();
        let dataset = location.new_dataset_builder().deflate(COMPRESSION)
            .with_data(&data).create(name)?;
        create_str_attr(&*dataset, "encoding-type", "string-array")?;
        create_str_attr(&*dataset, "encoding-version", self.version())?;
        Ok(dataset)
    }

    fn read(dataset: &Self::Container) -> Result<Self> {
        let data: Array1<VarLenUnicode> = dataset.read_1d()?;
        Ok(StrVec(data.into_iter().map(|x| x.parse().unwrap()).collect()))
    }

    fn update(&self, container: &Self::Container) -> Result<()> {
        let data: Array1<VarLenUnicode> = self.0.iter()
            .map(|x| x.parse::<VarLenUnicode>().unwrap()).collect();
        container.resize(self.0.len())?;
        container.write(&data)
    }

    fn dtype(&self) -> DataType { DataType::StringVector }
    fn version(&self) -> &str { "0.2.0" }
}
*/


/*
impl AsRef<CsrMatrix<T>> for dyn AnnDataType {
    fn as_ref(&self) -> &CsrMatrix<T> {
        match self.dtype() {
            DataType::CsrMatrix(_) =>
                unsafe { &*(self as *const dyn AnnDataType as *const CsrMatrix<T>) },
            _ => {
                panic!(
                    "implementation error, cannot get ref Group from {:?}",
                    self.dtype(),
                )
            }
        }
    }
}
*/

pub fn downcast_anndata<T>(val: Box<dyn DataIO>) -> Box<T>
where
    T: DataIO,
{
    let ptr = Box::into_raw(val);
    let type_expected = T::dtype();
    let type_actual = unsafe { ptr.as_ref().unwrap().get_dtype() };
    if type_expected == type_actual {
        unsafe { Box::from_raw(ptr as *mut T) }
    } else {
        panic!(
            "implementation error, cannot read {:?} from {:?}",
            type_expected,
            type_actual,
        )
    }
}