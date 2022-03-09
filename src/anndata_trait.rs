use crate::utils::{create_str_attr, read_str_attr, COMPRESSION};

use ndarray::{arr1, Array1, Array, Dimension};
use hdf5::{
    File, H5Type, Result, Group, Dataset,
    types::{VarLenUnicode, TypeDescriptor},
};
use sprs::CsMat;

#[derive(Debug, PartialEq)]
pub enum DataType {
    CsMatrix(TypeDescriptor),
    Vector(TypeDescriptor),
    Array(TypeDescriptor),
    StringVector,
    Unknown,
}

pub trait ContainerType {
    fn container_type(&self) -> &str;

    fn _encoding_type(&self) -> Result<String>;
}

impl ContainerType for Group {
    fn container_type(&self) -> &str { "group" }
    
    fn _encoding_type(&self) -> Result<String> {
        read_str_attr(self, "encoding-type")
    }
}

impl ContainerType for Dataset {
    fn container_type(&self) -> &str { "dataset" }

    fn _encoding_type(&self) -> Result<String> {
        read_str_attr(self, "encoding-type")
    }
}

impl dyn ContainerType {
    pub fn get_encoding_type(&self) -> Result<DataType> {
        match self._encoding_type()?.as_str() {
            "csr_matrix" | "csc_matrix" => {
                let group: &Group = self.as_ref();
                let ty = group.dataset("data")?.dtype()?.to_descriptor()?;
                Ok(DataType::CsMatrix(ty))
            },
            _ => todo!()
        }
    }
}

impl AsRef<Group> for dyn ContainerType {
    fn as_ref(&self) -> &Group {
        if self.container_type() == "group" {
            unsafe { &*(self as *const dyn ContainerType as *const Group) }
        } else {
            panic!(
                "implementation error, cannot get ref Group from {:?}",
                self.container_type(),
            )
        }
    }
}

impl AsRef<Dataset> for dyn ContainerType {
    fn as_ref(&self) -> &Dataset {
        if self.container_type() == "dataset" {
            unsafe { &*(self as *const dyn ContainerType as *const Dataset) }
        } else {
            panic!(
                "implementation error, cannot get ref Dataset from {:?}",
                self.container_type(),
            )
        }
    }
}

#[derive(Clone)]
pub struct StrVec(pub Vec<String>);

pub trait AnnDataType {
    fn write(&self, location: &Group, name: &str) -> Result<Box<dyn ContainerType>>;

    fn read(container: &Box<dyn ContainerType>) -> Result<Self> where Self: Sized;

    fn version(&self) -> &str;

    fn get_dtype(&self) -> DataType;

    fn dtype() -> DataType where Self: Sized;
}

impl<T> AnnDataType for CsMat<T>
where
    T: H5Type,
{
    fn write(&self, location: &Group, name: &str) -> Result<Box<dyn ContainerType>>
    {
        let group = location.create_group(name)?;
        let encoding = if self.is_csr() { "csr_matrix" } else { "csc_matrix" };
        create_str_attr(&group, "encoding-type", encoding)?;
        create_str_attr(&group, "encoding-version", self.version())?;

        group.new_attr_builder()
            .with_data(&[self.rows(), self.cols()]).create("shape")?;
        group.new_dataset_builder().deflate(COMPRESSION)
            .with_data(self.data()).create("data")?;

        // TODO: fix index type
        let indices: Array1<i32> = self.indices().iter()
            .map(|x| *x as i32).collect(); // scipy compatibility
        group.new_dataset_builder().deflate(COMPRESSION)
            .with_data(&indices).create("indices")?;

        let indptr: Array1<i32> = self.proper_indptr().into_owned().iter()
            .map(|x| *x as i32).collect();  // scipy compatibility
        group.new_dataset_builder().deflate(COMPRESSION)
            .with_data(&indptr).create("indptr")?;

        Ok(Box::new(group))
    }

    fn read(container: &Box<dyn ContainerType>) -> Result<Self> where Self: Sized {
        let dataset: &Group = container.as_ref().as_ref();
        let shape: Vec<usize> = dataset.attr("shape")?.read_raw()?;
        let data = dataset.dataset("data")?.read_raw()?;
        let indices: Vec<usize> = dataset.dataset("indices")?.read_raw()?;
        let indptr: Vec<usize> = dataset.dataset("indptr")?.read_raw()?;

        match dataset._encoding_type()?.as_str() {
            "csr_matrix" => Ok(CsMat::new(
                (shape[0], shape[1]),
                indptr,
                indices,
                data,
            )),
            "csc_matrix" => Ok(CsMat::new_csc(
                (shape[0], shape[1]),
                indptr,
                indices,
                data,
            )),
            _ => Err(hdf5::Error::from("not a sparse matrix!")),
        }
    }

    fn get_dtype(&self) -> DataType {
        DataType::CsMatrix(T::type_descriptor())
    }

    fn dtype() -> DataType {
        DataType::CsMatrix(T::type_descriptor())
    }

    fn version(&self) -> &str {
        "0.1.0"
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

impl<A, D> AnnDataType for Array<A, D>
where
    A: H5Type,
    D: Dimension,
{
    type Container = Dataset;

    fn write(&self, location: &Group, name: &str) -> Result<Self::Container>
    {
        let dataset = location.new_dataset_builder().deflate(COMPRESSION)
            .with_data(self).create(name)?;

        create_str_attr(&*dataset, "encoding-type", "array")?;
        create_str_attr(&*dataset, "encoding-version", self.version())?;

        Ok(dataset)
    }

    fn read(dataset: &Self::Container) -> Result<Self> { dataset.read() }

    fn update(&self, container: &Self::Container) -> Result<()> {
        todo!()
    }

    fn dtype(&self) -> DataType { DataType::Array(A::type_descriptor()) }
    fn version(&self) -> &str { "0.2.0" }
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

pub trait AnnDataView: AnnDataType {
    fn read_partial(
        container: &Box<dyn ContainerType>,
        ridx: Vec<usize>,
        cidx: Vec<usize>,
    ) -> Result<Self> where Self: Sized;

    fn to_view(&self, ridx: Vec<usize>, cidx: Vec<usize>) -> Box<dyn ContainerType>;
}

impl<T> AnnDataView for CsMat<T>
where
    T: H5Type,
{
    fn read_partial(
        container: &Box<dyn ContainerType>,
        ridx: Vec<usize>,
        cidx: Vec<usize>,
    ) -> Result<Self> where Self: Sized {
        todo!()
    }

    fn to_view(&self, ridx: Vec<usize>, cidx: Vec<usize>) -> Box<dyn ContainerType> {

    }

pub fn downcast_anndata<T>(val: Box<dyn AnnDataType>) -> Box<T>
where
    T: AnnDataType,
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