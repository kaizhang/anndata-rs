use crate::utils::{create_str_attr, read_str_attr, COMPRESSION};

use itertools::zip;
use ndarray::{Axis, Array1, Array2, Array, Dimension};
use hdf5::{H5Type, Result, Group, Dataset,
    types::TypeDescriptor,
};
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

impl<T> AnnDataType for CsrMatrix<T>
where
    T: H5Type,
{
    fn write(&self, location: &Group, name: &str) -> Result<Box<dyn ContainerType>>
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

    fn read(container: &Box<dyn ContainerType>) -> Result<Self> where Self: Sized {
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

impl<T, D> AnnDataType for Array<T, D>
where
    T: H5Type,
    D: Dimension,
{
    fn write(&self, location: &Group, name: &str) -> Result<Box<dyn ContainerType>>
    {
        let dataset = location.new_dataset_builder().deflate(COMPRESSION)
            .with_data(self).create(name)?;

        create_str_attr(&*dataset, "encoding-type", "array")?;
        create_str_attr(&*dataset, "encoding-version", self.version())?;
        Ok(Box::new(dataset))
    }

    fn read(container: &Box<dyn ContainerType>) -> Result<Self> where Self: Sized {
        let dataset: &Dataset = container.as_ref().as_ref();
        dataset.read()
    }

    fn get_dtype(&self) -> DataType { DataType::Array(T::type_descriptor()) }
    fn dtype() -> DataType { DataType::Array(T::type_descriptor()) }
    fn version(&self) -> &str { "0.2.0" }
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

pub trait AnnDataGetRow: AnnDataType {
    fn read_rows(
        container: &Box<dyn ContainerType>,
        idx: Option<&[usize]>,
    ) -> Self
    where Self: Sized,
    {
        let x: Self = AnnDataType::read(container).unwrap();
        x.get_rows(idx)
    }

    fn get_rows(&self, idx: Option<&[usize]>) -> Self where Self: Sized;
}

impl<T> AnnDataGetRow for Array2<T>
where
    T: H5Type + Clone,
{
    fn get_rows(&self, idx: Option<&[usize]>) -> Self {
        match idx {
            None => self.clone(),
            Some(i) => self.select(Axis(0), i),
        }
    }
}

impl<T> AnnDataGetRow for CsrMatrix<T>
where
    T: H5Type + Clone + Copy,
{
    fn get_rows(&self, idx: Option<&[usize]>) -> Self {
        match idx {
            None => self.clone(),
            Some(i) => {
                create_csr_from_rows(i.iter().map(|r| {
                    let row = self.get_row(*r).unwrap();
                    zip(row.col_indices(), row.values())
                        .map(|(x, y)| (*x, *y)).collect()
                }),
                self.ncols()
                )
            }
        }
    }
}

pub trait AnnDataGetCol: AnnDataType {
    fn read_columns(
        container: &Box<dyn ContainerType>,
        idx: Option<&[usize]>,
    ) -> Self
    where Self: Sized,
    {
        let x: Self = AnnDataType::read(container).unwrap();
        x.get_columns(idx)
    }

    fn get_columns(&self, idx: Option<&[usize]>) -> Self where Self: Sized;
}

impl<T> AnnDataGetCol for Array2<T>
where
    T: H5Type + Clone,
{
    fn get_columns(&self, idx: Option<&[usize]>) -> Self {
        match idx {
            None => self.clone(),
            Some(i) => self.select(Axis(1), i),
        }
    }
}

impl<T> AnnDataGetCol for CsrMatrix<T>
where
    T: H5Type + Clone,
{
    fn get_columns(&self, idx: Option<&[usize]>) -> Self {
        todo!()
    }
}

pub trait AnnDataSubset: AnnDataGetRow + AnnDataGetCol {
    fn read_partial(
        container: &Box<dyn ContainerType>,
        ridx: Option<&[usize]>,
        cidx: Option<&[usize]>,
    ) -> Self
    where Self: Sized,
    {
        let x: Self = AnnDataGetRow::read_rows(container, ridx);
        match cidx {
            None => x,
            _ => x.get_columns(cidx),
        }
    }

    fn subset(
        &self,
        ridx: Option<&[usize]>,
        cidx: Option<&[usize]>
    ) -> Self
    where Self: Sized,
    {
        self.get_rows(ridx).get_columns(cidx)
    }
}

impl<T> AnnDataSubset for CsrMatrix<T> where T: H5Type + Clone + Copy, {}
impl<T> AnnDataSubset for Array2<T> where T: H5Type + Clone, {}

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

/*
pub fn cast_datatype(val: Box<dyn AnnDataType>) -> Box<dyn AnnDataSubset> {
    let ptr = Box::into_raw(val);
    let ty = unsafe { ptr.as_ref().unwrap().get_dtype() };
    if true {
        unsafe { Box::from_raw(ptr as *mut dyn AnnDataSubset) }
    } else {
        panic!(
            "implementation error, cannot read {:?}",
            ty,
        )
    }
}
*/

fn create_csr_from_rows<I, T>(iter: I, num_col: usize) -> CsrMatrix<T>
where
    I: Iterator<Item = Vec<(usize, T)>>,
    T: H5Type,
{
    let mut data: Vec<T> = Vec::new();
    let mut indices: Vec<usize> = Vec::new();
    let mut indptr: Vec<usize> = Vec::new();

    let n = iter.fold(0, |r_idx, row| {
        indptr.push(r_idx);
        let new_idx = r_idx + row.len();
        let (mut a, mut b) = row.into_iter().unzip();
        indices.append(&mut a);
        data.append(&mut b);
        new_idx
    });
    indptr.push(n);
    CsrMatrix::try_from_csr_data(indptr.len() - 1, num_col, indptr, indices, data).unwrap()
}

