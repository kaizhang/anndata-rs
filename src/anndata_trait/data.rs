use crate::utils::{
    create_str_attr, read_str_attr, read_str_vec_attr, read_str_vec, COMPRESSION};

use ndarray::{Array1, ArrayD, ArrayView, Dimension};
use hdf5::{
    H5Type, Result, Group, Dataset,
    types::TypeDescriptor,
    types::VarLenUnicode,
};
use hdf5::types::TypeDescriptor::*;
use hdf5::types::IntSize;
use hdf5::types::FloatSize;
use nalgebra_sparse::csr::CsrMatrix;
use polars::{
    prelude::{NamedFromOwned, NamedFrom, IntoSeries},
    series::Series,
    datatypes::CategoricalChunkedBuilder,
    frame::DataFrame,
};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct StrVector(Vec<String>);

#[derive(Debug, Clone)]
pub struct CategoricalArray {
    codes: Vec<u32>,
    categories: Vec<String>,
}

impl<'a> FromIterator<&'a str> for CategoricalArray {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a str>,
    {
        let mut str_to_id = HashMap::new();
        let mut counter = 0;
        let codes = iter.into_iter().map(|x| {
            let str = x.to_string();
            match str_to_id.get(&str) {
                Some(v) => *v,
                None => {
                    let v = counter;
                    str_to_id.insert(str, v);
                    counter += 1;
                    v
                },
            }
        }).collect();
        let mut categories = str_to_id.drain().collect::<Vec<_>>();
        categories.sort_by_key(|x| x.1);
        CategoricalArray {
            codes, categories: categories.into_iter().map(|x| x.0).collect()
        }
    }

}

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    CsrMatrix(TypeDescriptor),
    CscMatrix(TypeDescriptor),
    Array(TypeDescriptor),
    Categorical,
    DataFrame,
    StringVector,
    Unknown,
}

#[derive(Debug, Clone)]
pub enum DataContainer {
    H5Group(Group),
    H5Dataset(Dataset),
}

impl DataContainer {
    pub fn open(group: &Group, name: &str) -> Result<Self> {
        match group.dataset(name) {
            Ok(x) => Ok(DataContainer::H5Dataset(x)),
            _ => match group.group(name) {
                Ok(x) => Ok(DataContainer::H5Group(x)),
                _ => Err(hdf5::Error::Internal(format!(
                    "Cannot open '{}' as group or dataset", name
                ))),
            },
        }
    }

    fn _encoding_type(&self) -> Result<String> {
        match self {
            Self::H5Group(group) => read_str_attr(group, "encoding-type"),
            Self::H5Dataset(dataset) => read_str_attr(dataset, "encoding-type"),
        }
    }

    pub fn get_encoding_type(&self) -> Result<DataType> {
        match self._encoding_type().unwrap_or("array".to_string()).as_ref() {
            "dataframe" => Ok(DataType::DataFrame),
            "categorical" => Ok(DataType::Categorical),
            "array" | "string-array" => {
                let dataset = self.get_dataset_ref()?;
                let ty = dataset.dtype()?.to_descriptor()?;
                Ok(DataType::Array(ty))
            },
            "csr_matrix" => {
                let group = self.get_group_ref()?;
                let ty = group.dataset("data")?.dtype()?.to_descriptor()?;
                Ok(DataType::CsrMatrix(ty))
            },
            "csc_matrix" => {
                let group = self.get_group_ref()?;
                let ty = group.dataset("data")?.dtype()?.to_descriptor()?;
                Ok(DataType::CscMatrix(ty))
            },
            ty => Err(hdf5::Error::Internal(format!(
                "type '{}' is not supported", ty 
            ))),
        }
    }

    pub fn get_group_ref(&self) -> Result<&Group> {
        match self {
            Self::H5Group(x) => Ok(&x),
            _ => Err(hdf5::Error::Internal(format!(
                "Expecting Group" 
            ))),
        }
    }

    pub fn get_dataset_ref(&self) -> Result<&Dataset> {
        match self {
            Self::H5Dataset(x) => Ok(&x),
            _ => Err(hdf5::Error::Internal(format!(
                "Expecting Dataset" 
            ))),
        }
    }
}

pub trait WriteData {
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer>;

    fn version(&self) -> &str;

    fn get_dtype(&self) -> DataType;

    fn dtype() -> DataType where Self: Sized;

    fn update(&self, container: &DataContainer) -> Result<DataContainer> {
        match container {
            DataContainer::H5Group(grp) => {
                let file = grp.file()?;
                let name = grp.name();
                let (path, obj) = name.as_str().rsplit_once("/")
                    .unwrap_or(("", name.as_str()));
                if path.is_empty() {
                    file.unlink(obj)?;
                    self.write(&file, obj)
                } else {
                    let g = file.group(path)?;
                    g.unlink(obj)?;
                    self.write(&g, obj)
                }
            },
            DataContainer::H5Dataset(data) => {
                let file = data.file()?;
                let name = data.name();
                let (path, obj) = name.as_str().rsplit_once("/").unwrap();
                let g = file.group(path)?;
                g.unlink(obj)?;
                self.write(&g, obj)
            },
        }
    }
}

pub trait ReadData {
    fn read(container: &DataContainer) -> Result<Self> where Self: Sized;
}

impl WriteData for CategoricalArray {
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer> {
        let group = location.create_group(name)?;
        create_str_attr(&group, "encoding-type", "categorical")?;
        create_str_attr(&group, "encoding-version", self.version())?;
        group.new_dataset_builder().deflate(COMPRESSION)
            .with_data(self.codes.as_slice()).create("codes")?;
        let cat: Vec<VarLenUnicode> = self.categories.iter()
            .map(|x| x.parse().unwrap()).collect();
        group.new_dataset_builder().deflate(COMPRESSION)
            .with_data(cat.as_slice()).create("categories")?;
        Ok(DataContainer::H5Group(group))
    }

    fn get_dtype(&self) -> DataType { DataType::Categorical }
    fn dtype() -> DataType { DataType::Categorical }
    fn version(&self) -> &str { "0.2.0" }
}

impl ReadData for CategoricalArray {
    fn read(container: &DataContainer) -> Result<Self> where Self: Sized {
        let group : &Group = container.get_group_ref()?;
        let categories: Vec<String> = read_str_vec(&group.dataset("categories")?)?;
        let codes: Vec<u32> = group.dataset("codes")?.read_raw()?;

        Ok(CategoricalArray { categories, codes })
    }
}

impl<T> WriteData for CsrMatrix<T>
where
    T: H5Type + Clone + Send + Sync,
{
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer>
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

        Ok(DataContainer::H5Group(group))
    }

    fn get_dtype(&self) -> DataType { DataType::CsrMatrix(T::type_descriptor()) }
    fn dtype() -> DataType { DataType::CsrMatrix(T::type_descriptor()) }
    fn version(&self) -> &str { "0.1.0" }
}

impl<T> ReadData for CsrMatrix<T>
where
    T: H5Type + Clone + Send + Sync,
{
    fn read(container: &DataContainer) -> Result<Self> where Self: Sized {
        let dataset: &Group = container.get_group_ref()?;
        let shape: Vec<usize> = dataset.attr("shape")?.read_raw()?;
        let data = dataset.dataset("data")?.read_raw()?;
        let indices: Vec<usize> = dataset.dataset("indices")?.read_raw()?;
        let indptr: Vec<usize> = dataset.dataset("indptr")?.read_raw()?;

        match container._encoding_type()?.as_str() {
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
}

impl<T> WriteData for ArrayD<T>
where
    T: H5Type + Clone + Send + Sync,
{
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer>
    {
        let dataset = location.new_dataset_builder().deflate(COMPRESSION)
            .with_data(self).create(name)?;

        create_str_attr(&*dataset, "encoding-type", "array")?;
        create_str_attr(&*dataset, "encoding-version", self.version())?;
        Ok(DataContainer::H5Dataset(dataset))
    }

    fn get_dtype(&self) -> DataType { DataType::Array(T::type_descriptor()) }
    fn dtype() -> DataType { DataType::Array(T::type_descriptor()) }
    fn version(&self) -> &str { "0.2.0" }
}

impl<T> ReadData for ArrayD<T>
where
    T: H5Type + Clone + Send + Sync,
{
    fn read(container: &DataContainer) -> Result<Self> where Self: Sized {
        let dataset: &Dataset = container.get_dataset_ref()?;
        dataset.read()
    }
}

impl<T> WriteData for Vec<T>
where
    T: H5Type + Clone + Send + Sync,
{
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer>
    {
        let dataset = location.new_dataset_builder().deflate(COMPRESSION)
            .with_data(self).create(name)?;

        create_str_attr(&*dataset, "encoding-type", "array")?;
        create_str_attr(&*dataset, "encoding-version", self.version())?;
        Ok(DataContainer::H5Dataset(dataset))
    }

    fn get_dtype(&self) -> DataType { DataType::Array(T::type_descriptor()) }
    fn dtype() -> DataType { DataType::Array(T::type_descriptor()) }
    fn version(&self) -> &str { "0.2.0" }
}

impl<T> ReadData for Vec<T>
where
    T: H5Type + Clone + Send + Sync,
{
    fn read(container: &DataContainer) -> Result<Self> where Self: Sized {
        let dataset: &Dataset = container.get_dataset_ref()?;
        let arr: Array1<_> = dataset.read()?;
        Ok(arr.into_raw_vec())
    }
}

impl WriteData for StrVector
{
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer>
    {
        let vec: Vec<VarLenUnicode> = self.0.iter()
            .map(|x| x.as_str().parse().unwrap()).collect();
        let dataset = location.new_dataset_builder().deflate(COMPRESSION)
            .with_data(vec.as_slice()).create(name)?;

        create_str_attr(&*dataset, "encoding-type", "string-array")?;
        create_str_attr(&*dataset, "encoding-version", self.version())?;
        Ok(DataContainer::H5Dataset(dataset))
    }

    fn get_dtype(&self) -> DataType { DataType::Array(VarLenUnicode) }
    fn dtype() -> DataType { DataType::Array(VarLenUnicode) }
    fn version(&self) -> &str { "0.2.0" }
}

impl ReadData for StrVector {
    fn read(container: &DataContainer) -> Result<Self> where Self: Sized {
        let dataset: &Dataset = container.get_dataset_ref()?;
        Ok(StrVector(read_str_vec(dataset)?))
    }
}

impl WriteData for DataFrame {
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer> {
        let group = location.create_group(name)?;
        create_str_attr(&group, "encoding-type", "dataframe")?;
        create_str_attr(&group, "encoding-version", self.version())?;

        let colnames = self.get_column_names();
        let index_name = colnames[0];
        create_str_attr(&group, "_index", index_name)?;

        let columns: Array1<hdf5::types::VarLenUnicode> = colnames.into_iter()
            .skip(1).map(|x| x.parse().unwrap()).collect();
        group.new_attr_builder().with_data(&columns).create("column-order")?;

        for series in self.iter() {
            let name = series.name();
            match series.dtype() {
                polars::datatypes::DataType::UInt8 =>
                    series.u8().unwrap().to_ndarray().unwrap().write(&group, name),
                polars::datatypes::DataType::UInt16 =>
                    series.u16().unwrap().to_ndarray().unwrap().write(&group, name),
                polars::datatypes::DataType::UInt32 =>
                    series.u32().unwrap().to_ndarray().unwrap().write(&group, name),
                polars::datatypes::DataType::UInt64 =>
                    series.u64().unwrap().to_ndarray().unwrap().write(&group, name),
                polars::datatypes::DataType::Int32 =>
                    series.i32().unwrap().to_ndarray().unwrap().write(&group, name),
                polars::datatypes::DataType::Int64 =>
                    series.i64().unwrap().to_ndarray().unwrap().write(&group, name),
                polars::datatypes::DataType::Float32 => 
                    series.f32().unwrap().to_ndarray().unwrap().write(&group, name),
                polars::datatypes::DataType::Float64 => 
                    series.f64().unwrap().to_ndarray().unwrap().write(&group, name),
                polars::datatypes::DataType::Utf8 => {
                    let vec: Vec<VarLenUnicode> = series.utf8().unwrap()
                        .into_iter().map(|x| x.unwrap().parse().unwrap()).collect();
                    let dataset = group.new_dataset_builder().deflate(COMPRESSION)
                        .with_data(vec.as_slice()).create(name)?;
                    create_str_attr(&*dataset, "encoding-type", "string-array")?;
                    create_str_attr(&*dataset, "encoding-version", "0.2.0")?;
                    Ok(DataContainer::H5Dataset(dataset))
                },
                polars::datatypes::DataType::Categorical(_) => series
                    .categorical().unwrap().iter_str().map(|x| x.unwrap())
                    .collect::<CategoricalArray>().write(&group, name),
                other => Err(hdf5::Error::Internal(
                    format!("Not implemented: writing Series of type '{:?}'", other)
                )),
            }?;
        }

        Ok(DataContainer::H5Group(group))
    }

    fn get_dtype(&self) -> DataType { DataType::DataFrame }
    fn dtype() -> DataType { DataType::DataFrame }
    fn version(&self) -> &str { "0.2.0" }
}

impl ReadData for DataFrame {
    fn read(container: &DataContainer) -> Result<Self> where Self: Sized {
        let group: &Group = container.get_group_ref()?;
        let index = read_str_attr(group, "_index")?;
        let columns = if group.attr("column-order")?.size() == 0 {
            Vec::new()
        } else {
            read_str_vec_attr(group, "column-order")?
        };

        std::iter::once(index).chain(columns).map(|x| {
            let name = x.as_str();
            let data = DataContainer::open(group, name)?;
            match data.get_encoding_type()? {
                DataType::Array(Unsigned(IntSize::U4)) =>
                    Ok(Series::from_vec(name, Vec::<u32>::read(&data).unwrap())),
                DataType::Array(Unsigned(IntSize::U8)) =>
                    Ok(Series::from_vec(name, Vec::<u64>::read(&data).unwrap())),
                DataType::Array(Integer(IntSize::U4)) =>
                    Ok(Series::from_vec(name, Vec::<i32>::read(&data).unwrap())),
                DataType::Array(Integer(IntSize::U8)) =>
                    Ok(Series::from_vec(name, Vec::<i64>::read(&data).unwrap())),
                DataType::Array(Float(FloatSize::U4)) =>
                    Ok(Series::from_vec(name, Vec::<f32>::read(&data).unwrap())),
                DataType::Array(Float(FloatSize::U8)) =>
                    Ok(Series::from_vec(name, Vec::<f64>::read(&data).unwrap())),
                DataType::Array(VarLenUnicode) =>
                    Ok(Series::new(name, read_str_vec(data.get_dataset_ref()?)?.as_slice())),
                DataType::Categorical => {
                    let arr = CategoricalArray::read(&data).unwrap();
                    let mut builder = CategoricalChunkedBuilder::new(name, arr.codes.len());
                    builder.drain_iter(arr.codes.into_iter()
                        .map(|i| Some(arr.categories[i as usize].as_str()))
                    );
                    Ok(builder.finish().into_series())
                },
                unknown => Err(hdf5::Error::Internal(
                    format!("Not implemented: reading Series from type '{:?}'", unknown)
                )),
            }
        }).collect()
    }
}

/*
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
*/

impl<'a, T, D> WriteData for ArrayView<'a, T, D>
where
    D: Dimension,
    T: H5Type,
{
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer> {
        let dataset = location.new_dataset_builder().deflate(COMPRESSION)
            .with_data(self).create(name)?;

        create_str_attr(&*dataset, "encoding-type", "array")?;
        create_str_attr(&*dataset, "encoding-version", "0.2.0")?;
        Ok(DataContainer::H5Dataset(dataset))
    }

    fn get_dtype(&self) -> DataType { DataType::Array(T::type_descriptor()) }
    fn dtype() -> DataType { DataType::Array(T::type_descriptor()) }
    fn version(&self) -> &str { "0.2.0" }
}