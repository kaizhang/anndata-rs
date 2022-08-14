use crate::{
    anndata_trait::{DataIO, read_dyn_data},
    utils::hdf5::{
        create_str_attr, read_str_attr, read_str_vec_attr, read_str_vec,
        create_dataset,
    },
};

use std::fmt;
use ndarray::{Array1, ArrayD, ArrayView, Dimension};
use hdf5::{
    H5Type, Group, Dataset, Result,
    types::{TypeDescriptor, VarLenUnicode, TypeDescriptor::*}
};
use nalgebra_sparse::csr::CsrMatrix;
use polars::{
    prelude::IntoSeries,
    series::Series,
    datatypes::CategoricalChunkedBuilder,
    frame::DataFrame,
};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scalar<T>(pub T);

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
    Mapping,
    CsrMatrix(TypeDescriptor),
    CscMatrix(TypeDescriptor),
    Array(TypeDescriptor),
    Categorical,
    DataFrame,
    Scalar(TypeDescriptor),
    String,
    Unknown,
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
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

    /// Determine the data type by reading the "encoding-type" metadata.
    /// If the metadata is not available, the data type is set to "mapping" for
    /// Group objects, or "scalar" for Dataset objects.
    fn _encoding_type(&self) -> String {
        match self {
            Self::H5Group(group) => read_str_attr(group, "encoding-type")
                .unwrap_or("mapping".to_string()),
            Self::H5Dataset(dataset) => match read_str_attr(dataset, "encoding-type") {
                Ok(ty) => ty,
                _ => if dataset.is_scalar() {
                    "scalar".to_string()
                } else {
                    "array".to_string()
                }
            },
        }
    }

    pub fn get_encoding_type(&self) -> Result<DataType> {
        match self._encoding_type().as_ref() {
            "mapping" => Ok(DataType::Mapping),
            "string" => Ok(DataType::String),
            "scalar" => {
                let dataset = self.get_dataset_ref()?;
                let ty = dataset.dtype()?.to_descriptor()?;
                Ok(DataType::Scalar(ty))
            },
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

    /// Delete the container.
    pub fn delete(self) -> Result<()> {
        let (file, name) = match &self {
            DataContainer::H5Group(grp) => (grp.file()?, grp.name()),
            DataContainer::H5Dataset(data) => (data.file()?, data.name()),
        };
        let (path, obj) = name.as_str().rsplit_once("/")
            .unwrap_or(("", name.as_str()));
        if path.is_empty() {
            file.unlink(obj)?;
        } else {
            let g = file.group(path)?;
            g.unlink(obj)?;
        }
        Ok(())
    }
}

pub trait WriteData {
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer>;

    fn version(&self) -> &str;

    fn get_dtype(&self) -> DataType;

    fn dtype() -> DataType where Self: Sized;

    fn update(&self, container: &DataContainer) -> Result<DataContainer> {
        let (file, name) = match container {
            DataContainer::H5Group(grp) => (grp.file()?, grp.name()),
            DataContainer::H5Dataset(data) => (data.file()?, data.name()),
        };
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
    }
}

pub trait ReadData {
    fn read(container: &DataContainer) -> Result<Self> where Self: Sized;
}

////////////////////////////////////////////////////////////////////////////////
// Scalar
////////////////////////////////////////////////////////////////////////////////
impl<T> WriteData for Scalar<T>
where
    T: H5Type + Copy,
{
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer> {
        let dataset = location.new_dataset::<T>().create(name)?;
        //create_str_attr(&*dataset, "encoding-type", "scalar")?;
        dataset.write_scalar(&self.0)?;
        Ok(DataContainer::H5Dataset(dataset))
    }

    fn get_dtype(&self) -> DataType { DataType::Scalar(T::type_descriptor()) }
    fn dtype() -> DataType { DataType::Scalar(T::type_descriptor()) }
    fn version(&self) -> &str { "0.2.0" }
}

impl<T> ReadData for Scalar<T>
where
    T: H5Type + Copy,
{
    fn read(container: &DataContainer) -> Result<Self> where Self: Sized {
        let dataset = container.get_dataset_ref()?;
        Ok(Scalar(dataset.read_scalar()?))
    }
}

////////////////////////////////////////////////////////////////////////////////
// String
////////////////////////////////////////////////////////////////////////////////
impl WriteData for String {
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer> {
        let dataset = location.new_dataset::<VarLenUnicode>().create(name)?;
        create_str_attr(&*dataset, "encoding-type", "string")?;
        let value: VarLenUnicode = self.parse().unwrap();
        dataset.write_scalar(&value)?;
        Ok(DataContainer::H5Dataset(dataset))
    }

    fn get_dtype(&self) -> DataType { DataType::String }
    fn dtype() -> DataType { DataType::String }
    fn version(&self) -> &str { "0.2.0" }
}

impl ReadData for String {
    fn read(container: &DataContainer) -> Result<Self> where Self: Sized {
        let dataset = container.get_dataset_ref()?;
        let result: VarLenUnicode = dataset.read_scalar()?;
        Ok(result.parse().unwrap())
    }
}

////////////////////////////////////////////////////////////////////////////////
// CategoricalArray
////////////////////////////////////////////////////////////////////////////////
impl WriteData for CategoricalArray {
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer> {
        let group = location.create_group(name)?;
        create_str_attr(&group, "encoding-type", "categorical")?;
        create_str_attr(&group, "encoding-version", self.version())?;
        group.new_attr::<bool>().create("ordered")?.write_scalar(&false)?;

        create_dataset(&group, "codes", self.codes.as_slice())?;
        let cat: Vec<VarLenUnicode> = self.categories.iter()
            .map(|x| x.parse().unwrap()).collect();
        create_dataset(&group, "categories", cat.as_slice())?;
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
        let codes: Vec<u32> = group.dataset("codes")?.read_1d()?.to_vec();

        Ok(CategoricalArray { categories, codes })
    }
}

////////////////////////////////////////////////////////////////////////////////
// CsrMatrix
////////////////////////////////////////////////////////////////////////////////
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
        create_dataset(&group, "data", self.values())?;

        let try_convert_indptr: Option<Vec<i32>> = self.row_offsets().iter()
            .map(|x| (*x).try_into().ok()).collect();
        match try_convert_indptr {
            Some(vec) => create_dataset(&group, "indptr", vec.as_slice())?,
            _ => create_dataset(&group, "indptr", self.row_offsets())?,
        };

        let try_convert_indices: Option<Vec<i32>> = self.col_indices().iter()
            .map(|x| (*x).try_into().ok()).collect();
        match try_convert_indices {
            Some(vec) => create_dataset(&group, "indices", vec.as_slice())?,
            _ => create_dataset(&group, "indices", self.col_indices())?,
        };

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
        let shape: Vec<usize> = dataset.attr("shape")?.read_1d()?.to_vec();
        let data = dataset.dataset("data")?.read_1d()?.to_vec();
        let indices: Vec<usize> = dataset.dataset("indices")?.read_1d()?.to_vec();
        let indptr: Vec<usize> = dataset.dataset("indptr")?.read_1d()?.to_vec();

        Ok(CsrMatrix::try_from_csr_data(
            shape[0],
            shape[1],
            indptr,
            indices,
            data,
        ).unwrap())
    }
}

////////////////////////////////////////////////////////////////////////////////
// Array
////////////////////////////////////////////////////////////////////////////////
impl<T> WriteData for ArrayD<T>
where
    T: H5Type + Clone + Send + Sync,
{
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer>
    {
        let dataset = create_dataset(location, name, &self.as_standard_layout())?;
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

////////////////////////////////////////////////////////////////////////////////
// Vector
////////////////////////////////////////////////////////////////////////////////
impl<T> WriteData for Vec<T>
where
    T: H5Type + Clone + Send + Sync,
{
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer>
    {
        let dataset = create_dataset(location, name, self)?;
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

////////////////////////////////////////////////////////////////////////////////
// DataFrame
////////////////////////////////////////////////////////////////////////////////
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

        self.iter().try_for_each(|x| x.write(&group, x.name()).map(|_| ()))?;

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
            let mut series = Series::read(&DataContainer::open(group, name)?)?;
            series.rename(name);
            Ok(series)
        }).collect()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Series
////////////////////////////////////////////////////////////////////////////////
impl WriteData for Series {
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer> {
        match self.dtype() {
            polars::datatypes::DataType::UInt8 =>
                self.u8().unwrap().to_ndarray().unwrap().write(location, name),
            polars::datatypes::DataType::UInt16 =>
                self.u16().unwrap().to_ndarray().unwrap().write(location, name),
            polars::datatypes::DataType::UInt32 =>
                self.u32().unwrap().to_ndarray().unwrap().write(location, name),
            polars::datatypes::DataType::UInt64 =>
                self.u64().unwrap().to_ndarray().unwrap().write(location, name),
            polars::datatypes::DataType::Int8 =>
                self.i8().unwrap().to_ndarray().unwrap().write(location, name),
            polars::datatypes::DataType::Int16 =>
                self.i16().unwrap().to_ndarray().unwrap().write(location, name),
            polars::datatypes::DataType::Int32 =>
                self.i32().unwrap().to_ndarray().unwrap().write(location, name),
            polars::datatypes::DataType::Int64 =>
                self.i64().unwrap().to_ndarray().unwrap().write(location, name),
            polars::datatypes::DataType::Float32 => 
                self.f32().unwrap().to_ndarray().unwrap().write(location, name),
            polars::datatypes::DataType::Float64 => 
                self.f64().unwrap().to_ndarray().unwrap().write(location, name),
            polars::datatypes::DataType::Boolean => 
                self.bool().unwrap().into_iter().flatten().collect::<Vec<_>>().write(location, name),
            polars::datatypes::DataType::Utf8 => {
                let vec: Vec<VarLenUnicode> = self.utf8().unwrap()
                    .into_iter().map(|x| x.unwrap().parse().unwrap()).collect();
                let dataset = create_dataset(location, name, vec.as_slice())?;
                create_str_attr(&*dataset, "encoding-type", "string-array")?;
                create_str_attr(&*dataset, "encoding-version", "0.2.0")?;
                Ok(DataContainer::H5Dataset(dataset))
            },
            polars::datatypes::DataType::Categorical(_) => self
                .categorical().unwrap().iter_str().map(|x| x.unwrap())
                .collect::<CategoricalArray>().write(location, name),
            other => Err(hdf5::Error::Internal(
                format!("Not implemented: writing Series of type '{:?}'", other)
            )),
        }
    }

    fn get_dtype(&self) -> DataType { todo!() }
    fn dtype() -> DataType { todo!() }
    fn version(&self) -> &str { "0.2.0" }
}

impl ReadData for Series {
    fn read(container: &DataContainer) -> Result<Self> where Self: Sized {
        macro_rules! _into_series {
            ($x:expr) => { Ok($x.iter().collect::<Series>()) };
        }

        match container.get_encoding_type()? {
            DataType::Scalar(VarLenUnicode) | DataType::Array(VarLenUnicode) =>
                Ok(read_str_vec(container.get_dataset_ref()?)?.into_iter().collect()),
            DataType::Scalar(ty) | DataType::Array(ty) => crate::proc_numeric_data!(
                ty,
                ReadData::read(container)?,
                _into_series,
                ArrayD
            ),
            DataType::Categorical => {
                let arr = CategoricalArray::read(container).unwrap();
                let mut builder = CategoricalChunkedBuilder::new("", arr.codes.len());
                builder.drain_iter(arr.codes.into_iter()
                    .map(|i| Some(arr.categories[i as usize].as_str()))
                );
                Ok(builder.finish().into_series())
            },
            unknown => Err(hdf5::Error::Internal(
                format!("Not implemented: reading Series from type '{:?}'", unknown)
            )),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Mapping
////////////////////////////////////////////////////////////////////////////////
#[derive(Clone)]
pub struct Mapping(pub HashMap<String, Box<dyn DataIO>>);

impl WriteData for Mapping {
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer> {
        let group = location.create_group(name)?;

        self.0.iter().try_for_each(|(k, v)|
            v.write(&group, k).map(|_| ())
        )?;

        Ok(DataContainer::H5Group(group))
    }

    fn get_dtype(&self) -> DataType { DataType::Mapping }
    fn dtype() -> DataType { DataType::Mapping }
    fn version(&self) -> &str { "0.2.0" }
}

impl ReadData for Mapping {
    fn read(container: &DataContainer) -> Result<Self> where Self: Sized {
        let group: &Group = container.get_group_ref()?;

        let m: Result<HashMap<_, _>>= get_all_data(group)
            .map(|(k, c)| Ok((k, read_dyn_data(&c)?))).collect();
        Ok(Mapping(m?))
    }
}

////////////////////////////////////////////////////////////////////////////////
// ArrayView
////////////////////////////////////////////////////////////////////////////////
impl<'a, T, D> WriteData for ArrayView<'a, T, D>
where
    D: Dimension,
    T: H5Type,
{
    fn write(&self, location: &Group, name: &str) -> Result<DataContainer> {
        let dataset = create_dataset(location, name, self)?;
        create_str_attr(&*dataset, "encoding-type", "array")?;
        create_str_attr(&*dataset, "encoding-version", "0.2.0")?;
        Ok(DataContainer::H5Dataset(dataset))
    }

    fn get_dtype(&self) -> DataType { DataType::Array(T::type_descriptor()) }
    fn dtype() -> DataType { DataType::Array(T::type_descriptor()) }
    fn version(&self) -> &str { "0.2.0" }
}

pub(crate) fn get_all_data(group: &Group) -> impl Iterator<Item=(String, DataContainer)> {
    let get_name = |x: String| std::path::Path::new(&x).file_name()
        .unwrap().to_str().unwrap().to_string();
    group.groups().unwrap().into_iter().map(move |x|
        (get_name(x.name()), DataContainer::H5Group(x))
    ).chain(group.datasets().unwrap().into_iter().map(move |x|
        (get_name(x.name()), DataContainer::H5Dataset(x))
    ))
}

#[cfg(test)]
mod data_io_test {
    use super::*;
    use hdf5::*;
    use tempfile::tempdir;
    use std::path::PathBuf;

    pub fn with_tmp_dir<T, F: Fn(PathBuf) -> T>(func: F) -> T {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();
        func(path)
    }

    fn with_tmp_path<T, F: Fn(PathBuf) -> T>(func: F) -> T {
        with_tmp_dir(|dir| func(dir.join("foo.h5")))
    }

    fn with_tmp_file<T, F: Fn(File) -> T>(func: F) -> T {
        with_tmp_path(|path| {
            let file = File::create(&path).unwrap();
            func(file)
        })
    }

    #[test]
    fn test_hdf5_read_write_scalar() {
        with_tmp_file::<Result<_>, _>(|file| {
            let val: f64 = 0.2;
            let dataset = file.new_dataset::<f64>().create("foo")?;
            dataset.write_scalar(&val)?;
            let val_back = dataset.read_scalar()?;
            assert_eq!(val, val_back);
            Ok(())
        })
        .unwrap()
    }

    #[test]
    fn test_scalar() -> Result<()> {
        with_tmp_file::<Result<_>, _>(|file| {
            let value = Scalar(0.2);
            let group = file.create_group("test")?;
            let container = value.write(&group, "data")?;
            let value_read = Scalar::read(&container)?;
            assert_eq!(value, value_read);
            Ok(())
        })
    }
}