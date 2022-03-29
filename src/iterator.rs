use crate::utils::{ResizableVectorData, COMPRESSION, create_str_attr};
use crate::anndata_trait::{DataType, DataContainer};
use crate::base::AnnData;
use crate::element::MatrixElem;

use ndarray::{arr1, Array};
use hdf5::{Group, H5Type, Result};
use itertools::Itertools;

pub trait RowIterator {
    fn write(self, location: &Group, name: &str) -> Result<(DataContainer, usize)>;

    fn version(&self) -> &str;

    fn get_dtype(&self) -> DataType;

    fn ncols(&self) -> usize;

    fn update(self, container: &DataContainer) -> Result<(DataContainer, usize)>
    where Self: Sized,
    {
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

pub struct CsrIterator<I> {
    iterator: I,
    num_col: usize,
}

impl<I, D> RowIterator for CsrIterator<I>
where
    I: Iterator<Item = Vec<(usize, D)>>,
    D: H5Type,
{
    fn write(self, location: &Group, name: &str) -> Result<(DataContainer, usize)> {
        let group = location.create_group(name)?;
        create_str_attr(&group, "encoding-type", "csr_matrix")?;
        create_str_attr(&group, "encoding-version", self.version())?;
        create_str_attr(&group, "h5sparse_format", "csr")?;
        let data: ResizableVectorData<D> =
            ResizableVectorData::new(&group, "data", 10000)?;
        let mut indptr: Vec<usize> = vec![0];
        let iter = self.iterator.scan(0, |state, x| {
            *state = *state + x.len();
            Some((*state, x))
        });

        if self.num_col <= (i32::MAX as usize) {
            let indices: ResizableVectorData<i32> =
                ResizableVectorData::new(&group, "indices", 10000)?;
            for chunk in &iter.chunks(10000) {
                let (a, b): (Vec<i32>, Vec<D>) = chunk.map(|(x, vec)| {
                    indptr.push(x);
                    vec
                }).flatten().map(|(x, y)| -> (i32, D) {(
                    x.try_into().expect(&format!("cannot convert '{}' to i32", x)),
                    y
                ) }).unzip();
                indices.extend(a.into_iter())?;
                data.extend(b.into_iter())?;
            }

            let num_rows = indptr.len() - 1;
            group.new_attr_builder()
                .with_data(&arr1(&[num_rows, self.num_col]))
                .create("shape")?;

            let try_convert_indptr: Option<Vec<i32>> = indptr.iter()
                .map(|x| (*x).try_into().ok()).collect();
            match try_convert_indptr {
                Some(vec) => {
                    group.new_dataset_builder().deflate(COMPRESSION)
                        .with_data(&Array::from_vec(vec)).create("indptr")?;
                },
                _ => {
                    let vec: Vec<i64> = indptr.into_iter()
                        .map(|x| x.try_into().unwrap()).collect();
                    group.new_dataset_builder().deflate(COMPRESSION)
                        .with_data(&Array::from_vec(vec)).create("indptr")?;
                },
            }
            Ok((DataContainer::H5Group(group), num_rows))
        } else {
            let indices: ResizableVectorData<i64> =
                ResizableVectorData::new(&group, "indices", 10000)?;
            for chunk in &iter.chunks(10000) {
                let (a, b): (Vec<i64>, Vec<D>) = chunk.map(|(x, vec)| {
                    indptr.push(x);
                    vec
                }).flatten().map(|(x, y)| -> (i64, D) {(
                    x.try_into().expect(&format!("cannot convert '{}' to i64", x)),
                    y
                ) }).unzip();
                indices.extend(a.into_iter())?;
                data.extend(b.into_iter())?;
            }

            let num_rows = indptr.len() - 1;
            group.new_attr_builder()
                .with_data(&arr1(&[num_rows, self.num_col]))
                .create("shape")?;

            let vec: Vec<i64> = indptr.into_iter()
                .map(|x| x.try_into().unwrap()).collect();
            group.new_dataset_builder().deflate(COMPRESSION)
                .with_data(&Array::from_vec(vec)).create("indptr")?;
            Ok((DataContainer::H5Group(group), num_rows))
        }
    }

    fn ncols(&self) -> usize { self.num_col }
    fn get_dtype(&self) -> DataType { DataType::CsrMatrix(D::type_descriptor()) }
    fn version(&self) -> &str { "0.1.0" }
}

impl AnnData {
    pub fn set_x_from_row_iter<I>(&mut self, data: I) -> Result<()>
    where
        I: RowIterator,
    {
        if self.n_vars == 0 { self.n_vars = data.ncols(); }
        assert!(
            self.n_vars == data.ncols(),
            "Number of variables mismatched, expecting {}, but found {}",
            self.n_vars, data.ncols(),
        );

        if self.x.is_some() { self.file.unlink("X")?; }
        let (container, nrows) = data.write(&self.file, "X")?;
        if self.n_obs == 0 { self.n_obs = nrows; }
        assert!(
            self.n_obs == nrows,
            "Number of observations mismatched, expecting {}, but found {}",
            self.n_obs, nrows,
        );
        self.x = Some(MatrixElem::new(container)?);
        Ok(())
    }

    pub fn add_obsm_from_row_iter<I>(&mut self, key: &str, data: I) -> Result<()>
    where
        I: RowIterator,
    {
       let obsm = match self.file.group("obsm") {
            Ok(x) => x,
            _ => self.file.create_group("obsm").unwrap(),
        };
        if self.obsm.contains_key(key) { obsm.unlink(key)?; } 
        let (container, nrows) = data.write(&obsm, key)?;
        if self.n_obs == 0 { self.n_obs = nrows; }

        assert!(
            self.n_obs == nrows,
            "Number of observations mismatched, expecting {}, but found {}",
            self.n_obs, nrows,
        );
 
        let elem = MatrixElem::new(container)?;
        self.obsm.insert(key.to_string(), elem);
        Ok(())
    }
}