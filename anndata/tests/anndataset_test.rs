use anndata::*;
use anndata_hdf5::H5;

use proptest::prelude::*;

use anyhow::Result;
use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{Axis, Array, Array1, Array2, Array3, concatenate};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::fmt::Debug;
use tempfile::tempdir;
use std::path::PathBuf;

pub fn with_tmp_dir<T, F: FnMut(PathBuf) -> T>(mut func: F) -> T {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    func(path)
}

fn with_tmp_path<T, F: Fn(PathBuf) -> T>(func: F) -> T {
    with_tmp_dir(|dir| func(dir.join("temp.h5")))
}

fn rand_csr(nrow: usize, ncol: usize, nnz: usize) -> CsrMatrix<i32> {
    let mut rng = rand::thread_rng();
    let values: Vec<i32> = Array::random((nnz,), Uniform::new(0, 100)).to_vec();

    let (row_indices, col_indices) = (0..nnz)
        .map(|_| (rng.gen_range(0..nrow), rng.gen_range(0..ncol)))
        .unzip();
    (&CooMatrix::try_from_triplets(nrow, ncol, row_indices, col_indices, values).unwrap()).into()
}

fn uns_io<T>(input: T)
where
    T: Eq + Debug + Into<Data> + Clone,
    Data: TryInto<T>,
    <Data as TryInto<T>>::Error: Debug,
{
    with_tmp_path(|file| {
        let adata: AnnData<H5> = AnnData::new(file, 0, 0).unwrap();
        adata.add_uns("test", Data::from(&input)).unwrap();
        assert_eq!(input, adata.fetch_uns::<Data>("test").unwrap().unwrap().try_into().unwrap());
    });
}

#[test]
fn test_basic() -> Result<()> {
    with_tmp_dir(|dir| -> Result<()> {
        let arr = Array::random((30, 50), Uniform::new(-100, 100));
        let merged = concatenate(Axis(0), &[arr.view(), arr.view(), arr.view()])?;
        
        let d1: AnnData<H5> = AnnData::new(dir.join("1.h5ad"), 0, 0)?;
        let d2: AnnData<H5> = AnnData::new(dir.join("2.h5ad"), 0, 0)?;
        let d3: AnnData<H5> = AnnData::new(dir.join("3.h5ad"), 0, 0)?;
        d1.set_x(&arr)?;
        d2.set_x(&arr)?;
        d3.set_x(&arr)?;

        let dataset = AnnDataSet::new([("1", d1), ("2", d2), ("3", d3)], dir.join("dataset.h5ads"), "key")?;

        assert_eq!(merged, dataset.read_x::<Array<i32, _>>()?.unwrap());

        Ok(())
    })
}

fn test_slice<B: Backend>() -> Result<()> {
    with_tmp_dir(|dir| -> Result<()> {
        let arr1 = Array::random((97, 50), Uniform::new(-100, 100));
        let arr2 = Array::random((13, 50), Uniform::new(-100, 100));
        let arr3 = Array::random((71, 50), Uniform::new(-100, 100));
        let merged = concatenate(Axis(0), &[arr1.view(), arr2.view(), arr3.view()])?;

        let d1: AnnData<H5> = AnnData::new(dir.join("1.h5ad"), 0, 0)?;
        let d2: AnnData<H5> = AnnData::new(dir.join("2.h5ad"), 0, 0)?;
        let d3: AnnData<H5> = AnnData::new(dir.join("3.h5ad"), 0, 0)?;
        d1.set_x(&arr1)?;
        d2.set_x(&arr2)?;
        d3.set_x(&arr3)?;
        let dataset = AnnDataSet::new([("1", d1), ("2", d2), ("3", d3)], dir.join("dataset.h5ads"), "key")?;

        let mut x: Array2<i32> = dataset.read_x_slice(s![3..33, 4..44])?.unwrap();
        assert_eq!(x, merged.slice(ndarray::s![3..33, 4..44]).to_owned());

        x = dataset.read_x_slice(s![0..181, 4..44])?.unwrap();
        assert_eq!(x, merged.slice(ndarray::s![0..181, 4..44]).to_owned());

        // Fancy index
        let indices1 = Array::random(300, Uniform::new(0, 181)).to_vec();
        let indices2 = Array::random(300, Uniform::new(0, 50)).to_vec();
        let expected = merged
            .select(ndarray::Axis(0), indices1.as_slice())
            .select(ndarray::Axis(1), indices2.as_slice());
        x = dataset.read_x_slice(s![indices1, indices2])?.unwrap();
        assert_eq!(x, expected);

        Ok(())
    })
}

#[test]
fn test_slice_h5() -> Result<()> {
    test_slice::<H5>()
}