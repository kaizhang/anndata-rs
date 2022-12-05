use anndata_rs::*;
use anndata_rs::backend::hdf5::H5;

use proptest::prelude::*;

use anyhow::Result;
use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{Array, Array1, Array2, Array3};
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
        assert_eq!(input, adata.fetch_uns::<Data>("test").unwrap().unwrap());
    });
}

#[test]
fn test_basic() -> Result<()> {
    with_tmp_path(|file| -> Result<()> {
        let adata: AnnData<H5> = AnnData::new(file, 0, 0)?;

        adata.add_uns("test", 3i32)?;
        assert_eq!(3i32, adata.fetch_uns::<i32>("test")?.unwrap());
        adata.add_uns("test", 3.0f32)?;
        assert_eq!(3.0f32, adata.fetch_uns::<f32>("test")?.unwrap());

        let arr = Array::random((2, 5), Uniform::new(0, 100));
        adata.add_obsm("test", &arr)?;
        adata.add_varm("test", &arr)?;

        let arr_x = Array::random((2, 2), Uniform::new(-100, 100));
        adata.set_x(&arr_x)?;
        assert_eq!(arr_x, adata.read_x::<Array2<i32>>()?.unwrap());
        adata.del_x()?;

        let csr_x = rand_csr(2, 2, 1);
        adata.set_x(&csr_x)?;
        assert_eq!(csr_x, adata.read_x::<CsrMatrix<i32>>()?.unwrap());

        assert!(adata.add_obsp("test", &arr).is_err());

        Ok(())
    })
}

#[test]
fn test_slice() -> Result<()> {
    with_tmp_path(|file| -> Result<()> {
        let adata: AnnData<H5> = AnnData::new(file, 0, 0)?;

        let arr: Array3<i32> = Array::random((40, 50, 10), Uniform::new(0, 100));
        adata.set_x(&arr)?;
        let x: Array3<i32> = adata.read_x_slice(s![3..33, 4..44, ..])?.unwrap();
        assert_eq!(x, arr.slice(ndarray::s![3..33, 4..44, ..]).to_owned());
        Ok(())
    })
}

#[test]
fn test_fancy_index() -> Result<()> {
    with_tmp_path(|file| -> Result<()> {
        let adata: AnnData<H5> = AnnData::new(file, 0, 0)?;

        {
            let arr: Array2<i32> = Array::random((40, 1), Uniform::new(0, 100));
            adata.set_x(&arr)?;

            let idx  = vec![1, 3, 5, 7, 9];
            let expected = arr.select(ndarray::Axis(0), idx.as_slice());
            let actual: Array2<i32> = adata.read_x_slice(s![idx, ..])?.unwrap();
            assert_eq!(expected, actual);
        }

        adata.del_x()?;

        {
            let arr: Array3<i32> = Array::random((40, 50, 10), Uniform::new(0, 100));
            let i1: Array1<usize> = Array::random((100,), Uniform::new(0, 39));
            let i2: Array1<usize> = Array::random((100,), Uniform::new(0, 49));
            adata.set_x(&arr)?;

            let expected = arr
                .select(ndarray::Axis(0), i1.as_slice().unwrap())
                .select(ndarray::Axis(1), i2.as_slice().unwrap());
            let actual: Array3<i32> = adata.read_x_slice(s![i1, i2, ..])?.unwrap();
            assert_eq!(expected, actual);
        }

        Ok(())
    })
}


#[test]
fn test_uns_io_array() {
    let arr = Array::random((2, 5), Uniform::new(0, 100)).into_dyn();
    uns_io(arr);
}

proptest! {
    #[test]
    fn test_uns_io(input: (i64, String)) {
        uns_io(input.0);
        uns_io(input.1);
    }
}