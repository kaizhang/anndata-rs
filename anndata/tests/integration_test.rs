use anndata::*;
use anndata::backend::Backend;
use anndata::AnnDataIterator;
use anndata_hdf5::H5;
use anndata_n5::N5;

use polars::df;
use polars::prelude::{DataFrame, NamedFrom};
use proptest::prelude::*;

use anyhow::Result;
use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{Array, Array1, Array2, Array3};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use nalgebra::base::DMatrix;
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

fn rand_csr(nrow: usize, ncol: usize, nnz: usize) -> CsrMatrix<i64> {
    let mut rng = rand::thread_rng();
    let values: Vec<i64> = Array::random((nnz,), Uniform::new(-10000, 10000)).to_vec();

    let (row_indices, col_indices) = (0..nnz)
        .map(|_| (rng.gen_range(0..nrow), rng.gen_range(0..ncol)))
        .unzip();
    (&CooMatrix::try_from_triplets(nrow, ncol, row_indices, col_indices, values).unwrap()).into()
}

fn rand_chunks(nrow: usize, ncol: usize, chunk_size: usize) -> (CsrMatrix<i64>, impl Iterator<Item = CsrMatrix<i64>>) {
    let n = nalgebra::Dynamic::new(nrow);
    let c = nalgebra::Dynamic::new(ncol);
    let mat: DMatrix<i64> = DMatrix::new_random_generic(n, c);
    let csr = CsrMatrix::from(&mat);
    let iter = (0..nrow).into_iter().step_by(chunk_size)
        .map(move |i| {
            let j = (i + chunk_size).min(nrow);
            let m = mat.index((i..j, ..));
            CsrMatrix::from(&m)
        });
    (csr, iter)
}

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

fn obs_io<B>(adata: &AnnData<B>, input: DataFrame) -> Result<()>
where
    B: Backend,
{
    adata.set_obs(input.clone())?;
    assert_eq!(input, adata.read_obs()?);
    Ok(())
}

fn var_io<B>(adata: &AnnData<B>, input: DataFrame) -> Result<()>
where
    B: Backend,
{
    adata.set_var(input.clone())?;
    assert_eq!(input, adata.read_var()?);
    Ok(())
}

fn uns_io<B, T>(adata: &AnnData<B>, input: T) -> Result<()>
where
    B: Backend,
    T: Eq + Debug + Into<Data> + ReadData + Clone + TryFrom<Data>,
    <T as TryFrom<Data>>::Error: Into<anyhow::Error>, 
{
    adata.add_uns("test", Data::from(&input))?;
    assert_eq!(input, adata.fetch_uns::<T>("test")?.unwrap());
    Ok(())
}

fn obsm_io<B, T>(adata: &AnnData<B>, input: T) -> Result<()>
where
    B: Backend,
    T: Eq + Debug + HasShape + Into<ArrayData> + ReadArrayData + WriteArrayData + Clone + TryFrom<ArrayData>,
    <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>, 
{
    adata.add_obsm("test", input.clone())?;
    assert_eq!(input, adata.fetch_obsm::<T>("test")?.unwrap());
    Ok(())
}

fn obsp_io<B, T>(adata: &AnnData<B>, input: T) -> Result<()>
where
    B: Backend,
    T: Eq + Debug + HasShape + Into<ArrayData> + ReadArrayData + WriteArrayData + Clone + TryFrom<ArrayData>,
    <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>, 
{
    adata.add_obsp("test", input.clone())?;
    assert_eq!(input, adata.fetch_obsp::<T>("test")?.unwrap());
    Ok(())
}

fn varm_io<B, T>(adata: &AnnData<B>, input: T) -> Result<()>
where
    B: Backend,
    T: Eq + Debug + HasShape + Into<ArrayData> + ReadArrayData + WriteArrayData + Clone + TryFrom<ArrayData>,
    <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>, 
{
    adata.add_varm("test", input.clone())?;
    assert_eq!(input, adata.fetch_varm::<T>("test")?.unwrap());
    Ok(())
}

fn varp_io<B, T>(adata: &AnnData<B>, input: T) -> Result<()>
where
    B: Backend,
    T: Eq + Debug + HasShape + Into<ArrayData> + ReadArrayData + WriteArrayData + Clone + TryFrom<ArrayData>,
    <T as TryFrom<ArrayData>>::Error: Into<anyhow::Error>, 
{
    adata.add_varp("test", input.clone())?;
    assert_eq!(input, adata.fetch_varp::<T>("test")?.unwrap());
    Ok(())
}

fn test_iterator<B: Backend>() -> Result<()> {
    with_tmp_path(|file| -> Result<()> {
        let adata: AnnData<B> = AnnData::new(file, 0, 0)?;
        let (csr, chunks) = rand_chunks(997, 20, 41);

        adata.set_x_from_iter(chunks)?;

        assert_eq!(adata.n_obs(), 997);
        assert_eq!(adata.n_vars(), 20);
        assert_eq!(adata.read_x::<CsrMatrix<i64>>()?.unwrap(), csr);

        adata.add_obsm_from_iter::<_, CsrMatrix<i64>>("key", adata.read_x_iter(111).map(|x| x.0))?;
        assert_eq!(adata.fetch_obsm::<CsrMatrix<i64>>("key")?.unwrap(), csr);

        Ok(())
    })
}

fn test_io<B: Backend>() -> Result<()> {
    with_tmp_path(|file| -> Result<()> {
        let adata: AnnData<B> = AnnData::new(file, 0, 0)?;

        let mut arr_x: Array2<usize> = Array::random((0, 0), Uniform::new(0, 100));
        let mut csr_x = rand_csr(0, 0, 0);
        adata.set_x(&csr_x)?;
        assert_eq!(csr_x, adata.read_x::<CsrMatrix<i64>>()?.unwrap());
        adata.set_x(&arr_x)?;
        assert_eq!(arr_x, adata.read_x::<Array2<usize>>()?.unwrap());
        adata.del_x()?;

        arr_x = Array::random((2, 3), Uniform::new(0, 100));
        csr_x = rand_csr(2, 3, 2);
        adata.set_x(&csr_x)?;
        assert_eq!(csr_x, adata.read_x::<CsrMatrix<i64>>()?.unwrap());
        adata.set_x(&arr_x)?;
        assert_eq!(arr_x, adata.read_x::<Array2<usize>>()?.unwrap());

        obs_io(&adata, df!(
            "Fruit" => &["Apple", "Pear"],
            "Color" => &["Red", "Green"]
        ).unwrap())?;
        var_io(&adata, df!(
            "Fruit" => &["Apple", "Apple", "Pear"],
            "Color" => &["Red", "Yellow", "Green"]
        ).unwrap())?;

        uns_io(&adata, 3i32)?;
        uns_io(&adata, "test".to_string())?;

        obsm_io(&adata, Array::random((2, 5), Uniform::new(0, 100)))?;
        varm_io(&adata, Array::random((3, 5), Uniform::new(0, 100)))?;

        obsp_io(&adata, Array::random((2, 2), Uniform::new(0, 100)))?;
        varp_io(&adata, Array::random((3, 3), Uniform::new(0, 100)))?;

        with_tmp_path(|file| -> Result<()> { adata.write::<B, _>(file) })?;

        adata.del_x()?;
        adata.del_obs()?;
        adata.del_var()?;

        Ok(())
    })
}

fn test_slice<B: Backend>() -> Result<()> {
    with_tmp_path(|file| -> Result<()> {
        let adata: AnnData<B> = AnnData::new(file, 0, 0)?;

        let arr: Array3<i32> = Array::random((40, 50, 10), Uniform::new(0, 100));
        adata.set_x(&arr)?;
        let x: Array3<i32> = adata.read_x_slice(s![3..33, 4..44, ..])?.unwrap();
        assert_eq!(x, arr.slice(ndarray::s![3..33, 4..44, ..]).to_owned());

        let csr: CsrMatrix<i64> = rand_csr(40, 50, 500);
        adata.set_x(&csr)?;
        assert_eq!(
            adata.read_x_slice::<CsrMatrix<i64>, _>(s![3..33, 4..44])?.unwrap(),
            csr_select(&csr, 3..33, 4..44),
        );
        Ok(())
    })
}

fn test_fancy_index<B: Backend>() -> Result<()> {
    with_tmp_path(|file| -> Result<()> {
        let adata: AnnData<B> = AnnData::new(file, 0, 0)?;

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

        {
            let csr: CsrMatrix<i64> = rand_csr(40, 50, 500);
            let i1: Vec<usize> = Array::random((100,), Uniform::new(0, 39)).to_vec();
            let i2: Vec<usize> = Array::random((100,), Uniform::new(0, 49)).to_vec();
            adata.set_x(&csr)?;
            assert_eq!(
                csr_select(&csr, i1.clone().into_iter(), i2.clone().into_iter()),
                adata.read_x_slice::<CsrMatrix<i64>, _>(s![i1, i2])?.unwrap(),
            );
        }

        Ok(())
    })
}

fn test_subset<B: Backend>() -> Result<()> {
    with_tmp_path(|file| -> Result<()> {
        let adata: AnnData<B> = AnnData::new(file, 0, 0)?;

        let csr: CsrMatrix<i64> = rand_csr(100, 150, 2000);
        let arr1: Array2<i32> = Array::random((100, 200), Uniform::new(0, 100));
        let arr2: Array2<i32> = Array::random((100, 100), Uniform::new(0, 100));
        let arr3: Array2<i32> = Array::random((150, 50), Uniform::new(0, 100));
        let arr4: Array2<i32> = Array::random((150, 150), Uniform::new(0, 100));
        let obs = df!(
            "Fruit" => (0..100).map(|i| format!("Fruit {}", i)).collect::<Vec<_>>(),
            "Color" => (0..100).map(|i| format!("Color {}", i)).collect::<Vec<_>>()
        ).unwrap();
        let var = df!(
            "Fruit" => (0..150).map(|i| format!("Fruit {}", i)).collect::<Vec<_>>(),
            "Color" => (0..150).map(|i| format!("Color {}", i)).collect::<Vec<_>>()
        ).unwrap();

        adata.set_x(&csr)?;
        adata.set_obs(obs.clone())?;
        adata.set_var(var.clone())?;
        adata.add_obsm("test1", &arr1)?;
        adata.add_obsp("test1", &arr2)?;
        adata.add_varm("test1", &arr3)?;
        adata.add_varp("test1", &arr4)?;
        adata.add_uns("test1", &arr1)?;

        adata.subset(s![..50, ..100])?;

        assert_eq!(adata.n_obs(), 50);
        assert_eq!(adata.n_vars(), 100);
        assert_eq!(adata.read_obs()?, obs.take_iter(0..50).unwrap());
        assert_eq!(adata.read_var()?, var.take_iter(0..100).unwrap());
        assert_eq!(adata.read_x::<CsrMatrix<i64>>()?.unwrap(), csr_select(&csr, 0..50, 0..100));
        assert_eq!(adata.fetch_obsm::<Array2<i32>>("test1")?.unwrap(), arr1.slice(ndarray::s![0..50, ..]));
        assert_eq!(adata.fetch_obsp::<Array2<i32>>("test1")?.unwrap(), arr2.slice(ndarray::s![0..50, 0..50]));
        assert_eq!(adata.fetch_varm::<Array2<i32>>("test1")?.unwrap(), arr3.slice(ndarray::s![0..100, ..]));
        assert_eq!(adata.fetch_varp::<Array2<i32>>("test1")?.unwrap(), arr4.slice(ndarray::s![0..100, 0..100]));
        assert_eq!(adata.fetch_uns::<Array2<i32>>("test1")?.unwrap(), arr1);

        Ok(())
    })
}

// Begin of tests

// H5 backend tests

#[test]
fn test_io_h5() -> Result<()> {
    test_io::<H5>()
}

#[test]
fn test_iterator_h5() -> Result<()> {
    test_iterator::<H5>()
}

#[test]
fn test_slice_h5() -> Result<()> {
    test_slice::<H5>()
}

#[test]
fn test_fancy_index_h5() -> Result<()> {
    test_fancy_index::<H5>()
}

#[test]
fn test_subset_h5() -> Result<()> {
    test_subset::<H5>()
}