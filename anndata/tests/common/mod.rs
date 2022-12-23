use anndata::*;

use anndata::backend::{BackendData, ScalarType};
use nalgebra::Scalar;
use proptest::prelude::*;
use anyhow::Result;
use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra::base::DMatrix;
use ndarray::{ArrayD, Axis, Array, Array1, Array2, Array3, concatenate};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::Rng;
use tempfile::tempdir;
use std::path::PathBuf;
use std::fmt::Debug;

pub fn with_tmp_dir<T, F: FnMut(PathBuf) -> T>(mut func: F) -> T {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    func(path)
}

pub fn with_tmp_path<T, F: FnMut(PathBuf) -> T>(mut func: F) -> T {
    with_tmp_dir(|dir| func(dir.join("temp.h5")))
}

pub fn with_empty_adata<B, F, T>(mut func: F) -> T
where
    B: Backend,
    F: FnMut(AnnData<B>) -> T,
{
    with_tmp_path(|file| {
        let adata: AnnData<B> = AnnData::new(file, 0, 0).unwrap();
        func(adata)
    })
}

pub fn empty_adata<B: Backend>(file: PathBuf) -> impl Strategy<Value = AnnData<B>> {
    let n_obs = any::<u16>().prop_filter("reason for filtering", |x| x < &1000u16);
    let n_vars = any::<u16>().prop_filter("reason for filtering", |x| x < &1000u16);

    (n_obs, n_vars).prop_map(move |(n_obs, n_vars)| {
        let adata: AnnData<B> = AnnData::new(file.clone(), n_obs as usize, n_vars as usize).unwrap();
        adata
    })
}

pub fn array_data_strat() -> impl Strategy<Value = ArrayData> {
    prop_oneof![
        array_strat(),
        csr_strat(),
    ]
}

fn array_strat() -> impl Strategy<Value = ArrayData> {
    let range: std::ops::Range<usize> = 0..20;
    proptest::collection::vec(range, 2..4).prop_flat_map(|shape| {
        prop_oneof![
            Just(rand_array::<usize>(shape.clone())),
            /*
            Just(rand_array::<u8>(shape.clone())),
            Just(rand_array::<u16>(shape.clone())),
            Just(rand_array::<u32>(shape.clone())),
            Just(rand_array::<u64>(shape.clone())),
            //Just(rand_array::<usize>(shape.clone())),
            Just(rand_array::<i8>(shape.clone())),
            Just(rand_array::<i16>(shape.clone())),
            Just(rand_array::<i32>(shape.clone())),
            Just(rand_array::<i64>(shape.clone())),
            Just(rand_array::<f32>(shape.clone())),
            Just(rand_array::<f64>(shape.clone())),
            Just(rand_array::<bool>(shape.clone())),
            Just(rand_array::<String>(shape.clone())),
            */
        ]
    })
}


fn csr_strat() -> impl Strategy<Value = ArrayData> {
    ((0 as usize ..100), (0 as usize ..100)).prop_map(|(x, y)| {
        let max = x * y; 
        let nnz = (max / 10).max(1).min(max);
        rand_csr(x, y, nnz).into()
    })
}

fn rand_array<B: BackendData>(shape: Vec<usize>) -> ArrayData {
    match B::DTYPE {
        ScalarType::U8 => Array::random(shape, Uniform::new(0u8, 255u8)).into(),
        ScalarType::U16 => Array::random(shape, Uniform::new(0u16, 255u16)).into(),
        ScalarType::U32 => Array::random(shape, Uniform::new(0u32, 255u32)).into(),
        ScalarType::U64 => Array::random(shape, Uniform::new(0u64, 255u64)).into(),
        ScalarType::Usize => Array::random(shape, Uniform::new(0usize, 255usize)).into(),
        ScalarType::I8 => Array::random(shape, Uniform::new(-128i8, 127i8)).into(),
        ScalarType::I16 => Array::random(shape, Uniform::new(-128i16, 127i16)).into(),
        ScalarType::I32 => Array::random(shape, Uniform::new(-128i32, 127i32)).into(),
        ScalarType::I64 => Array::random(shape, Uniform::new(-128i64, 127i64)).into(),
        ScalarType::F32 => Array::random(shape, Uniform::new(-128f32, 127f32)).into(),
        ScalarType::F64 => Array::random(shape, Uniform::new(-128f64, 127f64)).into(),
        ScalarType::Bool => Array::random(shape, Uniform::new(0u8, 1u8)).mapv(|x| x == 1).into(),
        ScalarType::String => Array::random(shape, Uniform::new(-1000f32, 1000f32)).mapv(|x| x.to_string()).into(),
    }
}

pub fn rand_csr(nrow: usize, ncol: usize, nnz: usize) -> CsrMatrix<i64> {
    let mut rng = rand::thread_rng();
    let values: Vec<i64> = Array::random((nnz,), Uniform::new(0, 100)).to_vec();

    let (row_indices, col_indices) = (0..nnz)
        .map(|_| (rng.gen_range(0..nrow), rng.gen_range(0..ncol)))
        .unzip();
    (&CooMatrix::try_from_triplets(nrow, ncol, row_indices, col_indices, values).unwrap()).into()
}

pub fn csr_select<I1, I2>(
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