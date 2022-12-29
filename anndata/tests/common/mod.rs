use anndata::*;

use anndata::backend::{BackendData, ScalarType, DatasetOp};
use anndata::data::{SelectInfoElem, BoundedSelectInfo, BoundedSelectInfoElem};
use itertools::Itertools;
use nalgebra::{Scalar, ClosedAdd};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Zero;
use proptest::prelude::*;
use anyhow::Result;
use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra::base::DMatrix;
use ndarray::{ArrayD, Axis, Array, Array1, Array2, Array3, concatenate, Dimension, RemoveAxis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::Rng;
use tempfile::tempdir;
use std::path::{PathBuf, Path};
use std::fmt::Debug;
use rand::seq::{SliceRandom, IteratorRandom};
use proptest::strategy::BoxedStrategy;

pub fn with_tmp_dir<T, F: FnMut(PathBuf) -> T>(mut func: F) -> T {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    func(path)
}

pub fn with_tmp_path<T, F: FnMut(PathBuf) -> T>(mut func: F) -> T {
    with_tmp_dir(|dir| func(dir.join("temp.h5")))
}

pub fn empty_adata<B>() -> AnnData<B>
where
    B: Backend,
{
    let file = tempfile::NamedTempFile::new().unwrap();
    AnnData::new(file.path()).unwrap()
}


////////////////////////////////////////////////////////////////////////////////
/// Strategies
////////////////////////////////////////////////////////////////////////////////

/// Strategy for generating a random AnnData
pub fn anndata_strat<B: Backend, P: AsRef<Path> + Clone>(file: P, n_obs: usize, n_vars: usize) -> impl Strategy<Value = AnnData<B>> {
    let x = array_strat(&vec![n_obs, n_vars]);
    let obsm = proptest::collection::vec(0 as usize ..100, 0..3).prop_flat_map(move |shapes|
        shapes.into_iter().map(|d| array_strat(&vec![n_obs, d])).collect::<Vec<_>>()
    );
    let obsp = (0 as usize ..3).prop_flat_map(move |d| 
        std::iter::repeat_with(|| array_strat(&vec![n_obs, n_obs])).take(d).collect::<Vec<_>>()
    );
    let varm = proptest::collection::vec(0 as usize ..100, 0..3).prop_flat_map(move |shapes|
        shapes.into_iter().map(|d| array_strat(&vec![n_vars, d])).collect::<Vec<_>>()
    );
    let varp = (0 as usize ..3).prop_flat_map(move |d| 
        std::iter::repeat_with(|| array_strat(&vec![n_vars, n_vars])).take(d).collect::<Vec<_>>()
    );
    (x, obsm, obsp, varm, varp).prop_map(move |(x, obsm, obsp, varm, varp)| {
        let adata: AnnData<B> = AnnData::new(file.clone()).unwrap();
        adata.set_x(x).unwrap();
        obsm.into_iter().enumerate().for_each(|(i, arr)| {
            adata.obsm().add(&format!("varm_{}", i), arr).unwrap();
        });
        obsp.into_iter().enumerate().for_each(|(i, arr)| {
            adata.obsp().add(&format!("obsp_{}", i), arr).unwrap();
        });
        varm.into_iter().enumerate().for_each(|(i, arr)| {
            adata.varm().add(&format!("varm_{}", i), arr).unwrap();
        });
        varp.into_iter().enumerate().for_each(|(i, arr)| {
            adata.varp().add(&format!("varp_{}", i), arr).unwrap();
        });
        adata
    })
}

pub fn array_slice_strat(shape: &Vec<usize>) -> impl Strategy<Value = (ArrayData, Vec<SelectInfoElem>)> {
    array_strat(&shape).prop_flat_map(|x| {
        let select = x.shape().as_ref().iter().map(|&s| select_strat(s)).collect::<Vec<_>>();
        (Just(x), select)
    })
}

pub fn array_strat(shape: &Vec<usize>) -> impl Strategy<Value = ArrayData> {
    prop_oneof![
        csr_strat(shape[0], shape[1]),
        dense_array_strat(shape),
    ]
}

/// Strategy for generating a random CsrMatrix
fn csr_strat(num_rows: usize, num_cols: usize) -> impl Strategy<Value = ArrayData> {
    let max = num_rows * num_cols; 
    let nnz = max / 10;
    prop_oneof![
        Just(rand_csr::<u8>(num_rows, num_cols, nnz, 1, 100).into()),
        Just(rand_csr::<u16>(num_rows, num_cols, nnz, 1, 100).into()),
        Just(rand_csr::<u32>(num_rows, num_cols, nnz, 1, 2550).into()),
        Just(rand_csr::<u64>(num_rows, num_cols, nnz, 1, 25500).into()),
        Just(rand_csr::<i8>(num_rows, num_cols, nnz, 1, 100).into()),
        Just(rand_csr::<i16>(num_rows, num_cols, nnz, 1, 100).into()),
        Just(rand_csr::<i32>(num_rows, num_cols, nnz, 1, 2550).into()),
        Just(rand_csr::<i64>(num_rows, num_cols, nnz, 1, 25500).into()),
        Just(rand_csr::<f32>(num_rows, num_cols, nnz, 1.0, 255.0).into()),
        Just(rand_csr::<f64>(num_rows, num_cols, nnz, 1.0, 255.0).into()),
    ]
}

fn dense_array_strat(shape: &Vec<usize>) -> impl Strategy<Value = ArrayData> {
    let s: Vec<_> = shape.clone().into_iter().rev().collect();
    prop_oneof![
        Just(Array::random(shape.clone(), Uniform::new(0u8, 255u8)).into()),
        Just(Array::random(shape.clone(), Uniform::new(0u16, 255u16)).into()),
        Just(Array::random(shape.clone(), Uniform::new(0u32, 255u32)).into()),
        Just(Array::random(shape.clone(), Uniform::new(0u64, 255u64)).into()),
        Just(Array::random(shape.clone(), Uniform::new(-128i8, 127i8)).into()),
        Just(Array::random(shape.clone(), Uniform::new(-128i16, 127i16)).into()),
        Just(Array::random(shape.clone(), Uniform::new(-128i32, 127i32)).into()),
        Just(Array::random(shape.clone(), Uniform::new(-128i64, 127i64)).into()),
        Just(Array::random(shape.clone(), Uniform::new(-128f32, 127f32)).into()),
        Just(Array::random(shape.clone(), Uniform::new(-128f64, 127f64)).into()),
        Just(Array::random(shape.clone(), Uniform::new(0u8, 1u8)).mapv(|x| x == 1).into()),
        Just(Array::random(shape.clone(), Uniform::new(-1000f32, 1000f32)).mapv(|x| x.to_string()).into()),
        Just(Array::random(s.clone(), Uniform::new(0u8, 255u8)).reversed_axes().into()),
        Just(Array::random(s.clone(), Uniform::new(0u16, 255u16)).reversed_axes().into()),
        Just(Array::random(s.clone(), Uniform::new(0u32, 255u32)).reversed_axes().into()),
        Just(Array::random(s.clone(), Uniform::new(0u64, 255u64)).reversed_axes().into()),
        Just(Array::random(s.clone(), Uniform::new(-128i8, 127i8)).reversed_axes().into()),
        Just(Array::random(s.clone(), Uniform::new(-128i16, 127i16)).reversed_axes().into()),
        Just(Array::random(s.clone(), Uniform::new(-128i32, 127i32)).reversed_axes().into()),
        Just(Array::random(s.clone(), Uniform::new(-128i64, 127i64)).reversed_axes().into()),
        Just(Array::random(s.clone(), Uniform::new(-128f32, 127f32)).reversed_axes().into()),
        Just(Array::random(s.clone(), Uniform::new(-128f64, 127f64)).reversed_axes().into()),
        Just(Array::random(s.clone(), Uniform::new(0u8, 1u8)).mapv(|x| x == 1).reversed_axes().into()),
        Just(Array::random(s.clone(), Uniform::new(-1000f32, 1000f32)).mapv(|x| x.to_string()).reversed_axes().into()),
    ]
}

fn rand_csr<T>(nrow: usize, ncol: usize, nnz: usize, low: T, high: T) -> CsrMatrix<T>
where
    T: Scalar + Zero + ClosedAdd + SampleUniform,
{
    let mut rng = rand::thread_rng();
    let values: Vec<T> = Array::random((nnz,), Uniform::new(low, high)).to_vec();
    let (row_indices, col_indices) = (0..nrow).cartesian_product(0..ncol).choose_multiple(&mut rng, nnz).into_iter().unzip();
    (&CooMatrix::try_from_triplets(nrow, ncol, row_indices, col_indices, values).unwrap()).into()
}

pub fn select_strat(n: usize) -> BoxedStrategy<SelectInfoElem> {
    if n == 0 {
        Just(Vec::new().into()).boxed()
    } else {
        let indices = proptest::collection::vec(0..n, 0..2*n).prop_map(|i| i.into());
        let slice = (0..n).prop_flat_map(move |start| (Just(start), (start+1)..=n).prop_map(|(start, stop)| (start..stop).into()));
        prop_oneof![
            indices,
            slice,
        ].boxed()
    }
}

////////////////////////////////////////////////////////////////////////////////
/// AnnData operations
////////////////////////////////////////////////////////////////////////////////

pub fn anndata_eq<B1: Backend, B2: Backend>(adata1: &AnnData<B1>, adata2: &AnnData<B2>) -> Result<bool> {
    let is_equal = adata1.n_obs() == adata2.n_obs() &&
        adata1.n_vars() == adata2.n_vars() &&
        adata1.obs_names().names == adata2.obs_names().names &&
        adata1.var_names().names == adata2.var_names().names &&
        adata1.read_obs()? == adata2.read_obs()? &&
        adata1.read_var()? == adata2.read_var()? &&
        adata1.read_x::<ArrayData>()? == adata2.read_x()? &&
        adata1.obsm().keys().iter().all(|k| {
            adata1.obsm().get::<ArrayData>(k).unwrap() == adata2.obsm().get(k).unwrap()
        }) &&
        adata1.obsp().keys().iter().all(|k| {
            adata1.obsp().get::<ArrayData>(k).unwrap() == adata2.obsp().get(k).unwrap()
        }) &&
        adata1.varm().keys().iter().all(|k| {
            adata1.varm().get::<ArrayData>(k).unwrap() == adata2.varm().get(k).unwrap()
        }) &&
        adata1.varp().keys().iter().all(|k| {
            adata1.varp().get::<ArrayData>(k).unwrap() == adata2.varp().get(k).unwrap()
        }) &&
        adata1.uns().keys().iter().all(|k| {
            adata1.uns().get::<Data>(k).unwrap() == adata2.uns().get(k).unwrap()
        });
    Ok(is_equal)
}

////////////////////////////////////////////////////////////////////////////////
/// Array operations
////////////////////////////////////////////////////////////////////////////////

pub fn array_select(arr: &ArrayData, select: &[SelectInfoElem]) -> ArrayData {
    match arr {
        ArrayData::Array(data::DynArray::Bool(arr)) => dense_array_select(arr, select).into(),
        ArrayData::Array(data::DynArray::U8(arr)) => dense_array_select(arr, select).into(),
        ArrayData::Array(data::DynArray::U16(arr)) => dense_array_select(arr, select).into(),
        ArrayData::Array(data::DynArray::U32(arr)) => dense_array_select(arr, select).into(),
        ArrayData::Array(data::DynArray::U64(arr)) => dense_array_select(arr, select).into(),
        ArrayData::Array(data::DynArray::I8(arr)) => dense_array_select(arr, select).into(),
        ArrayData::Array(data::DynArray::I16(arr)) => dense_array_select(arr, select).into(),
        ArrayData::Array(data::DynArray::I32(arr)) => dense_array_select(arr, select).into(),
        ArrayData::Array(data::DynArray::I64(arr)) => dense_array_select(arr, select).into(),
        ArrayData::Array(data::DynArray::F32(arr)) => dense_array_select(arr, select).into(),
        ArrayData::Array(data::DynArray::F64(arr)) => dense_array_select(arr, select).into(),
        ArrayData::Array(data::DynArray::String(arr)) => dense_array_select(arr, select).into(),
        ArrayData::CsrMatrix(data::DynCsrMatrix::U8(arr)) => csr_select(arr, select).into(),
        ArrayData::CsrMatrix(data::DynCsrMatrix::U16(arr)) => csr_select(arr, select).into(),
        ArrayData::CsrMatrix(data::DynCsrMatrix::U32(arr)) => csr_select(arr, select).into(),
        ArrayData::CsrMatrix(data::DynCsrMatrix::U64(arr)) => csr_select(arr, select).into(),
        ArrayData::CsrMatrix(data::DynCsrMatrix::I8(arr)) => csr_select(arr, select).into(),
        ArrayData::CsrMatrix(data::DynCsrMatrix::I16(arr)) => csr_select(arr, select).into(),
        ArrayData::CsrMatrix(data::DynCsrMatrix::I32(arr)) => csr_select(arr, select).into(),
        ArrayData::CsrMatrix(data::DynCsrMatrix::I64(arr)) => csr_select(arr, select).into(),
        ArrayData::CsrMatrix(data::DynCsrMatrix::F32(arr)) => csr_select(arr, select).into(),
        ArrayData::CsrMatrix(data::DynCsrMatrix::F64(arr)) => csr_select(arr, select).into(),
        _ => todo!()
    }
}

fn csr_select<T: Scalar + Zero + Copy>(
    csr: &CsrMatrix<T>,
    select: &[SelectInfoElem],
) -> CsrMatrix<T>
{
    let i = BoundedSelectInfoElem::new(&select[0], csr.nrows()).to_vec();
    let j = BoundedSelectInfoElem::new(&select[1], csr.ncols()).to_vec();
    let mut dm = DMatrix::<T>::zeros(csr.nrows(), csr.ncols());
    csr.triplet_iter().for_each(|(r, c, v)| dm[(r, c)] = *v);
    CsrMatrix::from(&dm.select_rows(&i).select_columns(&j))
}

fn dense_array_select<T: Clone, D: Dimension + RemoveAxis>(
    array: &Array<T, D>,
    select: &[SelectInfoElem],
) -> Array<T, D>
{
    let mut result = array.clone();
    array.shape().into_iter().enumerate().for_each(|(i, &dim)| {
        let idx = BoundedSelectInfoElem::new(&select[i], dim).to_vec();
        result = result.select(Axis(i), idx.as_slice());
    });
    result
}