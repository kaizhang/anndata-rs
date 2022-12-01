use anndata_rs::*;
use anndata_rs::backend::hdf5::H5;

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray::{Array, Array1, Array2, Array3};
use criterion::BenchmarkId;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hdf5::*;
use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use rand::Rng;
use std::path::PathBuf;
use tempfile::tempdir;

pub fn with_tmp_dir<T, F: FnMut(PathBuf) -> T>(mut func: F) -> T {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    func(path)
}

fn with_tmp_path<T, F: FnMut(PathBuf) -> T>(mut func: F) -> T {
    with_tmp_dir(|dir| func(dir.join("foo.h5")))
}

fn with_tmp_file<T, F: FnMut(File) -> T>(mut func: F) -> T {
    with_tmp_path(|path| {
        let file = File::create(&path).unwrap();
        func(file)
    })
}

fn bench_array_io(c: &mut Criterion) {
    with_tmp_path(|file| {
        let mut group = c.benchmark_group("Array IO");

        // Prepare data
        let n = 200;
        let m = 300;
        let z = 10;
        let adata: AnnData<H5> = AnnData::new(file, n, m).unwrap();
        let arr: Array3<i32> = Array::random((n, m, z), Uniform::new(-100, 100));
        let fancy_d1: Array1<usize> = Array::random((30,), Uniform::new(0, n-1));
        let fancy_d2: Array1<usize> = Array::random((40,), Uniform::new(0, m-1));
        adata.set_x(&arr).unwrap();

        group.bench_function(
            BenchmarkId::new("read full", "200 x 300 x 10"),
            |b| b.iter(|| adata.read_x::<ArrayData>().unwrap().unwrap()),
        );

        group.bench_function(
            BenchmarkId::new("read slice", "30 x 40 x 10"),
            |b| b.iter(|| adata.read_x_slice::<ArrayData, _>(s![3..33, 4..44, ..]).unwrap().unwrap())
        );

        group.bench_function(
            BenchmarkId::new("read fancy", "30 x 40 x 10"),
            |b| b.iter(|| adata.read_x_slice::<ArrayData, _>(s![&fancy_d1, &fancy_d2, ..]).unwrap().unwrap())
        );

        group.finish();
    })
}

/*
fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Read_CSR");
    group.sample_size(10);
    let mut rng = rand::thread_rng();
    let n: usize = 10000;
    let m: usize = 10000;
    let nnz: usize = 5000000;
    let mat: CsrMatrix<i64> = {
        let values: Vec<i64> = vec![1; nnz];
        let (row_indices, col_indices) = (0..nnz)
            .map(|_| (rng.gen_range(0..n), rng.gen_range(0..m)))
            .unzip();
        (&CooMatrix::try_from_triplets(n, m, row_indices, col_indices, values).unwrap()).into()
    };
    let row_indices: Vec<usize> = (0..9000).map(|_| rng.gen_range(0..n)).collect();
    let mut col_indices: Vec<usize> = (0..9000).map(|_| rng.gen_range(0..m)).collect();

    let input = with_tmp_file(|file| {
        let g = file.create_group("foo").unwrap();
        mat.write(&g, "x").unwrap()
    });

    let dataset = input.get_group_ref().unwrap().dataset("data").unwrap();
    println!(
        "dataset contains {} chunks with shape: {:#?}",
        dataset.num_chunks().unwrap_or(0),
        dataset.chunk().unwrap_or(Vec::new()),
    );
    println!("{:#?}", dataset.access_plist().unwrap());

    group.bench_with_input(BenchmarkId::new("full", ""), &input, |b, i| {
        b.iter(|| {
            CsrMatrix::<i64>::read(i).unwrap();
        });
    });

    group.bench_with_input(BenchmarkId::new("slice", ""), &input, |b, i| {
        b.iter(|| {
            CsrMatrix::<i64>::read_row_slice(i, 10..1000);
        });
    });

    group.bench_with_input(BenchmarkId::new("rows", ""), &input, |b, i| {
        b.iter(|| {
            CsrMatrix::<i64>::read_rows(i, row_indices.as_slice());
        });
    });

    group.bench_with_input(
        BenchmarkId::new("columns (baseline)", ""),
        &input,
        |b, i| {
            b.iter(|| {
                CsrMatrix::<i64>::read(i)
                    .unwrap()
                    .get_columns(col_indices.as_slice())
            })
        },
    );
    group.bench_with_input(BenchmarkId::new("columns", ""), &input, |b, i| {
        b.iter(|| CsrMatrix::<i64>::read_columns(i, col_indices.as_slice()))
    });
    col_indices.sort();
    group.bench_with_input(
        BenchmarkId::new("columns (pre-sorted)", ""),
        &input,
        |b, i| {
            b.iter(|| {
                CsrMatrix::<i64>::read_columns(i, col_indices.as_slice());
            });
        },
    );

    group.finish();
}

fn criterion_multi_threading(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_threading");
    group.sample_size(10);
    let mut rng = rand::thread_rng();
    let n: usize = 10000;
    let m: usize = 10000;
    let nnz: usize = 1000000;
    let mat: Box<dyn MatrixData> = {
        let values: Vec<i64> = vec![1; nnz];
        let (row_indices, col_indices) = (0..nnz)
            .map(|_| (rng.gen_range(0..n), rng.gen_range(0..m)))
            .unzip();
        let m: CsrMatrix<i64> =
            (&CooMatrix::try_from_triplets(n, m, row_indices, col_indices, values).unwrap()).into();
        Box::new(m)
    };
    let row_indices: Vec<usize> = (0..9000).map(|_| rng.gen_range(0..n * 3)).collect();

    with_tmp_dir(|dir| {
        let data = (0..3)
            .into_iter()
            .map(|i| {
                let d = AnnData::new(dir.join(format!("{}.h5ad", i)), 0, 0).unwrap();
                d.set_x(Some(&mat)).unwrap();
                (format!("{}", i), d)
            })
            .collect();
        let input = AnnDataSet::new(data, dir.join("1.h5ads"), "sample").unwrap();
        group.bench_with_input(BenchmarkId::new("full", ""), &input, |b, i| {
            b.iter(|| {
                i.write_subset(Some(row_indices.as_slice()), None, dir.join("subset"))
                    .unwrap();
            });
        });
    });

    group.finish();
}
*/

criterion_group!(benches, bench_array_io);
criterion_main!(benches);
