use anndata::*;
use anndata_hdf5::H5;
use anndata_n5::N5;

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray::{Array, Array1, Array2, Array3};
use criterion::BenchmarkId;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use proptest::array;
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

fn array_io<B: Backend>(name: &str, c: &mut Criterion) {
    with_tmp_path(|file| {
        let mut group = c.benchmark_group(name);

        // Prepare data
        let n = 200;
        let m = 300;
        let z = 10;
        let adata: AnnData<H5> = AnnData::new(file, n, m).unwrap();
        let arr: Array3<i32> = Array::random((n, m, z), Uniform::new(-100, 100));
        let mut fancy_d1: Vec<usize> = Array::random((30,), Uniform::new(0, n-1)).to_vec();
        let mut fancy_d2: Vec<usize> = Array::random((40,), Uniform::new(0, m-1)).to_vec();
        adata.set_x(&arr).unwrap();

        group.bench_function(
            BenchmarkId::new("read full", "200 x 300 x 10"),
            |b| b.iter(|| adata.read_x::<ArrayData>().unwrap().unwrap()),
        );

        group.bench_function(
            BenchmarkId::new("read slice", "30 x 40 x 10"),
            |b| b.iter(|| adata.read_x_slice::<ArrayData, _>(s![3..33, 4..44, ..]).unwrap().unwrap())
        );

        /*
        group.bench_function(
            BenchmarkId::new("read fancy", "30 x 40 x 10"),
            |b| b.iter(|| adata.read_x_slice::<ArrayData, _>(s![&fancy_d1, &fancy_d2, ..]).unwrap().unwrap())
        );

        fancy_d1.sort();
        fancy_d2.sort();

        group.bench_function(
            BenchmarkId::new("read fancy (sorted)", "30 x 40 x 10"),
            |b| b.iter(|| adata.read_x_slice::<ArrayData, _>(s![&fancy_d1, &fancy_d2, ..]).unwrap().unwrap())
        );
        */

        group.finish();
    })
}

fn parallel_io<B: Backend>(name: &str, c: &mut Criterion) {
    with_tmp_dir(|dir| {
        let mut group = c.benchmark_group(name);

        let n = 50;
        let m = 10000;
        let arr: Array2<i32> = Array::random((n, m), Uniform::new(-100, 100));

        let d1: AnnData<B> = AnnData::new(dir.join("1.h5ad"), 0, 0).unwrap();
        let d2: AnnData<B> = AnnData::new(dir.join("2.h5ad"), 0, 0).unwrap();
        let d3: AnnData<B> = AnnData::new(dir.join("3.h5ad"), 0, 0).unwrap();
        d1.set_x(&arr).unwrap();
        d2.set_x(&arr).unwrap();
        d3.set_x(&arr).unwrap();
        let dataset = AnnDataSet::new(
            [("1", d1), ("2", d2), ("3", d3)],
            dir.join("dataset.h5ads"), "key"
        ).unwrap();

        group.bench_function(
            BenchmarkId::new("Serial read", "50 x 10000 (3)"),
            |b| b.iter(|| dataset.get_x().data::<ArrayData>().unwrap()),
        );

        group.bench_function(
            BenchmarkId::new("Parallel read", "50 x 10000 (3)"),
            |b| b.iter(|| dataset.get_x().par_data::<ArrayData>().unwrap()),
        );
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

fn array_io_h5(c: &mut Criterion) { array_io::<H5>("Array IO (HDF5 backend)", c); }
fn array_io_n5(c: &mut Criterion) { array_io::<N5>("Array IO (N5 backend)", c); }
fn parallel_io_h5(c: &mut Criterion) { parallel_io::<H5>("Parallel IO (HDF5 backend)", c); }
fn parallel_io_n5(c: &mut Criterion) { parallel_io::<N5>("Parallel IO (N5 backend)", c); }

criterion_group!(benches, array_io_h5, array_io_n5, parallel_io_h5, parallel_io_n5);
criterion_main!(benches);
