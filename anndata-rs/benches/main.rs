use anndata_rs::anndata_trait::*;
use anndata_rs::anndata::{AnnData, AnnDataSet};

use criterion::{criterion_group, criterion_main, Criterion};
use criterion::BenchmarkId;
use hdf5::*;
use tempfile::tempdir;
use std::path::PathBuf;
use rand::Rng;
use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::csr::CsrMatrix;

pub fn with_tmp_dir<T, F: FnMut(PathBuf) -> T>(mut func: F) -> T {
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

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Read_CSR");
    group.sample_size(10);
    let mut rng = rand::thread_rng();
    let n: usize = 10000;
    let m: usize = 10000;
    let nnz: usize = 5000000;
    let mat: CsrMatrix<i64> = {
        let values: Vec<i64> = vec![1; nnz];
        let (row_indices, col_indices) = (0..nnz).map(|_| (rng.gen_range(0..n), rng.gen_range(0..m) )).unzip();
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
        |b, i| { b.iter(|| CsrMatrix::<i64>::read(i).unwrap().get_columns(col_indices.as_slice())) },
    );
    group.bench_with_input(
        BenchmarkId::new("columns", ""),
        &input,
        |b, i| { b.iter(|| CsrMatrix::<i64>::read_columns(i, col_indices.as_slice())) },
    );
    col_indices.sort();
    group.bench_with_input(BenchmarkId::new("columns (pre-sorted)", ""), &input, |b, i| {
        b.iter(|| {
            CsrMatrix::<i64>::read_columns(i, col_indices.as_slice());
        });
    });

    group.finish();
}

fn criterion_multi_threading(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_threading");
    group.sample_size(10);
    let mut rng = rand::thread_rng();
    let n: usize = 10000;
    let m: usize = 10000;
    let nnz: usize = 1000000;
    let mat: Box<dyn DataPartialIO> = {
        let values: Vec<i64> = vec![1; nnz];
        let (row_indices, col_indices) = (0..nnz).map(|_| (rng.gen_range(0..n), rng.gen_range(0..m) )).unzip();
        let m: CsrMatrix<i64> = (&CooMatrix::try_from_triplets(n, m, row_indices, col_indices, values).unwrap()).into();
        Box::new(m)
    };
    let row_indices: Vec<usize> = (0..9000).map(|_| rng.gen_range(0..n*3)).collect();

    with_tmp_dir(|dir| {
        let data = (0..3).into_iter().map(|i| {
            let d = AnnData::new(dir.join(format!("{}.h5ad", i)), 0, 0).unwrap();
            d.set_x(Some(&mat)).unwrap();
            (format!("{}", i), d)
        }).collect();
        let input = AnnDataSet::new(data, dir.join("1.h5ads"), "sample").unwrap();
        group.bench_with_input(BenchmarkId::new("full", ""), &input, |b, i| {
            b.iter(|| {
                i.write_subset(
                    Some(row_indices.as_slice()),
                    None,
                    dir.join("subset"),
                ).unwrap();
            });
        });
    });

    group.finish();
}



criterion_group!(benches, criterion_benchmark, criterion_multi_threading);
criterion_main!(benches);