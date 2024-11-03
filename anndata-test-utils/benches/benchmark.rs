use std::hint::black_box;
use anndata_hdf5::H5;
use anndata_zarr::Zarr;
use criterion::{criterion_group, criterion_main, Criterion};
use anndata_test_utils::with_tmp_dir;
use anndata::*;
use ndarray::{Array, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn bench_array_io(n: usize, c: &mut Criterion) {
    with_tmp_dir(|dir| {
        let output = dir.join("test");
        let arr = Array::random([n, n], Uniform::new(-128i64, 127i64));

        let adata = AnnData::<H5>::open(H5::new(&output).unwrap()).unwrap();
        c.bench_function(&format!("Write {} x {} (H5)", n, n), |b| b.iter(||
            adata.set_x(black_box(arr.clone())).unwrap()
        ));
        c.bench_function(&format!("Read {} x {} (H5)",n, n), |b| b.iter(||
            adata.x().get::<Array2<i64>>().unwrap()
        ));

        let adata = AnnData::<Zarr>::open(Zarr::new(&output).unwrap()).unwrap();
        c.bench_function(&format!("Write {} x {} (Zarr)", n, n), |b| b.iter(||
            adata.set_x(black_box(arr.clone())).unwrap()
        ));
        c.bench_function(&format!("Read {} x {} (Zarr)",n, n), |b| b.iter(||
            adata.x().get::<Array2<i64>>().unwrap()
        ));
    })
}

fn bench_array_100(c: &mut Criterion) {
    bench_array_io(100, c);
}

fn bench_array_1000(c: &mut Criterion) {
    bench_array_io(1000, c);
}

fn bench_array_2000(c: &mut Criterion) {
    bench_array_io(2000, c);
}

fn bench_array_3000(c: &mut Criterion) {
    bench_array_io(3000, c);
}

fn bench_array_slice(n: usize, c: &mut Criterion) {
    with_tmp_dir(|dir| {
        let output = dir.join("test");
        let arr = Array::random([n, n], Uniform::new(-128i64, 127i64));

        let adata = AnnData::<H5>::open(H5::new(&output).unwrap()).unwrap();
        adata.set_x(arr.clone()).unwrap();
        c.bench_function(&format!("Slice {} x {} (H5)", n, n), |b| b.iter(||
            adata.x().slice::<Array2<i64>, _>(s![5..91, 5..91]).unwrap()
        ));

        let adata = AnnData::<Zarr>::open(Zarr::new(&output).unwrap()).unwrap();
        adata.set_x(arr).unwrap();
        c.bench_function(&format!("Slice {} x {} (Zarr)", n, n), |b| b.iter(||
            adata.x().slice::<Array2<i64>, _>(s![5..91, 5..91]).unwrap()
        ));
    })
}

fn slice_100(c: &mut Criterion) {
    bench_array_slice(100, c);
}

fn slice_1000(c: &mut Criterion) {
    bench_array_slice(1000, c);
}

fn slice_2000(c: &mut Criterion) {
    bench_array_slice(2000, c);
}

criterion_group!(
    benches,
    bench_array_100, bench_array_1000, bench_array_2000, bench_array_3000,
    slice_100, slice_1000, slice_2000,
);
criterion_main!(benches);