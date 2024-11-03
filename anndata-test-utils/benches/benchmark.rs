use std::hint::black_box;
use anndata_hdf5::H5;
use anndata_zarr::Zarr;
use criterion::{criterion_group, criterion_main, Criterion};
use anndata_test_utils::with_tmp_dir;
use anndata::*;
use ndarray::{Array, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn bench_array_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("Basic IO");
    group.sample_size(30);

    for n in [100usize, 1000, 2000, 3000].into_iter() {
        with_tmp_dir(|dir| {
            let output = dir.join("test");
            let arr = Array::random([n, n], Uniform::new(-128i64, 127i64));

            let adata = AnnData::<H5>::open(H5::new(&output).unwrap()).unwrap();
            group.bench_with_input(format!("Write H5 ({} x {})", n, n), &arr, |b, arr|
                b.iter(|| adata.set_x(black_box(arr.clone())).unwrap())
            );
            group.bench_function(format!("Read H5 ({} x {})", n, n), |b|
                b.iter(|| adata.x().get::<Array2<i64>>().unwrap())
            );

            let adata = AnnData::<Zarr>::open(Zarr::new(&output).unwrap()).unwrap();
            group.bench_with_input(format!("Write Zarr ({} x {})", n, n), &arr, |b, arr|
                b.iter(|| adata.set_x(black_box(arr.clone())).unwrap())
            );
            group.bench_function(format!("Read Zarr ({} x {})", n, n), |b|
                b.iter(|| adata.x().get::<Array2<i64>>().unwrap())
            );
        })
    }
    group.finish();
}

fn bench_array_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("Slice");

    for n in [1000usize, 2000, 3000].into_iter() {
        with_tmp_dir(|dir| {
            let output = dir.join("test");
            let arr = Array::random([n, n], Uniform::new(-128i64, 127i64));

            let adata = AnnData::<H5>::open(H5::new(&output).unwrap()).unwrap();
            adata.set_x(arr.clone()).unwrap();
            group.bench_function(&format!("H5 {} x {}", n, n), |b| b.iter(||
                adata.x().slice::<Array2<i64>, _>(s![5..591, 5..591]).unwrap()
            ));

            let adata = AnnData::<Zarr>::open(Zarr::new(&output).unwrap()).unwrap();
            adata.set_x(arr).unwrap();
            group.bench_function(&format!("Zarr {} x {}", n, n), |b| b.iter(||
                adata.x().slice::<Array2<i64>, _>(s![5..591, 5..591]).unwrap()
            ));

        })
    }
    group.finish();
}

fn bench_par_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("Multi-read");
    group.sample_size(30);

    for n in [1000usize, 2000].into_iter() {
        with_tmp_dir(|dir| {
            let arr = Array::random([n, n], Uniform::new(-128i64, 127i64));
            
            let adatas = (0..10).map(|i| {
                let output = dir.join(format!("test_{}", i));
                let adata = AnnData::<H5>::open(H5::new(&output).unwrap()).unwrap();
                adata.set_x(arr.clone()).unwrap();
                (i.to_string(), adata)
            }).collect::<Vec<_>>();
            let dataset = AnnDataSet::new(adatas, dir.join("dataset"), "key").unwrap();

            group.bench_function(&format!("H5 series 10 x {} x {}", n, n), |b| b.iter(||
                dataset.x().data::<Array2<i64>>().unwrap().unwrap()
            ));
            group.bench_function(&format!("H5 parallel 10 x {} x {}", n, n), |b| b.iter(||
                dataset.x().par_data::<Array2<i64>>().unwrap().unwrap()
            ));

            let adatas = (0..10).map(|i| {
                let output = dir.join(format!("test_{}", i));
                let adata = AnnData::<Zarr>::open(Zarr::new(&output).unwrap()).unwrap();
                adata.set_x(arr.clone()).unwrap();
                (i.to_string(), adata)
            }).collect::<Vec<_>>();
            let dataset = AnnDataSet::new(adatas, dir.join("dataset"), "key").unwrap();

            group.bench_function(&format!("Zarr series 10 x {} x {}", n, n), |b| b.iter(||
                dataset.x().data::<Array2<i64>>().unwrap().unwrap()
            ));
            group.bench_function(&format!("Zarr parallel 10 x {} x {}", n, n), |b| b.iter(||
                dataset.x().par_data::<Array2<i64>>().unwrap().unwrap()
            ));
        })
    }
    group.finish();
}

criterion_group!(benches, bench_array_io, bench_array_slice, bench_par_read);
criterion_main!(benches);