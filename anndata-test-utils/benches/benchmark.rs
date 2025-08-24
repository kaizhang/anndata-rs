use std::io::{Read, Write};
use std::path::Path;

use anndata_hdf5::H5;
use anndata_zarr::Zarr;
use bincode::config::Configuration;
use criterion::{criterion_group, criterion_main, Criterion};
use anndata_test_utils::with_tmp_dir;
use anndata::*;
use ndarray::{Array, Array2, Array3, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

fn save_arr_split(arr: &Array3<i64>, output: &str) -> Result<Vec<usize>, std::io::Error> {
    let mut sizes = Vec::new();
    let mut file = std::fs::File::create(output)?;
    arr.axis_iter(Axis(0)).for_each(|slice| {
        let data = bincode::serde::encode_to_vec::<_, Configuration>(slice, Configuration::default()).unwrap();
        let data = zstd::bulk::Compressor::new(9).unwrap().compress(&data).unwrap();
        sizes.push(data.len());
        file.write_all(&data).unwrap();
    });
        
    file.flush()?;
    Ok(sizes)
}

fn save_arr_split_lz(arr: &Array3<i64>, output: &str) -> Result<Vec<usize>, std::io::Error> {
    let mut sizes = Vec::new();
    let mut file = std::fs::File::create(output)?;
    arr.axis_iter(Axis(0)).for_each(|slice| {
        let data = bincode::serde::encode_to_vec::<_, Configuration>(slice, Configuration::default()).unwrap();
        let data = lz4_flex::compress_prepend_size(&data);
        sizes.push(data.len());
        file.write_all(&data).unwrap();
    });
        
    file.flush()?;
    Ok(sizes)
}

fn save_arr_lz(arr: &Array3<i64>, output: &str) {
    let mut file = std::fs::File::create(output).unwrap();
    let data = bincode::serde::encode_to_vec::<_, Configuration>(arr, Configuration::default()).unwrap();
    let data = lz4_flex::compress_prepend_size(&data);
    file.write_all(&data).unwrap();
    file.flush().unwrap();
}

fn save_arr(arr: &Array3<i64>, output: &str) {
    let mut file = std::fs::File::create(output).unwrap();
    let data = bincode::serde::encode_to_vec::<_, Configuration>(arr, Configuration::default()).unwrap();
    let data = zstd::bulk::Compressor::new(9).unwrap().compress(&data).unwrap();
    file.write_all(&data).unwrap();
    file.flush().unwrap();
}

fn load_arr_split(input: impl AsRef<Path>, sizes: &[usize]) -> Array3<i64> {
    let mut file = std::fs::File::open(input).unwrap();
    let values: Vec<_> = sizes.iter().map(|size| {
        let mut buffer = vec![0; *size];
        file.read_exact(&mut buffer).unwrap();
        let mut decoder = zstd::stream::Decoder::new(std::io::Cursor::new(buffer)).unwrap();
        let data: Array2<i64> = bincode::serde::decode_from_std_read::<_, Configuration, _>(
            &mut decoder,
            Configuration::default(),
        ).unwrap();
        data.insert_axis(Axis(2))
    }).collect();
    let values = values.iter().map(|arr| arr.view()).collect::<Vec<_>>();
    let arr = ndarray::concatenate(Axis(2), &values).unwrap();
    arr.as_standard_layout().into_owned()
}

fn load_arr_split_lz(input: impl AsRef<Path>, sizes: &[usize]) -> Array3<i64> {
    let mut file = std::fs::File::open(input).unwrap();
    let values: Vec<_> = sizes.iter().map(|size| {
        let mut buffer = vec![0; *size];
        file.read_exact(&mut buffer).unwrap();
        let decoder = lz4_flex::decompress_size_prepended(buffer.as_slice()).unwrap();
        let data: Array2<i64> = bincode::serde::decode_from_slice::<_, Configuration>(
            &decoder,
            Configuration::default(),
        ).unwrap().0;
        data.insert_axis(Axis(2))
    }).collect();
    let values = values.iter().map(|arr| arr.view()).collect::<Vec<_>>();
    let arr = ndarray::concatenate(Axis(2), &values).unwrap();
    arr.as_standard_layout().into_owned()
}

fn par_load_arr_split(input: impl AsRef<Path>, sizes: &[usize]) -> Array3<i64> {
    let mut file = std::fs::File::open(input).unwrap();
    let raw_bytes: Vec<_> = sizes.iter().map(|size| {
        let mut buffer = vec![0; *size];
        file.read_exact(&mut buffer).unwrap();
        buffer
    }).collect();

    let values: Vec<_> = raw_bytes.into_par_iter().map(|buffer| {
        let mut decoder = zstd::stream::Decoder::new(std::io::Cursor::new(buffer)).unwrap();
        let data: Array2<i64> = bincode::serde::decode_from_std_read::<_, Configuration, _>(
            &mut decoder,
            Configuration::default(),
        ).unwrap();
        data.insert_axis(Axis(2))
    }).collect();
    let values = values.iter().map(|arr| arr.view()).collect::<Vec<_>>();
    let arr = ndarray::concatenate(Axis(2), &values).unwrap();
    arr.as_standard_layout().into_owned()
}

fn par_load_arr_split_lz(input: impl AsRef<Path>, sizes: &[usize]) -> Array3<i64> {
    let mut file = std::fs::File::open(input).unwrap();
    let raw_bytes: Vec<_> = sizes.iter().map(|size| {
        let mut buffer = vec![0; *size];
        file.read_exact(&mut buffer).unwrap();
        buffer
    }).collect();

    let values: Vec<_> = raw_bytes.into_par_iter().map(|buffer| {
        let decoder = lz4_flex::decompress_size_prepended(buffer.as_slice()).unwrap();
        let data: Array2<i64> = bincode::serde::decode_from_slice::<_, Configuration>(
            &decoder,
            Configuration::default(),
        ).unwrap().0;
        data.insert_axis(Axis(2))
    }).collect();
    let values = values.iter().map(|arr| arr.view()).collect::<Vec<_>>();
    let arr = ndarray::concatenate(Axis(2), &values).unwrap();
    arr.as_standard_layout().into_owned()
}

fn load_arr(input: impl AsRef<Path>) -> Array3<i64> {
    let mut file = std::fs::File::open(input).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();
    let mut decoder = zstd::stream::Decoder::new(std::io::Cursor::new(buffer)).unwrap();
    bincode::serde::decode_from_std_read::<_, Configuration, _>(
        &mut decoder,
        Configuration::default(),
    ).unwrap()
}

fn load_arr_lz(input: impl AsRef<Path>) -> Array3<i64> {
    let mut file = std::fs::File::open(input).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();
    let decoder = lz4_flex::decompress_size_prepended(buffer.as_slice()).unwrap();
    bincode::serde::decode_from_slice::<_, Configuration>(
        &decoder,
        Configuration::default(),
    ).unwrap().0
}

fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("Compression");
    group.sample_size(10);

    for n in [10usize, 50, 100].into_iter() {
        with_tmp_dir(|dir| {
            let arr = Array::random([128, 1534, n], Uniform::new(-128i64, 127i64));

            let file_zst_split = dir.join("test_split.zstd");
            let file_zst = dir.join("test.zstd");
            let sizes = save_arr_split(&arr, file_zst_split.to_str().unwrap()).unwrap();
            save_arr(&arr, file_zst.to_str().unwrap());

            group.bench_with_input(format!("Array Zstd Split ({})", n), &(file_zst_split.clone(), sizes.clone()), |b, (file, sizes)|
                b.iter(|| load_arr_split(file, &sizes))
            );
            group.bench_with_input(format!("Array Zstd Split Parallel ({})", n), &(file_zst_split, sizes), |b, (file, sizes)|
                b.iter(|| par_load_arr_split(file, &sizes))
            );
            group.bench_with_input(format!("Array Zstd Batch ({})", n), &file_zst, |b, file|
                b.iter(|| load_arr(file))
            );

            let file_lz_split = dir.join("test_split.lz");
            let file_lz = dir.join("test.lz");
            let sizes = save_arr_split_lz(&arr, file_lz_split.to_str().unwrap()).unwrap();
            save_arr_lz(&arr, file_lz.to_str().unwrap());

            group.bench_with_input(format!("Array Lz4 Split ({})", n), &(file_lz_split.clone(), sizes.clone()), |b, (file, sizes)|
                b.iter(|| load_arr_split_lz(file, &sizes))
            );
            group.bench_with_input(format!("Array Lz4 Split Parallel ({})", n), &(file_lz_split, sizes), |b, (file, sizes)|
                b.iter(|| par_load_arr_split_lz(file, &sizes))
            );
            group.bench_with_input(format!("Array Lz4 Batch ({})", n), &(file_lz.clone()), |b, file|
                b.iter(|| load_arr_lz(file))
            );
        })
    }
    group.finish();
}

fn bench_array_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("Basic IO");
    group.sample_size(30);

    for n in [100usize, 1000, 2000, 3000].into_iter() {
        with_tmp_dir(|dir| {
            let output = dir.join("test");
            let arr = Array::random([n, n], Uniform::new(-128i64, 127i64));

            let adata = AnnData::<H5>::open(H5::new(&output).unwrap()).unwrap();
            group.bench_with_input(format!("Write H5 ({} x {})", n, n), &arr, |b, arr|
                b.iter(|| adata.set_x(arr.clone()).unwrap())
            );
            group.bench_function(format!("Read H5 ({} x {})", n, n), |b|
                b.iter(|| adata.x().get::<Array2<i64>>().unwrap())
            );

            let adata = AnnData::<Zarr>::open(Zarr::new(&output).unwrap()).unwrap();
            group.bench_with_input(format!("Write Zarr ({} x {})", n, n), &arr, |b, arr|
                b.iter(|| adata.set_x(arr.clone()).unwrap())
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

//criterion_group!(benches, bench_zstd_compression, bench_array_io, bench_array_slice, bench_par_read);
criterion_group!(benches, bench_compression);
criterion_main!(benches);