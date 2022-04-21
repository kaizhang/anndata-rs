pub mod conversion;
pub mod instance;

use flate2::read::MultiGzDecoder;
use std::fs::File;
use crate::utils::instance::isinstance_of_arr;
use pyo3::{
    PyResult,
    PyAny,
    Python,
};

/// Determine if a file is gzipped.
pub(crate) fn is_gzipped(file: &str) -> bool {
    MultiGzDecoder::new(File::open(file).unwrap()).header().is_some()
}

pub(crate) fn to_indices<'py>(py: Python<'py>, input: &'py PyAny, length: usize) -> PyResult<Vec<usize>> {
    if input.is_instance_of::<pyo3::types::PySlice>()? {
        let slice = input.downcast::<pyo3::types::PySlice>()?.indices(
            length.try_into().unwrap()
        )?;
        Ok(
            (slice.start.try_into().unwrap() ..= slice.stop.try_into().unwrap())
            .step_by(slice.step.try_into().unwrap()).collect()
        )
    } else if isinstance_of_arr(py, input)? {
        match input.getattr("dtype")?.getattr("name")?.extract::<&str>()? {
            "bool" => {
                let arr = input.extract::<numpy::PyReadonlyArrayDyn<bool>>()?.to_owned_array();
                let ndim = arr.ndim();
                let len = arr.len();
                if ndim == 1 && len == length {
                    Ok(
                        arr.into_iter().enumerate().filter_map(|(i, x)|
                            if x { Some(i) } else { None }).collect()
                    )
                } else {
                    panic!("dimension mismatched")
                }
            },
            "int64" => {
                let arr = input.extract::<numpy::PyReadonlyArrayDyn<i64>>()?.to_owned_array();
                let ndim = arr.ndim();
                if ndim == 1 {
                    Ok(arr.into_iter().map(|x| x.try_into().unwrap()).collect())
                } else {
                    panic!("dimension mismatched")
                }
            }
            ty => panic!("{}", ty),
        }
    } else if input.is_instance_of::<pyo3::types::PyList>()? {
        Ok(input.extract::<Vec<i64>>()?.into_iter().map(|x| x.try_into().unwrap()).collect())
    } else {
        todo!()
    }
}