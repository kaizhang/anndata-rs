pub mod conversion;
pub mod instance;

use flate2::read::MultiGzDecoder;
use std::fs::File;
use crate::utils::instance::*;
use pyo3::{types::PyIterator, PyResult, PyAny, Python};

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
    } else if input.is_instance_of::<pyo3::types::PyInt>()? {
        Ok(vec![input.extract::<usize>()?])
    } else if isinstance_of_arr(py, input)? && input.getattr("dtype")?.getattr("name")?.extract::<&str>()? == "bool" {
        let arr = input.extract::<numpy::PyReadonlyArrayDyn<bool>>()?.to_owned_array();
        let ndim = arr.ndim();
        let len = arr.len();
        if ndim == 1 && len == length {
            Ok(boolean_mask_to_indices(arr.into_iter()))
        } else {
            panic!("boolean mask dimension mismatched")
        }
    } else {
        let boolean_mask: PyResult<Vec<bool>> = PyIterator::from_object(py, input)?
            .map(|x| x.unwrap().extract()).collect();
        match boolean_mask {
            Ok(mask) => if mask.len() == length {
                Ok(boolean_mask_to_indices(mask.into_iter()))
            } else if mask.len() == 0 {
                Ok(Vec::new())
            } else {
                panic!("boolean mask dimension mismatched")
            },
            _ => PyIterator::from_object(py, input)?.map(|x| x.unwrap().extract()).collect(),
        }
    }
}

fn boolean_mask_to_indices<I>(iter: I) -> Vec<usize>
where
    I: Iterator<Item = bool>
{
    iter.enumerate().filter_map(|(i, x)| if x { Some(i) } else { None }).collect()
}