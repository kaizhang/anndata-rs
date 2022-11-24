pub mod conversion;
pub mod instance;

use crate::utils::instance::*;
use flate2::read::MultiGzDecoder;
use pyo3::{PyAny, PyResult, Python};
use std::fs::File;
use std::path::Path;

/// Determine if a file is gzipped.
pub(crate) fn is_gzipped<P: AsRef<Path>>(file: P) -> bool {
    MultiGzDecoder::new(File::open(file).unwrap())
        .header()
        .is_some()
}

pub(crate) fn open_file<P: AsRef<Path>>(file: P) -> Box<dyn std::io::Read> {
    if is_gzipped(&file) {
        Box::new(MultiGzDecoder::new(File::open(file).unwrap()))
    } else {
        Box::new(File::open(file).unwrap())
    }
}

/*
pub fn to_range<'py>(py: Python<'py>, input: &'py PyAny, length: usize) -> PyResult<Option<Range<usize>>> {
    let res = if input.is_instance_of::<pyo3::types::PySlice>()? {
        let slice = input.downcast::<pyo3::types::PySlice>()?.indices(length.try_into().unwrap())?;
        if slice.step == 0 { Some(slice.start..=slice.stop) } else { None }
    } else {
        None
    };
    Ok(res)
}
*/

pub(crate) fn to_indices<'py>(
    py: Python<'py>,
    input: &'py PyAny,
    length: usize,
) -> PyResult<(Option<Vec<usize>>, bool)> {
    let indices = if is_none_slice(py, input)? {
        (None, false)
    } else if input.is_instance_of::<pyo3::types::PySlice>()? {
        let slice = input
            .downcast::<pyo3::types::PySlice>()?
            .indices(length.try_into().unwrap())?;
        let indices = (slice.start.try_into().unwrap()..slice.stop.try_into().unwrap())
            .step_by(slice.step.try_into().unwrap())
            .collect();
        (Some(indices), false)
    } else if input.is_instance_of::<pyo3::types::PyInt>()? {
        (Some(vec![input.extract::<usize>()?]), false)
    } else if isinstance_of_arr(py, input)?
        && input.getattr("dtype")?.getattr("name")?.extract::<&str>()? == "bool"
    {
        let arr = input
            .extract::<numpy::PyReadonlyArrayDyn<bool>>()?
            .to_owned_array();
        let ndim = arr.ndim();
        let len = arr.len();
        if ndim == 1 && len == length {
            (Some(boolean_mask_to_indices(arr.into_iter())), true)
        } else {
            panic!("boolean mask dimension mismatched")
        }
    } else {
        let boolean_mask: PyResult<Vec<bool>> =
            input.iter()?.map(|x| x.unwrap().extract()).collect();
        match boolean_mask {
            Ok(mask) => {
                if mask.len() == length {
                    (Some(boolean_mask_to_indices(mask.into_iter())), true)
                } else if mask.len() == 0 {
                    (Some(Vec::new()), false)
                } else {
                    panic!("boolean mask dimension mismatched")
                }
            }
            _ => (
                Some(
                    input
                        .iter()?
                        .map(|x| x.unwrap().extract())
                        .collect::<PyResult<_>>()?,
                ),
                false,
            ),
        }
    };
    Ok(indices)
}

fn boolean_mask_to_indices<I>(iter: I) -> Vec<usize>
where
    I: Iterator<Item = bool>,
{
    iter.enumerate()
        .filter_map(|(i, x)| if x { Some(i) } else { None })
        .collect()
}
