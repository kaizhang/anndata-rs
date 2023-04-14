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