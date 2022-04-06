pub mod conversion;
pub mod instance;

use flate2::read::MultiGzDecoder;
use std::fs::File;

/// Determine if a file is gzipped.
pub fn is_gzipped(file: &str) -> bool {
    MultiGzDecoder::new(File::open(file).unwrap()).header().is_some()
}