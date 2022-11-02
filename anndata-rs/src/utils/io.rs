use crate::data::DataPartialIO;

use std::error::Error;
use std::fmt;
use std::io;

use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::csr::CsrMatrix;

#[derive(Debug)]
pub(crate) enum IoError {
    Io(io::Error),
    BadMatrixMarketFile,
    UnsupportedMatrixMarketFormat,
}

use self::IoError::*;

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Io(ref err) => err.fmt(f),
            Self::BadMatrixMarketFile | Self::UnsupportedMatrixMarketFormat => {
                write!(f, "Bad matrix market file.")
            }
        }
    }
}

impl Error for IoError {}

impl From<io::Error> for IoError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum DataType {
    Integer,
    Real,
    Complex,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum SymmetryMode {
    General,
    Hermitian,
    Symmetric,
    SkewSymmetric,
}

pub(crate) fn read_sorted_mm_body_from_bufread<R, T>(
    reader: &mut R,
) -> (usize, usize, impl Iterator<Item = (usize, usize, T)> + '_)
where
    R: io::BufRead,
    T: Copy + std::str::FromStr,
{
    // MatrixMarket format specifies lines of at most 1024 chars
    let mut line = String::with_capacity(1024);

    // The header is followed by any number of comment or empty lines, skip
    'header: loop {
        line.clear();
        let len = reader.read_line(&mut line).unwrap();
        if len == 0 || line.starts_with('%') {
            continue 'header;
        }
        break;
    }
    // read shape and number of entries
    // this is a line like:
    // rows cols entries
    // with arbitrary amounts of whitespace
    let (rows, cols, entries) = {
        let mut infos = line
            .split_whitespace()
            .filter_map(|s| s.parse::<usize>().ok());
        let rows = infos.next().ok_or(BadMatrixMarketFile).unwrap();
        let cols = infos.next().ok_or(BadMatrixMarketFile).unwrap();
        let entries = infos.next().ok_or(BadMatrixMarketFile).unwrap();
        if infos.next().is_some() {
            panic!("BadMatrixMarketFile");
        }
        (rows, cols, entries)
    };
    // one non-zero entry per non-empty line
    let iter = std::iter::repeat_with(move || {
        // skip empty lines (no comment line should appear)
        'empty_lines: loop {
            line.clear();
            let len = reader.read_line(&mut line).unwrap();
            // check for an all whitespace line
            if len != 0 && line.split_whitespace().next() == None {
                continue 'empty_lines;
            }
            break;
        }
        // Non-zero entries are lines of the form:
        // row col value
        // if the data type is integer of real, and
        // row col real imag
        // if the data type is complex.
        // Again, this is with arbitrary amounts of whitespace
        let mut entry = line.split_whitespace();
        let row = entry
            .next()
            .ok_or(BadMatrixMarketFile)
            .and_then(|s| s.parse::<usize>().or(Err(BadMatrixMarketFile))).unwrap();
        let col = entry
            .next()
            .ok_or(BadMatrixMarketFile)
            .and_then(|s| s.parse::<usize>().or(Err(BadMatrixMarketFile))).unwrap();
        // MatrixMarket indices are 1-based
        let row = row.checked_sub(1).ok_or(BadMatrixMarketFile).unwrap();
        let col = col.checked_sub(1).ok_or(BadMatrixMarketFile).unwrap();
        let val: T = entry.next().ok_or(BadMatrixMarketFile).and_then(|s| {
            s.parse::<T>().or(Err(BadMatrixMarketFile))
        }).unwrap();

        if entry.next().is_some() {
            panic!("BadMatrixMarketFile");
        }
        (row, col, val)
    }).take(entries);
    (rows, cols, iter)
}

pub(crate) fn read_matrix_market_from_bufread<R>(
    reader: &mut R,
) -> Result<Box<dyn DataPartialIO>, IoError>
where
    R: io::BufRead,
{
    let (sym_mode, data_type) = read_header(reader)?;
    if data_type == DataType::Complex {
        // we currently don't support complex
        return Err(UnsupportedMatrixMarketFormat);
    }
    if sym_mode == SymmetryMode::Hermitian {
        // support for Hermitian requires complex support
        return Err(UnsupportedMatrixMarketFormat);
    }
    match data_type {
        DataType::Integer => {
            let coo: CooMatrix<i64> = read_mtx_body(reader, sym_mode)?;
            Ok(Box::new(CsrMatrix::from(&coo)))
        }
        DataType::Real => {
            let coo: CooMatrix<f64> = read_mtx_body(reader, sym_mode)?;
            Ok(Box::new(CsrMatrix::from(&coo)))
        }
        DataType::Complex => unreachable!(),
    }
}

fn read_mtx_body<T, R>(reader: &mut R, sym_mode: SymmetryMode) -> Result<CooMatrix<T>, IoError>
where
    R: io::BufRead,
    T: Copy + std::str::FromStr,
{
    // MatrixMarket format specifies lines of at most 1024 chars
    let mut line = String::with_capacity(1024);

    // The header is followed by any number of comment or empty lines, skip
    'header: loop {
        line.clear();
        let len = reader.read_line(&mut line)?;
        if len == 0 || line.starts_with('%') {
            continue 'header;
        }
        break;
    }
    // read shape and number of entries
    // this is a line like:
    // rows cols entries
    // with arbitrary amounts of whitespace
    let (rows, cols, entries) = {
        let mut infos = line
            .split_whitespace()
            .filter_map(|s| s.parse::<usize>().ok());
        let rows = infos.next().ok_or(BadMatrixMarketFile)?;
        let cols = infos.next().ok_or(BadMatrixMarketFile)?;
        let entries = infos.next().ok_or(BadMatrixMarketFile)?;
        if infos.next().is_some() {
            return Err(BadMatrixMarketFile);
        }
        (rows, cols, entries)
    };
    let nnz_max = if sym_mode == SymmetryMode::General {
        entries
    } else {
        2 * entries
    };
    let mut row_inds = Vec::with_capacity(nnz_max);
    let mut col_inds = Vec::with_capacity(nnz_max);
    let mut data = Vec::with_capacity(nnz_max);
    // one non-zero entry per non-empty line
    for _ in 0..entries {
        // skip empty lines (no comment line should appear)
        'empty_lines: loop {
            line.clear();
            let len = reader.read_line(&mut line)?;
            // check for an all whitespace line
            if len != 0 && line.split_whitespace().next() == None {
                continue 'empty_lines;
            }
            break;
        }
        // Non-zero entries are lines of the form:
        // row col value
        // if the data type is integer of real, and
        // row col real imag
        // if the data type is complex.
        // Again, this is with arbitrary amounts of whitespace
        let mut entry = line.split_whitespace();
        let row = entry
            .next()
            .ok_or(BadMatrixMarketFile)
            .and_then(|s| s.parse::<usize>().or(Err(BadMatrixMarketFile)))?;
        let col = entry
            .next()
            .ok_or(BadMatrixMarketFile)
            .and_then(|s| s.parse::<usize>().or(Err(BadMatrixMarketFile)))?;
        // MatrixMarket indices are 1-based
        let row = row.checked_sub(1).ok_or(BadMatrixMarketFile)?;
        let col = col.checked_sub(1).ok_or(BadMatrixMarketFile)?;
        let val: T = entry.next().ok_or(BadMatrixMarketFile).and_then(|s| {
            s.parse::<T>().or(Err(BadMatrixMarketFile))
        })?;
        row_inds.push(row);
        col_inds.push(col);
        data.push(val);
        if sym_mode != SymmetryMode::General && row != col {
            if sym_mode == SymmetryMode::Hermitian {
                unreachable!();
            } else {
                row_inds.push(col);
                col_inds.push(row);
                data.push(val);
            }
        }
        if sym_mode == SymmetryMode::SkewSymmetric && row == col {
            return Err(BadMatrixMarketFile);
        }
        if entry.next().is_some() {
            return Err(BadMatrixMarketFile);
        }
    }

    CooMatrix::try_from_triplets(rows, cols, row_inds, col_inds, data)
        .map_err(|_| BadMatrixMarketFile)
}

pub(crate) fn read_header<R>(
    reader: &mut R,
) -> Result<(SymmetryMode, DataType), IoError>
where
    R: io::BufRead,
{
    // MatrixMarket format specifies lines of at most 1024 chars
    let mut line = String::with_capacity(1024);

    // Parse the header line, all tags are case insensitive.
    reader.read_line(&mut line)?;
    let header = line.to_lowercase();
    parse_header(&header)
}

fn parse_header(header: &str) -> Result<(SymmetryMode, DataType), IoError> {
    if !header.starts_with("%%matrixmarket matrix coordinate") {
        return Err(BadMatrixMarketFile);
    }
    let data_type = if header.contains("real") {
        DataType::Real
    } else if header.contains("integer") {
        DataType::Integer
    } else if header.contains("complex") {
        DataType::Complex
    } else {
        return Err(BadMatrixMarketFile);
    };
    let sym_mode = if header.contains("general") {
        SymmetryMode::General
    } else if header.contains("symmetric") {
        SymmetryMode::Symmetric
    } else if header.contains("skew-symmetric") {
        SymmetryMode::SkewSymmetric
    } else if header.contains("hermitian") {
        SymmetryMode::Hermitian
    } else {
        return Err(BadMatrixMarketFile);
    };
    Ok((sym_mode, data_type))
}