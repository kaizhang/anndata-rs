use crate::data::utils::to_csr_data;
use crate::{data::array::DataFrameIndex, AnnDataOp, ArrayData};

use anyhow::Result;
use flate2::read::MultiGzDecoder;
use itertools::Itertools;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use std::path::Path;
use std::{error::Error, fmt, io};
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

pub struct MMReader {
    reader: Box<dyn BufRead>,
    obs_names: Option<DataFrameIndex>,
    var_names: Option<DataFrameIndex>,
    sorted: bool,
}

impl MMReader {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Self {
            reader: open_file(path)?,
            obs_names: None,
            var_names: None,
            sorted: false,
        })
    }

    pub fn obs_names<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        let reader = open_file(path)?;
        let obs_names: Result<DataFrameIndex> = reader
            .lines()
            .map(|line| Ok(line?.split('\t').next().unwrap().to_string()))
            .collect();
        self.obs_names = Some(obs_names?);
        Ok(self)
    }

    pub fn var_names<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        let reader = open_file(path)?;
        let var_names: Result<DataFrameIndex> = reader
            .lines()
            .map(|line| Ok(line?.split('\t').next().unwrap().to_string()))
            .collect();
        self.var_names = Some(var_names?);
        Ok(self)
    }

    pub fn is_sorted(mut self) -> Self {
        self.sorted = true;
        self
    }

    pub fn finish<O: AnnDataOp>(mut self, output: &O) -> Result<()> {
        if self.sorted {
            let (_, cols, iter) = read_sorted_mm_body_from_bufread::<_, f64>(&mut self.reader);
            output.set_x_from_iter(
                iter
                    .chunk_by(|x| x.0)
                    .into_iter()
                    .map(|x| x.1.map(|(_, j, v)| (j, v)).collect::<Vec<_>>())
                    .chunks(2000)
                    .into_iter()
                    .map(|x| {
                        let (r, c, indptr, indices, data) = to_csr_data(x.into_iter().collect::<Vec<_>>(), cols);
                        CsrMatrix::try_from_csr_data(r, c, indptr, indices, data).unwrap()
                    })
            )?;
        } else {
            output.set_x(read_matrix_market_from_bufread(&mut self.reader)?)?;
        }
        if let Some(obs_names) = self.obs_names {
            output.set_obs_names(obs_names)?;
        }
        if let Some(var_names) = self.var_names {
            output.set_var_names(var_names)?;
        }
        Ok(())
    }
}

fn open_file<P: AsRef<Path>>(file: P) -> Result<Box<dyn BufRead>> {
    fn is_gzipped<P: AsRef<Path>>(file: P) -> Result<bool> {
        Ok(MultiGzDecoder::new(File::open(file)?).header().is_some())
    }

    let reader: Box<dyn BufRead> = if is_gzipped(&file)? {
        Box::new(BufReader::new(MultiGzDecoder::new(File::open(file)?)))
    } else {
        Box::new(BufReader::new(File::open(file)?))
    };
    Ok(reader)
}

/*
// TODO: fix dataframe index
pub fn import_csv<P>(
    &self,
    path: P,
    has_header: bool,
    index_column: Option<usize>,
    delimiter: u8,
) -> Result<()>
where
    P: Into<PathBuf>,
{
    let mut df = polars::prelude::CsvReader::from_path(path)?
        .has_header(has_header)
        .with_delimiter(delimiter)
        .finish()?;
    let mut colnames = df.get_column_names_owned();
    if let Some(idx_col) = index_column {
        let series = df.drop_in_place(&colnames.remove(idx_col))?;
        self.set_obs(Some(DataFrame::new(vec![series])?))?;
    }
    if has_header {
        self.set_var(Some(DataFrame::new(vec![Series::new("Index", colnames)])?))?;
    }
    let data: Box<dyn MatrixData> = Box::new(
        df.to_ndarray::<polars::datatypes::Float64Type>()?
            .into_dyn(),
    );
    self.set_x(Some(data))?;
    Ok(())
}
*/

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

fn read_sorted_mm_body_from_bufread<R, T>(
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
            .and_then(|s| s.parse::<usize>().or(Err(BadMatrixMarketFile)))
            .unwrap();
        let col = entry
            .next()
            .ok_or(BadMatrixMarketFile)
            .and_then(|s| s.parse::<usize>().or(Err(BadMatrixMarketFile)))
            .unwrap();
        // MatrixMarket indices are 1-based
        let row = row.checked_sub(1).ok_or(BadMatrixMarketFile).unwrap();
        let col = col.checked_sub(1).ok_or(BadMatrixMarketFile).unwrap();
        let val: T = entry
            .next()
            .ok_or(BadMatrixMarketFile)
            .and_then(|s| s.parse::<T>().or(Err(BadMatrixMarketFile)))
            .unwrap();

        if entry.next().is_some() {
            panic!("BadMatrixMarketFile");
        }
        (row, col, val)
    }).take(entries);
    (rows, cols, iter)
}


fn read_matrix_market_from_bufread<R>(reader: &mut R) -> Result<ArrayData, IoError>
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
            Ok(CsrMatrix::from(&coo).into())
        }
        DataType::Real => {
            let coo: CooMatrix<f64> = read_mtx_body(reader, sym_mode)?;
            Ok(CsrMatrix::from(&coo).into())
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
        let val: T = entry
            .next()
            .ok_or(BadMatrixMarketFile)
            .and_then(|s| s.parse::<T>().or(Err(BadMatrixMarketFile)))?;
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

fn read_header<R>(reader: &mut R) -> Result<(SymmetryMode, DataType), IoError>
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
