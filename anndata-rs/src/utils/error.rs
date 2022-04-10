use std::{error, fmt, io};


#[derive(Debug)]
pub enum Error {
    IO(io::Error),
    HDF5(hdf5::Error),
    Internal(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<&str> for Error {
    fn from(desc: &str) -> Self {
        Self::Internal(desc.into())
    }
}

impl From<String> for Error {
    fn from(desc: String) -> Self {
        Self::Internal(desc)
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Self::IO(err)
    }
}

impl From<hdf5::Error> for Error {
    fn from(err: hdf5::Error) -> Self {
        Self::HDF5(err)
    }
}

impl error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::IO(ref err) => err.fmt(f),
            Self::HDF5(ref err) => err.fmt(f),
            Self::Internal(ref err) => err.fmt(f),
            /*
            Self::BadMatrixMarketFile | Self::UnsupportedMatrixMarketFormat => {
                write!(f, "Bad matrix market file.")
            }
            */
        }
    }
}



/*
#[derive(Debug)]
pub enum AnnDataIOError {
    IO(io::Error),
    MatrixIO(),
}

impl fmt::Display for AnnDataIOError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::IO(ref err) => err.fmt(f),
            Self::BadMatrixMarketFile | Self::UnsupportedMatrixMarketFormat => {
                write!(f, "Bad matrix market file.")
            }
        }
    }
}

impl Error for AnnDataIOError {}

impl From<io::Error> for AnnDataError {
    fn from(err: io::Error) -> Self {
        Self::IO(err)
    }
}
*/