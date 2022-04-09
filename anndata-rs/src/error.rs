#[derive(Debug)]
pub enum AnnDataError {
    IO(io::Error),
}

impl fmt::Display for AnnDataError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::IO(ref err) => err.fmt(f),
            Self::BadMatrixMarketFile | Self::UnsupportedMatrixMarketFormat => {
                write!(f, "Bad matrix market file.")
            }
        }
    }
}

impl Error for IoError {}

impl From<io::Error> for AnnDataError {
    fn from(err: io::Error) -> Self {
        Self::IO(err)
    }
}


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