//! Errors carrying the Python exception class they should surface as.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    ValueError(String),
    TypeError(String),
    IndexError(String),
    NotImplemented(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn message(&self) -> &str {
        match self {
            Error::ValueError(m)
            | Error::TypeError(m)
            | Error::IndexError(m)
            | Error::NotImplemented(m) => m,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for Error {}
