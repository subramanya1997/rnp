//! Errors carrying the Python exception class they should surface as.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    ValueError(String),
    TypeError(String),
    IndexError(String),
    /// numpy's `np.exceptions.AxisError` (a subclass of ValueError/IndexError).
    AxisError(String),
    /// numpy's `np.exceptions.DTypePromotionError`.
    DTypePromotionError(String),
    /// numpy's `numpy._core._exceptions._UFuncNoLoopError` (displayed as
    /// `UFuncTypeError`, a `TypeError` subclass). Raised by the ufuncs whose
    /// type resolver refuses to upcast the operands -- numpy's
    /// `PyUFunc_SimpleUniformOperationTypeResolver` and friends.
    UFuncNoLoop {
        /// The ufunc's name, e.g. `"gcd"`.
        ufunc: String,
        /// The resolved input dtype names, e.g. `["float64", "float64"]`.
        dtypes: Vec<String>,
        /// numpy's rendering of the above, used when the shim has not
        /// registered a richer factory.
        message: String,
    },
    NotImplemented(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn message(&self) -> &str {
        match self {
            Error::ValueError(m)
            | Error::TypeError(m)
            | Error::IndexError(m)
            | Error::AxisError(m)
            | Error::DTypePromotionError(m)
            | Error::NotImplemented(m) => m,
            Error::UFuncNoLoop { message, .. } => message,
        }
    }
}

/// Build numpy's `_UFuncNoLoopError` for `ufunc` over the given input dtype
/// names. The message is byte-identical to numpy's `__str__`, which prints the
/// *DType classes* rather than the dtype instances.
pub fn ufunc_no_loop(ufunc: &str, dtypes: &[&str]) -> Error {
    let rendered: Vec<String> = dtypes
        .iter()
        .map(|d| format!("<class 'numpy.dtypes.{}'>", dtype_class_name(d)))
        .collect();
    // numpy unpacks a 1-tuple, so a unary ufunc prints a bare class.
    let inputs = if rendered.len() == 1 {
        rendered[0].clone()
    } else {
        format!("({})", rendered.join(", "))
    };
    Error::UFuncNoLoop {
        ufunc: ufunc.to_string(),
        dtypes: dtypes.iter().map(|d| d.to_string()).collect(),
        message: format!(
            "ufunc {ufunc:?} did not contain a loop with signature matching \
             types {inputs} -> None"
        ),
    }
}

/// `"float64"` -> `"Float64DType"`, matching `numpy.dtypes`.
fn dtype_class_name(name: &str) -> String {
    match name {
        "bool" => "BoolDType".to_string(),
        _ => {
            let mut out = String::new();
            let mut chars = name.chars().peekable();
            // `uint8` -> `UInt8DType`, everything else just title-cases.
            if name.starts_with("uint") {
                out.push_str("UInt");
                for _ in 0..4 {
                    chars.next();
                }
            } else if let Some(c) = chars.next() {
                out.extend(c.to_uppercase());
            }
            out.extend(chars);
            out.push_str("DType");
            out
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for Error {}
