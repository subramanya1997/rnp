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
    /// numpy's `_UFuncBinaryResolutionError` (also a `UFuncTypeError`),
    /// raised by the datetime type resolvers when no loop combination fits:
    /// `ufunc 'add' cannot use operands with types dtype('<M8[s]') and ...`.
    UFuncBinaryResolution {
        ufunc: String,
        /// The two operand dtypes, as `dtype.str`-style strings.
        dtypes: Vec<String>,
        message: String,
    },
    /// numpy's `_UFuncInputCastingError` (also a `UFuncTypeError`): the type
    /// resolver picked a loop, but one *input* cannot be cast to the dtype that
    /// loop wants under the ufunc's casting rule.
    UFuncInputCasting {
        ufunc: String,
        /// The casting rule name, e.g. `"same_kind"`.
        casting: String,
        /// `dtype.str`-style spelling of the operand's own dtype.
        from_: String,
        /// `dtype.str`-style spelling of the dtype the loop needs.
        to: String,
        /// Which input (0-based).
        i: usize,
        message: String,
    },
    OverflowError(String),
    RuntimeError(String),
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
            | Error::OverflowError(m)
            | Error::RuntimeError(m)
            | Error::NotImplemented(m) => m,
            Error::UFuncNoLoop { message, .. } => message,
            Error::UFuncBinaryResolution { message, .. } => message,
            Error::UFuncInputCasting { message, .. } => message,
        }
    }
}

/// Build numpy's `_UFuncInputCastingError` for input `i` of `ufunc`. `from_`
/// and `to` are `dtype.str`-style spellings, e.g. `"<m8[Y]"`; `nin` is the
/// ufunc's input count, because numpy omits the index for unary ufuncs.
pub fn ufunc_input_casting(
    ufunc: &str,
    casting: &str,
    from_: &str,
    to: &str,
    i: usize,
    nin: usize,
) -> Error {
    let i_str = if nin != 1 {
        format!("{i} ")
    } else {
        String::new()
    };
    Error::UFuncInputCasting {
        ufunc: ufunc.to_string(),
        casting: casting.to_string(),
        from_: from_.to_string(),
        to: to.to_string(),
        i,
        message: format!(
            "Cannot cast ufunc {ufunc:?} input {i_str}from dtype({from_:?}) \
             to dtype({to:?}) with casting rule {casting:?}"
        ),
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

/// Build numpy's `_UFuncBinaryResolutionError` message for `ufunc` over two
/// operand dtypes, spelled the way `repr(np.dtype(...))` spells them.
pub fn ufunc_binary_resolution(ufunc: &str, a: &str, b: &str) -> Error {
    Error::UFuncBinaryResolution {
        ufunc: ufunc.to_string(),
        dtypes: vec![a.to_string(), b.to_string()],
        message: format!(
            "ufunc {ufunc:?} cannot use operands with types dtype('{a}') and dtype('{b}')"
        ),
    }
}
