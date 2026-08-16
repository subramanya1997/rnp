//! `rnp-core` — a pure-Rust ndarray engine mirroring NumPy's memory model,
//! dtype system and broadcasting rules. No Python dependency lives here.

pub mod array;
pub mod buffer;
pub mod dtype;
pub mod element;
pub mod error;
pub mod iter;
pub mod ops;
pub mod printing;

pub use array::{Flags, NdArray};
pub use buffer::Buffer;
pub use dtype::{promote, promote_for_division, DType, Kind, ALL_DTYPES};
pub use element::{Element, NpBool, Scalar};
pub use error::{Error, Result};
pub use ops::{binary, BinOp};
