//! `rnp-core` — a pure-Rust ndarray engine mirroring NumPy's memory model,
//! dtype system and broadcasting rules. No Python dependency lives here.

pub mod array;
pub mod buffer;
pub mod casting;
pub mod datetime;
pub mod datetime_ops;
pub mod descr;
pub mod dtype;
pub mod element;
pub mod error;
pub mod fpe;
pub mod indexing;
pub mod iter;
pub mod loops;
pub mod ops;
pub mod printing;
pub mod reduce;
pub mod sort;
pub mod ufunc;

pub use array::{Flags, NdArray};
pub use buffer::Buffer;
pub use casting::{can_cast, common_type, min_scalar_type, result_type, Casting, TypeArg, WeakKind};
pub use descr::{ByteOrder, Descr, Field, FieldSpec, StructDef, SubArrayDef};
pub use dtype::{promote, promote_for_division, DType, Kind, ALL_DTYPES};
pub use element::{Element, NpBool, Scalar, F16};
pub use indexing::{IndexItem, Indexed, SliceSpec, TakeMode};
pub use reduce::{reduce_all, reduce_axis, reduce_dtype, ReduceOp};
pub use error::{Error, Result};
pub use ops::{binary, divmod, BinOp};
pub use ufunc::{unary, UnOp};
