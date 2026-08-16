//! The full dtype *descriptor*: a storage [`DType`] plus a byte order, which
//! together are what numpy's `np.dtype` object models.
//!
//! Structured and subarray dtypes are interned: [`registry`] hands out one id
//! per structurally distinct definition, so two descriptors compare equal
//! exactly when numpy's dtypes would. That keeps `Descr` (and therefore
//! `DType`) `Copy`, which the whole engine relies on.
//!
//! The repr/str algorithms below are transcriptions of
//! `upstream/numpy/_core/_dtype.py`, which is where real numpy implements
//! them.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::dtype::{DType, Kind};
use crate::error::{Error, Result};

/// Byte order, already normalised for the host (which we assume is
/// little-endian, as every platform numpy 2.x targets in CI is).
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum ByteOrder {
    /// `=` — native.
    Native,
    /// `<` — explicitly little-endian. Semantically identical to `Native` on
    /// a little-endian host, but numpy keeps the distinction: parsing `'<i4'`
    /// normalises to `=`, while `newbyteorder('<')` leaves a literal `<` that
    /// shows up in `byteorder` and in the repr.
    Little,
    /// `>` — byte-swapped relative to the host.
    Big,
    /// `|` — byte order is not applicable (1-byte and `S`/`V` types).
    NotApplicable,
}

impl ByteOrder {
    pub fn as_char(self) -> char {
        match self {
            ByteOrder::Native => '=',
            ByteOrder::Little => '<',
            ByteOrder::Big => '>',
            ByteOrder::NotApplicable => '|',
        }
    }

    /// `<` and `=` mean the same layout on a little-endian host; equality and
    /// hashing use this normalised form, as numpy's do.
    fn canonical(self) -> ByteOrder {
        match self {
            ByteOrder::Little => ByteOrder::Native,
            other => other,
        }
    }

    /// True when the descriptor's bytes are in the host's order.
    pub fn is_native(self) -> bool {
        self != ByteOrder::Big
    }

    /// `_byte_order_str`: `<`/`>`/`''`, used by the construction reprs.
    pub fn repr_prefix(self) -> &'static str {
        match self {
            ByteOrder::Native | ByteOrder::Little => "<",
            ByteOrder::Big => ">",
            ByteOrder::NotApplicable => "",
        }
    }

    fn swapped(self) -> ByteOrder {
        match self {
            ByteOrder::Native | ByteOrder::Little => ByteOrder::Big,
            // Swapping back lands on an explicit `<`, not on `=`.
            ByteOrder::Big => ByteOrder::Little,
            ByteOrder::NotApplicable => ByteOrder::NotApplicable,
        }
    }

    pub fn from_char(c: char) -> Option<ByteOrder> {
        Some(match c {
            '<' | '=' => ByteOrder::Native,
            '>' => ByteOrder::Big,
            '|' => ByteOrder::NotApplicable,
            _ => return None,
        })
    }
}

/// A C-type spelling that shares a storage dtype with another but keeps its
/// own type number and char code, as `long long` does on platforms where it
/// is the same width as `long`.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, PartialOrd, Ord, Default)]
pub enum Alias {
    #[default]
    None,
    /// `'q'` / `'longlong'`: int64 storage, `num` 9, `char` `'q'`.
    LongLong,
    /// `'Q'` / `'ulonglong'`: uint64 storage, `num` 10, `char` `'Q'`.
    ULongLong,
    /// `'c'`: an `S1` whose char code stays `'c'`.
    Char,
    /// `'g'` / `'longdouble'`: on this platform numpy's long double *is* an
    /// IEEE double, but it keeps `num` 13 and `char` `'g'`.
    LongDouble,
    /// `'G'` / `'clongdouble'`: `num` 16, `char` `'G'`.
    CLongDouble,
}

/// One field of a structured dtype.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Field {
    pub name: String,
    pub descr: Descr,
    pub offset: usize,
    pub title: Option<String>,
}

/// The interned body of a structured dtype.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StructDef {
    pub fields: Vec<Field>,
    pub itemsize: usize,
    pub alignment: usize,
    /// numpy's `isalignedstruct`.
    pub aligned: bool,
}

/// The interned body of a subarray dtype (`('f4', (2, 2))`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SubArrayDef {
    pub base: Descr,
    pub shape: Vec<isize>,
}

/// Process-wide interning of the compound dtype bodies.
pub mod registry {
    use super::*;

    struct Interner<T> {
        items: Vec<Arc<T>>,
        index: HashMap<Arc<T>, u32>,
    }

    impl<T: Eq + std::hash::Hash> Interner<T> {
        fn new() -> Self {
            Interner {
                items: Vec::new(),
                index: HashMap::new(),
            }
        }

        fn intern(&mut self, value: T) -> u32 {
            let arc = Arc::new(value);
            if let Some(&id) = self.index.get(&arc) {
                return id;
            }
            let id = self.items.len() as u32;
            self.items.push(arc.clone());
            self.index.insert(arc, id);
            id
        }
    }

    static STRUCTS: OnceLock<RwLock<Interner<StructDef>>> = OnceLock::new();
    static SUBARRAYS: OnceLock<RwLock<Interner<SubArrayDef>>> = OnceLock::new();

    fn structs() -> &'static RwLock<Interner<StructDef>> {
        STRUCTS.get_or_init(|| RwLock::new(Interner::new()))
    }

    fn subarrays() -> &'static RwLock<Interner<SubArrayDef>> {
        SUBARRAYS.get_or_init(|| RwLock::new(Interner::new()))
    }

    pub fn intern_struct(def: StructDef) -> u32 {
        structs().write().unwrap().intern(def)
    }

    pub fn struct_def(id: u32) -> Arc<StructDef> {
        structs().read().unwrap().items[id as usize].clone()
    }

    pub fn intern_subarray(def: SubArrayDef) -> u32 {
        subarrays().write().unwrap().intern(def)
    }

    pub fn subarray_def(id: u32) -> Arc<SubArrayDef> {
        subarrays().read().unwrap().items[id as usize].clone()
    }
}

/// A complete dtype descriptor: what `np.dtype(...)` builds.
///
/// Equality and hashing ignore the `<` vs `=` spelling of a native byte
/// order, exactly as numpy's do.
#[derive(Copy, Clone, Debug)]
pub struct Descr {
    pub dt: DType,
    pub bo: ByteOrder,
    /// Only affects `num`/`char`; ignored by equality and hashing, exactly
    /// as numpy's `dtype('q') == dtype('l')` is true with different `num`s.
    pub alias: Alias,
}

impl PartialEq for Descr {
    fn eq(&self, other: &Self) -> bool {
        self.dt == other.dt && self.bo.canonical() == other.bo.canonical()
    }
}

impl Eq for Descr {}

impl std::hash::Hash for Descr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dt.hash(state);
        self.bo.canonical().hash(state);
    }
}

impl Descr {
    /// Build a descriptor, normalising the byte order the way numpy does
    /// (`|` for anything where byte order is meaningless, and an explicit
    /// `|` on a multi-byte type means "native").
    pub fn new(dt: DType, bo: ByteOrder) -> Descr {
        let bo = if !dt.byteorder_matters() {
            ByteOrder::NotApplicable
        } else if bo == ByteOrder::NotApplicable {
            ByteOrder::Native
        } else {
            bo
        };
        Descr {
            dt,
            bo,
            alias: Alias::None,
        }
    }

    /// As `new`, tagging the descriptor with a C-type alias.
    pub fn with_alias(dt: DType, bo: ByteOrder, alias: Alias) -> Descr {
        Descr {
            alias,
            ..Descr::new(dt, bo)
        }
    }

    pub fn native(dt: DType) -> Descr {
        Descr::new(dt, ByteOrder::Native)
    }

    pub fn itemsize(&self) -> usize {
        self.dt.itemsize()
    }

    pub fn alignment(&self) -> usize {
        self.dt.alignment()
    }

    pub fn kind(&self) -> char {
        self.dt.kind()
    }

    pub fn num(&self) -> i32 {
        match self.alias {
            Alias::LongLong => 9,
            Alias::ULongLong => 10,
            Alias::LongDouble => 13,
            Alias::CLongDouble => 16,
            Alias::None | Alias::Char => self.dt.num(),
        }
    }

    pub fn char_code(&self) -> char {
        match self.alias {
            Alias::LongLong => 'q',
            Alias::ULongLong => 'Q',
            Alias::Char => 'c',
            Alias::LongDouble => 'g',
            Alias::CLongDouble => 'G',
            Alias::None => self.dt.char_code(),
        }
    }

    pub fn name(&self) -> String {
        self.dt.name()
    }

    /// numpy's `dtype.isnative`: false only for byte-swapped descriptors.
    pub fn isnative(&self) -> bool {
        match self.dt {
            DType::Struct(id) => registry::struct_def(id)
                .fields
                .iter()
                .all(|f| f.descr.isnative()),
            DType::SubArray(id) => registry::subarray_def(id).base.isnative(),
            _ => self.bo.is_native(),
        }
    }

    /// numpy's `dtype.str`, e.g. `<i8`, `|S5`, `<U3`, `|V12`.
    pub fn str_code(&self) -> String {
        let prefix = match self.bo {
            ByteOrder::NotApplicable => '|',
            ByteOrder::Native | ByteOrder::Little => '<',
            ByteOrder::Big => '>',
        };
        if self.dt == DType::Object {
            // numpy prints the object dtype as `|O`, with no size.
            return "|O".to_string();
        }
        if let DType::DateTime(u) | DType::TimeDelta(u) = self.dt {
            let unit = crate::dtype::DATETIME_UNITS[u as usize];
            let suffix = if unit.is_empty() {
                String::new()
            } else {
                format!("[{unit}]")
            };
            return format!("{}{}8{}", prefix, self.dt.kind(), suffix);
        }
        format!("{}{}{}", prefix, self.dt.str_kind(), self.dt.str_size())
    }

    /// The PEP 3118 format string numpy exposes through the buffer protocol,
    /// including the `>` prefix a byte-swapped descriptor carries.
    pub fn buffer_format(&self) -> String {
        let prefix = if self.bo == ByteOrder::Big { ">" } else { "" };
        match self.dt {
            // numpy spells the flexible kinds with a repeat count: `'3s'`,
            // `'3w'` (UCS4) and `'4x'` (opaque padding).
            DType::Bytes(n) => format!("{prefix}{n}s"),
            DType::Str(n) => format!("{prefix}{n}w"),
            DType::Void(n) => format!("{prefix}{n}x"),
            other => format!("{prefix}{}", other.buffer_format()),
        }
    }

    pub fn is_struct(&self) -> bool {
        matches!(self.dt, DType::Struct(_))
    }

    pub fn struct_def(&self) -> Option<Arc<StructDef>> {
        match self.dt {
            DType::Struct(id) => Some(registry::struct_def(id)),
            _ => None,
        }
    }

    pub fn subarray_def(&self) -> Option<Arc<SubArrayDef>> {
        match self.dt {
            DType::SubArray(id) => Some(registry::subarray_def(id)),
            _ => None,
        }
    }

    /// numpy's `dtype.base`: the element type of a subarray, else self.
    pub fn base(&self) -> Descr {
        match self.subarray_def() {
            Some(d) => d.base,
            None => *self,
        }
    }

    /// numpy's `dtype.shape`.
    pub fn shape(&self) -> Vec<isize> {
        match self.subarray_def() {
            Some(d) => d.shape.clone(),
            None => Vec::new(),
        }
    }

    pub fn isalignedstruct(&self) -> bool {
        self.struct_def().map(|d| d.aligned).unwrap_or(false)
    }

    /// `newbyteorder(order)`; `None` means numpy's default `'S'` (swap).
    pub fn newbyteorder(&self, order: Option<char>) -> Result<Descr> {
        let target = match order {
            None | Some('S') | Some('s') => self.bo.swapped(),
            Some('<') | Some('L') | Some('l') => ByteOrder::Little,
            Some('=') | Some('N') | Some('n') => ByteOrder::Native,
            Some('>') | Some('B') | Some('b') => ByteOrder::Big,
            // numpy's NPY_IGNORE: leave the byte order alone.
            Some('|') | Some('I') | Some('i') => self.bo,
            Some(c) => {
                return Err(Error::ValueError(format!(
                    "{c} is an unrecognized byteorder"
                )))
            }
        };
        // Compound dtypes push the change down to their members.
        match self.dt {
            DType::Struct(id) => {
                let def = registry::struct_def(id);
                let mut fields = Vec::with_capacity(def.fields.len());
                for f in &def.fields {
                    fields.push(Field {
                        name: f.name.clone(),
                        descr: f.descr.newbyteorder(order)?,
                        offset: f.offset,
                        title: f.title.clone(),
                    });
                }
                let new = StructDef {
                    fields,
                    itemsize: def.itemsize,
                    alignment: def.alignment,
                    aligned: def.aligned,
                };
                Ok(Descr::new(
                    DType::Struct(registry::intern_struct(new)),
                    ByteOrder::NotApplicable,
                ))
            }
            DType::SubArray(id) => {
                let def = registry::subarray_def(id);
                Ok(make_subarray(def.base.newbyteorder(order)?, def.shape.clone()))
            }
            _ => Ok(Descr::new(self.dt, target)),
        }
    }

    // ---- repr / str, transcribed from numpy/_core/_dtype.py ------------

    /// `_scalar_str(dtype, short)`.
    fn scalar_str(&self, short: bool) -> String {
        let bo = self.bo.repr_prefix();
        match self.dt {
            DType::Bool => {
                if short {
                    "'?'".into()
                } else {
                    "'bool'".into()
                }
            }
            DType::Bytes(n) => {
                if n == 0 {
                    "'S'".into()
                } else {
                    format!("'S{n}'")
                }
            }
            DType::Str(n) => {
                if n == 0 {
                    format!("'{bo}U'")
                } else {
                    format!("'{bo}U{n}'")
                }
            }
            DType::Void(n) => {
                if n == 0 {
                    "'V'".into()
                } else {
                    format!("'V{n}'")
                }
            }
            // numpy always reprs a datetime dtype in its `<M8[unit]` form,
            // never as `datetime64[unit]`.
            DType::DateTime(u) | DType::TimeDelta(u) => {
                let unit = crate::dtype::DATETIME_UNITS[u as usize];
                let suffix = if unit.is_empty() {
                    String::new()
                } else {
                    format!("[{unit}]")
                };
                format!("'{}{}8{}'", bo, self.dt.kind(), suffix)
            }
            d => {
                if short || !matches!(self.bo, ByteOrder::Native | ByteOrder::NotApplicable) {
                    format!("'{}{}{}'", bo, d.kind(), d.itemsize())
                } else {
                    format!("'{}'", d.name())
                }
            }
        }
    }

    /// `_construction_repr`.
    pub fn construction_repr(&self, include_align: bool, short: bool) -> String {
        if self.is_struct() {
            self.struct_str(include_align)
        } else if let Some(sub) = self.subarray_def() {
            format!(
                "({}, {})",
                sub.base.construction_repr(false, true),
                tuple_str(&sub.shape)
            )
        } else {
            self.scalar_str(short)
        }
    }

    /// `_is_packed`.
    fn is_packed(&self) -> bool {
        let def = match self.struct_def() {
            Some(d) => d,
            None => return true,
        };
        let align = def.aligned;
        let mut max_alignment = 1usize;
        let mut total = 0usize;
        for f in &def.fields {
            if align {
                let a = f.descr.alignment().max(1);
                total = total.div_ceil(a) * a;
                max_alignment = max_alignment.max(a);
            }
            if f.offset != total {
                return false;
            }
            total += f.descr.itemsize();
        }
        if align {
            total = total.div_ceil(max_alignment) * max_alignment;
        }
        total == def.itemsize
    }

    /// `_struct_list_str`.
    fn struct_list_str(&self) -> String {
        let def = self.struct_def().unwrap();
        let items: Vec<String> = def
            .fields
            .iter()
            .map(|f| {
                let mut item = String::from("(");
                match &f.title {
                    Some(t) => item.push_str(&format!("({}, {}), ", py_repr(t), py_repr(&f.name))),
                    None => item.push_str(&format!("{}, ", py_repr(&f.name))),
                }
                match f.descr.subarray_def() {
                    Some(sub) => item.push_str(&format!(
                        "{}, {}",
                        sub.base.construction_repr(false, true),
                        tuple_str(&sub.shape)
                    )),
                    None => item.push_str(&f.descr.construction_repr(false, true)),
                }
                item.push(')');
                item
            })
            .collect();
        format!("[{}]", items.join(", "))
    }

    /// `_struct_dict_str`.
    fn struct_dict_str(&self, include_aligned_flag: bool) -> String {
        let def = self.struct_def().unwrap();
        let names: Vec<String> = def.fields.iter().map(|f| py_repr(&f.name)).collect();
        let formats: Vec<String> = def
            .fields
            .iter()
            .map(|f| f.descr.construction_repr(false, true))
            .collect();
        let offsets: Vec<String> = def.fields.iter().map(|f| f.offset.to_string()).collect();
        let mut ret = format!("{{'names': [{}]", names.join(", "));
        ret.push_str(&format!(", 'formats': [{}]", formats.join(", ")));
        ret.push_str(&format!(", 'offsets': [{}]", offsets.join(", ")));
        if def.fields.iter().any(|f| f.title.is_some()) {
            let titles: Vec<String> = def
                .fields
                .iter()
                .map(|f| match &f.title {
                    Some(t) => py_repr(t),
                    None => "None".to_string(),
                })
                .collect();
            ret.push_str(&format!(", 'titles': [{}]", titles.join(", ")));
        }
        ret.push_str(&format!(", 'itemsize': {}", def.itemsize));
        if include_aligned_flag && def.aligned {
            ret.push_str(", 'aligned': True}");
        } else {
            ret.push('}');
        }
        ret
    }

    /// `_struct_str`.
    fn struct_str(&self, include_align: bool) -> String {
        let aligned = self.isalignedstruct();
        if !(include_align && aligned) && self.is_packed() {
            self.struct_list_str()
        } else {
            self.struct_dict_str(include_align)
        }
    }

    /// `repr(dtype)`.
    pub fn repr_string(&self) -> String {
        let mut arg = self.construction_repr(false, false);
        if self.isalignedstruct() {
            arg.push_str(", align=True");
        }
        format!("dtype({arg})")
    }

    /// `str(dtype)`.
    pub fn str_string(&self) -> String {
        if self.is_struct() {
            self.struct_str(true)
        } else if let Some(sub) = self.subarray_def() {
            format!(
                "({}, {})",
                sub.base.construction_repr(false, true),
                tuple_str(&sub.shape)
            )
        } else if self.dt.is_flexible()
            || !matches!(self.bo, ByteOrder::Native | ByteOrder::NotApplicable)
        {
            self.str_code()
        } else {
            self.dt.name()
        }
    }

    // ---- parsing --------------------------------------------------------

    /// Parse any of numpy's dtype *strings*: names, char codes, byte-order
    /// prefixed codes, `S`/`U`/`V` sizes, comma-separated struct formats and
    /// subarray shape prefixes.
    pub fn parse(s: &str) -> Option<Descr> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }
        if has_top_level_comma(s) {
            return parse_comma_struct(s);
        }
        parse_single(s)
    }
}

/// Python's `repr()` for a plain string, which is all the dtype reprs need.
fn py_repr(s: &str) -> String {
    if s.contains('\'') && !s.contains('"') {
        format!("\"{s}\"")
    } else {
        format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
    }
}

/// Python's tuple repr for a shape: `(2,)`, `(2, 3)`.
fn tuple_str(shape: &[isize]) -> String {
    if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn has_top_level_comma(s: &str) -> bool {
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

/// `'i4,f8'` -> a struct with fields `f0`, `f1`.
fn parse_comma_struct(s: &str) -> Option<Descr> {
    let parts = split_top_level_commas(s);
    let mut specs = Vec::with_capacity(parts.len());
    for (i, p) in parts.iter().enumerate() {
        let p = p.trim();
        if p.is_empty() {
            return None;
        }
        specs.push(FieldSpec {
            name: format!("f{i}"),
            descr: parse_single(p)?,
            title: None,
            offset: None,
        });
    }
    make_struct(specs, None, false).ok()
}

/// A single format item: an optional repeat/shape prefix then a scalar spec.
fn parse_single(s: &str) -> Option<Descr> {
    let s = s.trim();
    let (shape, rest) = split_shape_prefix(s)?;
    let base = parse_scalar_spec(rest)?;
    match shape {
        None => Some(base),
        Some(sh) => Some(make_subarray(base, sh)),
    }
}

/// Split a leading `(2,3)` or `3` repeat prefix off a format item.
fn split_shape_prefix(s: &str) -> Option<(Option<Vec<isize>>, &str)> {
    let bytes = s.as_bytes();
    if bytes.first() == Some(&b'(') {
        let end = s.find(')')?;
        let inner = &s[1..end];
        let mut dims = Vec::new();
        for part in inner.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            dims.push(part.parse::<isize>().ok()?);
        }
        return Some((Some(dims), &s[end + 1..]));
    }
    // A leading run of digits is a repeat count -- but only if something
    // follows it that is not itself a size (`'4'` alone is not a dtype).
    let ndigits = bytes.iter().take_while(|c| c.is_ascii_digit()).count();
    if ndigits > 0 && ndigits < s.len() {
        let n: isize = s[..ndigits].parse().ok()?;
        // `'3f8'` is 3 x float64; but `'8'`-style leading digits never occur
        // on their own in a valid spec.
        return Some((Some(vec![n]), &s[ndigits..]));
    }
    Some((None, s))
}

/// `'<f8'`, `'S5'`, `'U'`, `'int32'`, `'?'`.
fn parse_scalar_spec(s: &str) -> Option<Descr> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut bo = None;
    let mut rest = s;
    if let Some(c) = s.chars().next() {
        if let Some(b) = ByteOrder::from_char(c) {
            bo = Some(b);
            rest = &s[c.len_utf8()..];
        }
    }
    if rest.is_empty() {
        return None;
    }
    let bo = bo.unwrap_or(ByteOrder::Native);

    // numpy's `'c'` is a one-byte character, i.e. `S1`.
    if rest == "c" {
        return Some(Descr::with_alias(DType::Bytes(1), bo, Alias::Char));
    }
    // Flexible: S/a/U/V with an optional size.
    let head = rest.as_bytes()[0] as char;
    // ('a' was the old alias for 'S'; numpy 2.x rejects it, so we do too.)
    if matches!(head, 'S' | 'U' | 'V') {
        let tail = &rest[1..];
        // Reject `'V'` followed by junk, but allow the bare letter.
        let n: u32 = if tail.is_empty() {
            0
        } else {
            tail.parse().ok()?
        };
        let dt = match head {
            'S' => DType::Bytes(n),
            'U' => DType::Str(n),
            _ => DType::Void(n),
        };
        return Some(Descr::new(dt, bo));
    }
    // datetime64 / timedelta64, with an optional `[unit]`.
    if let Some(d) = parse_datetime(rest, bo) {
        return Some(d);
    }
    if matches!(rest, "q" | "longlong") {
        return Some(Descr::with_alias(DType::I64, bo, Alias::LongLong));
    }
    if matches!(rest, "Q" | "ulonglong") {
        return Some(Descr::with_alias(DType::U64, bo, Alias::ULongLong));
    }
    // On macOS/arm64 numpy's long double is an IEEE double; the port models
    // it as float64 with numpy's own num/char (a documented M1 gap).
    if matches!(rest, "g" | "longdouble" | "longfloat" | "float128" | "f16") {
        return Some(Descr::with_alias(DType::F64, bo, Alias::LongDouble));
    }
    if matches!(rest, "G" | "clongdouble" | "clongfloat" | "complex256" | "c32") {
        return Some(Descr::with_alias(DType::C128, bo, Alias::CLongDouble));
    }
    let dt = DType::from_plain_name(rest)?;
    Some(Descr::new(dt, bo))
}

/// `M`, `M8`, `M8[ns]`, `datetime64`, `datetime64[us]` and the `m` forms.
fn parse_datetime(rest: &str, bo: ByteOrder) -> Option<Descr> {
    let (head, tail) = if let Some(t) = rest.strip_prefix("datetime64") {
        ('M', t)
    } else if let Some(t) = rest.strip_prefix("timedelta64") {
        ('m', t)
    } else {
        let mut c = rest.chars();
        let first = c.next()?;
        if first != 'M' && first != 'm' {
            return None;
        }
        let t = &rest[first.len_utf8()..];
        // `M8[...]` and `M[...]` are both accepted; `M4` is not a dtype.
        let t = t.strip_prefix('8').unwrap_or(t);
        (first, t)
    };
    let tail = tail.strip_prefix('8').unwrap_or(tail);
    let unit = if tail.is_empty() {
        0
    } else {
        let inner = tail.strip_prefix('[')?.strip_suffix(']')?;
        crate::dtype::datetime_unit_index(inner)?
    };
    let dt = if head == 'M' {
        DType::DateTime(unit)
    } else {
        DType::TimeDelta(unit)
    };
    Some(Descr::new(dt, bo))
}

// ---- structured / subarray construction --------------------------------

/// One field as described by the caller, before layout is computed.
#[derive(Clone, Debug)]
pub struct FieldSpec {
    pub name: String,
    pub descr: Descr,
    pub title: Option<String>,
    pub offset: Option<usize>,
}

fn align_up(offset: usize, alignment: usize) -> usize {
    let a = alignment.max(1);
    offset.div_ceil(a) * a
}

/// Build a structured dtype, laying out any fields whose offsets were not
/// given the way numpy's `_convert_from_*` does.
pub fn make_struct(specs: Vec<FieldSpec>, itemsize: Option<usize>, align: bool) -> Result<Descr> {
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for s in &specs {
        if !seen.insert(s.name.as_str()) {
            return Err(Error::ValueError(format!(
                "field '{}' occurs more than once",
                s.name
            )));
        }
    }

    let mut fields = Vec::with_capacity(specs.len());
    let mut cursor = 0usize;
    let mut max_alignment = 1usize;
    for s in &specs {
        let fa = s.descr.alignment().max(1);
        max_alignment = max_alignment.max(fa);
        let offset = match s.offset {
            Some(o) => o,
            None => {
                if align {
                    cursor = align_up(cursor, fa);
                }
                cursor
            }
        };
        cursor = offset + s.descr.itemsize();
        fields.push(Field {
            name: s.name.clone(),
            descr: s.descr,
            offset,
            title: s.title.clone(),
        });
    }

    let natural = if align {
        align_up(
            fields
                .iter()
                .map(|f| f.offset + f.descr.itemsize())
                .max()
                .unwrap_or(0),
            max_alignment,
        )
    } else {
        fields
            .iter()
            .map(|f| f.offset + f.descr.itemsize())
            .max()
            .unwrap_or(0)
    };
    let size = match itemsize {
        Some(n) => {
            if n < natural {
                return Err(Error::ValueError(format!(
                    "itemsize {n} is too small for the given fields"
                )));
            }
            n
        }
        None => natural,
    };

    let def = StructDef {
        fields,
        itemsize: size,
        alignment: if align { max_alignment } else { 1 },
        aligned: align,
    };
    Ok(Descr::new(
        DType::Struct(registry::intern_struct(def)),
        ByteOrder::NotApplicable,
    ))
}

/// Build a subarray dtype.
///
/// A subarray of a subarray *nests* rather than flattening, matching numpy:
/// `np.dtype((np.dtype(('i4', (2,))), (3,)))` keeps `('<i4', (2,))` as its
/// base. Because the nesting is held through an interned id, arbitrarily
/// deep chains cost O(1) each to build.
pub fn make_subarray(base: Descr, shape: Vec<isize>) -> Descr {
    if shape.is_empty() {
        return base;
    }
    let def = SubArrayDef { base, shape };
    Descr::new(
        DType::SubArray(registry::intern_subarray(def)),
        ByteOrder::NotApplicable,
    )
}

/// numpy's `dtype.kind` grouping used by `np.issubdtype`-style questions.
pub fn kind_of(d: Descr) -> Kind {
    d.dt.category()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Descr {
        Descr::parse(s).unwrap_or_else(|| panic!("failed to parse {s:?}"))
    }

    #[test]
    fn byteorder_round_trips() {
        // Probed from numpy 2.5.2.
        assert_eq!(p("<f8").repr_string(), "dtype('float64')");
        assert_eq!(p(">i4").repr_string(), "dtype('>i4')");
        assert_eq!(p("=f4").repr_string(), "dtype('float32')");
        assert_eq!(p("|b1").repr_string(), "dtype('bool')");
        assert_eq!(p(">i4").str_code(), ">i4");
        assert_eq!(p("<i2").str_code(), "<i2");
        assert_eq!(p("|b1").str_code(), "|b1");
        assert_eq!(p("<f8").bo.as_char(), '=');
        assert_eq!(p("|b1").bo.as_char(), '|');
        assert!(!p(">f8").isnative());
        assert!(p("<f8").isnative());
    }

    #[test]
    fn newbyteorder_matches_numpy() {
        assert_eq!(p("i4").newbyteorder(None).unwrap(), p(">i4"));
        assert_eq!(p("i4").newbyteorder(Some('>')).unwrap(), p(">i4"));
        assert_eq!(p(">i4").newbyteorder(Some('=')).unwrap(), p("i4"));
        assert_eq!(p("i4").newbyteorder(Some('S')).unwrap(), p(">i4"));
        // `newbyteorder('<')` keeps a literal `<`, which the repr shows.
        assert_eq!(
            p("i4").newbyteorder(Some('<')).unwrap().repr_string(),
            "dtype('<i4')"
        );
        assert_eq!(p("i4").newbyteorder(Some('<')).unwrap(), p("i4"));
        // `'|'` means "ignore": the byte order is untouched.
        assert_eq!(
            p(">i4").newbyteorder(Some('|')).unwrap().repr_string(),
            "dtype('>i4')"
        );
        // Single-byte types have no byte order to change.
        assert_eq!(p("b").newbyteorder(Some('>')).unwrap(), p("b"));
    }

    #[test]
    fn longlong_keeps_its_own_type_number() {
        // Probed: np.dtype('q').num == 9 and char == 'q', yet it compares
        // equal to (and hashes like) int64.
        assert_eq!(p("q").num(), 9);
        assert_eq!(p("q").char_code(), 'q');
        assert_eq!(p("Q").num(), 10);
        assert_eq!(p("Q").char_code(), 'Q');
        assert_eq!(p("longlong").num(), 9);
        assert_eq!(p("q"), p("l"));
        assert_eq!(p("q").name(), "int64");
        assert_eq!(p("q").repr_string(), "dtype('int64')");
        assert_eq!(p("l").num(), 7);
        // np.dtype('c') is an S1 that keeps its own char code.
        assert_eq!(p("c").char_code(), 'c');
        assert_eq!(p("c"), p("S1"));
        assert_eq!(p("c").repr_string(), "dtype('S1')");
    }

    #[test]
    fn flexible_specs() {
        assert_eq!(p("S5").dt, DType::Bytes(5));
        assert_eq!(p("S5").repr_string(), "dtype('S5')");
        assert_eq!(p("S5").str_code(), "|S5");
        assert_eq!(p("U3").dt, DType::Str(3));
        assert_eq!(p("U3").repr_string(), "dtype('<U3')");
        assert_eq!(p("U3").str_code(), "<U3");
        assert_eq!(p("U3").itemsize(), 12);
        assert_eq!(p("V10").repr_string(), "dtype('V10')");
        assert_eq!(p("S").repr_string(), "dtype('S')");
        assert_eq!(p("U").repr_string(), "dtype('<U')");
        assert_eq!(p("V").repr_string(), "dtype('V')");
        assert_eq!(p(">U3").repr_string(), "dtype('>U3')");
    }

    #[test]
    fn comma_and_shape_specs() {
        // np.dtype('i4,f8')
        let d = p("i4,f8");
        assert_eq!(d.repr_string(), "dtype([('f0', '<i4'), ('f1', '<f8')])");
        assert_eq!(d.itemsize(), 12);
        assert_eq!(d.str_code(), "|V12");
        // np.dtype('(2,2)f4')
        let s = p("(2,2)f4");
        assert_eq!(s.repr_string(), "dtype(('<f4', (2, 2)))");
        assert_eq!(s.itemsize(), 16);
        assert_eq!(s.alignment(), 4);
        // np.dtype('3f8')
        let t = p("3f8");
        assert_eq!(t.repr_string(), "dtype(('<f8', (3,)))");
        assert_eq!(t.itemsize(), 24);
    }

    #[test]
    fn structured_layout_and_repr() {
        let spec = |n: &str, d: &str| FieldSpec {
            name: n.into(),
            descr: p(d),
            title: None,
            offset: None,
        };
        // np.dtype([('a','i4'),('b','f8')])
        let d = make_struct(vec![spec("a", "i4"), spec("b", "f8")], None, false).unwrap();
        assert_eq!(d.repr_string(), "dtype([('a', '<i4'), ('b', '<f8')])");
        assert_eq!(d.itemsize(), 12);
        assert_eq!(d.alignment(), 1);
        // np.dtype([('a','i1'),('b','f8')], align=True)
        let a = make_struct(vec![spec("a", "i1"), spec("b", "f8")], None, true).unwrap();
        assert_eq!(a.repr_string(), "dtype([('a', 'i1'), ('b', '<f8')], align=True)");
        assert_eq!(a.itemsize(), 16);
        assert_eq!(a.alignment(), 8);
        assert_eq!(a.struct_def().unwrap().fields[1].offset, 8);
        // Explicit offsets force the dict repr.
        let o = make_struct(
            vec![
                FieldSpec {
                    name: "a".into(),
                    descr: p("i4"),
                    title: None,
                    offset: Some(0),
                },
                FieldSpec {
                    name: "b".into(),
                    descr: p("f8"),
                    title: None,
                    offset: Some(8),
                },
            ],
            Some(20),
            false,
        )
        .unwrap();
        assert_eq!(
            o.repr_string(),
            "dtype({'names': ['a', 'b'], 'formats': ['<i4', '<f8'], 'offsets': [0, 8], 'itemsize': 20})"
        );
    }

    #[test]
    fn interning_makes_equality_structural() {
        let d1 = Descr::parse("i4,f8").unwrap();
        let d2 = Descr::parse("i4,f8").unwrap();
        assert_eq!(d1, d2);
        assert_eq!(d1.dt, d2.dt);
        let d3 = Descr::parse("i4,f4").unwrap();
        assert_ne!(d1, d3);
    }

    #[test]
    fn nested_and_subarray_fields() {
        let inner = make_struct(
            vec![
                FieldSpec {
                    name: "x".into(),
                    descr: p("i4"),
                    title: None,
                    offset: None,
                },
                FieldSpec {
                    name: "y".into(),
                    descr: p("f4"),
                    title: None,
                    offset: None,
                },
            ],
            None,
            false,
        )
        .unwrap();
        let outer = make_struct(
            vec![
                FieldSpec {
                    name: "a".into(),
                    descr: inner,
                    title: None,
                    offset: None,
                },
                FieldSpec {
                    name: "b".into(),
                    descr: p("f8"),
                    title: None,
                    offset: None,
                },
            ],
            None,
            false,
        )
        .unwrap();
        assert_eq!(
            outer.repr_string(),
            "dtype([('a', [('x', '<i4'), ('y', '<f4')]), ('b', '<f8')])"
        );
        assert_eq!(outer.itemsize(), 16);

        // np.dtype([('a','f4',(2,3)),('b','i8')])
        let sub = make_struct(
            vec![
                FieldSpec {
                    name: "a".into(),
                    descr: make_subarray(p("f4"), vec![2, 3]),
                    title: None,
                    offset: None,
                },
                FieldSpec {
                    name: "b".into(),
                    descr: p("i8"),
                    title: None,
                    offset: None,
                },
            ],
            None,
            false,
        )
        .unwrap();
        assert_eq!(
            sub.repr_string(),
            "dtype([('a', '<f4', (2, 3)), ('b', '<i8')])"
        );
        assert_eq!(sub.itemsize(), 32);
    }

    #[test]
    fn str_string_matches_numpy() {
        // str(np.dtype('i4')) == 'int32'; str(np.dtype('>i4')) == '>i4'
        assert_eq!(p("i4").str_string(), "int32");
        assert_eq!(p(">i4").str_string(), ">i4");
        assert_eq!(p("S5").str_string(), "|S5");
    }
}
