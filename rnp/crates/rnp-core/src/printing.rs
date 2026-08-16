//! A numpy-flavoured `repr` for arrays.
//!
//! This is deliberately a subset of numpy's `arrayprint` (M5 owns the real
//! thing): common-width element alignment, decimal-point alignment for
//! floats, line wrapping at 75 columns, and `...` summarisation past 1000
//! elements.

use crate::{DType, NdArray, Scalar};

const LINE_WIDTH: usize = 75;
const THRESHOLD: usize = 1000;
const EDGE_ITEMS: usize = 3;

/// dtypes numpy omits from the repr because they are the platform defaults.
fn is_default_dtype(d: DType) -> bool {
    matches!(d, DType::Bool | DType::I64 | DType::F64 | DType::C128)
}

/// The shortest decimal string that round-trips through `dtype`.
///
/// Rust's `{}` gives the shortest round-trip for `f64`, which is wrong for
/// narrower dtypes: `np.float32(0.1)` prints as `0.1`, not as the `f64`
/// spelling `0.10000000149011612` of the same bits.
fn shortest_for_dtype(v: f64, dtype: DType) -> f64 {
    if dtype == DType::F64 || !v.is_finite() {
        return v;
    }
    let matches = |cand: f64| -> bool {
        match dtype {
            DType::F32 => (cand as f32).to_bits() == (v as f32).to_bits(),
            DType::F16 => {
                crate::element::f64_to_f16_bits(cand) == crate::element::f64_to_f16_bits(v)
            }
            _ => cand.to_bits() == v.to_bits(),
        }
    };
    for prec in 1..=17 {
        let s = format!("{:.*e}", prec - 1, v);
        if let Ok(cand) = s.parse::<f64>() {
            if matches(cand) {
                return cand;
            }
        }
    }
    v
}

/// Shortest round-trip decimal for a float, in numpy's style: a trailing
/// `.` instead of `.0`, and `inf`/`nan` spelled out.
fn fmt_float(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let s = format!("{}", v);
    if s.contains('e') || s.contains('.') {
        s
    } else {
        format!("{}.", s)
    }
}

/// Split a formatted float into (integer-ish part, fractional part) so a
/// column of them can be decimal-point aligned.
fn split_at_point(s: &str) -> (&str, &str) {
    match s.find('.') {
        Some(i) => (&s[..=i], &s[i + 1..]),
        None => (s, ""),
    }
}

struct Formatter {
    strings: Vec<String>,
    width: usize,
}

impl Formatter {
    /// Format every element up-front so widths can be reconciled, which is
    /// what numpy does too.
    fn build(arr: &NdArray) -> Formatter {
        let values = arr.to_vec();
        let mut strings: Vec<String> = match arr.dtype() {
            // numpy always reserves the width of "False" for bools, so a
            // lone True still prints as `array([ True])`.
            DType::Bool => values
                .iter()
                .map(|v| match v {
                    Scalar::Bool(true) => " True".to_string(),
                    _ => "False".to_string(),
                })
                .collect(),
            d if d.is_integer() => values
                .iter()
                .map(|v| match v {
                    Scalar::Int(i) => i.to_string(),
                    Scalar::Uint(u) => u.to_string(),
                    other => format!("{:?}", other),
                })
                .collect(),
            d if d.is_float() => {
                let raw: Vec<String> = values
                    .iter()
                    .map(|v| match v {
                        Scalar::Float(f) => fmt_float(shortest_for_dtype(*f, d)),
                        other => format!("{:?}", other),
                    })
                    .collect();
                align_decimals(&raw)
            }
            d if d.is_complex() => {
                let comp = if d == DType::C64 { DType::F32 } else { DType::F64 };
                let (re, im): (Vec<f64>, Vec<f64>) = values
                    .iter()
                    .map(|v| match v {
                        Scalar::Complex(c) => (c.re, c.im),
                        _ => (0.0, 0.0),
                    })
                    .unzip();
                let re_s = align_decimals(
                    &re.iter()
                        .map(|&x| fmt_float(shortest_for_dtype(x, comp)))
                        .collect::<Vec<_>>(),
                );
                let im_s = align_decimals(
                    &im.iter()
                        .map(|&x| fmt_float(shortest_for_dtype(x.abs(), comp)))
                        .collect::<Vec<_>>(),
                );
                re_s.iter()
                    .zip(im_s.iter())
                    .zip(im.iter())
                    .map(|((r, i), &imv)| {
                        let sign = if imv < 0.0 || (imv == 0.0 && imv.is_sign_negative()) {
                            '-'
                        } else {
                            '+'
                        };
                        format!("{}{}{}j", r, sign, i)
                    })
                    .collect()
            }
            // Flexible dtypes have no numeric formatter yet (M5 owns the
            // real arrayprint); show the raw bytes' repr placeholder.
            _ => values.iter().map(|v| format!("{v:?}")).collect(),
        };
        let width = strings.iter().map(|s| s.len()).max().unwrap_or(0);
        for s in &mut strings {
            // Right-align, as numpy does for every numeric column.
            while s.len() < width {
                s.insert(0, ' ');
            }
        }
        Formatter { strings, width }
    }
}

/// Pad a column of float strings so their decimal points line up.
fn align_decimals(raw: &[String]) -> Vec<String> {
    let int_w = raw.iter().map(|s| split_at_point(s).0.len()).max().unwrap_or(0);
    let frac_w = raw.iter().map(|s| split_at_point(s).1.len()).max().unwrap_or(0);
    raw.iter()
        .map(|s| {
            let (a, b) = split_at_point(s);
            format!("{:>iw$}{:<fw$}", a, b, iw = int_w, fw = frac_w)
        })
        .collect()
}

/// Which indices along an axis to print, with `None` marking the `...` gap.
fn axis_indices(len: isize, summarize: bool) -> Vec<Option<isize>> {
    if !summarize || len <= 2 * EDGE_ITEMS as isize {
        return (0..len).map(Some).collect();
    }
    let mut v: Vec<Option<isize>> = (0..EDGE_ITEMS as isize).map(Some).collect();
    v.push(None);
    v.extend((len - EDGE_ITEMS as isize..len).map(Some));
    v
}

struct Ctx {
    fmt: Formatter,
    shape: Vec<isize>,
    summarize: bool,
}

impl Ctx {
    /// Flat C-order position of a multi-index.
    fn flat(&self, index: &[isize]) -> usize {
        let mut acc = 0usize;
        for (i, &ix) in index.iter().enumerate() {
            acc = acc * self.shape[i] as usize + ix as usize;
        }
        acc
    }

    fn render(&self, index: &mut Vec<isize>, indent: usize) -> String {
        let depth = index.len();
        let ndim = self.shape.len();
        if depth == ndim {
            return self.fmt.strings[self.flat(index)].clone();
        }
        let items = axis_indices(self.shape[depth], self.summarize);
        if depth == ndim - 1 {
            // Innermost row: join with ", " and wrap at LINE_WIDTH.
            let mut line = String::from("[");
            let mut col = indent + 1;
            let mut first = true;
            for it in items {
                let s = match it {
                    None => "...".to_string(),
                    Some(i) => {
                        index.push(i);
                        let s = self.render(index, 0);
                        index.pop();
                        s
                    }
                };
                if !first {
                    line.push(',');
                    col += 1;
                    if col + 1 + s.len() > LINE_WIDTH {
                        line.push('\n');
                        line.push_str(&" ".repeat(indent + 1));
                        col = indent + 1;
                    } else {
                        line.push(' ');
                        col += 1;
                    }
                }
                line.push_str(&s);
                col += s.len();
                first = false;
            }
            line.push(']');
            return line;
        }
        // Outer axis: one child per line, blank lines between higher blocks.
        let gap = "\n".repeat(ndim - depth - 1);
        let mut parts = Vec::new();
        for it in items {
            match it {
                None => parts.push("...".to_string()),
                Some(i) => {
                    index.push(i);
                    parts.push(self.render(index, indent + 1));
                    index.pop();
                }
            }
        }
        format!(
            "[{}]",
            parts.join(&format!(",{}{}", gap, " ".repeat(indent + 1)))
        )
    }
}

/// `repr(array)` in numpy's style.
pub fn repr(arr: &NdArray) -> String {
    let dtype_suffix = if is_default_dtype(arr.dtype()) && arr.size() > 0 {
        String::new()
    } else {
        format!(", dtype={}", arr.dtype().name())
    };

    if arr.size() == 0 {
        // numpy: array([], dtype=float64) / array([], shape=(0, 3), ...)
        let shape_part = if arr.ndim() == 1 {
            String::new()
        } else {
            format!(
                ", shape=({})",
                arr.shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        return format!("array([]{}{})", shape_part, dtype_suffix);
    }

    // numpy appends the shape whenever summarisation hides elements, so the
    // reader can still tell how big the array was.
    let shape_suffix = if arr.size() > THRESHOLD {
        format!(
            ", shape=({}{})",
            arr.shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            if arr.ndim() == 1 { "," } else { "" }
        )
    } else {
        String::new()
    };

    let body = body_string(arr, "array(".len());
    format!("array({}{}{})", body, shape_suffix, dtype_suffix)
}

/// `str(array)` in numpy's style: the body only, no `array(...)` wrapper.
pub fn to_str(arr: &NdArray) -> String {
    if arr.size() == 0 {
        return "[]".to_string();
    }
    body_string(arr, 0)
}

fn body_string(arr: &NdArray, indent: usize) -> String {
    let fmt = Formatter::build(arr);
    let _ = fmt.width;
    let ctx = Ctx {
        fmt,
        shape: arr.shape.clone(),
        summarize: arr.size() > THRESHOLD,
    };
    if arr.ndim() == 0 {
        return ctx.fmt.strings[0].trim().to_string();
    }
    let mut index = Vec::new();
    ctx.render(&mut index, indent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Scalar;

    fn f(v: &[f64]) -> NdArray {
        let s: Vec<Scalar> = v.iter().map(|&x| Scalar::Float(x)).collect();
        NdArray::from_scalars(&s, DType::F64).unwrap()
    }

    #[test]
    fn float_repr_matches_numpy() {
        // repr(np.arange(3.0))
        assert_eq!(repr(&f(&[0.0, 1.0, 2.0])), "array([0., 1., 2.])");
        // repr(np.array([0.0, 0.5]))
        assert_eq!(repr(&f(&[0.0, 0.5])), "array([0. , 0.5])");
    }

    #[test]
    fn two_d_repr_indents_like_numpy() {
        let a = NdArray::zeros(vec![2, 3], DType::F64).unwrap();
        assert_eq!(
            repr(&a),
            "array([[0., 0., 0.],\n       [0., 0., 0.]])"
        );
    }

    #[test]
    fn int_repr_shows_non_default_dtype() {
        let a = NdArray::arange(0.0, 6.0, 1.0, DType::I32)
            .unwrap()
            .reshape(&[2, 3])
            .unwrap();
        assert_eq!(
            repr(&a),
            "array([[0, 1, 2],\n       [3, 4, 5]], dtype=int32)"
        );
        let b = NdArray::arange(0.0, 3.0, 1.0, DType::I64).unwrap();
        assert_eq!(repr(&b), "array([0, 1, 2])");
    }

    #[test]
    fn bool_repr_pads_true() {
        let a = NdArray::from_scalars(&[Scalar::Bool(true), Scalar::Bool(false)], DType::Bool)
            .unwrap();
        assert_eq!(repr(&a), "array([ True, False])");
    }

    #[test]
    fn zero_d_and_empty() {
        let a = NdArray::zeros(vec![], DType::I64).unwrap();
        assert_eq!(repr(&a), "array(0)");
        let e = NdArray::zeros(vec![0], DType::F64).unwrap();
        assert_eq!(repr(&e), "array([], dtype=float64)");
    }

    #[test]
    fn complex_repr() {
        let a = NdArray::from_scalars(
            &[Scalar::Complex(num_complex::Complex::new(1.0, 2.0))],
            DType::C128,
        )
        .unwrap();
        assert_eq!(repr(&a), "array([1.+2.j])");
    }

    #[test]
    fn large_arrays_summarize() {
        let a = NdArray::arange(0.0, 2000.0, 1.0, DType::I64).unwrap();
        let r = repr(&a);
        assert!(r.contains("..."), "{r}");
        assert_eq!(
            r,
            "array([   0,    1,    2, ..., 1997, 1998, 1999], shape=(2000,))"
        );
    }
}
