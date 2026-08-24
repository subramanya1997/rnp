//! Numeric sum-of-products loop for the pure-Python einsum parser.
//!
//! Parsing and planning deliberately stay in the NumPy-derived shim.  This
//! function only transcribes the hot C core loop: operands are already cast
//! to one common dtype and repeated axes are already combined into views.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use rnp_core::ops::binary_scalar;
use rnp_core::{BinOp, NdArray, Scalar};

use crate::pyarray::PyNdArray;

fn read_operand(array: &NdArray, operand_positions: &[usize], index: &[isize]) -> Scalar {
    let mut byte_offset = array.byte_offset;
    for (axis, &position) in operand_positions.iter().enumerate() {
        let coordinate = if array.shape[axis] == 1 {
            0
        } else {
            index[position]
        };
        byte_offset += coordinate * array.strides[axis];
    }
    array.read_at(byte_offset)
}

fn float_add(dtype: rnp_core::DType, a: f64, b: f64) -> f64 {
    if dtype == rnp_core::DType::F32 {
        ((a as f32) + (b as f32)) as f64
    } else {
        a + b
    }
}

fn float_mul(dtype: rnp_core::DType, a: f64, b: f64) -> f64 {
    if dtype == rnp_core::DType::F32 {
        ((a as f32) * (b as f32)) as f64
    } else {
        a * b
    }
}

fn float_mul_add(dtype: rnp_core::DType, a: f64, b: f64, c: f64) -> f64 {
    if dtype == rnp_core::DType::F32 {
        (a as f32).mul_add(b as f32, c as f32) as f64
    } else {
        a.mul_add(b, c)
    }
}

fn vector_sum(dtype: rnp_core::DType, lanes: &[f64]) -> f64 {
    if lanes.len() == 4 {
        float_add(
            dtype,
            float_add(dtype, lanes[0], lanes[1]),
            float_add(dtype, lanes[2], lanes[3]),
        )
    } else {
        float_add(dtype, lanes[0], lanes[1])
    }
}

#[pyfunction]
pub fn _einsum_numeric(
    operands: &Bound<'_, PyList>,
    positions: Vec<Vec<usize>>,
    iter_shape: Vec<isize>,
    output_ndim: usize,
    result: &Bound<'_, PyNdArray>,
) -> PyResult<()> {
    if operands.len() != positions.len() {
        return Err(PyValueError::new_err(
            "einsum operand position table has the wrong length",
        ));
    }
    if output_ndim > iter_shape.len() {
        return Err(PyValueError::new_err(
            "einsum output rank exceeds iterator rank",
        ));
    }

    let mut arrays = Vec::<NdArray>::with_capacity(operands.len());
    for operand in operands.iter() {
        let array = operand.cast::<PyNdArray>()?;
        arrays.push(array.borrow().arr.clone());
    }
    let dtype = result.borrow().arr.dtype();
    if arrays.iter().any(|array| array.dtype() != dtype) {
        return Err(PyValueError::new_err(
            "einsum numeric operands must share the output dtype",
        ));
    }

    let checked_size = |shape: &[isize]| {
        shape.iter().try_fold(1usize, |n, &dim| {
            usize::try_from(dim)
                .ok()
                .and_then(|d| n.checked_mul(d))
                .ok_or_else(|| PyValueError::new_err("invalid einsum iterator shape"))
        })
    };
    let output_size = checked_size(&iter_shape[..output_ndim])?;
    let reduction_size = checked_size(&iter_shape[output_ndim..])?;
    let mut index = vec![0isize; iter_shape.len()];

    for (array, operand_positions) in arrays.iter().zip(&positions) {
        if operand_positions.len() != array.ndim() {
            return Err(PyValueError::new_err(
                "einsum operand rank does not match its position table",
            ));
        }
    }

    let scalar_op = |a, b, op| -> PyResult<Scalar> {
        Ok(binary_scalar(a, dtype, b, dtype, op)
            .expect("numeric dtype has the requested scalar loop")
            .map_err(crate::err)?
            .1)
    };

    for output_linear in 0..output_size {
        let mut rem = output_linear;
        for axis in (0..output_ndim).rev() {
            let dim = iter_shape[axis] as usize;
            index[axis] = (rem % dim) as isize;
            rem /= dim;
        }

        let output = result.borrow();
        let mut output_offset = output.arr.byte_offset;
        for axis in 0..output_ndim {
            output_offset += index[axis] * output.arr.strides[axis];
        }
        let mut accumulated = output.arr.read_at(output_offset);

        if matches!(dtype, rnp_core::DType::F32 | rnp_core::DType::F64) {
            let mut float_accumulated = accumulated.as_f64();
            let reduction_rank = iter_shape.len() - output_ndim;
            let contiguous_reduction = reduction_rank == 1
                && arrays
                    .iter()
                    .zip(&positions)
                    .all(|(array, operand_positions)| {
                        operand_positions
                            .iter()
                            .position(|&position| position == output_ndim)
                            .is_some_and(|axis| array.strides[axis] == array.itemsize() as isize)
                    });
            let lanes = if dtype == rnp_core::DType::F32 { 4 } else { 2 };

            for chunk_start in (0..reduction_size).step_by(8192) {
                let chunk_end = (chunk_start + 8192).min(reduction_size);
                let mut chunk = 0.0f64;

                if contiguous_reduction && arrays.len() <= 2 {
                    let mut lane_accum = vec![0.0f64; lanes];
                    let mut reduction_linear = chunk_start;
                    let block = lanes * 4;
                    while reduction_linear + block <= chunk_end {
                        let mut values = vec![vec![0.0f64; lanes]; 4];
                        for group in 0..4 {
                            for lane in 0..lanes {
                                index[output_ndim] =
                                    (reduction_linear + group * lanes + lane) as isize;
                                let first =
                                    read_operand(&arrays[0], &positions[0], &index).as_f64();
                                values[group][lane] = if arrays.len() == 1 {
                                    first
                                } else {
                                    let second =
                                        read_operand(&arrays[1], &positions[1], &index).as_f64();
                                    float_mul(dtype, first, second)
                                };
                            }
                        }
                        for lane in 0..lanes {
                            if arrays.len() == 1 {
                                let pair01 = float_add(dtype, values[0][lane], values[1][lane]);
                                let pair23 = float_add(dtype, values[2][lane], values[3][lane]);
                                lane_accum[lane] = float_add(
                                    dtype,
                                    float_add(dtype, pair01, pair23),
                                    lane_accum[lane],
                                );
                            } else {
                                for group in (0..4).rev() {
                                    index[output_ndim] =
                                        (reduction_linear + group * lanes + lane) as isize;
                                    let first =
                                        read_operand(&arrays[0], &positions[0], &index).as_f64();
                                    let second =
                                        read_operand(&arrays[1], &positions[1], &index).as_f64();
                                    lane_accum[lane] =
                                        float_mul_add(dtype, first, second, lane_accum[lane]);
                                }
                            }
                        }
                        reduction_linear += block;
                    }
                    while reduction_linear < chunk_end {
                        for lane in 0..lanes {
                            let position = reduction_linear + lane;
                            if position >= chunk_end {
                                break;
                            }
                            index[output_ndim] = position as isize;
                            let first = read_operand(&arrays[0], &positions[0], &index).as_f64();
                            if arrays.len() == 1 {
                                lane_accum[lane] = float_add(dtype, first, lane_accum[lane]);
                            } else {
                                let second =
                                    read_operand(&arrays[1], &positions[1], &index).as_f64();
                                lane_accum[lane] =
                                    float_mul_add(dtype, first, second, lane_accum[lane]);
                            }
                        }
                        reduction_linear += lanes;
                    }
                    chunk = vector_sum(dtype, &lane_accum);
                } else {
                    for reduction_linear in chunk_start..chunk_end {
                        let mut rem = reduction_linear;
                        for axis in (output_ndim..iter_shape.len()).rev() {
                            let dim = iter_shape[axis] as usize;
                            index[axis] = (rem % dim) as isize;
                            rem /= dim;
                        }
                        let first = read_operand(&arrays[0], &positions[0], &index).as_f64();
                        if arrays.len() == 1 {
                            chunk = float_add(dtype, first, chunk);
                        } else {
                            let mut product = first;
                            for (array, operand_positions) in arrays[1..arrays.len() - 1]
                                .iter()
                                .zip(&positions[1..arrays.len() - 1])
                            {
                                product = float_mul(
                                    dtype,
                                    product,
                                    read_operand(array, operand_positions, &index).as_f64(),
                                );
                            }
                            let last = read_operand(
                                arrays.last().expect("einsum has operands"),
                                positions.last().expect("einsum has position tables"),
                                &index,
                            )
                            .as_f64();
                            chunk = float_mul_add(dtype, product, last, chunk);
                        }
                    }
                }
                float_accumulated = float_add(dtype, float_accumulated, chunk);
            }
            output
                .arr
                .write_at(output_offset, Scalar::Float(float_accumulated));
            continue;
        }

        if dtype == rnp_core::DType::F16 {
            let mut half_accumulated = accumulated.as_f64() as f32;
            for chunk_start in (0..reduction_size).step_by(8192) {
                let chunk_end = (chunk_start + 8192).min(reduction_size);
                let mut chunk = 0.0f32;
                for reduction_linear in chunk_start..chunk_end {
                    let mut rem = reduction_linear;
                    for axis in (output_ndim..iter_shape.len()).rev() {
                        let dim = iter_shape[axis] as usize;
                        index[axis] = (rem % dim) as isize;
                        rem /= dim;
                    }
                    let mut product = None;
                    for (array, operand_positions) in arrays.iter().zip(&positions) {
                        let value = read_operand(array, operand_positions, &index).as_f64() as f32;
                        product = Some(match product {
                            None => value,
                            Some(previous) => previous * value,
                        });
                    }
                    chunk += product.expect("einsum always has at least one operand");
                }
                half_accumulated += chunk;
            }
            output
                .arr
                .write_at(output_offset, Scalar::Float(half_accumulated as f64));
            continue;
        }

        // NumPy deliberately omits NPY_ITER_GROWINNER for einsum.  The
        // buffered external loop therefore accumulates bounded chunks before
        // adding each chunk to the output, giving its characteristic partly
        // stable (but non-pairwise) floating-point result.
        for chunk_start in (0..reduction_size).step_by(8192) {
            let chunk_end = (chunk_start + 8192).min(reduction_size);
            let mut chunk = Scalar::Int(0).cast(dtype);
            for reduction_linear in chunk_start..chunk_end {
                let mut rem = reduction_linear;
                for axis in (output_ndim..iter_shape.len()).rev() {
                    let dim = iter_shape[axis] as usize;
                    index[axis] = (rem % dim) as isize;
                    rem /= dim;
                }

                let mut product = None;
                for (array, operand_positions) in arrays.iter().zip(&positions) {
                    let value = read_operand(array, operand_positions, &index);
                    product = Some(match product {
                        None => value,
                        Some(previous) => scalar_op(previous, value, BinOp::Mul)?,
                    });
                }
                chunk = scalar_op(
                    chunk,
                    product.expect("einsum always has at least one operand"),
                    BinOp::Add,
                )?;
            }
            accumulated = scalar_op(accumulated, chunk, BinOp::Add)?;
        }
        output.arr.write_at(output_offset, accumulated);
    }
    Ok(())
}
