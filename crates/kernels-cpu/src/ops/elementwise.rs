// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Add` and `Mul` over F32 rows: `y = a ∘ b`, one IEEE operation per element, rounded once,
//! as `reference::add` and `reference::mul` define them with `Operand::Rows` (RFC-0040 §3.1).
//! The output may be exactly `a` or `b` (in place); the parent module's docs give the rest.

use syntropy_core::{Bytes, DType, QuantScheme};
use syntropy_kernels::{Error, OpArgs, OpCaps, OpKey, OpKind, ShapeSig};

use super::{active_rows, at_level, elementwise_caps, plain_key, rows_span, run_for_level};
use crate::device::internal;
use crate::gemv::Level;
use crate::gemv::entry::packed;
use crate::launch::{Kernel, Launch, Plan};

/// `y[i] = a[i] + b[i]` for every `i < y.len()`, at `level`.
///
/// # Panics
///
/// When `a` or `b` is shorter than `y`.
pub fn add(level: Level, a: &[f32], b: &[f32], y: &mut [f32]) {
    apply(level, |a, b| a + b, a, b, y);
}

/// `y[i] = a[i] · b[i]` for every `i < y.len()`, at `level`.
///
/// # Panics
///
/// When `a` or `b` is shorter than `y`.
pub fn mul(level: Level, a: &[f32], b: &[f32], y: &mut [f32]) {
    apply(level, |a, b| a * b, a, b, y);
}

fn apply(level: Level, f: impl Fn(f32, f32) -> f32, a: &[f32], b: &[f32], y: &mut [f32]) {
    let n = y.len();
    assert!(
        a.len() >= n && b.len() >= n,
        "an input is shorter than the output"
    );
    // SAFETY: each slice holds at least `n` aligned values, and `y`, a mutable borrow, overlaps
    // neither shared borrow.
    unsafe { apply_raw(level, f, a.as_ptr(), b.as_ptr(), y.as_mut_ptr(), n) }
}

/// `y[i] = f(a[i], b[i])` over `n` elements through raw pointers, at `level`. `y` may be `a`
/// or `b` exactly.
///
/// # Safety
///
/// `a`, `b` and `y` each address `n` valid, aligned F32 values; `y` is writable, and it either
/// overlaps neither input or equals one of them.
unsafe fn apply_raw(
    level: Level,
    f: impl Fn(f32, f32) -> f32,
    a: *const f32,
    b: *const f32,
    y: *mut f32,
    n: usize,
) {
    at_level(level, || {
        for i in 0..n {
            // SAFETY: `i < n`, and the caller guarantees `n` values at each pointer. Each
            // element is read before it is written, so an output equal to an input is fine.
            unsafe { *y.add(i) = f(*a.add(i), *b.add(i)) };
        }
    });
}

/// The `Add` or `Mul` kernel for `isa`'s level.
pub(crate) fn kernel(kind: OpKind, isa: syntropy_kernels::CpuIsa) -> Kernel {
    let level = Level::for_isa(isa);
    Kernel {
        plan,
        caps,
        check,
        run: match kind {
            OpKind::Mul => run_for_level!(run_mul, level),
            _ => run_for_level!(run_add, level),
        },
    }
}

fn is_f32(s: &QuantScheme) -> bool {
    matches!(
        s,
        QuantScheme::Plain(DType::F32) | QuantScheme::Ggml(syntropy_core::GgmlType::F32)
    )
}

fn plan(key: &OpKey, shape: &ShapeSig) -> Option<Plan> {
    let ok = matches!(key.op, OpKind::Add | OpKind::Mul)
        && plain_key(key)
        && key.schemes.iter().all(is_f32)
        && shape.dims.len() == 1;
    ok.then(|| Plan {
        weight: None,
        ..Plan::for_key(key, shape)
    })
}

fn caps(key: &OpKey, shape: &ShapeSig) -> Option<OpCaps> {
    plan(key, shape).map(|_| elementwise_caps(Bytes(0), DType::F32))
}

fn check(plan: &Plan, args: &OpArgs<'_>) -> Result<(), Error> {
    let n = plan.dims[0] as usize;
    let ([a, b], [y], [], []) = (args.inputs, args.outputs, args.weights, args.statics) else {
        return Err(internal(
            "Add and Mul bind two inputs, one output, no weights and no constants".to_owned(),
        ));
    };
    if ![a, b, y].iter().all(|t| packed(t, plan.rows, n)) {
        return Err(internal(format!(
            "Add and Mul: a, b and y must be packed F32 [{}, {n}]",
            plan.rows
        )));
    }
    Ok(())
}

fn run_add(level: Level, l: &Launch<'_>) -> Result<(), Error> {
    run(level, l, "Add", |a, b| a + b)
}

fn run_mul(level: Level, l: &Launch<'_>) -> Result<(), Error> {
    run(level, l, "Mul", |a, b| a * b)
}

fn run(level: Level, l: &Launch<'_>, op: &str, f: impl Fn(f32, f32) -> f32) -> Result<(), Error> {
    let n = l.plan().dims[0] as usize;
    let rows = active_rows(l, op)?;
    if rows == 0 || n == 0 {
        return Ok(());
    }
    let (Some(a), Some(b), Some(y)) = (l.input(0), l.input(1), l.output(0)) else {
        return Err(internal(format!("{op}: missing operands")));
    };
    let (a, b, y) = (
        rows_span(a, rows, n * 4, 4, op)?,
        rows_span(b, rows, n * 4, 4, op)?,
        rows_span(y, rows, n * 4, 4, op)?,
    );
    for x in [a, b] {
        if y.overlaps(&x) && y != x {
            return Err(internal(format!(
                "{op}: the output overlaps an input without being it"
            )));
        }
    }
    // SAFETY: each span was resolved inside a live host buffer (`Launch`'s docs) and checked
    // to hold `rows · n` F32 values at 4-byte alignment; the output overlaps neither input or
    // is one of them, which `apply_raw` allows.
    unsafe {
        apply_raw(
            level,
            f,
            a.addr as *const f32,
            b.addr as *const f32,
            y.addr as *mut f32,
            rows * n,
        );
    }
    Ok(())
}
