// SPDX-License-Identifier: MIT OR Apache-2.0

//! A launch's operands reach the kernel as host memory (COMP-E3-T03; RFC-0040 §3.5, §7.3).
//!
//! `launch` checks the operands, and the push resolves each buffer to its address: at once for
//! a direct launch, and at each replay for a captured one, under the rules a captured copy
//! follows (a buffer freed since the capture fails the replay, and a failed replay queues
//! nothing). The kernel here doubles each active row of its input and adds its constant.

#![allow(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct, 344 B inline [measured: size_of, debug, x86-64 Linux, ce2044ae] (COMP-E13-T15)"
)]

use smallvec::smallvec;
use syntropy_core::{BufferView, Bytes, DType, DeviceId, ErrorClass, HostRegion, NodeId, ViewKind};
use syntropy_kernels::step::StepDesc;
use syntropy_kernels::{
    AllocRequest, BufferSlice, DeviceCtx, Error, KernelProvider, LayoutVersion, MemKind, OpArgs,
    OpKey, OpKind, OpenOptions, PowerHint, PreparedOp, Scalar, ShapeClass, ShapeSig, Stream,
    StreamPlan, StreamPriority, TensorArg,
};
use syntropy_kernels_cpu::{CpuProvider, KillSwitches, Launch, Run};

/// `y[r][i] = 2 · x[r][i] + c` for every active row, `x` and `y` packed F32 `[rows, n]`.
fn double_plus(l: &Launch<'_>) -> Result<(), Error> {
    let (x, y) = (l.input(0).unwrap(), l.output(0).unwrap());
    let c = match l.statics(0) {
        Some(Scalar::F32(c)) => c,
        _ => 0.0,
    };
    let rows = l.rows_active()?.unwrap_or(l.plan().rows) as usize;
    let n = x.dims[1] as usize;
    for i in 0..rows * n {
        // SAFETY: both tensors were resolved inside their buffers, which the test keeps alive,
        // and span `rows · n` F32 values; they do not overlap.
        unsafe {
            let v = std::ptr::read_unaligned((x.addr as *const f32).add(i));
            std::ptr::write_unaligned((y.addr as *mut f32).add(i), 2.0 * v + c);
        }
    }
    Ok(())
}

fn options() -> OpenOptions {
    OpenOptions {
        streams: StreamPlan {
            compute: 1,
            copy_in: 0,
            copy_out: 0,
            collective: 0,
        },
        doorbell_slots: 8,
        capture: true,
        // Stand-in kernels declare no capabilities, which a device opened deterministic
        // refuses (RFC-0040 §3.3).
        deterministic: false,
        power: PowerHint::Mains,
    }
}

struct Fixture {
    p: CpuProvider,
    ctx: DeviceCtx,
    s: Stream,
    op: PreparedOp,
}

const ROWS: u32 = 4;
const N: u32 = 8;

fn fixture() -> Fixture {
    let p = CpuProvider::new(KillSwitches::NONE).with_kernel(OpKind::Add, double_plus);
    let ctx = p
        .open(
            DeviceId {
                node: NodeId([5; 32]),
                ordinal: 0,
            },
            &options(),
        )
        .unwrap();
    let s = p.stream(&ctx, StreamPriority::Compute).unwrap();
    let key = OpKey {
        op: OpKind::Add,
        schemes: smallvec![],
        layout: LayoutVersion::SOURCE,
        variant: 0,
    };
    let shape = ShapeSig {
        class: ShapeClass::Decode,
        rows: ROWS,
        dims: smallvec![N],
    };
    let op = p.prepare(&ctx, &key, &shape).unwrap();
    Fixture { p, ctx, s, op }
}

fn import(f: &Fixture, bytes: &mut [u8]) -> u64 {
    f.p.import(
        &f.ctx,
        HostRegion {
            addr: bytes.as_mut_ptr() as u64,
            len: Bytes(bytes.len() as u64),
        },
    )
    .unwrap()
    .handle
}

fn alloc(f: &Fixture, bytes: u64) -> syntropy_kernels::DeviceBuffer {
    f.p.alloc(
        &f.ctx,
        AllocRequest {
            bytes: Bytes(bytes),
            kind: MemKind::HostMapped,
            align: Bytes(64),
            reserve_only: false,
        },
    )
    .unwrap()
}

fn tensor(buffer: u64) -> TensorArg {
    let slice = BufferSlice {
        buffer,
        offset: Bytes(0),
        bytes: Bytes(u64::from(ROWS * N * 4)),
    };
    TensorArg::packed(slice, DType::F32, [ROWS, N, 1, 1]).unwrap()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn input_bytes() -> Vec<u8> {
    (0..ROWS * N)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect()
}

#[test]
fn a_launch_reads_and_writes_its_operands_in_host_memory() {
    let f = fixture();
    let mut x = input_bytes();
    let mut y = vec![0u8; (ROWS * N * 4) as usize];
    let (xb, yb) = (import(&f, &mut x), import(&f, &mut y));
    let (inputs, outputs) = ([tensor(xb)], [tensor(yb)]);
    let args = OpArgs {
        inputs: &inputs,
        outputs: &outputs,
        statics: &[Scalar::F32(0.5)],
        ..OpArgs::none()
    };
    f.p.launch(&f.ctx, &f.s, &f.op, &args).unwrap();
    assert_eq!(f.p.run(&f.ctx, &f.s).unwrap(), Run::Idle);
    let want: Vec<f32> = (0..ROWS * N).map(|i| 2.0 * i as f32 + 0.5).collect();
    assert_eq!(f32s(&y), want);
}

#[test]
fn rows_active_comes_from_the_step_descriptor() {
    let f = fixture();
    let mut x = input_bytes();
    let mut y = vec![0u8; (ROWS * N * 4) as usize];
    let mut step = vec![0u8; size_of::<StepDesc>()];
    let at = std::mem::offset_of!(StepDesc, rows_active);
    step[at..at + 4].copy_from_slice(&1u32.to_le_bytes());
    let (xb, yb, sb) = (
        import(&f, &mut x),
        import(&f, &mut y),
        import(&f, &mut step),
    );
    let (inputs, outputs) = ([tensor(xb)], [tensor(yb)]);
    let args = OpArgs {
        inputs: &inputs,
        outputs: &outputs,
        step: BufferSlice {
            buffer: sb,
            offset: Bytes(0),
            bytes: Bytes(step.len() as u64),
        },
        ..OpArgs::none()
    };
    f.p.launch(&f.ctx, &f.s, &f.op, &args).unwrap();
    f.p.run(&f.ctx, &f.s).unwrap();
    let got = f32s(&y);
    assert_eq!(
        &got[..N as usize],
        (0..N).map(|i| 2.0 * i as f32).collect::<Vec<_>>()
    );
    assert!(
        got[N as usize..].iter().all(|&v| v == 0.0),
        "rows past rows_active untouched"
    );
}

/// A captured launch names its buffers, not their addresses: each replay resolves them again,
/// so it reads the buffers' current contents, and a buffer freed since the capture fails the
/// replay, which queues nothing, as a captured copy does.
#[test]
fn a_captured_launch_is_resolved_at_each_replay() {
    let f = fixture();
    let mut x = input_bytes();
    let xb = import(&f, &mut x);
    let y = alloc(&f, u64::from(ROWS * N * 4));
    let (inputs, outputs) = ([tensor(xb)], [tensor(y.handle)]);
    let args = OpArgs {
        inputs: &inputs,
        outputs: &outputs,
        ..OpArgs::none()
    };
    f.p.begin_capture(&f.ctx, &f.s).unwrap();
    f.p.launch(&f.ctx, &f.s, &f.op, &args).unwrap();
    let graph = f.p.end_capture(&f.ctx, &f.s).unwrap();
    assert_eq!(
        f.p.run(&f.ctx, &f.s).unwrap(),
        Run::Idle,
        "capture queues nothing"
    );
    f.p.replay(&f.ctx, &f.s, &graph).unwrap();
    f.p.run(&f.ctx, &f.s).unwrap();
    // The device's buffer is read back through a copy into test memory.
    let read = |f: &Fixture, y: &syntropy_kernels::DeviceBuffer| {
        let mut out = vec![0u8; (ROWS * N * 4) as usize];
        let ob = import(f, &mut out);
        let whole = |buffer| BufferSlice {
            buffer,
            offset: Bytes(0),
            bytes: Bytes(u64::from(ROWS * N * 4)),
        };
        f.p.copy(&f.ctx, &f.s, whole(y.handle), whole(ob)).unwrap();
        f.p.run(&f.ctx, &f.s).unwrap();
        f32s(&out)
    };
    assert_eq!(read(&f, &y)[3], 6.0);
    // New input contents, same buffers: the replay reads them.
    x[12..16].copy_from_slice(&10.0f32.to_le_bytes());
    f.p.replay(&f.ctx, &f.s, &graph).unwrap();
    f.p.run(&f.ctx, &f.s).unwrap();
    assert_eq!(read(&f, &y)[3], 20.0);
    // The output buffer is freed: the replay fails before anything is queued.
    f.p.free(&f.ctx, y);
    let error = f.p.replay(&f.ctx, &f.s, &graph).unwrap_err();
    assert_eq!(error.class, ErrorClass::Internal);
    assert!(error.message.contains("no buffer"), "{error}");
    assert_eq!(
        f.p.run(&f.ctx, &f.s).unwrap(),
        Run::Idle,
        "nothing was queued"
    );
}

#[test]
fn launch_refuses_operands_it_cannot_bind() {
    let f = fixture();
    let mut x = input_bytes();
    let xb = import(&f, &mut x);
    let ok = tensor(xb);
    // A tensor longer than its slice: refused at the push, nothing queued.
    let mut long = ok;
    long.buf.bytes = Bytes(16);
    let inputs = [long];
    let args = OpArgs {
        inputs: &inputs,
        outputs: &inputs,
        ..OpArgs::none()
    };
    let error = f.p.launch(&f.ctx, &f.s, &f.op, &args).unwrap_err();
    assert!(error.message.contains("past its"), "{error}");
    // An unknown buffer.
    let mut gone = ok;
    gone.buf.buffer = 999;
    let inputs = [gone];
    let args = OpArgs {
        inputs: &inputs,
        ..OpArgs::none()
    };
    assert!(f.p.launch(&f.ctx, &f.s, &f.op, &args).is_err());
    // More tensors than a launch holds.
    let many = [ok; 7];
    let args = OpArgs {
        inputs: &many,
        ..OpArgs::none()
    };
    let error = f.p.launch(&f.ctx, &f.s, &f.op, &args).unwrap_err();
    assert!(error.message.contains("at most"), "{error}");
    // A weight view the CPU cannot read.
    let weights = [BufferView {
        kind: ViewKind::Device(DeviceId {
            node: NodeId([5; 32]),
            ordinal: 1,
        }),
        addr: 64,
        len: Bytes(64),
    }];
    let args = OpArgs {
        weights: &weights,
        ..OpArgs::none()
    };
    let error = f.p.launch(&f.ctx, &f.s, &f.op, &args).unwrap_err();
    assert!(error.message.contains("not host memory"), "{error}");
    assert_eq!(
        f.p.run(&f.ctx, &f.s).unwrap(),
        Run::Idle,
        "nothing was queued"
    );
    assert!(f.p.health(&f.ctx) == syntropy_kernels::DeviceHealth::Ok);
}
