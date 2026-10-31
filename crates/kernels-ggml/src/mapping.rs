// SPDX-License-Identifier: MIT OR Apache-2.0

//! The op mapping layer: one Syntropy op as a small ggml graph (RFC-0040 §8.2; ADR-0403 items 1,
//! 3).
//!
//! [`CpuOps::prepare`] builds a **template graph** for an (`OpKey`, `ShapeSig`): a `ggml_context`
//! made with `no_alloc`, so its tensors are descriptors with no data, the ggml ops of the
//! mapping, and a `ggml_cgraph`. [`CpuOps::query`] builds the same template and asks the ggml CPU
//! backend whether it supports every node (`ggml_backend_supports_op`). It answers `Some` only
//! when all are. [`CpuOps::launch`] writes the `data` pointers of the template's tensors from the
//! `OpArgs` buffers and computes the graph. Those are field writes: a launch allocates nothing.
//!
//! **Memory.** Every byte a launch touches is the caller's (ADR-0002 item 4). Inputs, outputs and
//! weights are the `OpArgs` buffers. The intermediate tensors of a multi-node template, and the
//! work buffer ggml's CPU compute plan asks for (`ggml_graph_plan`), sit in the `OpArgs`
//! workspace, at offsets fixed at `prepare`. `OpCaps::workspace` declares their sum (RFC-0040
//! §5.3). The compute runs on a one-thread ggml threadpool made at `prepare`, so `ggml_graph_compute`
//! never creates one per call.
//!
//! **Rows.** Templates are built at the bucket's row count, `ShapeSig::rows`. When `OpArgs::step`
//! holds a step descriptor, a launch reads its `rows_active` and sets the row extent of every
//! row-bucketed tensor to it, so ggml computes only the active rows (RFC-0040 §3.2, §3.5). With no
//! descriptor (an empty `step` slice) it computes the whole bucket.
//!
//! **The mappings, on the CPU backend** (operands in `OpArgs` order; `[rows, n]` is a dense
//! row-major F32 tensor; `W` is a weight view in the source layout, `LayoutVersion::SOURCE`):
//!
//! | `OpKind` | dims | ggml template | operands |
//! |---|---|---|---|
//! | `Gemv`, `Gemm`, `Logits` | `[n_out, k_in]` | `mul_mat` | in `x [rows, k_in]`; weight `W [n_out, k_in]`; out `[rows, n_out]` |
//! | `GateUpAct { act: Silu }` | `[n_out, k_in]` | `mul_mat`, then `swiglu` | in `x [rows, k_in]`; weight `W [2·n_out, k_in]`, gate rows first, then up rows; out `[rows, n_out]` |
//! | `RmsNorm` | `[hidden]` | `rms_norm`, then `mul` | in `x [rows, hidden]`; weight `g [hidden]` F32; static `eps` (F32); out `[rows, hidden]` |
//! | `Act(Silu)` | `[n]` | `silu` | in `x`; out `[rows, n]` |
//! | `Add`, `Mul` | `[n]` | `add`, `mul` | in `x`, `y`; out `[rows, n]` |
//! | `RowGather` | `[table_rows, width]` | `get_rows` | in ids `[rows]` (I32, each below `table_rows`); weight `T [table_rows, width]`; out `[rows, width]` |
//! | `Convert` | `[n]` | `cpy` (`ggml_cast`) | in `x [rows, n]` of `schemes[0]`; out `[rows, n]` of the last scheme; F32, F16 and BF16 |
//!
//! A matrix product's weight scheme is `OpKey::schemes[0]`: `Plain(F32)`, `Plain(BF16)` or one of
//! the GGUF block types `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` and `Q2_K`–`Q6_K`. ggml's CPU
//! backend rounds or quantises the activation to the weight type's `vec_dot_type` first; `query`
//! reads that type from ggml and reports it as `Numerics::act_in` (RFC-0042 §4.3). `Plain(F16)`
//! weights are not mapped: ggml rounds their activations to F16, which `ActQuant` cannot state.
//!
//! Not mapped yet, so `query` returns `None` and `prepare` refuses them: `Rope`, `MoeRoute`,
//! `MoeGrouped`, `AttnDecode` and `AttnPrefill`, whose operands (positions, paged KV, routing)
//! come through the step descriptor and the KV store, and the GELU variants of `Act` and
//! `GateUpAct`, whose GELU form `kernels::reference` has not fixed. Every op §8.2 marks as having
//! no mapping (`Pool`, `EmbedInject`, the sampling ops, …) returns `None` too, as does any key with
//! a fusion bit set or a layout other than `SOURCE`.
//!
//! **Capture safety** ([`capture_safe`]). On the CPU every mapped op is capture-safe. On CUDA and
//! HIP, `Gemm`, and any `mul_mat` over float weights or at a prefill or batch bucket, may reach
//! cuBLAS or hipBLAS, so they are not (RFC-0040 §8.1, §8.2).

use std::cell::RefCell;
use std::fmt;
use std::mem::offset_of;
use std::ptr;

use smallvec::SmallVec;
use syntropy_core::resource::GgmlBackend;
use syntropy_core::{
    AccumKind, ActQuant, Blake3, BufferView, Bytes, DType, Domain, ErrorClass, ErrorCode, GgmlType,
    Numerics, QuantScheme, ViewKind,
};
use syntropy_kernels::error::codes;
use syntropy_kernels::step::StepDesc;
use syntropy_kernels::{
    ActKind, BufferSlice, Error, GatedAct, LayoutVersion, OpArgs, OpCaps, OpKey, OpKind,
    PreparedOp, Scalar, ShapeClass, ShapeSig, TensorArg, unsupported,
};

use crate::sys;

/// Alignment of each region of the workspace: a cache line, and every SIMD width ggml's CPU
/// kernels load.
const ALIGN: usize = 64;

/// The most tensors a template holds: `GateUpAct` and `RmsNorm` have four.
const MAX_TENSORS: usize = 6;

/// Where the bytes of the host buffer with a given handle live, for [`CpuOps::launch`].
///
/// The ggml provider's buffer table implements it (COMP-E7-T03 wraps tier-manager memory); a
/// test implements it over its own vectors.
pub trait HostMemory {
    /// The host range of the buffer whose `DeviceBuffer::handle` is `handle`, or `None` for a
    /// handle this memory does not know.
    fn range(&self, handle: u64) -> Option<HostRange>;
}

/// A host byte range: its address and length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HostRange {
    /// The address of the first byte.
    pub addr: u64,
    /// The length in bytes.
    pub len: u64,
}

/// What one op maps to, decided from its key and shape alone ([`mapping`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mapping {
    /// `mul_mat` of a weight `[n_out, k_in]` and the input: `Gemv`, `Gemm` and `Logits`.
    MatMul {
        /// The weight's storage type.
        weight: WeightType,
        /// Output columns: the weight's rows.
        n_out: u32,
        /// The reduction length.
        k_in: u32,
    },
    /// `mul_mat` of a fused gate-and-up weight `[2·n_out, k_in]`, then `swiglu`.
    GateUpSilu {
        /// The weight's storage type.
        weight: WeightType,
        /// Output columns: half the weight's rows.
        n_out: u32,
        /// The reduction length.
        k_in: u32,
    },
    /// `rms_norm`, then `mul` by the F32 gain.
    RmsNorm {
        /// The row width.
        hidden: u32,
    },
    /// `silu`.
    Silu {
        /// The row width.
        n: u32,
    },
    /// `add`.
    Add {
        /// The row width.
        n: u32,
    },
    /// `mul`.
    Mul {
        /// The row width.
        n: u32,
    },
    /// `get_rows` of a table `[table_rows, width]`.
    RowGather {
        /// The table's storage type.
        table: WeightType,
        /// The table's rows.
        table_rows: u32,
        /// The row width.
        width: u32,
    },
    /// `cpy` from one float type to another.
    Convert {
        /// The input's element type.
        from: DType,
        /// The output's element type.
        to: DType,
        /// The row width.
        n: u32,
    },
}

/// A weight's storage type, as this layer maps it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightType(sys::ggml_type);

impl WeightType {
    /// The ggml type of a weight scheme, for the schemes this layer maps: `Plain` F32, F16 and
    /// BF16, and the GGUF block types with an integer or float CPU dot product of known
    /// accumulation (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`–`Q6_K`).
    #[must_use]
    pub fn of(scheme: &QuantScheme) -> Option<Self> {
        use sys::ggml_type as T;
        let ty = match scheme {
            QuantScheme::Plain(DType::F32) => T::GGML_TYPE_F32,
            QuantScheme::Plain(DType::F16) => T::GGML_TYPE_F16,
            QuantScheme::Plain(DType::BF16) => T::GGML_TYPE_BF16,
            QuantScheme::Ggml(g) => match g {
                GgmlType::F32 => T::GGML_TYPE_F32,
                GgmlType::F16 => T::GGML_TYPE_F16,
                GgmlType::BF16 => T::GGML_TYPE_BF16,
                GgmlType::Q4_0 => T::GGML_TYPE_Q4_0,
                GgmlType::Q4_1 => T::GGML_TYPE_Q4_1,
                GgmlType::Q5_0 => T::GGML_TYPE_Q5_0,
                GgmlType::Q5_1 => T::GGML_TYPE_Q5_1,
                GgmlType::Q8_0 => T::GGML_TYPE_Q8_0,
                GgmlType::Q2_K => T::GGML_TYPE_Q2_K,
                GgmlType::Q3_K => T::GGML_TYPE_Q3_K,
                GgmlType::Q4_K => T::GGML_TYPE_Q4_K,
                GgmlType::Q5_K => T::GGML_TYPE_Q5_K,
                GgmlType::Q6_K => T::GGML_TYPE_Q6_K,
                _ => return None,
            },
            _ => return None,
        };
        Some(Self(ty))
    }

    /// Whether the weight is stored as floats (F32, F16, BF16), not in quantised blocks.
    #[must_use]
    pub fn is_float(self) -> bool {
        use sys::ggml_type as T;
        matches!(
            self.0,
            T::GGML_TYPE_F32 | T::GGML_TYPE_F16 | T::GGML_TYPE_BF16
        )
    }

    /// Elements per block: 1 for floats.
    fn block_elems(self) -> u32 {
        // SAFETY: a pure lookup in ggml's static type table, for a type it defines.
        let n = unsafe { sys::ggml_blck_size(self.0) };
        u32::try_from(n).expect("a block holds a positive number of elements")
    }

    /// The type ggml's CPU matrix product converts the activation to before its dot product.
    fn vec_dot_type(self) -> sys::ggml_type {
        // SAFETY: `ggml_get_type_traits_cpu` returns a pointer into ggml's static CPU type table,
        // valid for the program, for any type ggml defines.
        unsafe { (*sys::ggml_get_type_traits_cpu(self.0)).vec_dot_type }
    }

    /// The numerics of ggml's CPU matrix product over this weight type (RFC-0042 §4), or `None`
    /// when its activation handling has no `ActQuant`: F16 rounding.
    fn matmul_numerics(self) -> Option<Numerics> {
        use sys::ggml_type as T;
        let (act_in, accumulate) = match self.vec_dot_type() {
            T::GGML_TYPE_F32 => (ActQuant::None, AccumKind::F32),
            T::GGML_TYPE_BF16 => (ActQuant::Bf16, AccumKind::F32),
            T::GGML_TYPE_Q8_0 | T::GGML_TYPE_Q8_1 => {
                (ActQuant::Q8Block32, AccumKind::I32ThenF32 { max_block: 32 })
            }
            T::GGML_TYPE_Q8_K => (ActQuant::Q8K256, AccumKind::I32ThenF32 { max_block: 256 }),
            _ => return None,
        };
        Some(Numerics {
            accumulate,
            act_in,
            out: DType::F32,
            softmax_f32: false,
            state_f32: false,
        })
    }
}

/// The mapping of an op at a shape, or `None` when it has none: an unmapped kind, a fusion bit,
/// a layout other than `SOURCE`, an unmapped scheme, or dims that are empty, of the wrong count,
/// or not whole weight blocks.
#[must_use]
pub fn mapping(key: &OpKey, shape: &ShapeSig) -> Option<Mapping> {
    if key.layout != LayoutVersion::SOURCE || key.variant != 0 || shape.rows == 0 {
        return None;
    }
    let dims = shape.dims.as_slice();
    if dims.len() != key.op.dims().len() || dims.contains(&0) {
        return None;
    }
    // The activation and output schemes, where the key names them, must be F32.
    let f32_after_weight = || key.schemes.iter().skip(1).all(is_f32);
    let m = match key.op {
        OpKind::Gemv | OpKind::Gemm | OpKind::Logits | OpKind::GateUpAct { .. } => {
            let weight = WeightType::of(key.schemes.first()?)?;
            weight.matmul_numerics()?;
            let (n_out, k_in) = (dims[0], dims[1]);
            if !f32_after_weight() || !k_in.is_multiple_of(weight.block_elems()) {
                return None;
            }
            match key.op {
                OpKind::GateUpAct {
                    act: GatedAct::Silu,
                } => Mapping::GateUpSilu {
                    weight,
                    n_out,
                    k_in,
                },
                OpKind::GateUpAct { .. } => return None,
                _ => Mapping::MatMul {
                    weight,
                    n_out,
                    k_in,
                },
            }
        }
        OpKind::RmsNorm if key.schemes.iter().all(is_f32) => Mapping::RmsNorm { hidden: dims[0] },
        OpKind::Act(ActKind::Silu) if key.schemes.iter().all(is_f32) => {
            Mapping::Silu { n: dims[0] }
        }
        OpKind::Add if key.schemes.iter().all(is_f32) => Mapping::Add { n: dims[0] },
        OpKind::Mul if key.schemes.iter().all(is_f32) => Mapping::Mul { n: dims[0] },
        OpKind::RowGather => {
            let table = WeightType::of(key.schemes.first()?)?;
            let (table_rows, width) = (dims[0], dims[1]);
            if !f32_after_weight() || !width.is_multiple_of(table.block_elems()) {
                return None;
            }
            Mapping::RowGather {
                table,
                table_rows,
                width,
            }
        }
        OpKind::Convert => {
            let float = |s: &QuantScheme| match s {
                QuantScheme::Plain(d @ (DType::F32 | DType::F16 | DType::BF16)) => Some(*d),
                _ => None,
            };
            let from = float(key.schemes.first()?)?;
            let to = float(key.schemes.last()?)?;
            if from == to {
                return None;
            }
            Mapping::Convert {
                from,
                to,
                n: dims[0],
            }
        }
        _ => return None,
    };
    Some(m)
}

fn is_f32(s: &QuantScheme) -> bool {
    matches!(
        s,
        QuantScheme::Plain(DType::F32) | QuantScheme::Ggml(GgmlType::F32)
    )
}

/// Whether an op is safe to record in a captured graph on a ggml backend (RFC-0040 §4.4 rule 5,
/// §8.2's third column). On CUDA and HIP, ggml may route `Gemm`, and any `mul_mat` over float
/// weights or at a prefill or batch bucket, to cuBLAS or hipBLAS, whose handles and workspaces it
/// creates outside patch P3; those are not capture-safe. Everything else this layer maps is.
#[must_use]
pub fn capture_safe(backend: GgmlBackend, kind: OpKind, m: &Mapping, class: ShapeClass) -> bool {
    if !matches!(backend, GgmlBackend::Cuda | GgmlBackend::Hip) {
        return true;
    }
    let weight = match m {
        Mapping::MatMul { weight, .. } | Mapping::GateUpSilu { weight, .. } => *weight,
        _ => return true,
    };
    let large_rows = matches!(
        class,
        ShapeClass::Prefill { .. } | ShapeClass::PrefillStream { .. } | ShapeClass::Batch { .. }
    );
    !(kind == OpKind::Gemm || weight.is_float() || large_rows)
}

/// What a template tensor is bound to at launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bind {
    /// `OpArgs::inputs[i]`, with this element type.
    Input(usize, DType),
    /// `OpArgs::weights[i]`.
    Weight(usize),
    /// `OpArgs::outputs[i]`, with this element type.
    Output(usize, DType),
    /// The workspace, at this offset.
    Scratch(usize),
}

/// One tensor of a template, with how it is bound and which axis holds the rows.
#[derive(Clone, Copy, Debug)]
struct Slot {
    tensor: *mut sys::ggml_tensor,
    bind: Bind,
    /// The ggml axis (`ne` index) that counts token rows, if the tensor is row-bucketed.
    rows_axis: Option<usize>,
    /// The bytes the tensor spans at the bucket's row count.
    bytes: usize,
}

/// A prepared op: its ggml context, graph, compute plan and threadpool, and how each tensor is
/// bound. It owns the context and the threadpool and frees them on drop.
struct Template {
    ctx: *mut sys::ggml_context,
    graph: *mut sys::ggml_cgraph,
    pool: *mut sys::ggml_threadpool,
    plan: sys::ggml_cplan,
    slots: SmallVec<[Slot; MAX_TENSORS]>,
    /// The `rms_norm` node, whose `eps` a launch writes from `OpArgs::statics[0]`.
    eps_node: Option<*mut sys::ggml_tensor>,
    /// For `get_rows`: the slot of the ids and the table's rows, which every id must be below.
    gather: Option<(usize, u32)>,
    /// Offset of ggml's work buffer in the workspace.
    work_offset: usize,
    /// The whole workspace: scratch tensors, then the work buffer.
    workspace: usize,
    rows: u32,
    inputs: usize,
    weights: usize,
    outputs: usize,
}

impl Drop for Template {
    fn drop(&mut self) {
        // SAFETY: `pool` came from `ggml_threadpool_new` (or is null) and `ctx` from `ggml_init`;
        // each is freed only here, and nothing else holds them.
        unsafe {
            if !self.pool.is_null() {
                sys::ggml_threadpool_free(self.pool);
            }
            sys::ggml_free(self.ctx);
        }
    }
}

/// The ggml CPU backend's op mapping layer: `query`, `prepare` and `launch` for the ops of
/// [`mapping`] (RFC-0040 §8.2).
///
/// It holds the CPU backend, which answers the support check, and the templates `prepare` built,
/// indexed by `PreparedOp::handle`. It is neither `Send` nor `Sync`: a launch writes the
/// template's tensors, so one thread drives it, as one stream's runner does.
pub struct CpuOps {
    backend: sys::ggml_backend_t,
    prepared: RefCell<Vec<Template>>,
}

impl fmt::Debug for CpuOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuOps")
            .field("prepared", &self.prepared.borrow().len())
            .finish_non_exhaustive()
    }
}

impl Drop for CpuOps {
    fn drop(&mut self) {
        // The templates hold no reference to the backend; they go first all the same.
        self.prepared.get_mut().clear();
        // SAFETY: `backend` came from `ggml_backend_cpu_init` and is freed only here.
        unsafe { sys::ggml_backend_free(self.backend) };
    }
}

impl CpuOps {
    /// The CPU backend's mapping layer.
    ///
    /// # Errors
    ///
    /// `AllocFailed` when ggml cannot create its CPU backend.
    pub fn new() -> Result<Self, Error> {
        // SAFETY: `ggml_cpu_init` only builds ggml's lookup tables, once; it is safe to call again.
        unsafe { sys::ggml_cpu_init() };
        // SAFETY: no inputs; it returns a new backend, or null when it cannot allocate one.
        let backend = unsafe { sys::ggml_backend_cpu_init() };
        if backend.is_null() {
            return Err(Error::new(
                ErrorClass::Capacity,
                codes::ALLOC_FAILED,
                "ggml could not create its CPU backend",
            ));
        }
        Ok(Self {
            backend,
            prepared: RefCell::new(Vec::new()),
        })
    }

    /// What the op costs and how it computes on the CPU backend, or `None` when it has no
    /// mapping ([`mapping`]) or ggml's CPU backend does not support every node of its template.
    #[must_use]
    pub fn query(&self, key: &OpKey, shape: &ShapeSig) -> Option<OpCaps> {
        let m = mapping(key, shape)?;
        let t = Template::build(&m, shape.rows).ok()?;
        // SAFETY: `t.graph` is the template's live graph; `ggml_graph_node` is called with
        // indices below `ggml_graph_n_nodes`, and the backend only reads each node.
        let supported = unsafe {
            (0..sys::ggml_graph_n_nodes(t.graph)).all(|i| {
                sys::ggml_backend_supports_op(self.backend, sys::ggml_graph_node(t.graph, i))
            })
        };
        if !supported {
            return None;
        }
        let numerics = match m {
            Mapping::MatMul { weight, .. } | Mapping::GateUpSilu { weight, .. } => {
                weight.matmul_numerics()?
            }
            _ => Numerics {
                // ggml's CPU `rms_norm` sums the squares in F64 (`ggml_float`).
                accumulate: if matches!(m, Mapping::RmsNorm { .. }) {
                    AccumKind::F64
                } else {
                    AccumKind::F32
                },
                act_in: ActQuant::None,
                out: match m {
                    Mapping::Convert { to, .. } => to,
                    _ => DType::F32,
                },
                softmax_f32: false,
                state_f32: false,
            },
        };
        Some(OpCaps {
            workspace: Bytes(t.workspace as u64),
            est: None,
            layout: LayoutVersion::SOURCE,
            capture_safe: capture_safe(GgmlBackend::Cpu, key.op, &m, shape.class),
            // One thread, and ggml's CPU kernels reduce in an order fixed by the shape.
            deterministic: true,
            numerics,
            // The hashes an `OpCert` is keyed on (RFC-0040 §10.2) come with the ggml `OpCert`s
            // (COMP-E7-T06). Zero is never certified, so until then every ggml op runs uncertified.
            kernel_hash: Blake3([0; 32]),
            source_key: Blake3([0; 32]),
        })
    }

    /// Builds the op's template and keeps it; the returned op's handle names it.
    ///
    /// # Errors
    ///
    /// `OpUnsupported` when `query` would return `None`; `AllocFailed` when ggml cannot create the
    /// template's context or threadpool.
    pub fn prepare(&self, key: &OpKey, shape: &ShapeSig) -> Result<PreparedOp, Error> {
        let caps = self.query(key, shape).ok_or_else(unsupported)?;
        let m = mapping(key, shape).ok_or_else(unsupported)?;
        let mut t = Template::build(&m, shape.rows)?;
        // SAFETY: plain parameter construction; `params` outlives the call, which copies it.
        let pool = unsafe {
            let mut params = sys::ggml_threadpool_params_default(1);
            sys::ggml_threadpool_new(&mut params)
        };
        if pool.is_null() {
            return Err(Error::new(
                ErrorClass::Capacity,
                codes::ALLOC_FAILED,
                "ggml could not create a threadpool",
            ));
        }
        t.pool = pool;
        t.plan.threadpool = pool;
        let op = PreparedOp::with_handle(
            Bytes(t.workspace as u64),
            self.prepared.borrow().len() as u64,
        );
        op.check_workspace(&caps)?;
        self.prepared.borrow_mut().push(t);
        Ok(op)
    }

    /// Binds `args` to the prepared op's template and computes it on the calling thread.
    ///
    /// Each input and output must be a dense row-major tensor of the template's element type and
    /// shape at the bucket's row count, inside its buffer and aligned to its element size; each
    /// weight a host or unified view of at least the weight's bytes, 4-byte aligned; the
    /// workspace, when the op declares one, at least that many bytes, 4-byte aligned. A launch
    /// that breaks one of these is refused before ggml runs.
    ///
    /// # Errors
    ///
    /// An `Internal` error (`Kernels` code 0) for an unknown handle, arguments that do not match
    /// the template, or a compute that ggml reports failed.
    ///
    /// # Safety
    ///
    /// For the call's duration, every byte the arguments name, through `mem` for the buffer
    /// handles and directly for the weight views, must be host memory that is valid to read, and
    /// for the outputs and the workspace to write, and that nothing else accesses.
    pub unsafe fn launch(
        &self,
        op: &PreparedOp,
        args: &OpArgs<'_>,
        mem: &dyn HostMemory,
    ) -> Result<(), Error> {
        let prepared = self.prepared.borrow();
        let t = usize::try_from(op.handle)
            .ok()
            .and_then(|i| prepared.get(i))
            .ok_or_else(|| internal("no prepared ggml op with this handle"))?;
        if args.inputs.len() != t.inputs
            || args.weights.len() != t.weights
            || args.outputs.len() != t.outputs
        {
            return Err(internal("operand counts differ from the prepared op's"));
        }
        let rows = rows_active(args.step, mem)?.unwrap_or(t.rows);
        if rows > t.rows {
            return Err(internal("rows_active exceeds the prepared bucket"));
        }
        let workspace = match (t.workspace, args.workspace) {
            (0, _) => 0,
            (need, Some(ws)) => {
                let base = slice_addr(ws, mem)?;
                if ws.bytes.0 < need as u64 || !base.is_multiple_of(4) {
                    return Err(internal("workspace smaller than the prepared op declared"));
                }
                base
            }
            (_, None) => return Err(internal("the prepared op needs a workspace")),
        };
        let eps = match (t.eps_node, args.statics) {
            (Some(_), [Scalar::F32(eps)]) if *eps >= 0.0 => Some(*eps),
            (None, []) => None,
            _ => return Err(internal("statics differ from the prepared op's")),
        };
        // Every address is checked before any tensor is written.
        let mut addrs: SmallVec<[usize; MAX_TENSORS]> = SmallVec::new();
        for s in &t.slots {
            let addr = match s.bind {
                Bind::Input(i, dtype) => tensor_addr(&args.inputs[i], dtype, s, t.rows, mem)?,
                Bind::Output(i, dtype) => tensor_addr(&args.outputs[i], dtype, s, t.rows, mem)?,
                Bind::Weight(i) => weight_addr(&args.weights[i], s.bytes)?,
                Bind::Scratch(offset) => workspace + offset,
            };
            addrs.push(addr);
        }
        if rows == 0 {
            return Ok(());
        }
        if let Some((slot, table_rows)) = t.gather {
            // ggml's `get_rows` does not bound its indices, so a launch does.
            // SAFETY: the ids slot's address was checked above to span `t.rows` ≥ `rows` I32
            // values inside its buffer, which the caller guarantees readable.
            let ids =
                unsafe { std::slice::from_raw_parts(addrs[slot] as *const i32, rows as usize) };
            if ids
                .iter()
                .any(|&i| u32::try_from(i).map_or(true, |i| i >= table_rows))
            {
                return Err(internal("a row index lies outside the table"));
            }
        }
        // SAFETY: every slot's tensor is a live tensor of `t.ctx`, and the template is used by
        // this thread only (`CpuOps` is not `Sync`). Each address was checked above to span the
        // tensor's bytes at the bucket's rows inside its buffer, and the caller guarantees those
        // buffers for the call. Shrinking the row extent only shrinks what ggml reads and writes.
        // The plan's work buffer lies in the workspace, checked to hold `t.workspace` bytes, and
        // its threadpool is the template's own.
        let status = unsafe {
            for (s, &addr) in t.slots.iter().zip(&addrs) {
                (*s.tensor).data = addr as *mut _;
                if let Some(axis) = s.rows_axis {
                    set_rows(s.tensor, axis, rows);
                }
            }
            if let (Some(node), Some(eps)) = (t.eps_node, eps) {
                (*node).op_params[0] = eps.to_bits() as i32;
            }
            let mut plan = t.plan;
            plan.work_data = if plan.work_size == 0 {
                ptr::null_mut()
            } else {
                (workspace + t.work_offset) as *mut u8
            };
            sys::ggml_graph_compute(t.graph, &mut plan)
        };
        if status != sys::ggml_status::GGML_STATUS_SUCCESS {
            return Err(internal("the ggml graph failed"));
        }
        Ok(())
    }
}

/// Sets `tensor`'s extent on `axis` to `rows` and recomputes the strides above it.
///
/// # Safety
///
/// `tensor` must be a live ggml tensor that nothing else accesses.
unsafe fn set_rows(tensor: *mut sys::ggml_tensor, axis: usize, rows: u32) {
    // SAFETY: the caller's guarantee.
    let t = unsafe { &mut *tensor };
    t.ne[axis] = i64::from(rows);
    for i in axis + 1..4 {
        t.nb[i] = t.nb[i - 1] * t.ne[i - 1] as usize;
    }
}

/// `rows_active` from the step descriptor, or `None` when `step` is empty.
fn rows_active(step: BufferSlice, mem: &dyn HostMemory) -> Result<Option<u32>, Error> {
    if step.bytes.0 == 0 {
        return Ok(None);
    }
    const AT: usize = offset_of!(StepDesc, rows_active);
    if step.bytes.0 < (AT + 4) as u64 {
        return Err(internal("the step descriptor is too short"));
    }
    let base = slice_addr(step, mem)?;
    // SAFETY: `slice_addr` checked that the slice lies in its buffer, which the caller of
    // `launch` guarantees readable; the descriptor holds at least the field's four bytes there.
    let value = unsafe { ptr::read_unaligned((base + AT) as *const u32) };
    Ok(Some(value))
}

/// The address of a buffer slice's first byte, after checking that it lies in its buffer.
fn slice_addr(s: BufferSlice, mem: &dyn HostMemory) -> Result<usize, Error> {
    let r = mem
        .range(s.buffer)
        .ok_or_else(|| internal("an operand names an unknown buffer"))?;
    let end = s.offset.0.checked_add(s.bytes.0);
    if end.is_none_or(|e| e > r.len) || r.addr == 0 {
        return Err(internal("an operand slice lies outside its buffer"));
    }
    usize::try_from(r.addr + s.offset.0).map_err(|_| internal("an address does not fit"))
}

/// The address an input or output binds, after checking its type, shape and extent.
fn tensor_addr(
    a: &TensorArg,
    dtype: DType,
    s: &Slot,
    rows: u32,
    mem: &dyn HostMemory,
) -> Result<usize, Error> {
    // SAFETY: reading the extents of a live tensor of the template.
    let ne = unsafe { (*s.tensor).ne };
    // ggml's `ne[0]` is the innermost extent; a `TensorArg` lists dims outermost first.
    let dims = match s.rows_axis {
        Some(0) => [rows, 1, 1, 1],
        _ => [rows, u32::try_from(ne[0]).unwrap_or(u32::MAX), 1, 1],
    };
    if TensorArg::packed(a.buf, dtype, dims).as_ref() != Some(a) || !a.fits() {
        return Err(internal(
            "an operand's type or shape differs from the prepared op's",
        ));
    }
    debug_assert_eq!(a.span_bytes(), s.bytes as u64);
    let addr = slice_addr(a.buf, mem)?;
    if !(addr as u64).is_multiple_of(u64::from(dtype.bits() / 8)) {
        return Err(internal("an operand is not aligned to its element size"));
    }
    Ok(addr)
}

/// The address of a weight view, after checking that it is host-addressable and large enough.
fn weight_addr(w: &BufferView, bytes: usize) -> Result<usize, Error> {
    if !matches!(w.kind, ViewKind::Host | ViewKind::Unified) || w.addr == 0 {
        return Err(internal("a weight view is not host memory"));
    }
    if w.len.0 < bytes as u64 {
        return Err(internal("a weight view is smaller than the weight"));
    }
    if !w.addr.is_multiple_of(4) {
        return Err(internal("a weight view is not 4-byte aligned"));
    }
    usize::try_from(w.addr).map_err(|_| internal("an address does not fit"))
}

fn internal(message: &str) -> Error {
    Error::new(
        ErrorClass::Internal,
        ErrorCode::new(Domain::Kernels, 0),
        message,
    )
}

fn ggml_dtype(d: DType) -> sys::ggml_type {
    use sys::ggml_type as T;
    match d {
        DType::F16 => T::GGML_TYPE_F16,
        DType::BF16 => T::GGML_TYPE_BF16,
        _ => T::GGML_TYPE_F32,
    }
}

impl Template {
    /// Builds the template of `m` at `rows` rows: the context, the tensors and their bindings,
    /// the graph and its compute plan, and the workspace layout. The threadpool is `prepare`'s.
    fn build(m: &Mapping, rows: u32) -> Result<Self, Error> {
        let fail = || {
            Error::new(
                ErrorClass::Capacity,
                codes::ALLOC_FAILED,
                "ggml could not build the op's template",
            )
        };
        // SAFETY: plain size queries with no inputs.
        let mem_size =
            unsafe { sys::ggml_tensor_overhead() * (MAX_TENSORS + 2) + sys::ggml_graph_overhead() };
        let params = sys::ggml_init_params {
            mem_size,
            mem_buffer: ptr::null_mut(),
            no_alloc: true,
        };
        // SAFETY: with a null `mem_buffer`, ggml allocates `mem_size` bytes for the tensor and
        // graph headers itself and returns null if it cannot. `no_alloc` keeps tensor data out.
        let ctx = unsafe { sys::ggml_init(params) };
        if ctx.is_null() {
            return Err(fail());
        }
        let mut t = Template {
            ctx,
            graph: ptr::null_mut(),
            pool: ptr::null_mut(),
            // SAFETY: `ggml_cplan` is plain data (sizes, pointers, a callback option, a bool);
            // all zero is a valid value, replaced by `ggml_graph_plan` below.
            plan: unsafe { std::mem::zeroed() },
            slots: SmallVec::new(),
            eps_node: None,
            gather: None,
            work_offset: 0,
            workspace: 0,
            rows,
            inputs: 0,
            weights: 0,
            outputs: 0,
        };
        let r = i64::from(rows);
        let f32t = sys::ggml_type::GGML_TYPE_F32;
        // SAFETY: `ctx` is live and sized for `MAX_TENSORS + 2` tensors and a graph, more than
        // any template below creates; each constructor returns null only when the context is
        // full, which is checked once at the end through the output.
        let out = unsafe {
            let new2 =
                |ty, ne0: u32, ne1: i64| sys::ggml_new_tensor_2d(ctx, ty, i64::from(ne0), ne1);
            match *m {
                Mapping::MatMul {
                    weight,
                    n_out,
                    k_in,
                } => {
                    let w = new2(weight.0, k_in, i64::from(n_out));
                    let x = new2(f32t, k_in, r);
                    t.bind(w, Bind::Weight(0), None);
                    t.bind(x, Bind::Input(0, DType::F32), Some(1));
                    let y = sys::ggml_mul_mat(ctx, w, x);
                    t.bind(y, Bind::Output(0, DType::F32), Some(1));
                    y
                }
                Mapping::GateUpSilu {
                    weight,
                    n_out,
                    k_in,
                } => {
                    let w = new2(weight.0, k_in, 2 * i64::from(n_out));
                    let x = new2(f32t, k_in, r);
                    t.bind(w, Bind::Weight(0), None);
                    t.bind(x, Bind::Input(0, DType::F32), Some(1));
                    let h = sys::ggml_mul_mat(ctx, w, x);
                    t.scratch(h);
                    let y = sys::ggml_swiglu(ctx, h);
                    t.bind(y, Bind::Output(0, DType::F32), Some(1));
                    y
                }
                Mapping::RmsNorm { hidden } => {
                    let x = new2(f32t, hidden, r);
                    let g = sys::ggml_new_tensor_1d(ctx, f32t, i64::from(hidden));
                    t.bind(x, Bind::Input(0, DType::F32), Some(1));
                    t.bind(g, Bind::Weight(0), None);
                    let n = sys::ggml_rms_norm(ctx, x, 0.0);
                    t.scratch(n);
                    t.eps_node = Some(n);
                    let y = sys::ggml_mul(ctx, n, g);
                    t.bind(y, Bind::Output(0, DType::F32), Some(1));
                    y
                }
                Mapping::Silu { n } => {
                    let x = new2(f32t, n, r);
                    t.bind(x, Bind::Input(0, DType::F32), Some(1));
                    let y = sys::ggml_silu(ctx, x);
                    t.bind(y, Bind::Output(0, DType::F32), Some(1));
                    y
                }
                Mapping::Add { n } | Mapping::Mul { n } => {
                    let x = new2(f32t, n, r);
                    let z = new2(f32t, n, r);
                    t.bind(x, Bind::Input(0, DType::F32), Some(1));
                    t.bind(z, Bind::Input(1, DType::F32), Some(1));
                    let y = if matches!(m, Mapping::Add { .. }) {
                        sys::ggml_add(ctx, x, z)
                    } else {
                        sys::ggml_mul(ctx, x, z)
                    };
                    t.bind(y, Bind::Output(0, DType::F32), Some(1));
                    y
                }
                Mapping::RowGather {
                    table,
                    table_rows,
                    width,
                } => {
                    let tab = new2(table.0, width, i64::from(table_rows));
                    let ids = sys::ggml_new_tensor_1d(ctx, sys::ggml_type::GGML_TYPE_I32, r);
                    t.gather = Some((t.slots.len(), table_rows));
                    t.bind(ids, Bind::Input(0, DType::I32), Some(0));
                    t.bind(tab, Bind::Weight(0), None);
                    let y = sys::ggml_get_rows(ctx, tab, ids);
                    t.bind(y, Bind::Output(0, DType::F32), Some(1));
                    y
                }
                Mapping::Convert { from, to, n } => {
                    let x = new2(ggml_dtype(from), n, r);
                    t.bind(x, Bind::Input(0, from), Some(1));
                    let y = sys::ggml_cast(ctx, x, ggml_dtype(to));
                    t.bind(y, Bind::Output(0, to), Some(1));
                    y
                }
            }
        };
        if out.is_null() || t.slots.iter().any(|s| s.tensor.is_null()) {
            return Err(fail());
        }
        // SAFETY: `ctx` has room for the graph (`ggml_graph_overhead`); `out` and every tensor it
        // reads are live tensors of `ctx`. Planning reads the graph's shapes only.
        unsafe {
            t.graph = sys::ggml_new_graph(ctx);
            if t.graph.is_null() {
                return Err(fail());
            }
            sys::ggml_build_forward_expand(t.graph, out);
            t.plan = sys::ggml_graph_plan(t.graph, 1, ptr::null_mut());
        }
        t.work_offset = t.workspace;
        t.workspace = if t.plan.work_size == 0 {
            t.workspace
        } else {
            (t.workspace + t.plan.work_size).next_multiple_of(ALIGN)
        };
        t.inputs = t.count(|b| matches!(b, Bind::Input(..)));
        t.weights = t.count(|b| matches!(b, Bind::Weight(_)));
        t.outputs = t.count(|b| matches!(b, Bind::Output(..)));
        Ok(t)
    }

    /// Records how `tensor` is bound. A null tensor is recorded too, and refused after the build.
    fn bind(&mut self, tensor: *mut sys::ggml_tensor, bind: Bind, rows_axis: Option<usize>) {
        let bytes = if tensor.is_null() {
            0
        } else {
            // SAFETY: a live tensor of the template's context.
            unsafe { sys::ggml_nbytes(tensor) }
        };
        self.slots.push(Slot {
            tensor,
            bind,
            rows_axis,
            bytes,
        });
    }

    /// Places an intermediate tensor in the workspace, at the next aligned offset.
    fn scratch(&mut self, tensor: *mut sys::ggml_tensor) {
        let offset = self.workspace;
        self.bind(tensor, Bind::Scratch(offset), Some(1));
        let bytes = self.slots.last().map_or(0, |s| s.bytes);
        self.workspace = (offset + bytes).next_multiple_of(ALIGN);
    }

    fn count(&self, f: impl Fn(&Bind) -> bool) -> usize {
        self.slots.iter().filter(|s| f(&s.bind)).count()
    }
}

#[cfg(test)]
mod tests;
