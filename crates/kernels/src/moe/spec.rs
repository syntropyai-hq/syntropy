// SPDX-License-Identifier: MIT OR Apache-2.0

//! The one specification of the split rule (RFC-0023 §6; RFC-0040 §3.4; ADR-0404 item 4), and
//! the printers that turn it into the Rust version (`src/moe/split.rs`), the C version
//! (`include/syntropy/moe_assign.h`) and the Metal Shading Language version the Metal provider
//! compiles (`crates/kernels-metal/shaders/moe_assign.metal`).
//!
//! The rule is written once, below, as a small statement tree with only the constructs every
//! kernel language has: `u32` and `u64` integers, the `syn_route_slot` struct, counted loops,
//! `if`, and the operators C and Rust spell alike. Each printer maps the tree one statement to
//! one statement, so the versions differ only in spelling. The CUDA, HIP and Slang printers
//! join it with their providers (COMP-E1-T12).
//!
//! The walk of RFC-0023 §6 is written as a rank, so each route entry is decided on its own,
//! which is how a kernel evaluates it with one thread per entry: a missed, eligible expert's
//! copy-slot index is the number of missed, eligible experts before it in the walk's order,
//! and it is copied while that index is below the copy-slot count and the copied bytes with
//! it stay at or below `B_copy(l)`. Every expert of a layer has the same size `s_l`, so the
//! byte test is monotone in the index, and the rank gives the walk's copied set.

use std::fmt::Write as _;

use crate::step::cgen::{CType, Header};

/// The generated Rust version, relative to the crate root.
pub(super) const RUST_PATH: &str = "src/moe/split.rs";
/// The generated C version, relative to the crate root.
pub(super) const C_PATH: &str = "include/syntropy/moe_assign.h";
/// The generated MSL version, relative to this crate's root: it belongs to the Metal provider,
/// which compiles it, and the `syntropy-kernels-metal` tests run it against the Rust version.
pub(super) const MSL_PATH: &str = "../kernels-metal/shaders/moe_assign.metal";

/// A value's type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ty {
    U32,
    U64,
    /// `syn_route_slot`.
    Route,
}

/// A parameter's type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Param {
    /// Read-only `syn_route_slot` array.
    Routes,
    /// Read-only `u32` array.
    Words,
    /// Written `u32` array.
    WordsMut,
    Scalar(Ty),
}

/// An expression. Every operator is one C and Rust spell and evaluate alike for unsigned
/// operands: comparisons, `&&` and `||` (both short-circuit), `&`, `>>`, `+` and `*`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum E {
    Var(&'static str),
    /// A constant of `moe_doorbell.h`, named without its `SYN_` prefix.
    Const(&'static str),
    Lit(u32),
    /// `var.field`.
    Field(&'static str, &'static str),
    /// `array[index]`.
    Index(&'static str, Box<E>),
    Bin(&'static str, Box<E>, Box<E>),
    /// A widening cast.
    Cast(Ty, Box<E>),
}

/// A statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum S {
    Let {
        name: &'static str,
        ty: Ty,
        mutable: bool,
        init: E,
    },
    /// `name += 1`.
    Inc(&'static str),
    /// `array[index] = value`.
    Store(&'static str, E, E),
    If(E, Vec<S>, Vec<S>),
    /// `for var in 0..end`, with `var` a `u32`.
    For(&'static str, E, Vec<S>),
}

/// The specified function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Func {
    pub(super) c_name: &'static str,
    pub(super) rust_name: &'static str,
    pub(super) docs: &'static [&'static str],
    pub(super) params: &'static [(&'static str, Param)],
    pub(super) body: Vec<S>,
}

fn v(name: &'static str) -> E {
    E::Var(name)
}

fn bin(op: &'static str, a: E, b: E) -> E {
    E::Bin(op, Box::new(a), Box::new(b))
}

fn and(a: E, b: E) -> E {
    bin("&&", a, b)
}

fn or(a: E, b: E) -> E {
    bin("||", a, b)
}

fn index(array: &'static str, i: E) -> E {
    E::Index(array, Box::new(i))
}

fn cast(ty: Ty, e: E) -> E {
    E::Cast(ty, Box::new(e))
}

/// `e < n_experts && bit e of the layer's bitmap`: an expert outside the bitmap is never
/// eligible, and the bitmap is read only inside it.
fn eligible(e: &'static str) -> E {
    let word = index("eligible", bin(">>", v(e), E::Lit(5)));
    let bit = bin("&", bin(">>", word, bin("&", v(e), E::Lit(31))), E::Lit(1));
    and(bin("<", v(e), v("n_experts")), bin("==", bit, E::Lit(1)))
}

/// The split rule: `kernels::moe::assign`'s one definition.
pub(super) fn assign() -> Func {
    // Entry j comes before entry i in the walk's order: more tokens first, then the lower
    // expert index, then (for a malformed doorbell that repeats an expert) the earlier entry.
    let before = or(
        bin(">", E::Field("q", "tokens"), E::Field("r", "tokens")),
        and(
            bin("==", E::Field("q", "tokens"), E::Field("r", "tokens")),
            or(
                bin("<", v("f"), v("e")),
                and(bin("==", v("f"), v("e")), bin("<", v("j"), v("i"))),
            ),
        ),
    );
    let rank = S::For(
        "j",
        v("n_route"),
        vec![
            S::Let {
                name: "q",
                ty: Ty::Route,
                mutable: false,
                init: index("route", v("j")),
            },
            S::Let {
                name: "f",
                ty: Ty::U32,
                mutable: false,
                init: cast(Ty::U32, E::Field("q", "expert")),
            },
            S::If(
                and(
                    and(
                        bin("==", E::Field("q", "cache_slot"), E::Const("MOE_MISS")),
                        eligible("f"),
                    ),
                    before,
                ),
                vec![S::Inc("c")],
                vec![],
            ),
        ],
    );
    // `expert_bytes <= b_copy` first: with it, `(c + 1) * expert_bytes` stays below 2^64,
    // because `c + 1 < 2^32` and `b_copy < 2^32`.
    let b_copy = || cast(Ty::U64, v("b_copy"));
    let fits = and(
        bin("<", v("c"), v("copy_slots")),
        and(
            bin("<=", v("expert_bytes"), b_copy()),
            bin(
                "<=",
                bin(
                    "*",
                    bin("+", cast(Ty::U64, v("c")), E::Lit(1)),
                    v("expert_bytes"),
                ),
                b_copy(),
            ),
        ),
    );
    let missed_eligible = vec![
        S::Let {
            name: "c",
            ty: Ty::U32,
            mutable: true,
            init: E::Lit(0),
        },
        rank,
        S::If(
            fits,
            vec![S::Store("out", v("i"), v("c"))],
            vec![S::Store("out", v("i"), E::Const("MOE_CPU"))],
        ),
    ];
    let body = vec![S::For(
        "i",
        v("n_route"),
        vec![
            S::Let {
                name: "r",
                ty: Ty::Route,
                mutable: false,
                init: index("route", v("i")),
            },
            S::Let {
                name: "e",
                ty: Ty::U32,
                mutable: false,
                init: cast(Ty::U32, E::Field("r", "expert")),
            },
            S::If(
                bin("!=", E::Field("r", "cache_slot"), E::Const("MOE_MISS")),
                vec![S::Store("out", v("i"), E::Const("MOE_HIT"))],
                vec![S::If(
                    eligible("e"),
                    missed_eligible,
                    vec![S::Store("out", v("i"), E::Const("MOE_CPU"))],
                )],
            ),
        ],
    )];
    Func {
        c_name: "syn_moe_assign",
        rust_name: "assign_codes",
        docs: &[
            "Splits one MoE layer's routed experts between the three sites (RFC-0023 §6;",
            "RFC-0040 §3.4): writes `out[i]` for each of the `n_route` entries of `route`.",
            "",
            "- `MOE_HIT` when the entry's `cache_slot` is not `MOE_MISS`.",
            "- A copy-slot index `c` when the expert missed, is marked in `eligible` (bit e % 32",
            "  of word e / 32, for e below `n_experts`), and is the c-th such expert in the walk's",
            "  order (more tokens first, then the lower expert index, then the earlier entry),",
            "  with `c < copy_slots` and `(c + 1) * expert_bytes <= b_copy`.",
            "- `MOE_CPU` for every other miss.",
        ],
        params: &[
            ("route", Param::Routes),
            ("n_route", Param::Scalar(Ty::U32)),
            ("eligible", Param::Words),
            ("n_experts", Param::Scalar(Ty::U32)),
            ("b_copy", Param::Scalar(Ty::U32)),
            ("expert_bytes", Param::Scalar(Ty::U64)),
            ("copy_slots", Param::Scalar(Ty::U32)),
            ("out", Param::WordsMut),
        ],
        body,
    }
}

/// The languages the specification prints in today.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lang {
    Rust,
    C,
    /// Metal Shading Language: C's spelling with MSL's type names and `device` pointers.
    Msl,
}

impl Lang {
    fn ty(self, ty: Ty) -> &'static str {
        match (self, ty) {
            (Self::Rust, Ty::U32) => "u32",
            (Self::Rust, Ty::U64) => "u64",
            (Self::Rust, Ty::Route) => "RouteSlot",
            (Self::C, Ty::U32) => "uint32_t",
            (Self::C, Ty::U64) => "uint64_t",
            (Self::C, Ty::Route) => "struct syn_route_slot",
            (Self::Msl, Ty::U32) => "uint",
            (Self::Msl, Ty::U64) => "ulong",
            (Self::Msl, Ty::Route) => "syn_route_slot",
        }
    }

    fn param(self, name: &str, p: Param) -> String {
        match (self, p) {
            (Self::Rust, Param::Routes) => format!("{name}: &[RouteSlot]"),
            (Self::Rust, Param::Words) => format!("{name}: &[u32]"),
            (Self::Rust, Param::WordsMut) => format!("{name}: &mut [u32]"),
            (Self::C, Param::Routes) => format!("const struct syn_route_slot *{name}"),
            (Self::C, Param::Words) => format!("const uint32_t *{name}"),
            (Self::C, Param::WordsMut) => format!("uint32_t *{name}"),
            (Self::Msl, Param::Routes) => format!("device const syn_route_slot *{name}"),
            (Self::Msl, Param::Words) => format!("device const uint *{name}"),
            (Self::Msl, Param::WordsMut) => format!("device uint *{name}"),
            (_, Param::Scalar(ty)) => match self {
                Self::Rust => format!("{name}: {}", self.ty(ty)),
                Self::C | Self::Msl => format!("{} {name}", self.ty(ty)),
            },
        }
    }

    /// `e` as an operand: parenthesized unless it is atomic.
    fn operand(self, e: &E) -> String {
        match e {
            E::Bin(..) | E::Cast(..) => format!("({})", self.expr(e)),
            _ => self.expr(e),
        }
    }

    fn expr(self, e: &E) -> String {
        match e {
            E::Var(n) => (*n).to_owned(),
            E::Const(n) => match self {
                Self::Rust => (*n).to_owned(),
                Self::C | Self::Msl => format!("SYN_{n}"),
            },
            E::Lit(n) => match self {
                Self::Rust => n.to_string(),
                Self::C | Self::Msl => format!("{n}u"),
            },
            E::Field(var, field) => format!("{var}.{field}"),
            E::Index(array, i) => match self {
                Self::Rust => format!("{array}[{} as usize]", self.operand(i)),
                Self::C | Self::Msl => format!("{array}[{}]", self.expr(i)),
            },
            E::Bin(op, a, b) => format!("{} {op} {}", self.operand(a), self.operand(b)),
            E::Cast(ty, x) => match self {
                Self::Rust => format!("{} as {}", self.operand(x), self.ty(*ty)),
                Self::C | Self::Msl => format!("({}){}", self.ty(*ty), self.operand(x)),
            },
        }
    }

    fn stmts(self, out: &mut String, depth: usize, body: &[S]) {
        for s in body {
            self.stmt(out, depth, s);
        }
    }

    fn stmt(self, out: &mut String, depth: usize, s: &S) {
        let pad = "    ".repeat(depth);
        match s {
            S::Let {
                name,
                ty,
                mutable,
                init,
            } => {
                let init = self.expr(init);
                let ty = self.ty(*ty);
                let _ = match (self, mutable) {
                    (Self::Rust, true) => writeln!(out, "{pad}let mut {name}: {ty} = {init};"),
                    (Self::Rust, false) => writeln!(out, "{pad}let {name}: {ty} = {init};"),
                    (Self::C | Self::Msl, true) => writeln!(out, "{pad}{ty} {name} = {init};"),
                    (Self::C | Self::Msl, false) => {
                        writeln!(out, "{pad}const {ty} {name} = {init};")
                    }
                };
            }
            S::Inc(name) => {
                let one = self.expr(&E::Lit(1));
                let _ = writeln!(out, "{pad}{name} += {one};");
            }
            S::Store(array, i, value) => {
                let target = self.expr(&index(array, i.clone()));
                let _ = writeln!(out, "{pad}{target} = {};", self.expr(value));
            }
            S::If(cond, then, els) => {
                let _ = writeln!(out, "{pad}{} {{", self.cond("if", cond));
                self.stmts(out, depth + 1, then);
                let mut els = els;
                // An `else` holding only an `if` prints as `else if`.
                while let [S::If(cond, then, rest)] = els.as_slice() {
                    let _ = writeln!(out, "{pad}}} else {} {{", self.cond("if", cond));
                    self.stmts(out, depth + 1, then);
                    els = rest;
                }
                if !els.is_empty() {
                    let _ = writeln!(out, "{pad}}} else {{");
                    self.stmts(out, depth + 1, els);
                }
                let _ = writeln!(out, "{pad}}}");
            }
            S::For(var, end, body) => {
                let end = self.expr(end);
                let _ = match self {
                    Self::Rust => writeln!(out, "{pad}for {var} in 0..{end} {{"),
                    Self::C | Self::Msl => writeln!(
                        out,
                        "{pad}for ({} {var} = 0u; {var} < {end}; {var}++) {{",
                        self.ty(Ty::U32)
                    ),
                };
                self.stmts(out, depth + 1, body);
                let _ = writeln!(out, "{pad}}}");
            }
        }
    }

    fn cond(self, keyword: &str, cond: &E) -> String {
        match self {
            Self::Rust => format!("{keyword} {}", self.expr(cond)),
            Self::C | Self::Msl => format!("{keyword} ({})", self.expr(cond)),
        }
    }
}

/// The Rust version of `f`.
pub(super) fn rust(f: &Func) -> String {
    let mut out = String::from(
        "// SPDX-License-Identifier: MIT OR Apache-2.0\n\
         // @generated from the specification in `src/moe/spec.rs` by the `syntropy-kernels`\n\
         // tests. Do not edit: change the specification and run `SYNTROPY_BLESS_GENERATED=1\n\
         // cargo test -p syntropy-kernels moe::`.\n\
         \n\
         //! The split rule in Rust (RFC-0023 §6; RFC-0040 §3.4; ADR-0404 item 4), printed from\n\
         //! the same specification as `include/syntropy/moe_assign.h`.\n\
         \n\
         use super::mirror::{MOE_CPU, MOE_HIT, MOE_MISS, RouteSlot};\n\n",
    );
    for doc in f.docs {
        if doc.is_empty() {
            out.push_str("///\n");
        } else {
            let _ = writeln!(out, "/// {doc}");
        }
    }
    out.push_str(
        "// The printer keeps the specification's shape: indexed loops and one parameter per\n\
         // kernel argument.\n\
         #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]\n",
    );
    let _ = writeln!(out, "pub(super) fn {}(", f.rust_name);
    for &(name, p) in f.params {
        let _ = writeln!(out, "    {},", Lang::Rust.param(name, p));
    }
    out.push_str(") {\n");
    Lang::Rust.stmts(&mut out, 1, &f.body);
    out.push_str("}\n");
    out
}

/// The C version of `f`: a header with one `static inline` function.
pub(super) fn c(f: &Func) -> String {
    let mut out = String::from(
        "/* SPDX-License-Identifier: MIT OR Apache-2.0 */\n\
         \n\
         /*\n\
         \x20* @generated from the specification in `crates/kernels/src/moe/spec.rs` by the\n\
         \x20* `syntropy-kernels` tests. Do not edit: change the specification and run\n\
         \x20* `SYNTROPY_BLESS_GENERATED=1 cargo test -p syntropy-kernels moe::`.\n\
         \x20*\n\
         \x20* The split rule in C (RFC-0023 §6; RFC-0040 §3.4; ADR-0404 item 4), printed from the\n\
         \x20* same specification as `crates/kernels/src/moe/split.rs`. The tests compile it as C11\n\
         \x20* and C++11 and run it against the Rust version.\n\
         \x20*/\n\
         \n\
         #ifndef SYNTROPY_MOE_ASSIGN_H\n\
         #define SYNTROPY_MOE_ASSIGN_H\n\
         \n\
         #include <stdint.h>\n\
         \n\
         #include \"syntropy/moe_doorbell.h\"\n\
         \n",
    );
    for doc in f.docs {
        if doc.is_empty() {
            out.push_str("///\n");
        } else {
            let _ = writeln!(out, "/// {doc}");
        }
    }
    let params: Vec<String> = f
        .params
        .iter()
        .map(|&(name, p)| format!("    {}", Lang::C.param(name, p)))
        .collect();
    let _ = writeln!(
        out,
        "static inline void {}(\n{}) {{",
        f.c_name,
        params.join(",\n")
    );
    Lang::C.stmts(&mut out, 1, &f.body);
    out.push_str("}\n\n/* SYNTROPY_MOE_ASSIGN_H */\n#endif\n");
    out
}

/// The MSL name of a header member's type.
fn msl_member(ty: CType) -> &'static str {
    match ty {
        CType::U8 => "uchar",
        CType::U16 => "ushort",
        CType::U32 => "uint",
        CType::I32 => "int",
        CType::U64 => "ulong",
        CType::F32 => "float",
    }
}

/// Doc lines as `///` comments at `indent`.
fn msl_docs<T: AsRef<str>>(out: &mut String, indent: &str, docs: &[T]) {
    for doc in docs {
        let doc = doc.as_ref();
        if doc.is_empty() {
            let _ = writeln!(out, "{indent}///");
        } else {
            let _ = writeln!(out, "{indent}/// {doc}");
        }
    }
}

/// The MSL version of `f`, for the Metal provider. It holds `header`'s constants and structs
/// (the doorbell's one definition, `moe_doorbell.h`, which an MSL source compiled at run time
/// cannot include), with `static_assert`s of the layout the header gives them; `f`'s outer
/// loop body as `<c_name>_entry`, which decides route entry `i` on its own; and the kernel
/// `<c_name>`, one thread per route entry.
///
/// # Panics
///
/// If `f`'s body is not one loop `for i in 0..n_route`, the shape the kernel splits across
/// threads, or a constant does not fit in a `uint`.
pub(super) fn msl(f: &Func, header: &Header) -> String {
    let [S::For("i", E::Var("n_route"), entry)] = f.body.as_slice() else {
        panic!("the MSL printer needs a body of one loop `for i in 0..n_route`");
    };
    let mut out = String::from(
        "// SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         // @generated from the specification in `crates/kernels/src/moe/spec.rs` and the layouts\n\
         // of `crates/kernels/include/syntropy/moe_doorbell.h` by the `syntropy-kernels` tests. Do\n\
         // not edit: change the specification or the header and run\n\
         // `SYNTROPY_BLESS_GENERATED=1 cargo test -p syntropy-kernels moe::`.\n\
         //\n\
         // The split rule in Metal Shading Language (RFC-0023 §6; RFC-0040 §3.4; ADR-0404 item 4),\n\
         // printed from the same specification as `crates/kernels/src/moe/split.rs` and\n\
         // `crates/kernels/include/syntropy/moe_assign.h`. The `syntropy-kernels-metal` tests\n\
         // compile it with the Metal compiler and run it against the Rust version.\n\
         \n\
         #include <metal_stdlib>\n\
         \n\
         using namespace metal;\n",
    );
    for d in &header.defines {
        let value = u32::try_from(d.value).expect("an MSL constant fits in a uint");
        out.push('\n');
        msl_docs(&mut out, "", &d.docs);
        let _ = writeln!(out, "constant uint {} = {value}u;", d.name);
    }
    for s in &header.structs {
        out.push('\n');
        msl_docs(&mut out, "", &s.docs);
        let _ = writeln!(out, "struct {} {{", s.c_name);
        for m in &s.fields {
            msl_docs(&mut out, "    ", &m.docs);
            let _ = writeln!(out, "    {} {};", msl_member(m.ty), m.name);
        }
        out.push_str("};\n");
    }
    out.push_str(
        "\n// The layout `moe_doorbell.h` gives each struct, checked by the Metal compiler.\n",
    );
    for s in &header.structs {
        let tag = &s.c_name;
        let _ = writeln!(
            out,
            "static_assert(sizeof({tag}) == {}, \"{tag} size\");",
            s.size
        );
        let _ = writeln!(
            out,
            "static_assert(alignof({tag}) == {}, \"{tag} align\");",
            s.align
        );
        for m in &s.fields {
            let _ = writeln!(
                out,
                "static_assert(__builtin_offsetof({tag}, {0}) == {1}, \"{tag}.{0}\");",
                m.name, m.offset
            );
        }
    }
    out.push('\n');
    msl_docs(&mut out, "", f.docs);
    out.push_str(
        "///\n\
         /// This function decides entry `i` alone, reading every entry of `route`: it is the body\n\
         /// of the specification's loop over `i`.\n",
    );
    let mut params: Vec<String> = f
        .params
        .iter()
        .map(|&(name, p)| format!("    {}", Lang::Msl.param(name, p)))
        .collect();
    params.push(format!("    {} i", Lang::Msl.ty(Ty::U32)));
    let _ = writeln!(
        out,
        "static inline void {}_entry(\n{}) {{",
        f.c_name,
        params.join(",\n")
    );
    Lang::Msl.stmts(&mut out, 1, entry);
    out.push_str("}\n\n");
    let _ = writeln!(
        out,
        "/// `{}_entry` for every route entry, one thread per entry: dispatch at least\n\
         /// `n_route` threads in one dimension. Threads at or past `n_route` write nothing.",
        f.c_name
    );
    let mut params = Vec::new();
    let mut args = Vec::new();
    for (k, &(name, p)) in f.params.iter().enumerate() {
        let decl = match p {
            Param::Scalar(ty) => format!("constant {} &{name}", Lang::Msl.ty(ty)),
            _ => Lang::Msl.param(name, p),
        };
        params.push(format!("    {decl} [[buffer({k})]]"));
        args.push(name);
    }
    params.push(format!(
        "    {} i [[thread_position_in_grid]]",
        Lang::Msl.ty(Ty::U32)
    ));
    let _ = writeln!(out, "kernel void {}(\n{}) {{", f.c_name, params.join(",\n"));
    let _ = writeln!(
        out,
        "    if (i < n_route) {{\n        {}_entry({}, i);\n    }}\n}}",
        f.c_name,
        args.join(", ")
    );
    out
}
