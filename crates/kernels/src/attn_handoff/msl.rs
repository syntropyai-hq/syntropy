// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Metal Shading Language version of `include/syntropy/attn_handoff.h`, which an MSL source
//! compiled at run time cannot include: its constants and structs, with `static_assert`s of the
//! layout the header gives them, so the Metal compiler checks that it lays them out as C does.

use std::fmt::Write as _;

use crate::step::cgen::{CType, Header};

/// The Metal version's path, relative to the `syntropy-kernels` crate root.
pub(super) const MSL_PATH: &str = "../kernels-metal/shaders/attn_handoff.metal";

/// The MSL name of a header member's type.
fn member(ty: CType) -> &'static str {
    match ty {
        CType::U8 => "uchar",
        CType::U16 => "ushort",
        CType::U32 => "uint",
        CType::I32 => "int",
        CType::U64 => "ulong",
        CType::F32 => "float",
    }
}

fn docs(out: &mut String, indent: &str, lines: &[String]) {
    for line in lines {
        if line.is_empty() {
            let _ = writeln!(out, "{indent}///");
        } else {
            let _ = writeln!(out, "{indent}/// {line}");
        }
    }
}

/// The MSL version of `header`, the attention hand-off's.
///
/// # Panics
///
/// If a constant does not fit in a `uint`.
pub(super) fn generate(header: &Header) -> String {
    let mut out = String::from(
        "// SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         // @generated from `crates/kernels/include/syntropy/attn_handoff.h` by the\n\
         // `syntropy-kernels` tests. Do not edit: change the header and run\n\
         // `SYNTROPY_BLESS_GENERATED=1 cargo test -p syntropy-kernels attn_handoff::`.\n\
         //\n\
         // The attention hand-off's layouts in Metal Shading Language (RFC-0022 §5.3), for the\n\
         // Metal provider. The `syntropy-kernels-metal` tests compile it with the Metal compiler,\n\
         // which checks the `static_assert`s below.\n\
         \n\
         #include <metal_stdlib>\n\
         \n\
         using namespace metal;\n",
    );
    for d in &header.defines {
        let value = u32::try_from(d.value).expect("an MSL constant fits in a uint");
        out.push('\n');
        docs(&mut out, "", &d.docs);
        let _ = writeln!(out, "constant uint {} = {value}u;", d.name);
    }
    for s in &header.structs {
        out.push('\n');
        docs(&mut out, "", &s.docs);
        let _ = writeln!(out, "struct {} {{", s.c_name);
        for (i, m) in s.fields.iter().enumerate() {
            docs(&mut out, "    ", &m.docs);
            // The header raises a struct's alignment on its first member only.
            let raised = i == 0 && s.align > m.ty.size();
            let align = if raised {
                format!("alignas({}) ", s.align)
            } else {
                String::new()
            };
            let _ = writeln!(out, "    {align}{} {};", member(m.ty), m.name);
        }
        out.push_str("};\n");
    }
    out.push_str(
        "\n// The layout `attn_handoff.h` gives each struct, checked by the Metal compiler.\n",
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
    out
}
