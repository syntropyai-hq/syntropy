// SPDX-License-Identifier: MIT OR Apache-2.0

//! The generator of [`mirror`](super::mirror): it reads `include/syntropy/step_desc.h`, lays
//! its structs out by the C rules, and writes the Rust mirror (RFC-0040 §1, "kernel argument
//! structs have one source of truth"). The MoE doorbell's mirror
//! ([`moe::mirror`](crate::moe::mirror)) comes from `include/syntropy/moe_doorbell.h` the
//! same way ([`MOE_DOORBELL`]).
//!
//! It reads only the subset of C the header's comment names and fails closed: a preprocessor
//! directive it does not know (`#pragma pack` among them), any directive inside a struct, a
//! struct or constant under a condition, and a block comment sharing a line with code are all
//! errors, so a member the generator cannot lay out never passes silently. It needs no C
//! compiler; the tests also run one when the machine has it.

use std::fmt::Write as _;

/// The header, relative to the crate root.
pub(super) const HEADER_PATH: &str = "include/syntropy/step_desc.h";
/// The generated mirror, relative to the crate root.
pub(super) const MIRROR_PATH: &str = "src/step/mirror.rs";

/// One header and the mirror generated from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Target {
    /// The header, relative to the crate root.
    pub(crate) header: &'static str,
    /// The header as a translation unit includes it.
    pub(crate) include: &'static str,
    /// The generated mirror, relative to the crate root.
    pub(crate) mirror: &'static str,
    /// The sections the mirror's module documentation cites.
    pub(crate) cites: &'static str,
    /// The test filter the bless instruction names.
    pub(crate) filter: &'static str,
    /// The path of this module as the mirror names it.
    pub(crate) cgen: &'static str,
}

/// The step descriptor's header and mirror (RFC-0040 §3.5).
pub(crate) const STEP_DESC: Target = Target {
    header: HEADER_PATH,
    include: "syntropy/step_desc.h",
    mirror: MIRROR_PATH,
    cites: "RFC-0040 §1, §3.5",
    filter: "step::",
    cgen: "super::cgen",
};

/// The MoE doorbell's header and mirror (RFC-0023 §2; RFC-0040 §3.4).
pub(crate) const MOE_DOORBELL: Target = Target {
    header: "include/syntropy/moe_doorbell.h",
    include: "syntropy/moe_doorbell.h",
    mirror: "src/moe/mirror.rs",
    cites: "RFC-0023 §2; RFC-0040 §1, §3.4",
    filter: "moe::",
    cgen: "crate::step::cgen",
};

/// The table-delta ring's header and mirror (RFC-0023 §5.4).
pub(crate) const TABLE_DELTA_RING: Target = Target {
    header: "include/syntropy/table_delta_ring.h",
    include: "syntropy/table_delta_ring.h",
    mirror: "src/delta_ring/mirror.rs",
    cites: "RFC-0023 §5.4",
    filter: "delta_ring::",
    cgen: "crate::step::cgen",
};

/// The attention hand-off's header and mirror (RFC-0022 §5.3).
pub(crate) const ATTN_HANDOFF: Target = Target {
    header: "include/syntropy/attn_handoff.h",
    include: "syntropy/attn_handoff.h",
    mirror: "src/attn_handoff/mirror.rs",
    cites: "RFC-0022 §5.3; RFC-0040 §4.2",
    filter: "attn_handoff::",
    cgen: "crate::step::cgen",
};

/// A member type the generator knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CType {
    U8,
    U16,
    U32,
    I32,
    U64,
    F32,
}

impl CType {
    fn parse(word: &str) -> Option<Self> {
        match word {
            "uint8_t" => Some(Self::U8),
            "uint16_t" => Some(Self::U16),
            "uint32_t" => Some(Self::U32),
            "int32_t" => Some(Self::I32),
            "uint64_t" => Some(Self::U64),
            "float" => Some(Self::F32),
            _ => None,
        }
    }

    /// Size and alignment, which are equal for these types on every target Syntropy builds.
    pub(crate) fn size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 => 8,
        }
    }

    fn rust(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64 => "u64",
            Self::F32 => "f32",
        }
    }
}

/// `#define NAME value` with `NAME` starting `SYN_` and a value the generator can read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Define {
    pub(crate) docs: Vec<String>,
    pub(crate) name: String,
    /// The value as a Rust expression.
    pub(crate) expr: String,
    pub(crate) value: u64,
}

/// One struct member, with the offset C gives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Field {
    pub(crate) docs: Vec<String>,
    pub(crate) name: String,
    pub(crate) ty: CType,
    pub(crate) offset: usize,
}

/// One struct, laid out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CStruct {
    pub(crate) docs: Vec<String>,
    /// The C tag, `syn_step_desc`.
    pub(crate) c_name: String,
    pub(crate) fields: Vec<Field>,
    pub(crate) size: usize,
    pub(crate) align: usize,
}

impl CStruct {
    /// The Rust name: the tag without `syn_`, in upper camel case.
    pub(crate) fn rust_name(&self) -> String {
        let bare = self.c_name.strip_prefix("syn_").unwrap_or(&self.c_name);
        bare.split('_')
            .map(|w| {
                let mut c = w.chars();
                c.next()
                    .map(|f| f.to_ascii_uppercase().to_string() + c.as_str())
                    .unwrap_or_default()
            })
            .collect()
    }
}

/// A mirror's layout as the Rust compiler gives it: what the generated `LAYOUTS` lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MirrorLayout {
    /// The C tag.
    pub(crate) tag: &'static str,
    pub(crate) size: usize,
    pub(crate) align: usize,
    /// Each member's name and offset, in declaration order.
    pub(crate) fields: &'static [(&'static str, usize)],
}

/// What the header declares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Header {
    pub(crate) defines: Vec<Define>,
    pub(crate) structs: Vec<CStruct>,
}

/// Reads a `#define` value: a decimal literal with an optional `u` suffix, or `(1u << n)`.
fn parse_value(text: &str) -> Option<(String, u64)> {
    let literal = |s: &str| -> Option<u64> { s.trim_end_matches(['u', 'U']).parse().ok() };
    if let Some(inner) = text.strip_prefix('(').and_then(|t| t.strip_suffix(')')) {
        let (base, shift) = inner.split_once("<<")?;
        let (base, shift) = (literal(base.trim())?, literal(shift.trim())?);
        let value = base.checked_shl(u32::try_from(shift).ok()?)?;
        return Some((format!("{base} << {shift}"), value));
    }
    let value = literal(text)?;
    Some((value.to_string(), value))
}

/// Parses the step descriptor's header: [`parse_named`] with its path.
pub(super) fn parse(text: &str) -> Result<Header, String> {
    parse_named(HEADER_PATH, text)
}

/// Parses a header and lays its structs out: each member at the next multiple of its
/// alignment (raised by `SYN_ALIGNAS`), the struct's alignment the largest of its members',
/// its size rounded up to that alignment. `path` names the header in errors.
pub(crate) fn parse_named(path: &str, text: &str) -> Result<Header, String> {
    let mut defines = Vec::new();
    let mut structs = Vec::new();
    let mut docs: Vec<String> = Vec::new();
    let mut open: Option<CStruct> = None;
    let mut in_block_comment = false;
    // Depth of `#if`/`#ifndef` nesting; 1 is inside the include guard.
    let mut depth = 0u32;

    for (ix, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let at = |msg: &str| format!("{path}:{}: {msg}: `{line}`", ix + 1);
        // Block comments must fill their lines: code before `/*` or after `*/` on the same
        // line would otherwise be skipped with the comment.
        if in_block_comment {
            if let Some(end) = line.find("*/") {
                if !line[end + 2..].trim().is_empty() {
                    return Err(at("code after a block comment"));
                }
                in_block_comment = false;
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("/*") {
            match rest.find("*/") {
                None => in_block_comment = true,
                Some(end) if !rest[end + 2..].trim().is_empty() => {
                    return Err(at("code after a block comment"));
                }
                Some(_) => {}
            }
            continue;
        }
        if line.contains("/*") || line.contains("*/") {
            return Err(at("a block comment inside a line"));
        }
        if let Some(doc) = line.strip_prefix("///") {
            docs.push(doc.strip_prefix(' ').unwrap_or(doc).to_owned());
            continue;
        }
        if line.is_empty() {
            continue;
        }
        if let Some(directive) = line.strip_prefix('#') {
            if open.is_some() {
                return Err(at("a preprocessor directive inside a struct"));
            }
            let directive = directive.trim_start();
            let word = directive
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .next()
                .unwrap_or("");
            match word {
                "if" | "ifndef" => depth += 1,
                "endif" => {
                    depth = depth
                        .checked_sub(1)
                        .ok_or_else(|| at("unbalanced #endif"))?
                }
                "else" | "include" => {}
                "define" => {
                    let def = directive["define".len()..].trim_start();
                    let (name, value) = def.split_once(' ').unwrap_or((def, ""));
                    if name.starts_with("SYN_") && !name.contains('(') {
                        if depth > 1 {
                            return Err(at("a constant under a condition"));
                        }
                        let (expr, value) =
                            parse_value(value.trim()).ok_or_else(|| at("unreadable value"))?;
                        defines.push(Define {
                            docs: std::mem::take(&mut docs),
                            name: name.to_owned(),
                            expr,
                            value,
                        });
                    } else {
                        // A function-like or foreign macro is not read; its comment
                        // documents it, not the next item.
                        docs.clear();
                    }
                }
                _ => return Err(at("a preprocessor directive the generator does not read")),
            }
            continue;
        }
        if let Some(s) = open.as_mut() {
            if line == "};" {
                let mut s = open.take().expect("open struct");
                s.size = s.size.next_multiple_of(s.align);
                structs.push(s);
                continue;
            }
            let (align, rest) = match line.strip_prefix("SYN_ALIGNAS(") {
                Some(rest) => {
                    let (n, rest) = rest.split_once(')').ok_or_else(|| at("bad alignment"))?;
                    let n: usize = n.parse().map_err(|_| at("bad alignment"))?;
                    if !s.fields.is_empty() || !n.is_power_of_two() {
                        return Err(at("SYN_ALIGNAS only on the first member, a power of two"));
                    }
                    (n, rest.trim())
                }
                None => (0, line),
            };
            let decl = rest
                .strip_suffix(';')
                .ok_or_else(|| at("expected a member"))?;
            let (ty, name) = decl
                .split_once(' ')
                .ok_or_else(|| at("expected a member"))?;
            let ty = CType::parse(ty).ok_or_else(|| at("unsupported member type"))?;
            let name = name.trim();
            if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err(at("one member per line, no arrays"));
            }
            let field_align = ty.size().max(align);
            let offset = s.size.next_multiple_of(field_align);
            s.align = s.align.max(field_align);
            s.size = offset + ty.size();
            s.fields.push(Field {
                docs: std::mem::take(&mut docs),
                name: name.to_owned(),
                ty,
                offset,
            });
            continue;
        }
        if let Some(tag) = line
            .strip_prefix("struct ")
            .and_then(|l| l.strip_suffix(" {"))
        {
            if depth > 1 {
                return Err(at("a struct under a condition"));
            }
            open = Some(CStruct {
                docs: std::mem::take(&mut docs),
                c_name: tag.to_owned(),
                fields: Vec::new(),
                size: 0,
                align: 1,
            });
            continue;
        }
        return Err(at("outside the subset the generator reads"));
    }
    if open.is_some() {
        return Err(format!("{path}: unterminated struct"));
    }
    if in_block_comment {
        return Err(format!("{path}: unterminated block comment"));
    }
    if depth != 0 {
        return Err(format!("{path}: unbalanced #if"));
    }
    Ok(Header { defines, structs })
}

fn push_docs(out: &mut String, indent: &str, docs: &[String]) {
    for doc in docs {
        if doc.is_empty() {
            let _ = writeln!(out, "{indent}///");
        } else {
            let _ = writeln!(out, "{indent}/// {doc}");
        }
    }
}

/// The step descriptor's mirror: [`generate_for`] with [`STEP_DESC`].
pub(super) fn generate(header: &Header) -> String {
    generate_for(header, &STEP_DESC)
}

/// The Rust mirror of `header`, the header of `target`, formatted as rustfmt leaves it.
pub(crate) fn generate_for(header: &Header, target: &Target) -> String {
    let Target {
        header: path,
        cites,
        filter,
        cgen,
        ..
    } = *target;
    let mut out = String::new();
    let _ = write!(
        out,
        "// SPDX-License-Identifier: MIT OR Apache-2.0\n\
         // @generated from `{path}` by the `syntropy-kernels` tests. Do not\n\
         // edit: change the header and run `SYNTROPY_BLESS_GENERATED=1 cargo test -p\n\
         // syntropy-kernels {filter}`.\n\
         \n\
         //! The `#[repr(C)]` mirrors of `{path}` ({cites}), with\n\
         //! compile-time checks of the layout the generator computed from the header.\n",
    );
    for d in &header.defines {
        out.push('\n');
        push_docs(&mut out, "", &d.docs);
        let name = d.name.strip_prefix("SYN_").unwrap_or(&d.name);
        let _ = writeln!(out, "pub const {name}: u32 = {};", d.expr);
    }
    for s in &header.structs {
        let name = s.rust_name();
        out.push('\n');
        push_docs(&mut out, "", &s.docs);
        let _ = writeln!(out, "#[repr(C, align({}))]", s.align);
        // A float member has no total equality.
        if s.fields.iter().any(|f| f.ty == CType::F32) {
            out.push_str("#[derive(Clone, Copy, Debug, Default, PartialEq)]\n");
        } else {
            out.push_str("#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]\n");
        }
        let _ = writeln!(out, "pub struct {name} {{");
        for f in &s.fields {
            push_docs(&mut out, "    ", &f.docs);
            let _ = writeln!(out, "    pub {}: {},", f.name, f.ty.rust());
        }
        out.push_str("}\n");
    }
    out.push_str("\n// The layout the header declares, as C lays it out.\nconst _: () = {\n");
    for s in &header.structs {
        let name = s.rust_name();
        let _ = writeln!(out, "    assert!(size_of::<{name}>() == {});", s.size);
        let _ = writeln!(out, "    assert!(align_of::<{name}>() == {});", s.align);
        for f in &s.fields {
            let _ = writeln!(
                out,
                "    assert!(core::mem::offset_of!({name}, {}) == {});",
                f.name, f.offset
            );
        }
    }
    out.push_str("};\n");
    let _ = write!(
        out,
        "\n/// Each mirror's C tag, size, alignment and member offsets as the Rust compiler lays\n\
         /// them out, for the tests that compare them with the header and with a C compiler.\n\
         #[cfg(test)]\n\
         pub(super) const LAYOUTS: &[{cgen}::MirrorLayout] = &[\n",
    );
    for s in &header.structs {
        let name = s.rust_name();
        let _ = writeln!(out, "    {cgen}::MirrorLayout {{");
        let _ = writeln!(out, "        tag: \"{}\",", s.c_name);
        let _ = writeln!(out, "        size: size_of::<{name}>(),");
        let _ = writeln!(out, "        align: align_of::<{name}>(),");
        let _ = writeln!(out, "        fields: &[");
        for f in &s.fields {
            let _ = writeln!(
                out,
                "            (\"{0}\", core::mem::offset_of!({name}, {0})),",
                f.name
            );
        }
        let _ = writeln!(out, "        ],");
        let _ = writeln!(out, "    }},");
    }
    out.push_str("];\n");
    out
}

/// A C translation unit that includes the header and asserts, with `_Static_assert`, the
/// layout in `layouts` (the Rust compiler's): what a C compiler checks when one is present.
pub(super) fn c_layout_check(layouts: &[MirrorLayout]) -> String {
    c_layout_check_for(STEP_DESC.include, layouts)
}

/// [`c_layout_check`] for the header a translation unit includes as `include`.
pub(crate) fn c_layout_check_for(include: &str, layouts: &[MirrorLayout]) -> String {
    let mut out = format!(
        "#include <stddef.h>\n#include \"{include}\"\n#if defined(__cplusplus)\n\
         #define SYN_ASSERT static_assert\n#define SYN_ALIGNOF alignof\n#else\n\
         #define SYN_ASSERT _Static_assert\n#define SYN_ALIGNOF _Alignof\n#endif\n",
    );
    for &MirrorLayout {
        tag,
        size,
        align,
        fields,
    } in layouts
    {
        let _ = writeln!(
            out,
            "SYN_ASSERT(sizeof(struct {tag}) == {size}, \"{tag} size\");"
        );
        let _ = writeln!(
            out,
            "SYN_ASSERT(SYN_ALIGNOF(struct {tag}) == {align}, \"{tag} align\");"
        );
        for (field, offset) in fields {
            let _ = writeln!(
                out,
                "SYN_ASSERT(offsetof(struct {tag}, {field}) == {offset}, \"{tag}.{field}\");"
            );
        }
    }
    out
}
