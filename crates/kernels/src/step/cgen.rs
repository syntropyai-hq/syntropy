// SPDX-License-Identifier: MIT OR Apache-2.0

//! The generator of [`mirror`](super::mirror): it reads `include/syntropy/step_desc.h`, lays
//! its structs out by the C rules, and writes the Rust mirror (RFC-0040 §1, "kernel argument
//! structs have one source of truth").
//!
//! It reads only the subset of C the header's comment names, and rejects anything else inside
//! a struct, so a member the generator cannot lay out never passes silently. It needs no C
//! compiler; the tests also run one when the machine has it.

use std::fmt::Write as _;

/// The header, relative to the crate root.
pub(super) const HEADER_PATH: &str = "include/syntropy/step_desc.h";
/// The generated mirror, relative to the crate root.
pub(super) const MIRROR_PATH: &str = "src/step/mirror.rs";

/// A member type the generator knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CType {
    U32,
    I32,
    U64,
}

impl CType {
    fn parse(word: &str) -> Option<Self> {
        match word {
            "uint32_t" => Some(Self::U32),
            "int32_t" => Some(Self::I32),
            "uint64_t" => Some(Self::U64),
            _ => None,
        }
    }

    /// Size and alignment, which are equal for these types on every target Syntropy builds.
    pub(super) fn size(self) -> usize {
        match self {
            Self::U32 | Self::I32 => 4,
            Self::U64 => 8,
        }
    }

    fn rust(self) -> &'static str {
        match self {
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64 => "u64",
        }
    }
}

/// `#define NAME value` with `NAME` starting `SYN_` and a value the generator can read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Define {
    pub(super) docs: Vec<String>,
    pub(super) name: String,
    /// The value as a Rust expression.
    pub(super) expr: String,
    pub(super) value: u64,
}

/// One struct member, with the offset C gives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Field {
    pub(super) docs: Vec<String>,
    pub(super) name: String,
    pub(super) ty: CType,
    pub(super) offset: usize,
}

/// One struct, laid out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CStruct {
    pub(super) docs: Vec<String>,
    /// The C tag, `syn_step_desc`.
    pub(super) c_name: String,
    pub(super) fields: Vec<Field>,
    pub(super) size: usize,
    pub(super) align: usize,
}

impl CStruct {
    /// The Rust name: the tag without `syn_`, in upper camel case.
    pub(super) fn rust_name(&self) -> String {
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
pub(super) struct MirrorLayout {
    /// The C tag.
    pub(super) tag: &'static str,
    pub(super) size: usize,
    pub(super) align: usize,
    /// Each member's name and offset, in declaration order.
    pub(super) fields: &'static [(&'static str, usize)],
}

/// What the header declares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Header {
    pub(super) defines: Vec<Define>,
    pub(super) structs: Vec<CStruct>,
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

/// Parses the header and lays its structs out: each member at the next multiple of its
/// alignment (raised by `SYN_ALIGNAS`), the struct's alignment the largest of its members',
/// its size rounded up to that alignment.
pub(super) fn parse(text: &str) -> Result<Header, String> {
    let mut defines = Vec::new();
    let mut structs = Vec::new();
    let mut docs: Vec<String> = Vec::new();
    let mut open: Option<CStruct> = None;
    let mut in_block_comment = false;

    for (ix, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let at = |msg: &str| format!("{HEADER_PATH}:{}: {msg}: `{line}`", ix + 1);
        if in_block_comment {
            in_block_comment = !line.contains("*/");
            continue;
        }
        if let Some(doc) = line.strip_prefix("///") {
            docs.push(doc.strip_prefix(' ').unwrap_or(doc).to_owned());
            continue;
        }
        if line.starts_with("/*") {
            in_block_comment = !line.contains("*/");
            continue;
        }
        if line.is_empty() {
            continue;
        }
        if let Some(def) = line.strip_prefix("#define ") {
            let (name, value) = def.split_once(' ').unwrap_or((def, ""));
            if name.starts_with("SYN_") && !name.contains('(') {
                let (expr, value) =
                    parse_value(value.trim()).ok_or_else(|| at("unreadable value"))?;
                defines.push(Define {
                    docs: std::mem::take(&mut docs),
                    name: name.to_owned(),
                    expr,
                    value,
                });
            }
            continue;
        }
        if line.starts_with('#') {
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
        return Err(format!("{HEADER_PATH}: unterminated struct"));
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

/// The Rust mirror of `header`, formatted as rustfmt leaves it.
pub(super) fn generate(header: &Header) -> String {
    let mut out = String::new();
    out.push_str(
        "// SPDX-License-Identifier: MIT OR Apache-2.0\n\
         // @generated from `include/syntropy/step_desc.h` by the `syntropy-kernels` tests. Do not\n\
         // edit: change the header and run `SYNTROPY_BLESS_GENERATED=1 cargo test -p\n\
         // syntropy-kernels step::`.\n\
         \n\
         //! The `#[repr(C)]` mirrors of `include/syntropy/step_desc.h` (RFC-0040 §1, §3.5), with\n\
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
        out.push_str("#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]\n");
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
    out.push_str(
        "\n/// Each mirror's C tag, size, alignment and member offsets as the Rust compiler lays\n\
         /// them out, for the tests that compare them with the header and with a C compiler.\n\
         #[cfg(test)]\n\
         pub(super) const LAYOUTS: &[super::cgen::MirrorLayout] = &[\n",
    );
    for s in &header.structs {
        let name = s.rust_name();
        let _ = writeln!(out, "    super::cgen::MirrorLayout {{");
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
    let mut out = String::from(
        "#include <stddef.h>\n#include \"syntropy/step_desc.h\"\n#if defined(__cplusplus)\n\
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
