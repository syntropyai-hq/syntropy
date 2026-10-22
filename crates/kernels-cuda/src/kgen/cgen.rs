// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Rust mirrors of `csrc/params.h`, the one source of truth for the CUDA kernels'
//! parameter blocks (RFC-0040 §1, "kernel argument structs have one source of truth").
//!
//! The generator reads a small subset of C, lays its structs out by the C rules, and writes
//! `src/params.rs`: one `#[repr(C)]` struct per C struct, compile-time checks that Rust lays
//! each out as the header does, and a table of the Rust layouts that the tests compare with
//! the generator's and, where one is installed, with a C compiler's. It needs no C or CUDA
//! compiler.
//!
//! The subset: `///` doc lines, `/* … */` comments on lines of their own, `#ifndef`/`#define`/
//! `#endif` include guards, `#include`, `#define SYN_NAME <decimal>` constants, and
//! `struct syn_name {` blocks with one `uint32_t`, `int32_t`, `uint64_t` or `float` member per
//! line, closed by `};`. Anything else fails, so a member the generator cannot lay out never
//! passes silently. The step descriptor header of `syntropy-kernels` has its own generator of
//! the same kind; the two can merge once both have landed.

use std::fmt::Write as _;

/// The header, relative to the crate root.
pub const HEADER_PATH: &str = "csrc/params.h";
/// The generated mirror, relative to the crate root.
pub const MIRROR_PATH: &str = "src/params.rs";

/// A member type the generator knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CType {
    /// `uint32_t`.
    U32,
    /// `int32_t`.
    I32,
    /// `uint64_t`.
    U64,
    /// `float`, IEEE-754 binary32.
    F32,
}

impl CType {
    fn parse(word: &str) -> Option<Self> {
        match word {
            "uint32_t" => Some(Self::U32),
            "int32_t" => Some(Self::I32),
            "uint64_t" => Some(Self::U64),
            "float" => Some(Self::F32),
            _ => None,
        }
    }

    /// Size, equal to the alignment for these types on every target Syntropy builds.
    #[must_use]
    pub fn size(self) -> usize {
        match self {
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 => 8,
        }
    }

    fn rust(self) -> &'static str {
        match self {
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64 => "u64",
            Self::F32 => "f32",
        }
    }
}

/// `#define SYN_NAME <decimal>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Define {
    /// Its doc lines.
    pub docs: Vec<String>,
    /// The name, with its `SYN_` prefix.
    pub name: String,
    /// The value.
    pub value: u64,
}

/// One struct member, at the offset C gives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    /// Its doc lines.
    pub docs: Vec<String>,
    /// The member name.
    pub name: String,
    /// Its type.
    pub ty: CType,
    /// Its offset.
    pub offset: usize,
}

/// One struct, laid out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CStruct {
    /// Its doc lines.
    pub docs: Vec<String>,
    /// The C tag, `syn_gemv_params`.
    pub c_name: String,
    /// The members, in order.
    pub fields: Vec<Field>,
    /// `sizeof`.
    pub size: usize,
    /// `_Alignof`.
    pub align: usize,
}

impl CStruct {
    /// The Rust name: the tag without `syn_`, in upper camel case.
    #[must_use]
    pub fn rust_name(&self) -> String {
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
pub struct MirrorLayout {
    /// The C tag.
    pub tag: &'static str,
    /// `size_of`.
    pub size: usize,
    /// `align_of`.
    pub align: usize,
    /// Each member's name and offset, in declaration order.
    pub fields: &'static [(&'static str, usize)],
}

/// What the header declares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    /// The `SYN_` constants.
    pub defines: Vec<Define>,
    /// The structs.
    pub structs: Vec<CStruct>,
}

/// Parses the header and lays its structs out: each member at the next multiple of its
/// alignment, the struct's alignment the largest of its members', its size rounded up to it.
///
/// # Errors
///
/// A message naming the line, for anything outside the subset the module docs describe.
pub fn parse(text: &str) -> Result<Header, String> {
    let mut defines = Vec::new();
    let mut structs = Vec::new();
    let mut docs: Vec<String> = Vec::new();
    let mut open: Option<CStruct> = None;
    let mut in_comment = false;
    let mut depth = 0u32;
    for (ix, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let at = |msg: &str| format!("{HEADER_PATH}:{}: {msg}: `{line}`", ix + 1);
        if in_comment {
            if let Some(end) = line.find("*/") {
                if !line[end + 2..].trim().is_empty() {
                    return Err(at("code after a block comment"));
                }
                in_comment = false;
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("/*") {
            match rest.find("*/") {
                None => in_comment = true,
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
        if line.is_empty() || line.starts_with("//") {
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
                "ifndef" => depth += 1,
                "endif" => {
                    depth = depth
                        .checked_sub(1)
                        .ok_or_else(|| at("unbalanced #endif"))?;
                }
                "include" => {}
                "define" => {
                    let def = directive["define".len()..].trim_start();
                    let (name, value) = def.split_once(' ').unwrap_or((def, ""));
                    if value.trim().is_empty() {
                        // An include guard.
                    } else if name.starts_with("SYN_") {
                        if depth > 1 {
                            return Err(at("a constant under a condition"));
                        }
                        let value = value
                            .trim()
                            .trim_end_matches(['u', 'U'])
                            .parse()
                            .map_err(|_| at("a constant that is not a decimal integer"))?;
                        defines.push(Define {
                            docs: std::mem::take(&mut docs),
                            name: name.to_owned(),
                            value,
                        });
                    } else {
                        return Err(at("a macro with a value that is not a SYN_ constant"));
                    }
                }
                _ => return Err(at("a preprocessor directive the generator does not read")),
            }
            continue;
        }
        if let Some(s) = open.as_mut() {
            if line == "};" {
                let mut s = open.take().expect("open struct");
                if s.fields.is_empty() {
                    return Err(at("an empty struct"));
                }
                s.size = s.size.next_multiple_of(s.align);
                structs.push(s);
                continue;
            }
            let decl = line
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
            let offset = s.size.next_multiple_of(ty.size());
            s.align = s.align.max(ty.size());
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
            if !tag.starts_with("syn_") {
                return Err(at("a struct tag without the syn_ prefix"));
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
        return Err(format!("{HEADER_PATH}: unterminated struct"));
    }
    if in_comment {
        return Err(format!("{HEADER_PATH}: unterminated block comment"));
    }
    if depth != 0 {
        return Err(format!("{HEADER_PATH}: unbalanced #ifndef"));
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
#[must_use]
pub fn generate(header: &Header) -> String {
    let mut out = String::new();
    out.push_str(
        "// SPDX-License-Identifier: MIT OR Apache-2.0\n\
         // @generated from `csrc/params.h` by `cargo xtask kgen`. Do not edit: change the header\n\
         // and run `cargo xtask kgen`.\n\
         \n\
         //! The `#[repr(C)]` mirrors of `csrc/params.h`, the CUDA kernels' parameter blocks\n\
         //! (RFC-0040 §1, §7.1), with compile-time checks of the layout the generator computed\n\
         //! from the header.\n",
    );
    for d in &header.defines {
        out.push('\n');
        push_docs(&mut out, "", &d.docs);
        let name = d.name.strip_prefix("SYN_").unwrap_or(&d.name);
        let _ = writeln!(out, "pub const {name}: u32 = {};", d.value);
    }
    for s in &header.structs {
        let name = s.rust_name();
        out.push('\n');
        push_docs(&mut out, "", &s.docs);
        out.push_str("#[repr(C)]\n#[derive(Clone, Copy, Debug, Default, PartialEq)]\n");
        let _ = writeln!(out, "pub struct {name} {{");
        for f in &s.fields {
            push_docs(&mut out, "    ", &f.docs);
            let _ = writeln!(out, "    pub {}: {},", f.name, f.ty.rust());
        }
        out.push_str("}\n");
    }
    out.push_str(
        "\n// The layout the header declares, as C lays it out. Generated text, so rustfmt leaves it.\n#[rustfmt::skip]\nconst _: () = {\n",
    );
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
         #[rustfmt::skip]\n\
         pub const LAYOUTS: &[crate::kgen::cgen::MirrorLayout] = &[\n",
    );
    for s in &header.structs {
        let name = s.rust_name();
        out.push_str("    crate::kgen::cgen::MirrorLayout {\n");
        let _ = writeln!(out, "        tag: \"{}\",", s.c_name);
        let _ = writeln!(out, "        size: size_of::<{name}>(),");
        let _ = writeln!(out, "        align: align_of::<{name}>(),");
        out.push_str("        fields: &[\n");
        for f in &s.fields {
            let _ = writeln!(
                out,
                "            (\"{0}\", core::mem::offset_of!({name}, {0})),",
                f.name
            );
        }
        out.push_str("        ],\n    },\n");
    }
    out.push_str("];\n");
    out
}

/// A C translation unit that includes the header and asserts, with `_Static_assert`, the
/// layout in `layouts` (the Rust compiler's): what a C compiler checks when one is present.
#[must_use]
pub fn c_layout_check(layouts: &[MirrorLayout]) -> String {
    let mut out = String::from(
        "#include <stddef.h>\n#include \"params.h\"\n#if defined(__cplusplus)\n\
         #define SYN_ASSERT static_assert\n#define SYN_ALIGNOF alignof\n#else\n\
         #define SYN_ASSERT _Static_assert\n#define SYN_ALIGNOF _Alignof\n#endif\n",
    );
    for layout in layouts {
        let MirrorLayout {
            tag,
            size,
            align,
            fields,
        } = layout;
        let _ = writeln!(
            out,
            "SYN_ASSERT(sizeof(struct {tag}) == {size}, \"{tag} size\");"
        );
        let _ = writeln!(
            out,
            "SYN_ASSERT(SYN_ALIGNOF(struct {tag}) == {align}, \"{tag} align\");"
        );
        for (field, offset) in *fields {
            let _ = writeln!(
                out,
                "SYN_ASSERT(offsetof(struct {tag}, {field}) == {offset}, \"{tag}.{field}\");"
            );
        }
    }
    out
}
