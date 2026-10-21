// SPDX-License-Identifier: MIT OR Apache-2.0

//! `#[derive(ConfigSection)]`, the schema derive of RFC-0054 §10.3.
//!
//! Use it through `syntropy-config`, which re-exports it and documents the attributes. The
//! generated code names `::syntropy_config`, so a crate that derives a section depends on
//! `syntropy-config` directly.
//!
//! What fails to compile here (RFC-0054 §2.3 steps 9–10, §10.3; ADR-0500 decision 2):
//! - a section or key without a doc comment, a section without an owner or scope, a key
//!   without a default or reload class;
//! - a cluster-scope key without `introduced_at`;
//! - `authority = Member` on a key that RFC-0054 §2.3 step 10 reserves for keyholders.
//!
//! Two sections with one prefix fail the build where `syntropy_config::Registry::new` runs in
//! a `static` or `const`, since only a registry sees every section.

#![forbid(unsafe_code)]

use proc_macro::TokenStream;
use proc_macro2::{Literal, Span, TokenStream as TokenStream2, TokenTree};
use quote::{ToTokens, quote, quote_spanned};
use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{
    Attribute, Data, DeriveInput, Error, Expr, ExprLit, Fields, Ident, Lit, LitBool, LitFloat,
    LitInt, LitStr, Meta, Token, braced, bracketed, parenthesized, parse_macro_input, token,
};

/// Declares a configuration section. See `syntropy_config`'s crate documentation for the
/// attributes.
#[proc_macro_derive(ConfigSection, attributes(config))]
pub fn derive_config_section(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(&input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// Collects every error, so one build reports all of them.
#[derive(Default)]
struct Errors(Option<Error>);

impl Errors {
    fn push(&mut self, error: Error) {
        match &mut self.0 {
            Some(first) => first.combine(error),
            None => self.0 = Some(error),
        }
    }

    fn at(&mut self, span: Span, message: impl std::fmt::Display) {
        self.push(Error::new(span, message));
    }

    fn finish(self) -> syn::Result<()> {
        self.0.map_or(Ok(()), Err)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    Node,
    Cluster,
    Instance,
    Request,
}

impl Scope {
    fn parse(ident: &Ident) -> syn::Result<Self> {
        match ident.to_string().as_str() {
            "Node" => Ok(Scope::Node),
            "Cluster" => Ok(Scope::Cluster),
            "Instance" => Ok(Scope::Instance),
            "Request" => Ok(Scope::Request),
            _ => Err(Error::new(
                ident.span(),
                "expected a scope: `Node`, `Cluster`, `Instance` or `Request` (RFC-0054 §2.1)",
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Scope::Node => "node",
            Scope::Cluster => "cluster",
            Scope::Instance => "instance",
            Scope::Request => "request",
        }
    }

    fn tokens(self) -> TokenStream2 {
        match self {
            Scope::Node => quote!(::syntropy_config::Scope::Node),
            Scope::Cluster => quote!(::syntropy_config::Scope::Cluster),
            Scope::Instance => quote!(::syntropy_config::Scope::Instance),
            Scope::Request => quote!(::syntropy_config::Scope::Request),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Authority {
    Member,
    Keyholder,
}

/// One of `Name = Ident` choices, parsed into its tokens.
fn reload_tokens(ident: &Ident) -> syn::Result<TokenStream2> {
    match ident.to_string().as_str() {
        "Hot" => Ok(quote!(::syntropy_config::Reload::Hot)),
        "Instance" => Ok(quote!(::syntropy_config::Reload::Instance)),
        "Node" => Ok(quote!(::syntropy_config::Reload::Node)),
        _ => Err(Error::new(
            ident.span(),
            "expected a reload class: `Hot`, `Instance` or `Node` (RFC-0054 §9.1)",
        )),
    }
}

fn authority(ident: &Ident) -> syn::Result<Authority> {
    match ident.to_string().as_str() {
        "Member" => Ok(Authority::Member),
        "Keyholder" => Ok(Authority::Keyholder),
        _ => Err(Error::new(
            ident.span(),
            "expected an authority: `Member` or `Keyholder` (RFC-0054 §2.3 step 10)",
        )),
    }
}

/// A value written in an attribute: the shape of a TOML value.
enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Table(Vec<(String, Value)>),
}

impl Parse for Value {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(LitStr) {
            Ok(Value::Str(input.parse::<LitStr>()?.value()))
        } else if lookahead.peek(LitBool) {
            Ok(Value::Bool(input.parse::<LitBool>()?.value))
        } else if lookahead.peek(LitInt) || lookahead.peek(LitFloat) {
            number(input, false)
        } else if lookahead.peek(Token![-]) {
            input.parse::<Token![-]>()?;
            number(input, true)
        } else if lookahead.peek(token::Bracket) {
            let content;
            bracketed!(content in input);
            let items = content.parse_terminated(Value::parse, Token![,])?;
            Ok(Value::Array(items.into_iter().collect()))
        } else if lookahead.peek(token::Brace) {
            let content;
            braced!(content in input);
            let entries = content.parse_terminated(entry, Token![,])?;
            Ok(Value::Table(entries.into_iter().collect()))
        } else {
            Err(lookahead.error())
        }
    }
}

fn number(input: ParseStream, negative: bool) -> syn::Result<Value> {
    let sign = if negative { "-" } else { "" };
    if input.peek(LitInt) {
        let lit: LitInt = input.parse()?;
        let value: i64 = format!("{sign}{}", lit.base10_digits())
            .parse()
            .map_err(|_| Error::new(lit.span(), "the integer does not fit in 64 bits"))?;
        Ok(Value::Int(value))
    } else {
        let lit: LitFloat = input.parse()?;
        let value: f64 = format!("{sign}{}", lit.base10_digits())
            .parse()
            .map_err(|_| Error::new(lit.span(), "malformed float"))?;
        if !value.is_finite() {
            return Err(Error::new(lit.span(), "the float is not finite"));
        }
        Ok(Value::Float(value))
    }
}

/// `key = value` in a table; the key is an identifier or a string.
fn entry(input: ParseStream) -> syn::Result<(String, Value)> {
    let key = if input.peek(LitStr) {
        input.parse::<LitStr>()?.value()
    } else {
        input.call(Ident::parse_any)?.unraw().to_string()
    };
    input.parse::<Token![=]>()?;
    Ok((key, input.parse()?))
}

impl Value {
    fn tokens(&self) -> TokenStream2 {
        match self {
            Value::Bool(b) => quote!(::syntropy_config::DefaultValue::Bool(#b)),
            Value::Int(i) => {
                let lit = Literal::i64_suffixed(*i);
                quote!(::syntropy_config::DefaultValue::Int(#lit))
            }
            Value::Float(x) => {
                let lit = Literal::f64_suffixed(*x);
                quote!(::syntropy_config::DefaultValue::Float(#lit))
            }
            Value::Str(s) => quote!(::syntropy_config::DefaultValue::Str(#s)),
            Value::Array(items) => {
                let items = items.iter().map(Value::tokens);
                quote!(::syntropy_config::DefaultValue::Array(&[#(#items),*]))
            }
            Value::Table(entries) => {
                let entries = entries.iter().map(|(key, value)| {
                    let value = value.tokens();
                    quote!((#key, #value))
                });
                quote!(::syntropy_config::DefaultValue::Table(&[#(#entries),*]))
            }
        }
    }
}

/// `default = …`: a value, `unset` or `computed("rule")`.
enum Default {
    Value(Value),
    Unset,
    Computed(String),
}

impl Parse for Default {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(Ident) && !input.peek(LitBool) {
            let ident: Ident = input.parse()?;
            return match ident.to_string().as_str() {
                "unset" => Ok(Default::Unset),
                "computed" => {
                    let content;
                    parenthesized!(content in input);
                    let rule: LitStr = content.parse()?;
                    if rule.value().trim().is_empty() {
                        return Err(Error::new(rule.span(), "a computed default needs its rule"));
                    }
                    Ok(Default::Computed(rule.value()))
                }
                _ => Err(Error::new(
                    ident.span(),
                    "expected a default: a value, `unset` or `computed(\"rule\")`",
                )),
            };
        }
        input.parse().map(Default::Value)
    }
}

impl Default {
    fn tokens(&self) -> TokenStream2 {
        match self {
            Default::Value(value) => {
                let value = value.tokens();
                quote!(::syntropy_config::DefaultSpec::Value(#value))
            }
            Default::Unset => quote!(::syntropy_config::DefaultSpec::Unset),
            Default::Computed(rule) => quote!(::syntropy_config::DefaultSpec::Computed(#rule)),
        }
    }
}

/// The doc comment, with each line's leading space removed and the whole trimmed.
fn doc_of(attrs: &[Attribute]) -> Option<String> {
    let lines: Vec<String> = attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .filter_map(|a| match &a.meta {
            Meta::NameValue(nv) => match &nv.value {
                Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                }) => Some(s.value()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let text = lines
        .iter()
        .map(|line| line.strip_prefix(' ').unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n");
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

/// Rejects serde attributes that would make the key's name differ from the field's.
fn check_serde_names(attrs: &[Attribute], errors: &mut Errors) {
    for attr in attrs.iter().filter(|a| a.path().is_ident("serde")) {
        let Meta::List(list) = &attr.meta else {
            continue;
        };
        for tree in list.tokens.clone() {
            if let TokenTree::Ident(ident) = tree
                && matches!(
                    ident.to_string().as_str(),
                    "rename" | "rename_all" | "alias" | "flatten"
                )
            {
                errors.at(
                    ident.span(),
                    format!(
                        "`serde({ident})` is not supported on a config section: a key's name is \
                         its field's name, so the schema, the docs and the environment variable \
                         names match the file"
                    ),
                );
            }
        }
    }
}

/// `a.b.c` with each segment `[a-z][a-z0-9_]*`; empty for the top level.
fn valid_path(path: &str) -> bool {
    path.is_empty() || path.split('.').all(valid_segment)
}

fn valid_segment(segment: &str) -> bool {
    let mut bytes = segment.bytes();
    bytes.next().is_some_and(|b| b.is_ascii_lowercase())
        && bytes.all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// `RFC-NNNN`.
fn valid_owner(owner: &str) -> bool {
    owner
        .strip_prefix("RFC-")
        .is_some_and(|n| n.len() == 4 && n.bytes().all(|b| b.is_ascii_digit()))
}

/// The keyholder rule of RFC-0054 §2.3 step 10 that covers `path`, if any.
fn keyholder_rule(path: &str, scope: Scope) -> Option<String> {
    let under = |prefix: &str| {
        path == prefix
            || path
                .strip_prefix(prefix)
                .is_some_and(|rest| rest.starts_with('.'))
    };
    match scope {
        Scope::Cluster => {
            for prefix in ["security", "invite", "coordinator", "swim", "update"] {
                if under(prefix) {
                    return Some(format!("cluster keys under `{prefix}.*`"));
                }
            }
            for key in [
                "cluster.lan_admission",
                "cluster_aliases",
                "network_scope",
                "allow_approximate",
            ] {
                if path == key {
                    return Some(format!("the cluster key `{key}`"));
                }
            }
            None
        }
        Scope::Instance => {
            if under("approximate") {
                return Some("the approximate knobs `approximate.*`".to_owned());
            }
            for key in ["fidelity", "default_fidelity"] {
                if path == key {
                    return Some(format!("the exactness field `{key}`"));
                }
            }
            for key in ["pin", "tensor_override"] {
                if path == key {
                    return Some(format!("the pin field `{key}`"));
                }
            }
            None
        }
        Scope::Node | Scope::Request => None,
    }
}

#[derive(Default)]
struct SectionAttrs {
    section: Option<LitStr>,
    scope: Option<Scope>,
    owner: Option<LitStr>,
    shared: bool,
    /// An attribute failed to parse, so missing attributes are not reported on top.
    failed: bool,
}

fn section_attrs(attrs: &[Attribute], errors: &mut Errors) -> SectionAttrs {
    let mut out = SectionAttrs::default();
    for attr in attrs.iter().filter(|a| a.path().is_ident("config")) {
        let result = attr.parse_nested_meta(|meta| {
            let duplicate = || meta.error("this attribute is set twice");
            if meta.path.is_ident("section") {
                if out.section.is_some() {
                    return Err(duplicate());
                }
                out.section = Some(meta.value()?.parse()?);
            } else if meta.path.is_ident("scope") {
                if out.scope.is_some() {
                    return Err(duplicate());
                }
                out.scope = Some(Scope::parse(&meta.value()?.parse()?)?);
            } else if meta.path.is_ident("owner") {
                if out.owner.is_some() {
                    return Err(duplicate());
                }
                out.owner = Some(meta.value()?.parse()?);
            } else if meta.path.is_ident("shared") {
                out.shared = true;
            } else {
                return Err(meta.error(
                    "unknown section attribute; expected `section`, `scope`, `owner` or `shared`",
                ));
            }
            Ok(())
        });
        if let Err(e) = result {
            out.failed = true;
            errors.push(e);
        }
    }
    out
}

#[derive(Default)]
struct KeyAttrs {
    reload: Option<TokenStream2>,
    default: Option<Default>,
    min: Option<Value>,
    max: Option<Value>,
    scope: Option<Scope>,
    owner: Option<LitStr>,
    introduced_at: Option<Expr>,
    authority: Option<(Authority, Span)>,
    /// An attribute failed to parse, so missing attributes are not reported on top.
    failed: bool,
}

fn key_attrs(attrs: &[Attribute], errors: &mut Errors) -> KeyAttrs {
    let mut out = KeyAttrs::default();
    for attr in attrs.iter().filter(|a| a.path().is_ident("config")) {
        let result = attr.parse_nested_meta(|meta| {
            let duplicate = || meta.error("this attribute is set twice");
            macro_rules! once {
                ($slot:expr, $value:expr) => {{
                    if $slot.is_some() {
                        return Err(duplicate());
                    }
                    $slot = Some($value);
                }};
            }
            if meta.path.is_ident("reload") {
                once!(out.reload, reload_tokens(&meta.value()?.parse()?)?);
            } else if meta.path.is_ident("default") {
                once!(out.default, meta.value()?.parse()?);
            } else if meta.path.is_ident("min") {
                once!(out.min, meta.value()?.parse()?);
            } else if meta.path.is_ident("max") {
                once!(out.max, meta.value()?.parse()?);
            } else if meta.path.is_ident("scope") {
                once!(out.scope, Scope::parse(&meta.value()?.parse()?)?);
            } else if meta.path.is_ident("owner") {
                once!(out.owner, meta.value()?.parse()?);
            } else if meta.path.is_ident("introduced_at") {
                once!(out.introduced_at, meta.value()?.parse()?);
            } else if meta.path.is_ident("authority") {
                let ident: Ident = meta.value()?.parse()?;
                once!(out.authority, (authority(&ident)?, ident.span()));
            } else {
                return Err(meta.error(
                    "unknown key attribute; expected `reload`, `default`, `min`, `max`, `scope`, \
                     `owner`, `introduced_at` or `authority`",
                ));
            }
            Ok(())
        });
        if let Err(e) = result {
            out.failed = true;
            errors.push(e);
        }
    }
    out
}

/// The type as written, with token spacing tidied: `Option<Duration>`.
fn type_name(ty: &syn::Type) -> String {
    ty.to_token_stream()
        .to_string()
        .replace(' ', "")
        .replace(',', ", ")
}

fn expand(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let mut errors = Errors::default();

    if !input.generics.params.is_empty() {
        errors.at(
            input.generics.span(),
            "a config section cannot be generic: its schema is one constant",
        );
    }
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => named.named.iter().collect::<Vec<_>>(),
            _ => {
                return Err(Error::new(
                    name.span(),
                    "a config section is a struct with named fields, one per key",
                ));
            }
        },
        _ => {
            return Err(Error::new(
                name.span(),
                "a config section is a struct with named fields, one per key",
            ));
        }
    };

    let attrs = section_attrs(&input.attrs, &mut errors);
    check_serde_names(&input.attrs, &mut errors);
    let section_doc = doc_of(&input.attrs).unwrap_or_else(|| {
        errors.at(
            name.span(),
            format!("config section `{name}` needs a doc comment (RFC-0054 §10.3)"),
        );
        String::new()
    });
    let prefix = match &attrs.section {
        Some(lit) => {
            let prefix = lit.value();
            if !valid_path(&prefix) {
                errors.at(
                    lit.span(),
                    "a section is `\"\"` or dotted lower-case segments, such as \"api.auth\"",
                );
            }
            prefix
        }
        None => {
            if !attrs.failed {
                errors.at(
                    name.span(),
                    format!(
                        "config section `{name}` needs `#[config(section = \"<prefix>\")]`; use \
                         \"\" for top-level keys"
                    ),
                );
            }
            String::new()
        }
    };
    let section_scope = attrs.scope.unwrap_or_else(|| {
        if attrs.failed {
            return Scope::Node;
        }
        errors.at(
            name.span(),
            format!(
                "config section `{name}` needs `scope = Node | Cluster | Instance | Request` \
                 (RFC-0054 §2.1)"
            ),
        );
        Scope::Node
    });
    let check_owner = |lit: &LitStr, errors: &mut Errors| {
        if !valid_owner(&lit.value()) {
            errors.at(lit.span(), "an owner is an RFC, written `\"RFC-NNNN\"`");
        }
        lit.value()
    };
    let section_owner = match &attrs.owner {
        Some(lit) => check_owner(lit, &mut errors),
        None => {
            if !attrs.failed {
                errors.at(
                    name.span(),
                    format!(
                        "config section `{name}` needs `owner = \"RFC-NNNN\"`: the RFC that owns \
                         its keys (RFC-0054 §5)"
                    ),
                );
            }
            String::new()
        }
    };
    let shared = attrs.shared || prefix.is_empty();

    let mut key_tokens = Vec::new();
    let mut default_checks = Vec::new();
    let mut range_checks = Vec::new();
    for (index, field) in fields.iter().enumerate() {
        let Some(ident) = &field.ident else { continue };
        let span = ident.span();
        let key_name = ident.unraw().to_string();
        let path = if prefix.is_empty() {
            key_name.clone()
        } else {
            format!("{prefix}.{key_name}")
        };
        if !valid_segment(&key_name) {
            errors.at(
                span,
                "a config key's name is lower snake case, such as `ram_headroom`",
            );
        }
        check_serde_names(&field.attrs, &mut errors);
        let key = key_attrs(&field.attrs, &mut errors);

        let doc = doc_of(&field.attrs).unwrap_or_else(|| {
            errors.at(
                span,
                format!("config key `{path}` needs a doc comment (RFC-0054 §10.3)"),
            );
            String::new()
        });
        let reload = key.reload.clone().unwrap_or_else(|| {
            if key.failed {
                return quote!(::syntropy_config::Reload::Node);
            }
            errors.at(
                span,
                format!(
                    "config key `{path}` needs `reload = Hot | Instance | Node` (RFC-0054 §9.1)"
                ),
            );
            quote!(::syntropy_config::Reload::Node)
        });
        let default = match &key.default {
            Some(default) => default.tokens(),
            None => {
                if !key.failed {
                    errors.at(
                        span,
                        format!(
                            "config key `{path}` needs a `default`: a value, `unset` or \
                             `computed(\"rule\")` (RFC-0054 §10.3)"
                        ),
                    );
                }
                quote!(::syntropy_config::DefaultSpec::Unset)
            }
        };
        let scope = key.scope.unwrap_or(section_scope);
        let owner = key.owner.as_ref().map_or_else(
            || section_owner.clone(),
            |lit| check_owner(lit, &mut errors),
        );

        let log_scoped = matches!(scope, Scope::Cluster | Scope::Instance);
        let introduced_at = match (&key.introduced_at, scope) {
            (Some(expr), Scope::Cluster | Scope::Instance) => {
                quote_spanned!(expr.span()=> ::core::option::Option::Some(#expr))
            }
            (Some(expr), _) => {
                errors.at(
                    expr.span(),
                    format!(
                        "`introduced_at` applies to cluster and instance keys; `{path}` is a {} \
                         key",
                        scope.name()
                    ),
                );
                quote!(::core::option::Option::None)
            }
            (None, Scope::Cluster) if !key.failed => {
                errors.at(
                    span,
                    format!(
                        "cluster key `{path}` needs `introduced_at = <feature level>`, the \
                         cluster feature level that introduced it (RFC-0054 §2.3 step 9)"
                    ),
                );
                quote!(::core::option::Option::None)
            }
            (None, _) => quote!(::core::option::Option::None),
        };
        let rule = keyholder_rule(&path, scope);
        let authority = match (key.authority, log_scoped) {
            (Some((_, at)), false) => {
                errors.at(
                    at,
                    format!(
                        "`authority` applies to cluster and instance keys; `{path}` is a {} key",
                        scope.name()
                    ),
                );
                None
            }
            (Some((Authority::Member, at)), true) if rule.is_some() => {
                let rule = rule.as_deref().unwrap_or_default();
                errors.at(
                    at,
                    format!(
                        "`{path}` must have `authority = Keyholder`: RFC-0054 §2.3 step 10 \
                         reserves {rule} for keyholders"
                    ),
                );
                Some(Authority::Keyholder)
            }
            (Some((authority, _)), true) => Some(authority),
            (None, true) if rule.is_some() => Some(Authority::Keyholder),
            (None, true) => Some(Authority::Member),
            (None, false) => None,
        };
        let authority = match authority {
            Some(Authority::Member) => {
                quote!(::core::option::Option::Some(
                    ::syntropy_config::Authority::Member
                ))
            }
            Some(Authority::Keyholder) => {
                quote!(::core::option::Option::Some(
                    ::syntropy_config::Authority::Keyholder
                ))
            }
            None => quote!(::core::option::Option::None),
        };
        let env = if scope == Scope::Node {
            let var = format!("SYNTROPY_{}", path.to_ascii_uppercase().replace('.', "__"));
            quote!(::core::option::Option::Some(#var))
        } else {
            quote!(::core::option::Option::None)
        };
        let bound = |value: &Option<Value>| match value {
            Some(value) => {
                let value = value.tokens();
                quote!(::core::option::Option::Some(#value))
            }
            None => quote!(::core::option::Option::None),
        };
        let (min, max) = (bound(&key.min), bound(&key.max));
        let ty = &field.ty;
        let ty_name = type_name(ty);
        let scope_tokens = scope.tokens();

        key_tokens.push(quote! {
            ::syntropy_config::KeySchema {
                name: #key_name,
                path: #path,
                doc: #doc,
                ty: #ty_name,
                scope: #scope_tokens,
                reload: #reload,
                owner: #owner,
                default: #default,
                min: #min,
                max: #max,
                introduced_at: #introduced_at,
                authority: #authority,
                env: #env,
            }
        });

        let ranged = key.min.is_some() || key.max.is_some();
        if ranged {
            default_checks.push(quote_spanned! {ty.span()=>
                ::syntropy_config::__private::check_default_in_range::<#ty>(&keys[#index], &mut errors);
            });
            range_checks.push(quote_spanned! {ty.span()=>
                ::syntropy_config::__private::check_range(&self.#ident, &keys[#index], &mut errors);
            });
        } else {
            default_checks.push(quote_spanned! {ty.span()=>
                ::syntropy_config::__private::check_default::<#ty>(&keys[#index], &mut errors);
            });
        }
    }
    errors.finish()?;

    let type_name = name.to_string();
    let scope = section_scope.tokens();
    Ok(quote! {
        impl ::syntropy_config::ConfigSection for #name {
            const SCHEMA: ::syntropy_config::SectionSchema = ::syntropy_config::SectionSchema {
                type_name: #type_name,
                prefix: #prefix,
                shared: #shared,
                owner: #section_owner,
                scope: #scope,
                doc: #section_doc,
                keys: &[#(#key_tokens),*],
                check_defaults: {
                    fn check_defaults() -> ::std::vec::Vec<::syntropy_config::KeyError> {
                        let keys = <#name as ::syntropy_config::ConfigSection>::SCHEMA.keys;
                        let mut errors = ::std::vec::Vec::new();
                        #(#default_checks)*
                        errors
                    }
                    check_defaults
                },
            };

            fn check_ranges(&self) -> ::std::vec::Vec<::syntropy_config::KeyError> {
                let keys = <Self as ::syntropy_config::ConfigSection>::SCHEMA.keys;
                let mut errors = ::std::vec::Vec::new();
                #(#range_checks)*
                let _ = keys;
                errors
            }
        }
    })
}
