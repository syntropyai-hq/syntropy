// SPDX-License-Identifier: MIT OR Apache-2.0

//! SPDX licence expressions, as Cargo's `license` field and `third_party/licences.toml` write
//! them: licence identifiers joined by `AND` and `OR`, an optional `WITH` exception, and
//! parentheses. The `/` that older crates use for `OR` is read as `OR`, as Cargo reads it.

use std::fmt;

/// A parsed expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    /// One licence, such as `MIT`, or `Apache-2.0 WITH LLVM-exception`.
    Licence {
        id: String,
        exception: Option<String>,
    },
    /// Every part must be met.
    And(Vec<Expr>),
    /// Any one part is enough.
    Or(Vec<Expr>),
}

impl Expr {
    /// Parses `text`. An error says what was wrong and where.
    pub fn parse(text: &str) -> Result<Expr, String> {
        let tokens = tokenize(text)?;
        if tokens.is_empty() {
            return Err("the expression is empty".to_owned());
        }
        let mut parser = Parser { tokens, at: 0 };
        let expr = parser.or()?;
        match parser.tokens.get(parser.at) {
            None => Ok(expr),
            Some(token) => Err(format!("unexpected `{token}`")),
        }
    }

    /// Whether the licences in `allow` meet the expression: an `OR` needs one allowed part and
    /// an `AND` needs every part allowed. A licence with an exception matches only an `allow`
    /// entry that names the same exception, such as `Apache-2.0 WITH LLVM-exception`.
    pub fn allowed_by(&self, allow: &[String]) -> bool {
        match self {
            Expr::Licence { .. } => allow.iter().any(|entry| *entry == self.to_string()),
            Expr::And(parts) => parts.iter().all(|part| part.allowed_by(allow)),
            Expr::Or(parts) => parts.iter().any(|part| part.allowed_by(allow)),
        }
    }

    /// Whether `id` appears anywhere in the expression.
    pub fn mentions(&self, id: &str) -> bool {
        match self {
            Expr::Licence { id: own, .. } => own == id,
            Expr::And(parts) | Expr::Or(parts) => parts.iter().any(|part| part.mentions(id)),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let joined = |f: &mut fmt::Formatter<'_>, parts: &[Expr], op: &str| {
            for (index, part) in parts.iter().enumerate() {
                if index > 0 {
                    write!(f, " {op} ")?;
                }
                match part {
                    Expr::Licence { .. } => write!(f, "{part}")?,
                    _ => write!(f, "({part})")?,
                }
            }
            Ok(())
        };
        match self {
            Expr::Licence {
                id,
                exception: None,
            } => write!(f, "{id}"),
            Expr::Licence {
                id,
                exception: Some(exception),
            } => write!(f, "{id} WITH {exception}"),
            Expr::And(parts) => joined(f, parts, "AND"),
            Expr::Or(parts) => joined(f, parts, "OR"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Token {
    Open,
    Close,
    And,
    Or,
    With,
    Id(String),
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Open => write!(f, "("),
            Token::Close => write!(f, ")"),
            Token::And => write!(f, "AND"),
            Token::Or => write!(f, "OR"),
            Token::With => write!(f, "WITH"),
            Token::Id(id) => write!(f, "{id}"),
        }
    }
}

fn tokenize(text: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = text.char_indices().peekable();
    while let Some(&(start, c)) = chars.peek() {
        match c {
            c if c.is_whitespace() => {
                chars.next();
            }
            '(' => {
                chars.next();
                tokens.push(Token::Open);
            }
            ')' => {
                chars.next();
                tokens.push(Token::Close);
            }
            '/' => {
                chars.next();
                tokens.push(Token::Or);
            }
            c if is_id_char(c) => {
                let mut end = start;
                while let Some(&(at, c)) = chars.peek() {
                    if !is_id_char(c) {
                        break;
                    }
                    end = at + c.len_utf8();
                    chars.next();
                }
                tokens.push(match &text[start..end] {
                    "AND" => Token::And,
                    "OR" => Token::Or,
                    "WITH" => Token::With,
                    id => Token::Id(id.to_owned()),
                });
            }
            other => return Err(format!("unexpected character `{other}`")),
        }
    }
    Ok(tokens)
}

/// The characters of an SPDX identifier, `LicenseRef-` ones included, and the `+` suffix.
fn is_id_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '+' | ':')
}

struct Parser {
    tokens: Vec<Token>,
    at: usize,
}

impl Parser {
    fn next_is(&self, token: &Token) -> bool {
        self.tokens.get(self.at) == Some(token)
    }

    /// `and ( OR and )*`
    fn or(&mut self) -> Result<Expr, String> {
        let mut parts = vec![self.and()?];
        while self.next_is(&Token::Or) {
            self.at += 1;
            parts.push(self.and()?);
        }
        Ok(if parts.len() == 1 {
            parts.remove(0)
        } else {
            Expr::Or(parts)
        })
    }

    /// `atom ( AND atom )*`
    fn and(&mut self) -> Result<Expr, String> {
        let mut parts = vec![self.atom()?];
        while self.next_is(&Token::And) {
            self.at += 1;
            parts.push(self.atom()?);
        }
        Ok(if parts.len() == 1 {
            parts.remove(0)
        } else {
            Expr::And(parts)
        })
    }

    /// `( or )` or `id [ WITH id ]`
    fn atom(&mut self) -> Result<Expr, String> {
        let token = self.tokens.get(self.at).cloned();
        self.at += 1;
        match token {
            Some(Token::Open) => {
                let inner = self.or()?;
                if !self.next_is(&Token::Close) {
                    return Err("a `(` has no matching `)`".to_owned());
                }
                self.at += 1;
                Ok(inner)
            }
            Some(Token::Id(id)) => {
                let exception = if self.next_is(&Token::With) {
                    self.at += 1;
                    match self.tokens.get(self.at).cloned() {
                        Some(Token::Id(exception)) => {
                            self.at += 1;
                            Some(exception)
                        }
                        _ => return Err(format!("`{id} WITH` needs an exception")),
                    }
                } else {
                    None
                };
                Ok(Expr::Licence { id, exception })
            }
            Some(other) => Err(format!("expected a licence, found `{other}`")),
            None => Err("the expression ends where a licence is expected".to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allow(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|id| (*id).to_owned()).collect()
    }

    #[test]
    fn parses_and_prints_the_forms_crates_use() {
        for (text, shown) in [
            ("MIT", "MIT"),
            ("MIT OR Apache-2.0", "MIT OR Apache-2.0"),
            ("MIT/Apache-2.0", "MIT OR Apache-2.0"),
            (
                "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT",
                "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT",
            ),
            (
                "(MIT OR Apache-2.0) AND Unicode-3.0",
                "(MIT OR Apache-2.0) AND Unicode-3.0",
            ),
            ("MIT AND (BSD-2-Clause)", "MIT AND BSD-2-Clause"),
            ("LicenseRef-Vendor-1.0", "LicenseRef-Vendor-1.0"),
        ] {
            let expr = Expr::parse(text).unwrap_or_else(|e| panic!("`{text}`: {e}"));
            assert_eq!(expr.to_string(), shown, "`{text}`");
        }
    }

    #[test]
    fn rejects_malformed_expressions() {
        for text in [
            "",
            "   ",
            "MIT OR",
            "AND MIT",
            "(MIT",
            "MIT)",
            "MIT WITH",
            "MIT Apache-2.0",
            "MIT, Apache-2.0",
        ] {
            assert!(Expr::parse(text).is_err(), "`{text}` parsed");
        }
    }

    #[test]
    fn or_needs_one_allowed_part_and_and_needs_all() {
        let allowed = allow(&["MIT", "Apache-2.0"]);
        let check = |text: &str| Expr::parse(text).unwrap().allowed_by(&allowed);
        assert!(check("MIT"));
        assert!(check("MIT OR GPL-3.0-only"));
        assert!(check("GPL-3.0-only OR Apache-2.0"));
        assert!(!check("GPL-3.0-only"));
        assert!(check("MIT AND Apache-2.0"));
        assert!(!check("MIT AND GPL-2.0-only"));
        assert!(check("(MIT OR GPL-2.0-only) AND Apache-2.0"));
        assert!(!check("(MPL-2.0 OR GPL-2.0-only) AND Apache-2.0"));
    }

    #[test]
    fn an_exception_must_be_allowed_by_name() {
        let allowed = allow(&["Apache-2.0 WITH LLVM-exception"]);
        assert!(
            Expr::parse("Apache-2.0 WITH LLVM-exception")
                .unwrap()
                .allowed_by(&allowed)
        );
        assert!(!Expr::parse("Apache-2.0").unwrap().allowed_by(&allowed));
        assert!(
            !Expr::parse("GPL-2.0-only WITH Classpath-exception-2.0")
                .unwrap()
                .allowed_by(&allowed)
        );
        assert!(
            !Expr::parse("Apache-2.0 WITH LLVM-exception")
                .unwrap()
                .allowed_by(&allow(&["Apache-2.0"]))
        );
    }

    #[test]
    fn mentions_finds_a_licence_at_any_depth() {
        let expr = Expr::parse("MIT AND (Apache-2.0 OR MPL-2.0)").unwrap();
        assert!(expr.mentions("MPL-2.0"));
        assert!(expr.mentions("MIT"));
        assert!(!expr.mentions("GPL-3.0-only"));
    }
}
