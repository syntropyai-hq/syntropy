// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::{Path, PathBuf};

use super::generated::{self, Generator};
use super::markdown::{render, slug};
use super::origins::{self, CSP, is_remote};
use super::*;

/// A directory under the system temp directory, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let dir =
            std::env::temp_dir().join(format!("syntropy-xtask-docs-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn write(&self, path: &str, text: &str) {
        let full = self.0.join(path);
        std::fs::create_dir_all(full.parent().unwrap()).unwrap();
        std::fs::write(full, text).unwrap();
    }

    fn root(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------------------------------------
// Markdown.

fn html(source: &str) -> String {
    render(source).html
}

#[test]
fn headings_get_the_forges_anchors() {
    let document = render(
        "# How it works\n\n## 1.1 Decode is a bytes-moved problem\n\n## `Vram` & RAM: the tiers\n\n\
         ## Exactness\n\n## Exactness\n\nSetext\n======\n\n### Closed ###\n",
    );
    let ids: Vec<&str> = document.headings.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(
        ids,
        [
            "how-it-works",
            "11-decode-is-a-bytes-moved-problem",
            "vram--ram-the-tiers",
            "exactness",
            "exactness-1",
            "setext",
            "closed",
        ]
    );
    assert!(
        document
            .html
            .contains("<h2 id=\"vram--ram-the-tiers\"><code>Vram</code> &amp; RAM: the tiers</h2>")
    );
    assert!(document.html.contains("<h1 id=\"setext\">Setext</h1>"));
    assert_eq!(document.headings[0].level, 1);
    assert_eq!(slug("Über § 4 rule_8"), "über--4-rule_8");
}

#[test]
fn emphasis_follows_the_flanking_rules() {
    assert_eq!(
        html("**strong**, *em*, ***both***, snake_case_name and 2 * 3 * 4\n"),
        "<p><strong>strong</strong>, <em>em</em>, <em><strong>both</strong></em>, \
         snake_case_name and 2 * 3 * 4</p>\n"
    );
    assert_eq!(
        html("_em_ and __strong__\n"),
        "<p><em>em</em> and <strong>strong</strong></p>\n"
    );
    assert_eq!(html("**unclosed\n"), "<p>**unclosed</p>\n");
}

#[test]
fn code_is_escaped_and_never_parsed() {
    assert_eq!(
        html("Use `a <b> & *c*` here.\n"),
        "<p>Use <code>a &lt;b&gt; &amp; *c*</code> here.</p>\n"
    );
    assert_eq!(
        html("```rust\nlet x = a < b && *p;\n[not](a link)\n```\n"),
        "<pre><code class=\"language-rust\">let x = a &lt; b &amp;&amp; *p;\n[not](a link)\n\
         </code></pre>\n"
    );
    assert_eq!(
        html("    indented <code>\n"),
        "<pre><code>indented &lt;code&gt;\n</code></pre>\n"
    );
}

#[test]
fn lists_nest_and_are_tight_or_loose() {
    assert_eq!(
        html("- one\n- two\n  - nested **b**\n- three\n"),
        "<ul>\n<li>one</li>\n<li>two\n<ul>\n<li>nested <strong>b</strong></li>\n</ul></li>\n\
         <li>three</li>\n</ul>\n"
    );
    assert_eq!(
        html("1. first\n\n2. second\n"),
        "<ol>\n<li>\n<p>first</p>\n</li>\n<li>\n<p>second</p>\n</li>\n</ol>\n"
    );
    assert!(html("3) three\n4) four\n").starts_with("<ol start=\"3\">\n<li>three</li>"));
    // A lazy continuation line stays in the item.
    assert_eq!(html("- a\nb\n"), "<ul>\n<li>a\nb</li>\n</ul>\n");
}

#[test]
fn tables_split_on_unescaped_pipes() {
    let out = html("| A | B |\n|:---|---:|\n| `x|y` | a \\| b |\n| c |\n");
    assert!(out.contains("<th align=\"left\">A</th>"), "{out}");
    assert!(out.contains("<th align=\"right\">B</th>"), "{out}");
    assert!(
        out.contains("<td align=\"left\"><code>x|y</code></td>"),
        "{out}"
    );
    assert!(out.contains("<td align=\"right\">a | b</td>"), "{out}");
    assert!(out.contains("<td align=\"right\"></td>"), "{out}");
}

#[test]
fn links_and_images_keep_their_targets() {
    assert_eq!(
        html("[RFC](../plan/rfcs/x.md#2-design \"t\") and ![a *b*](media/x.svg)\n"),
        "<p><a href=\"../plan/rfcs/x.md#2-design\" title=\"t\">RFC</a> and \
         <img src=\"media/x.svg\" alt=\"a b\"></p>\n"
    );
    assert_eq!(
        html("[ref][r] and [r]\n\n[r]: <a b.md> \"T\"\n"),
        "<p><a href=\"a b.md\" title=\"T\">ref</a> and <a href=\"a b.md\" title=\"T\">r</a></p>\n"
    );
    assert_eq!(
        html("<https://example.org/a> and see https://example.org/b.\n"),
        "<p><a href=\"https://example.org/a\">https://example.org/a</a> and see \
         <a href=\"https://example.org/b\">https://example.org/b</a>.</p>\n"
    );
    assert_eq!(
        html("[not a link] and a<b\n"),
        "<p>[not a link] and a&lt;b</p>\n"
    );
}

#[test]
fn html_blocks_pass_through_and_comments_go() {
    assert_eq!(
        html(
            "<p align=\"center\"><img src=\"media/x.svg\" alt=\"x\"></p>\n\n<!-- a\nnote -->\nText <kbd>K</kbd>\n"
        ),
        "<p align=\"center\"><img src=\"media/x.svg\" alt=\"x\"></p>\n<p>Text <kbd>K</kbd></p>\n"
    );
    assert_eq!(
        html("> quoted *text*\n> more\n\n---\n"),
        "<blockquote>\n<p>quoted <em>text</em>\nmore</p>\n</blockquote>\n<hr>\n"
    );
}

#[test]
fn hard_breaks_and_escapes() {
    assert_eq!(
        html("a  \nb\\\nc \\*d\\*\n"),
        "<p>a<br>\nb<br>\nc *d*</p>\n"
    );
    assert_eq!(
        html("&copy; &amp; & &#x41;\n"),
        "<p>&copy; &amp; &amp; &#x41;</p>\n"
    );
}

// ---------------------------------------------------------------------------------------------
// The tag scanner.

#[test]
fn the_scanner_reads_attributes_and_raw_text() {
    let text = "<!-- <a href=x> --><A HREF='a&amp;b' data-x=y hidden>t</A><style>p{} </style><br/>";
    let scan = html::scan(text);
    let names: Vec<(&str, bool)> = scan
        .tags
        .iter()
        .map(|t| (t.name.as_str(), t.closing))
        .collect();
    assert_eq!(
        names,
        [
            ("a", false),
            ("a", true),
            ("style", false),
            ("style", true),
            ("br", false)
        ]
    );
    let a = &scan.tags[0];
    assert_eq!(a.attr("href").unwrap().value.as_deref(), Some("a&b"));
    assert_eq!(a.attr("data-x").unwrap().value.as_deref(), Some("y"));
    assert_eq!(a.attr("hidden").unwrap().value, None);
    assert_eq!(scan.raw, [("style".to_owned(), "p{} ".to_owned())]);
    let span = a.attr("href").unwrap().span.clone().unwrap();
    assert_eq!(&text[span.clone()], "a&amp;b");
    assert_eq!(
        html::rewrite(text, vec![(span, "x.html#a\"b".to_owned())]),
        text.replace("a&amp;b", "x.html#a&quot;b")
    );
}

// ---------------------------------------------------------------------------------------------
// The origin check.

fn page(body: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta http-equiv=\"Content-Security-Policy\" \
         content=\"{}\"></head><body>{body}</body></html>",
        markdown::escape(CSP)
    )
}

#[test]
fn a_clean_page_passes_and_links_out_are_navigation() {
    let text = page(
        "<a href=\"https://github.com/x\">out</a><img src=\"media/x.svg\" srcset=\"a.png 1x, \
         b.png 2x\"><link rel=\"stylesheet\" href=\"../style.css\">",
    );
    assert_eq!(
        origins::check_file("a/page.html", text.as_bytes()),
        Vec::<String>::new()
    );
    assert_eq!(
        origins::check_file("img/x.png", b"\x89PNG"),
        Vec::<String>::new()
    );
}

#[test]
fn every_way_to_fetch_from_another_origin_fails() {
    for (body, expected) in [
        ("<script src=\"app.js\"></script>", "a <script> element"),
        (
            "<img src=\"https://cdn.example/x.png\">",
            "fetches from another origin",
        ),
        (
            "<img src=\"//cdn.example/x.png\">",
            "fetches from another origin",
        ),
        (
            "<img srcset=\"a.png 1x, https://x.example/b.png 2x\">",
            "fetches from another origin",
        ),
        (
            "<link rel=\"preconnect\" href=\"https://fonts.example\">",
            "fetches from another origin",
        ),
        ("<iframe src=\"x.html\"></iframe>", "loads another document"),
        (
            "<base href=\"https://x.example/\">",
            "loads another document",
        ),
        (
            "<a href=\"x.html\" ping=\"https://t.example\">x</a>",
            "fetches from another origin",
        ),
        ("<a href=\"javascript:alert(1)\">x</a>", "`javascript:` URL"),
        ("<img src=\"a.png\" onerror=\"x()\">", "event handler"),
        (
            "<meta http-equiv=\"refresh\" content=\"0; url=x\">",
            "refresh <meta>",
        ),
        (
            "<p style=\"background: url(https://t.example/p.gif)\">x</p>",
            "fetches from another origin",
        ),
        ("<style>@import \"x.css\";</style>", "@import"),
        (
            "<style>@font-face { src: url(f.woff2) }</style>",
            "declares a font",
        ),
    ] {
        let problems = origins::check_file("page.html", page(body).as_bytes());
        assert!(
            problems.iter().any(|p| p.contains(expected)),
            "`{body}` gave {problems:?}, not `{expected}`"
        );
    }
}

#[test]
fn svg_css_and_file_types_are_checked_too() {
    let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\
               <image xlink:href=\"https://x.example/a.png\"/><a href=\"https://ok.example\">t</a></svg>";
    let problems = origins::check_file("media/x.svg", svg.as_bytes());
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("xlink:href"));
    let css = "body { background: url('https://x.example/a.png'); } p { background: url(a.png) }";
    assert_eq!(origins::check_file("style.css", css.as_bytes()).len(), 1);
    // Escapes, comments and `image-set()` strings are read as a browser reads them.
    for (css, expected) in [
        (
            "p { background: u\\72l(https://x.example/a.png) }",
            "url(https://x.example/a.png)",
        ),
        (
            "p { background: ur\\00006C (//x.example/a.png) }",
            "url(//x.example/a.png)",
        ),
        ("@\\69mport 'a.css';", "@import"),
        ("@im/* x */port 'a.css';", "@import"),
        // A `/*` inside a string starts no comment, so the URL after it is still seen.
        (
            "p { content: \"/*\" } q { background: url(https://x.example/a.png) } r { content: \"*/\" }",
            "url(https://x.example/a.png)",
        ),
        // A `*/` inside a string does not end a real comment early: the comment hides nothing.
        (
            "p { content: '*/\\'' } /* note */ q { background: url(//x.example/b.png) }",
            "url(//x.example/b.png)",
        ),
        (
            "p { background: image-set(\"a.png\" 1x, \"https://x.example/b.png\" 2x) }",
            "image-set(\"https://x.example/b.png\")",
        ),
    ] {
        let problems = origins::check_file("style.css", css.as_bytes());
        assert!(
            problems.iter().any(|p| p.contains(expected)),
            "`{css}` gave {problems:?}, not `{expected}`"
        );
    }
    let local = "p { background: image-set(\"a.png\" 1x, url(b.png) 2x); content: \"\\201C\" }";
    assert_eq!(
        origins::check_file("style.css", local.as_bytes()),
        Vec::<String>::new()
    );
    for file in ["app.js", "font.woff2", "data.json"] {
        let problems = origins::check_file(file, b"x");
        assert!(
            problems[0].contains("ships only HTML, CSS and images"),
            "{problems:?}"
        );
    }
    let no_policy = "<!doctype html><html><head></head><body></body></html>";
    let problems = origins::check_file("x.html", no_policy.as_bytes());
    assert!(
        problems[0].contains("content security policy"),
        "{problems:?}"
    );
}

#[test]
fn remote_urls_are_told_apart() {
    for url in [
        "https://a",
        "HTTP://a",
        "//a",
        "/\\a",
        "\\\\a",
        " ftp://a",
        "ws://a",
        "java\nscript:x",
    ] {
        assert!(is_remote(url), "{url}");
    }
    for url in [
        "a.png",
        "../style.css",
        "#top",
        "data:image/png;base64,AA",
        "a/b:c",
        "?x=y:z",
        "",
    ] {
        assert!(!is_remote(url), "{url}");
    }
}

// ---------------------------------------------------------------------------------------------
// The generated-section hooks.

fn shout(_: &Path) -> Result<String, String> {
    Ok("GENERATED TEXT\n\n\n".to_owned())
}

fn broken(_: &Path) -> Result<String, String> {
    Err("no schema".to_owned())
}

const TEST_GENERATORS: &[Generator] = &[
    Generator {
        name: "shout",
        task: "PROD-E13-T07",
        run: shout,
    },
    Generator {
        name: "broken",
        task: "PROD-E13-T07",
        run: broken,
    },
];

#[test]
fn no_generator_is_registered_yet() {
    assert!(generated::GENERATORS.is_empty());
}

#[test]
fn sections_are_found_outside_fenced_code() {
    let text = "a\n<!-- BEGIN GENERATED: shout -->\nold\n<!-- END GENERATED -->\n```\n\
                <!-- BEGIN GENERATED: in-code -->\n```\n";
    let sections = generated::sections(text).unwrap();
    assert_eq!(sections.len(), 1);
    assert_eq!(sections[0].name, "shout");
    assert_eq!(sections[0].line, 2);
    assert_eq!(&text[sections[0].content.clone()], "old\n");
    for (bad, expected) in [
        ("<!-- BEGIN GENERATED: a -->\n", "has no END GENERATED"),
        ("<!-- END GENERATED -->\n", "with no section open"),
        (
            "<!-- BEGIN GENERATED: Bad_Name -->\nx\n<!-- END GENERATED -->\n",
            "not a generator name",
        ),
        (
            "<!-- BEGIN GENERATED: a -->\n<!-- BEGIN GENERATED: b -->\n",
            "starts inside section `a`",
        ),
        ("<!-- GENERATED: a -->\n", "malformed"),
    ] {
        let error = generated::sections(bad).unwrap_err();
        assert!(error.contains(expected), "`{bad}` gave `{error}`");
    }
}

#[test]
fn a_stale_or_unknown_section_fails_and_generate_fixes_the_stale_one() {
    let scratch = Scratch::new("generated");
    let stale = "# P\n\n<!-- BEGIN GENERATED: shout -->\nold\n<!-- END GENERATED -->\n";
    scratch.write("docs/index.md", stale);
    let problems = check_generated(scratch.root(), TEST_GENERATORS);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("differs from its generator"));

    let changed = write_generated(scratch.root(), TEST_GENERATORS).unwrap();
    assert_eq!(changed, ["docs/index.md"]);
    let text = std::fs::read_to_string(scratch.root().join("docs/index.md")).unwrap();
    assert_eq!(
        text,
        "# P\n\n<!-- BEGIN GENERATED: shout -->\nGENERATED TEXT\n<!-- END GENERATED -->\n"
    );
    assert!(check_generated(scratch.root(), TEST_GENERATORS).is_empty());

    scratch.write(
        "docs/other.md",
        "<!-- BEGIN GENERATED: config-keys -->\n<!-- END GENERATED -->\n\
         <!-- BEGIN GENERATED: broken -->\n<!-- END GENERATED -->\n",
    );
    let problems = check_generated(scratch.root(), TEST_GENERATORS);
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(problems[0].contains("docs/other.md:1: no generator is named `config-keys`"));
    assert!(
        problems[1]
            .contains("docs/other.md:3: generator `broken` (PROD-E13-T07) failed: no schema")
    );
}

// ---------------------------------------------------------------------------------------------
// The site build.

const CONFIG_TEXT: &str = "title = \"Site\"\nrepository = \"https://forge.example/o/r\"\n\
                           branch = \"main\"\nnav = [\"index.md\", \"guide/page.md\"]\n";

/// A small site: two pages, an image, and a plan file outside `docs/` the pages link to.
fn site(name: &str) -> Scratch {
    let scratch = Scratch::new(name);
    scratch.write("docs/site.toml", CONFIG_TEXT);
    scratch.write(
        "docs/index.md",
        "# Home\n\nSee [the guide](guide/page.md#second-part), [the plan](../plan/a.md#21-design), \
         [the plans](../plan/), [a line](../plan/b.rs#L3), [out](https://example.org/x) and \
         [here](#home).\n\n![tiers](media/tiers.svg)\n",
    );
    scratch.write(
        "docs/guide/page.md",
        "# Guide\n\n## Second part\n\n<a id=\"raw-anchor\"></a>\n\nBack [home](../index.md) and \
         [raw](#raw-anchor).\n\n<p align=\"center\"><img src=\"../media/tiers.svg\" alt=\"t\"></p>\n",
    );
    scratch.write(
        "docs/media/tiers.svg",
        "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>",
    );
    scratch.write("plan/a.md", "# A\n\n## 2.1 Design\n");
    scratch.write("plan/b.rs", "fn main() {}\n");
    scratch
}

fn page_of<'a>(files: &'a [SiteFile], path: &str) -> &'a str {
    let (_, bytes) = files.iter().find(|(p, _)| p == path).unwrap();
    std::str::from_utf8(bytes).unwrap()
}

#[test]
fn a_site_builds_with_its_links_resolved_and_rewritten() {
    let scratch = site("good");
    let files = build(scratch.root()).unwrap();
    let paths: Vec<&str> = files.iter().map(|(p, _)| p.as_str()).collect();
    assert_eq!(
        paths,
        [
            "guide/page.html",
            "index.html",
            "media/tiers.svg",
            "style.css"
        ]
    );

    let index = page_of(&files, "index.html");
    for expected in [
        "<html lang=\"en\">",
        "<title>Site</title>",
        "href=\"guide/page.html#second-part\"",
        "href=\"https://forge.example/o/r/blob/main/plan/a.md#21-design\"",
        "href=\"https://forge.example/o/r/tree/main/plan\"",
        "href=\"https://forge.example/o/r/blob/main/plan/b.rs#L3\"",
        "href=\"https://example.org/x\"",
        "href=\"#home\"",
        "src=\"media/tiers.svg\"",
        "<link rel=\"stylesheet\" href=\"style.css\">",
        "<a href=\"index.html\" aria-current=\"page\">Home</a>",
    ] {
        assert!(
            index.contains(expected),
            "index.html lacks `{expected}`:\n{index}"
        );
    }
    let guide = page_of(&files, "guide/page.html");
    for expected in [
        "<title>Guide · Site</title>",
        "href=\"../index.html\"",
        "href=\"../style.css\"",
        "<a href=\"../guide/page.html\" aria-current=\"page\">Guide</a>",
    ] {
        assert!(
            guide.contains(expected),
            "guide/page.html lacks `{expected}`:\n{guide}"
        );
    }

    // The written site passes the origin check, and a rebuild replaces it.
    let out = scratch.root().join("out");
    write(&out, &files).unwrap();
    assert_eq!(check_site(&out), Vec::<String>::new());
    write(&out, &files).unwrap();
}

#[test]
fn write_leaves_a_directory_that_is_not_a_site_alone() {
    let scratch = Scratch::new("write");
    scratch.write("precious/notes.txt", "keep");
    let error = write(&scratch.root().join("precious"), &[]).unwrap_err();
    assert!(error.contains("not an earlier site build"), "{error}");
    assert!(scratch.root().join("precious/notes.txt").is_file());
}

/// Builds the small site with one page replaced, and returns the problems.
fn problems_with(name: &str, path: &str, text: &str) -> Vec<String> {
    let scratch = site(name);
    scratch.write(path, text);
    build(scratch.root()).unwrap_err()
}

#[test]
fn a_broken_link_or_anchor_fails_the_build() {
    for (name, text, expected) in [
        ("page", "[x](missing.md)", "no page docs/missing.md"),
        ("case", "[x](Guide/page.md)", "no page docs/Guide/page.md"),
        (
            "anchor",
            "[x](guide/page.md#no-such)",
            "docs/guide/page.md has no anchor `#no-such`",
        ),
        (
            "self",
            "[x](#nowhere)",
            "docs/index.md has no anchor `#nowhere`",
        ),
        (
            "image",
            "![x](media/missing.svg)",
            "no image docs/media/missing.svg",
        ),
        (
            "repo",
            "[x](../plan/missing.md)",
            "no file plan/missing.md in the repository",
        ),
        (
            "repo-case",
            "[x](../Plan/a.md)",
            "no file Plan/a.md in the repository",
        ),
        (
            "repo-anchor",
            "[x](../plan/a.md#3-nothing)",
            "plan/a.md has no anchor `#3-nothing`",
        ),
        ("repo-line", "[x](../plan/b.rs#main)", "cannot be checked"),
        ("absolute", "[x](/guide/page.md)", "a site-absolute link"),
        (
            "above",
            "[x](../../outside.md)",
            "above the repository's root",
        ),
        ("query", "[x](guide/page.md?x=1)", "a query string"),
        ("dir", "[x](./)", "a directory of the site"),
        (
            "raw-html",
            "<p><img src=\"media/nope.png\"></p>",
            "no image docs/media/nope.png",
        ),
        (
            "script-url",
            "[x](javascript:alert(1))",
            "a `javascript:` URL",
        ),
    ] {
        let problems = problems_with(name, "docs/index.md", &format!("# Home\n\n{text}\n"));
        assert!(
            problems.iter().any(|p| p.contains(expected)),
            "`{text}` gave {problems:?}, not `{expected}`"
        );
    }
}

#[test]
fn the_site_is_english_only_and_takes_pages_and_images_only() {
    let problems = problems_with("translation", "docs/index.de.md", "# Start\n");
    assert!(problems[0].contains("English only"), "{problems:?}");
    let problems = problems_with("translation-region", "docs/guide/page.pt-BR.md", "# P\n");
    assert!(problems[0].contains("English only"), "{problems:?}");
    let problems = problems_with("script", "docs/app.js", "alert(1)");
    assert!(
        problems[0].contains("Markdown pages and images"),
        "{problems:?}"
    );
    // A page whose name merely has a dot is not a translation.
    assert!(!is_translation("v1.2.md"));
    assert!(!is_translation("HOW_IT_WORKS.md"));
}

#[test]
fn the_configuration_is_checked() {
    for (text, expected) in [
        (
            "title = \"S\"\nrepository = \"http://forge.example/o/r\"\nbranch = \"main\"\nnav = [\"index.md\"]\n",
            "`repository` is",
        ),
        (
            "title = \"S\"\nrepository = \"https://forge.example/o/r\"\nbranch = \"a b\"\nnav = [\"index.md\"]\n",
            "`branch` is one branch name",
        ),
        (
            "title = \"S\"\nrepository = \"https://forge.example/o/r\"\nbranch = \"main\"\nnav = []\n",
            "must list the home page",
        ),
        (
            "title = \"S\"\nrepository = \"https://forge.example/o/r\"\nbranch = \"main\"\n\
             nav = [\"index.md\", \"gone.md\"]\n",
            "`nav` lists `gone.md`, which is not a page",
        ),
        ("title = \"S\"\nanalytics = \"x\"\n", "unknown field"),
    ] {
        let problems = problems_with("config", "docs/site.toml", text);
        assert!(
            problems.iter().any(|p| p.contains(expected)),
            "`{text}` gave {problems:?}, not `{expected}`"
        );
    }
}

#[test]
fn links_resolve_against_the_page_that_holds_them() {
    assert_eq!(
        resolve("a/b.md", "../c.md#x%20y"),
        Ok(Target::Page {
            source: "c.md".into(),
            fragment: Some("x y".into()),
        })
    );
    assert_eq!(
        resolve("a/b.md", "../../plan/r%20s.md"),
        Ok(Target::Repository {
            path: "plan/r s.md".into(),
            fragment: None,
        })
    );
    assert_eq!(
        resolve("b.md", "./media/x.svg"),
        Ok(Target::Asset {
            path: "media/x.svg".into(),
        })
    );
    assert_eq!(resolve("b.md", "mailto:a@b.example"), Ok(Target::External));
    assert_eq!(up("a/b/c.md"), "../../");
    assert_eq!(up("c.md"), "");
}
