# Syntropy

Syntropy runs one large language model across every machine a user owns and across every memory tier inside each machine. This site explains the design. None of it is built yet: the pages describe what is planned, and no figure on them is a measurement of Syntropy.

## Pages

- [How Syntropy works](HOW_IT_WORKS.md): the design in one page, from the memory tiers inside one machine to a ring of machines, with links to the RFCs and ADRs that specify each part.

## Where the design lives

The full design is the plan in the repository: start with [plan/README.md](../plan/README.md), then the product directives in [ADR-0000](../plan/decisions/ADR-0000-product-directives.md). The [README](../README.md) is the project's front page.

## About this site

The site is built from the Markdown pages in the repository's `docs/` directory, by `cargo xtask ci docs`. It is English only. It loads no fonts, scripts or analytics, and it makes no request to any other origin: a link to another site is followed only when you click it.
