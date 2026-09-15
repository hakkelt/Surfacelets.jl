---
name: my-localcoverage
description: Measure and report Julia package test coverage with LocalCoverage.jl. Use when checking coverage percentages, generating HTML reports, or enforcing a minimum coverage target.
---

# Julia Local Coverage

Use this skill to run coverage analysis for Contourlets.jl or any Julia package.

## Quick Setup

LocalCoverage.jl should not be a package dependency. Use a temporary environment:

```julia
using Pkg
Pkg.activate(temp=true)
Pkg.add("LocalCoverage")
Pkg.develop(path=pwd())
using LocalCoverage
```

## Workflows

- **Get coverage percentage and write `coverage/lcov.info`** — `references/workflow.md`
- **Generate an HTML report** (requires `lcov` system package) — `references/html.md`
- **Enforce a minimum target in CI** — `references/threshold.md`
- **Reuse existing `.cov` files without re-running tests** — `references/reuse.md`

## Quick Commands

```julia
# Full run: tests + coverage report
cov = generate_coverage("Contourlets")

# Identify uncovered lines
report_coverage_and_exit("Contourlets"; target_coverage=90, print_gaps=true)
```

All generated output (lcov.info, HTML) goes under `.temp/` (git-ignored).

## Notes

- `generate_coverage` calls `Pkg.test(; coverage=true)` internally — no need to run tests separately first.
- Pass `test_args=["tag1,tag2"]` to forward filter arguments to `runtests.jl`.
- HTML output needs `lcov` installed: `sudo apt install lcov` or `brew install lcov`.
- To scope the report to specific folders: `generate_coverage("Contourlets"; folder_list=["src", "ext"])`.

## Related Skills

- `julia-tests` — general test runner workflow (TestItemRunner, tag filters)
- `julia-jet` — static type/error analysis complementary to coverage
