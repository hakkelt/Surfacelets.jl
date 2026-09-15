# Coverage Workflow

## Default: Run Tests + Generate Report

```julia
cov = generate_coverage("Contourlets")
```

- Calls `Pkg.test(; coverage=true)` and processes the resulting `*.cov` files.
- Writes `coverage/lcov.info`.
- Returns a coverage summary object.

## Scoped Reports

Limit processing to specific folders or files:

```julia
cov = generate_coverage("Contourlets";
    folder_list = ["src", "ext"],
    file_list   = ["src/transforms/ct.jl"],
)
```

## Reuse Existing `.cov` Files

If tests already ran (e.g. from `Pkg.test(; coverage=true)` in another step):

```julia
cov = generate_coverage("Contourlets"; run_test=false)
```

## Forward Test Arguments

Pass tag/name filters to the test runner:

```julia
cov = generate_coverage("Contourlets"; test_args=[":ct,:nsct"])
```
