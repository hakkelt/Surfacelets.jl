# Threshold / CI Check

## Enforce a Minimum Coverage Target

```julia
report_coverage_and_exit("Contourlets"; target_coverage=90)
```

- Exit code `0` — target met.
- Exit code `1` — target missed.

Use `print_gaps=true` to list every uncovered line:

```julia
report_coverage_and_exit("Contourlets"; target_coverage=90, print_gaps=true)
```
