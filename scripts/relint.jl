#!/usr/bin/env julia
# Run ReLint on the given paths (default: src/) with project-specific suppressions.
#
# Suppressed rules (false positives for a performance-oriented numerical library):
#   @inbounds          — deliberate, tested bounds-safety
#   return type annotation — used for documentation/correctness, not coercion
#   in                 — RelationalAI-internal `tin` convention; stdlib `in` is fine
#   unsafe-logging     — RelationalAI-internal `@safe(...)` convention; not applicable here
#
# Usage:
#   julia --startup-file=no -e 'include("scripts/relint.jl")' [path ...]
#   # or via the relint shell app:
#   relint src/

using ReLint

const SUPPRESSED = Set(["@inbounds", "return type annotation", "in", "unsafe-logging"])

rules = filter(r -> !(r.name in SUPPRESSED), ReLint.ALL_RULES)
ctx = ReLint.LintContext(rules)

paths = isempty(ARGS) ? ["src/"] : ARGS
result = ReLint.LintGlobalReport()
for p in paths
    ReLint.run_lint(p; result, context = ctx)
end

n = result.violations_count + result.recommendations_count + result.fatal_violations_count
if iszero(n)
    println("ReLint: no findings.")
else
    println("\nReLint: $(n) finding(s) — see output above.")
end
exit(iszero(n) ? 0 : 1)
