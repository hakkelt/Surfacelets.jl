# Wire the unregistered sibling checkout of MultiscaleFilterBanks.jl into every
# environment of this workspace.
#
#     julia --project=. dev/bootstrap.jl
#
# `[sources]` in Project.toml would be the tidier mechanism, but it needs Julia
# ≥ 1.11 while this package supports 1.10 (and CI runs `lts`), so the path
# dependency is recorded in the gitignored Manifest.toml instead.  Run this once
# per fresh checkout; re-running it is harmless.
using Pkg

const CORE = normpath(joinpath(@__DIR__, "..", "..", "MultiscaleFilterBanks"))
isdir(CORE) || error("MultiscaleFilterBanks checkout not found at $CORE")

const ROOT = normpath(joinpath(@__DIR__, ".."))

for env in (ROOT, joinpath(ROOT, "test"), joinpath(ROOT, "docs"), joinpath(ROOT, "benchmark"))
    isfile(joinpath(env, "Project.toml")) || continue
    @info "developing MultiscaleFilterBanks into" env
    Pkg.activate(env)
    Pkg.develop(path = CORE)
end

Pkg.activate(ROOT)
