# Develop a sibling checkout of MultiscaleFilterBanks.jl into every environment
# of this workspace, in place of the registered release.
#
#     julia --project=. dev/bootstrap.jl
#
# The path is recorded in the gitignored manifests, so nothing tracked changes
# and CI keeps using the registry.  Re-running it is harmless.
using Pkg

const CORE = normpath(joinpath(@__DIR__, "..", "..", "MultiscaleFilterBanks"))
isdir(CORE) || error("MultiscaleFilterBanks checkout not found at $CORE")

const ROOT = normpath(joinpath(@__DIR__, ".."))

for env in (ROOT, joinpath(ROOT, "test"), joinpath(ROOT, "docs"), joinpath(ROOT, "benchmark"))
    isfile(joinpath(env, "Project.toml")) || continue
    @info "developing MultiscaleFilterBanks into" env
    # Julia ≥ 1.11 also writes the path into `[sources]`; keep the tracked file.
    project = joinpath(env, "Project.toml")
    tracked = read(project, String)
    Pkg.activate(env)
    Pkg.develop(path = CORE)
    write(project, tracked)
end

Pkg.activate(ROOT)
