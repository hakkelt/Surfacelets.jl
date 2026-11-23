# Develop the sibling checkouts of AbstractMultiscaleTransforms.jl and
# MultiscaleFilterBanks.jl into every environment of this workspace, in place of
# registered releases:
#
#     julia --project=. dev/bootstrap.jl
#
# AbstractMultiscaleTransforms is not registered, and this package needs the
# development version of MultiscaleFilterBanks. `[sources]` covers Julia ≥ 1.11
# only; Julia 1.10 does not read it, so the paths are recorded in the gitignored
# manifests. CI runs this after cloning both repositories next to this one.
# Re-running it is harmless.
using Pkg

const SIBLINGS = [
    normpath(joinpath(@__DIR__, "..", "..", name)) for name in ("AbstractMultiscaleTransforms", "MultiscaleFilterBanks")
]
for dir in SIBLINGS
    isdir(dir) || error("sibling checkout not found at $dir")
end

const ROOT = normpath(joinpath(@__DIR__, ".."))

# The test environment is left out: `Pkg.test` takes the developed paths from
# the root manifest, and on Julia 1.10 developed copies in the test manifest as
# well make it fail ("can not merge projects").
for env in (ROOT, joinpath(ROOT, "docs"), joinpath(ROOT, "benchmark"))
    isfile(joinpath(env, "Project.toml")) || continue
    @info "developing the sibling checkouts into" env
    # Julia 1.10 also needs this package itself developed into the other
    # environments, which list it.
    paths = env == ROOT || VERSION >= v"1.11" ? SIBLINGS : [SIBLINGS; ROOT]
    # Julia ≥ 1.11 also writes the paths into `[sources]`; keep the tracked file.
    project = joinpath(env, "Project.toml")
    tracked = read(project, String)
    Pkg.activate(env)
    Pkg.develop([PackageSpec(path = dir) for dir in paths])
    write(project, tracked)
end

Pkg.activate(ROOT)
