@testitem "Documentation doctests" tags = [:docs, :quality] begin
    using Documenter
    DocMeta.setdocmeta!(Surfacelets, :DocTestSetup, :(using Surfacelets, Random); recursive = true)
    doctest(Surfacelets; manual = false)
end

@testitem "Documentation structure (draft build)" tags = [:docs, :quality] begin
    using Documenter, MultiscaleFilterBanks
    using Documenter: Remotes
    pkg_root = pkgdir(Surfacelets)
    # @meta blocks evaluate `CurrentModule = Surfacelets` in Main; the module
    # must be importable there (same as `using Surfacelets` in make.jl).
    Base.eval(Main, :(using Surfacelets))
    DocMeta.setdocmeta!(
        MultiscaleFilterBanks, :DocTestSetup,
        :(using MultiscaleFilterBanks, Random); recursive = true
    )
    DocMeta.setdocmeta!(Surfacelets, :DocTestSetup, :(using Surfacelets, Random); recursive = true)
    build_dir = mktempdir()
    makedocs(;
        modules = [Surfacelets, MultiscaleFilterBanks],
        sitename = "Surfacelets.jl",
        # The repository has no `origin` remote in a fresh local clone, and
        # Documenter derives source links from it, so name it explicitly.
        repo = Remotes.GitHub("hakkelt", "Surfacelets.jl"),
        remotes = Dict(
            pkgdir(MultiscaleFilterBanks) =>
                Remotes.GitHub("hakkelt", "MultiscaleFilterBanks.jl"),
        ),
        root = joinpath(pkg_root, "docs"),
        source = "src",
        build = build_dir,
        checkdocs = :exports,
        draft = true,
        # Images referenced by `![](img.png)` are only generated when @example
        # blocks run (which draft=true skips), so suppress those link warnings.
        warnonly = [:cross_references],
    )
end
