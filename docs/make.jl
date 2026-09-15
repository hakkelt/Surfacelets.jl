using Documenter, Surfacelets, MultiscaleFilterBanks
using Documenter: Remotes

DocMeta.setdocmeta!(Surfacelets, :DocTestSetup, :(using Surfacelets, Random); recursive = true)
DocMeta.setdocmeta!(
    MultiscaleFilterBanks, :DocTestSetup,
    :(using MultiscaleFilterBanks, Random); recursive = true
)

makedocs(;
    # MultiscaleFilterBanks is in the list because Surfacelets re-exports its
    # public names (`FilterPair`, `CDF97`, the threading policies), and their
    # docstrings live there.
    modules = [Surfacelets, MultiscaleFilterBanks],
    sitename = "Surfacelets.jl",
    # The repository has no `origin` remote in a fresh local clone, and
    # Documenter derives source links from it, so name it explicitly.
    repo = Remotes.GitHub("hakkelt", "Surfacelets.jl"),
    remotes = Dict(
        pkgdir(MultiscaleFilterBanks) =>
            Remotes.GitHub("hakkelt", "MultiscaleFilterBanks.jl"),
    ),
    authors = "Tamás Hakkel",
    format = Documenter.HTML(;
        canonical = "https://hakkelt.github.io/Surfacelets.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "Transforms" => "guide/transforms.md",
        ],
        "Background" => [
            "Theory" => "theory.md",
        ],
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/hakkelt/Surfacelets.jl",
    devbranch = "master",
)
