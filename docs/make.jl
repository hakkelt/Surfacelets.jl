using Documenter, Surfacelets
using Documenter: Remotes

DocMeta.setdocmeta!(Surfacelets, :DocTestSetup, :(using Surfacelets, Random); recursive = true)

makedocs(;
    modules = [Surfacelets],
    sitename = "Surfacelets.jl",
    # The repository has no `origin` remote in a fresh local clone, and
    # Documenter derives source links from it, so name it explicitly.
    repo = Remotes.GitHub("hakkelt", "Surfacelets.jl"),
    authors = "Tamás Hakkel",
    format = Documenter.HTML(;
        canonical = "https://hakkelt.github.io/Surfacelets.jl",
    ),
    pages = [
        "Home" => "index.md",
        "How-to guides" => [
            "Use the transforms" => "guide/transforms.md",
        ],
        "Explanation" => [
            "Theory" => "theory.md",
        ],
        "Reference" => [
            "API" => "api.md",
        ],
    ],
    checkdocs = :public,
)

deploydocs(;
    repo = "github.com/hakkelt/Surfacelets.jl",
    devbranch = "master",
)
