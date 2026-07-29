using Double32s
using Documenter

DocMeta.setdocmeta!(Double32s, :DocTestSetup, :(using Double32s); recursive=true)

makedocs(;
    modules=[Double32s],
    authors="JeffreySarnoff <jeffrey.sarnoff@gmail.com> and contributors",
    sitename="Double32s.jl",
    checkdocs=:exports,
    format=Documenter.HTML(;
        # Pretty URLs put every page at `page/index.html`, which reads as a
        # directory link when the built docs are browsed from the filesystem.
        # Keep them for CI deployment only; a local build emits `page.html`
        # so every link resolves immediately.
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://JeffreySarnoff.github.io/Double32.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Guide" => [
            "Getting Started" => "guide/getting-started.md",
            "Using Double32 on a GPU" => "guide/gpu.md",
            "Arrays, on CPU and GPU" => "guide/arrays.md",
            "Accuracy Model" => "guide/accuracy.md",
        ],
        "Reference" => [
            "Public API" => "reference/api.md",
            "Internals" => "reference/internals.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/JeffreySarnoff/Double32.jl",
    devbranch="main",
)
