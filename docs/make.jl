using Double32s
using Documenter

DocMeta.setdocmeta!(Double32s, :DocTestSetup, :(using Double32s); recursive=true)

makedocs(;
    modules=[Double32s],
    authors="JeffreySarnoff <jeffrey.sarnoff@gmail.com> and contributors",
    sitename="Double32s.jl",
    checkdocs=:exports,
    format=Documenter.HTML(;
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
