using Double32
using Documenter

DocMeta.setdocmeta!(Double32, :DocTestSetup, :(using Double32); recursive=true)

makedocs(;
    modules=[Double32],
    authors="JeffreySarnoff <jeffrey.sarnoff@gmail.com> and contributors",
    sitename="Double32.jl",
    format=Documenter.HTML(;
        canonical="https://JeffreySarnoff.github.io/Double32.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JeffreySarnoff/Double32.jl",
    devbranch="main",
)
