using Test

include("setup.jl")

@testset "Double32s.jl" begin
    include("core/type.jl")
    include("core/eft.jl")
    include("core/exactness.jl")
    include("core/compare.jl")
    include("core/arithmetic.jl")
    include("core/interface.jl")
    include("core/invariants.jl")
    include("core/elementary.jl")
    include("core/decimal.jl")
    include("arrays.jl")
    include("foreign.jl")
    include("extensions.jl")
    include("kernels.jl")
    include("gates.jl")
end
