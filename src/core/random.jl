# core/random.jl — rand support.
#
# Design section 11.2. `Random` reaches float types through a sampler for the
# [0,1) interval; one method is enough.

using Random: Random, AbstractRNG

"""
    rand(rng, Double32) -> Double32

A uniform draw from a 48-bit grid in `[0, 1)`.

Every step is exact — 24-bit integers convert to `Float32` exactly and `ldexp`
by a power of two is exact — so this is a genuine 48-bit uniform, unlike the
naive `Double32(rand(Float32))` whose low word is systematically zero.
"""
function Random.rand(rng::AbstractRNG, ::Random.SamplerTrivial{Random.CloseOpen01{Double32}})
    u = rand(rng, UInt64) >> 16                # 48 random bits
    a = Float32((u >> 24) & 0x00ff_ffff)       # exact: 24 bits
    b = Float32(u & 0x00ff_ffff)               # exact: 24 bits
    return canonicalize(ldexp(a, -24), ldexp(b, -48))
end

Random.rand(rng::AbstractRNG, ::Random.SamplerType{Double32}) =
    rand(rng, Random.CloseOpen01{Double32}())
