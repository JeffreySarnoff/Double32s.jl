# core/type.jl — the scalar, its unchecked constructor, and the accessors.
#
# Design sections 6.1 (recommended scalar) and 6.3 (size, alignment, ABI).

"""
    Raw()

Internal tag selecting the non-normalizing `Double32` constructor. Passing
`Raw()` asserts that the caller has already established the canonical invariant
of design section 5.2. Not exported, not public API.
"""
struct Raw end

"""
    Double32(hi::Float32, lo::Float32) -> Double32

A double-word number represented as the unevaluated sum `hi + lo` of two
`Float32` words, canonicalized so that `hi == fl(hi + lo)`.

The represented value is the *exact* real sum `hi + lo`, not a `Float32`
evaluation of it. A canonical finite pair satisfies `|lo| <= ulp(hi)/2`, and
`hi` is the nearest `Float32` to the pair sum.

`Double32` carries roughly 48 significant bits but keeps the `Float32` exponent
range, so it is **not** an IEEE format with a uniform significand lattice: the
distance to the next representable pair is not `eps(Double32)` and can be as
small as `2f0^-149` beside a high word of `1.0f0`. See [`nominal_eps`](@ref).

All NaN payloads may be canonicalized to `NaN32` by any operation.

# Examples
```jldoctest
julia> Double32(1.0f0) / 3
Double32(0.33333334f0, -9.934108f-9)

julia> Float64(Double32(1.0f0) / 3)
0.33333333333333304
```
"""
struct Double32 <: AbstractFloat
    hi::Float32
    lo::Float32

    # The only inner constructor: no normalization, no checks, no branches.
    Double32(::Raw, hi::Float32, lo::Float32) = new(hi, lo)
end

"""
    raw(hi::Float32, lo::Float32) -> Double32

Construct without canonicalizing. Every call site must be able to justify the
canonical invariant from the algorithm that produced `hi` and `lo`.
Internal; use [`canonicalize`](@ref) or the `Double32` constructor instead.
"""
@inline raw(hi::Float32, lo::Float32) = Double32(Raw(), hi, lo)

"""
    Double32s.hi(x::Double32) -> Float32

High word of `x`. Not exported: `hi` is too common a local name.
"""
@inline hi(x::Double32) = x.hi

"""
    Double32s.lo(x::Double32) -> Float32

Low word of `x`. Not exported: `lo` is too common a local name.
"""
@inline lo(x::Double32) = x.lo

"""
    Double32s.words(x::Double32) -> Tuple{Float32,Float32}

Both words as a tuple. This is the layer-1/layer-2 adapter of design section
8.0 and compiles away entirely.
"""
@inline words(x::Double32) = (x.hi, x.lo)
