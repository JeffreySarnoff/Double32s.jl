# core/convert.jl — constructors in and conversions out.
#
# Design section 6.4. Note what is *absent*: `Base.convert`. Base already gives
# `convert(::Type{T}, x::Number) where {T<:Number} = T(x)`, so defining the
# constructors is sufficient and defining `convert` too is redundant at best
# and a source of ambiguities at worst.

# ---------------------------------------------------------------- in --------

"""
    Double32(x::Float32) -> Double32

Exact and branch free: `raw(x, 0.0f0)` is already correct for every `Float32`,
including `±Inf32`, `NaN32`, and `-0.0f0`.

Note that this preserves the sign of a zero, which the two-argument
`Double32(-0.0f0, 0.0f0)` also does but a bare `two_sum` would not.
"""
@inline Double32(x::Float32) = raw(x, 0.0f0)

"""
    Double32(x::Float16) -> Double32

Exact: `Float16 -> Float32` needs no residual.
"""
@inline Double32(x::Float16) = Double32(Float32(x))

@inline Double32(x::Double32) = x

"""
    Double32(hi::Float32, lo::Float32) -> Double32

Canonicalizing two-word constructor; see [`canonicalize`](@ref).
"""
@inline Double32(hi::Float32, lo::Float32) = canonicalize(hi, lo)

"""
    Double32(x::Float64) -> Double32

Round a `Float64` to the nearest `Double32` pair.

Not injective — this rounds a 53-bit value into the roughly 48-bit pair — and
that loss is intentional and documented. A finite `Float64` may also exceed
`Float32` range, in which case the high word is `±Inf32`.
"""
function Double32(x::Float64)
    h = Float32(x)
    isfinite(h) || return raw(h, 0.0f0)        # ±Inf32 or NaN32: canonical, exact
    l = Float32(x - Float64(h))                # exact: |x - h| <= ulp(h)/2
    return raw(h, l)                           # already canonical by construction
end

# Exact via Float64 for every integer that fits in 53 bits.
Double32(n::Union{Int8,UInt8,Int16,UInt16,Int32,UInt32}) = Double32(Float64(n))
Double32(b::Bool) = b ? one(Double32) : zero(Double32)

"""
    Double32(n::Integer) -> Double32

Host method. For 64-bit and wider integers `Float64(n)` may already round, so
the residual is formed in the integer domain, where it is exact.
"""
function Double32(n::Integer)
    h = Float32(n)
    isfinite(h) || return raw(h, 0.0f0)
    r = n - oftype(n, h)                       # exact when h is integral
    return canonicalize(h, Float32(r))
end

"""
    Double32(x::BigFloat) -> Double32

Host method, not device code. Splits at `Float32` precision.
"""
function Double32(x::BigFloat)
    h = Float32(x)
    isfinite(h) || return raw(h, 0.0f0)
    l = Float32(x - big(h))
    return canonicalize(h, l)
end

Double32(x::Rational) = Double32(BigFloat(x, precision = 96))
Double32(x::AbstractIrrational) = Double32(BigFloat(x, precision = 96))

# No `Double32(x::Real)` catch-all: it is ambiguous against Base's
# `(::Type{T})(::Rational{S}) where {S,T<:AbstractFloat}`. Every remaining
# `Real` reaches `Double32` through `convert`/`promote` anyway.
Double32(x::AbstractFloat) = Double32(Float64(x))

Double32() = zero(Double32)

# --------------------------------------------------------------- out --------

"""
    Float32(x::Double32) -> Float32

The nearest `Float32`, which is `x.hi` by invariant 3 of design section 5.2 —
`hi` already *is* the correctly rounded `Float32` of the pair sum.
"""
@inline Base.Float32(x::Double32) = x.hi

"""
    Float64(x::Double32) -> Float64

`Float64(x.hi) + Float64(x.lo)`, correct to within one `Float64` rounding.

Exact **only** when the pair spans at most 53 bits; see
[`float64_reference_is_exact`](@ref). It is inexact both when `lo` is subnormal
under a not-tiny `hi` and when `lo` is normal but far below its permitted
bound. Use `Double32s.reference(x)` for an always-exact conversion.

The sign of a zero comes from the high word, not from the IEEE sum: `NEG_ZERO`
is `(-0.0f0, +0.0f0)` and `(-0.0f0) + (+0.0f0)` is `+0.0`. Without this,
`hash` would disagree on two pairs that `isequal` calls identical.
"""
@inline function Base.Float64(x::Double32)
    s = Float64(x.hi) + Float64(x.lo)
    return ifelse(iszero(s) & iszero(x.hi), Float64(x.hi), s)
end

Base.Float16(x::Double32) = Float16(x.hi)
Base.BigFloat(x::Double32; kw...) = big(x.hi) + big(x.lo)

Base.AbstractFloat(x::Double32) = x
Base.float(x::Double32) = x
Base.widen(::Type{Double32}) = Float64

# `<:Integer` rather than a bare `T`, and Bool excluded: `Bool(::Real)` in Base
# would otherwise be ambiguous against it.
(::Type{T})(x::Double32) where {T<:Union{Signed,Unsigned}} = T(x.hi) + T(x.lo)
Base.Bool(x::Double32) = Bool(x.hi) & iszero(x.lo)
