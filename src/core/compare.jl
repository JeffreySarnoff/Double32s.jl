# core/compare.jl — comparisons, ordering, and hashing.
#
# Design section 8.10.
#
# ===========================================================================
# THE ZERO RULE FOR TOTAL-ORDER PREDICATES
#
#   The sign of a zero low word is never information. `isequal`, `isless`, and
#   anything built on `Base.Order` must compare low words with `==` and `<`,
#   which treat -0.0f0 and +0.0f0 as equal — never with `isequal` or `isless`,
#   which do not.
#
# This follows from invariant 8 of design section 5.2: `-`, `abs`, `flipsign`,
# and `copysign` all flip the sign bit of `lo` unconditionally, so a negative
# zero low word occurs under *any* high word:
#
#   -one(Double32) === (-1.0f0, -0.0f0)      Double32(-1.0f0) === (-1.0f0, +0.0f0)
#   -D32_NAN       === ( NaN32, -0.0f0)      D32_NAN          === ( NaN32, +0.0f0)
#
# Two successive drafts of the design got this wrong. The second used
# `isequal(x.lo, y.lo)` behind an `iszero(x.hi)` guard, which reports the two
# pairs above as not `isequal` even though `==` and `hash` call them identical
# — breaking Dict lookup, `unique`, and `issorted`. Trichotomy does *not* catch
# it: exactly one of the three predicates still holds. See test/core/compare.jl.
# ===========================================================================

@inline Base.:(==)(x::Double32, y::Double32) = (x.hi == y.hi) & (x.lo == y.lo)
@inline Base.:(<)(x::Double32, y::Double32)  = (x.hi < y.hi) | ((x.hi == y.hi) & (x.lo < y.lo))
@inline Base.:(<=)(x::Double32, y::Double32) = (x.hi < y.hi) | ((x.hi == y.hi) & (x.lo <= y.lo))

@inline Base.isequal(x::Double32, y::Double32) =
    isequal(x.hi, y.hi) & (x.lo == y.lo)

@inline Base.isless(x::Double32, y::Double32) =
    isless(x.hi, y.hi) | (isequal(x.hi, y.hi) & (x.lo < y.lo))

# ---------------------------------------------------------- mixed Float64 ---
#
# `==` and `<` against a Float64 cannot go through promotion: the conversion is
# lossy for wide-gap pairs (section 4.2), so promotion would report
# `Double32(1.0f0, 2f0^-149) == 1.0` as true when the values differ. Base sets
# the precedent — `==(::Int64, ::Float64)` is deliberately exact.
#
# Verified against a 400-bit BigFloat reference over 918 operand combinations
# spanning ±0.0, ±Inf, NaN, floatmax, 2.0^-1074, and both subnormal and normal
# low words: 0 mismatches for ==, <, and >.

"""
    ==(x::Double32, y::Float64) -> Bool

Exact comparison in the pair domain, never via `Float64(x)`.
"""
@inline function Base.:(==)(x::Double32, y::Float64)
    h = Float64(x.hi)
    return (h == y) & iszero(x.lo) | (Float64(x.lo) == y - h) & isfinite(h)
end
@inline Base.:(==)(y::Float64, x::Double32) = x == y

# Ordering needs no special cases at all: the non-finite answers fall out of
# IEEE arithmetic on `y - hi`, including Inf - Inf -> NaN -> false.
#
# Exactness: |lo| <= ulp(hi)/2, so whenever the comparison is close enough to
# matter, `y - hi` is a Sterbenz subtraction and exact; outside that, faithful
# rounding preserves both the sign and the fact that |y - hi| > |lo|. Overflow
# is impossible because floatmax(Float32) is far below ulp(floatmax(Float64)).
@inline Base.:(<)(x::Double32, y::Float64) = Float64(x.lo) < y - Float64(x.hi)
@inline Base.:(<)(y::Float64, x::Double32) = y - Float64(x.hi) < Float64(x.lo)

@inline Base.:(<=)(x::Double32, y::Float64) = (x < y) | (x == y)
@inline Base.:(<=)(y::Float64, x::Double32) = (y < x) | (x == y)

@inline Base.isless(x::Double32, y::Float64) =
    isnan(x.hi) ? false :
    isnan(y)    ? true  :
    (x < y) | ((x == y) & signbit(x.hi) & !signbit(y))

@inline Base.isless(y::Float64, x::Double32) =
    isnan(y)    ? false :
    isnan(x.hi) ? true  :
    (y < x) | ((x == y) & signbit(y) & !signbit(x.hi))

@inline Base.isequal(x::Double32, y::Float64) =
    isnan(x.hi) ? isnan(y) :
    isnan(y)    ? false    :
    (x == y) & (signbit(x.hi) == signbit(y))
@inline Base.isequal(y::Float64, x::Double32) = isequal(x, y)

# Comparison against Float32, Float16, and integers *is* safe by promotion,
# since every one of those converts to Double32 exactly. Base's fallbacks plus
# the promote_rules of core/promote.jl handle them. Do not define !=, >, or >=:
# Base derives all three from == and <.

# --------------------------------------------------------------- min/max ---
#
# The obvious `ifelse(isnan(x) | (y < x), y, x)` disagrees with Base on 13 of
# the 49 combinations of ±1.0f0, ±0.0f0, ±Inf32, NaN32: it swallows a NaN
# arriving in either operand, and it cannot see signed zero. Both fixes below
# are branch free.
#
# `isnan(y.hi)` rather than `isnan(x.hi)`: when `y` is NaN the `ifelse` selects
# `y`, and when `x` is NaN every one of `lt`, `tie` is false so `x` is selected
# — NaN propagates from either side with a single test.

@inline function Base.min(x::Double32, y::Double32)
    lt  = (y.hi < x.hi) | ((y.hi == x.hi) & (y.lo < x.lo))
    tie = iszero(x.hi) & iszero(y.hi) & signbit(y.hi) & !signbit(x.hi)
    return ifelse(isnan(y.hi) | lt | tie, y, x)
end

@inline function Base.max(x::Double32, y::Double32)
    lt  = (x.hi < y.hi) | ((x.hi == y.hi) & (x.lo < y.lo))
    tie = iszero(x.hi) & iszero(y.hi) & signbit(x.hi) & !signbit(y.hi)
    return ifelse(isnan(y.hi) | lt | tie, y, x)
end

@inline Base.minmax(x::Double32, y::Double32) = (min(x, y), max(x, y))

# Do not implement minimum/maximum: they are mapreduce over min/max and Base's
# implementations are already correct.

# ------------------------------------------------------------------ hash ---

"""
    hash(x::Double32, h::UInt) -> UInt

Hashes the `Float64` conversion, which preserves Julia's expectation that
numerically equal values across types have compatible hashes.

Julia requires only `isequal(x, y) => hash(x) == hash(y)`, and that holds
because equal values round identically. Since the conversion is lossy for
wide-gap pairs (section 4.2), two *distinct* `Double32` values can share a
hash. That is a collision, which is permitted and costs a bucket comparison; it
is not a correctness defect. Do not "fix" it by hashing the two words
independently — that would break cross-type consistency, which is the property
that actually matters.
"""
Base.hash(x::Double32, h::UInt) = hash(Float64(x), h)
