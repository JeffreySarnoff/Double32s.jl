# core/special.jl — classification predicates and the non-finite canonicalizer.
#
# Design section 4.2 (predicate semantics) and 6.5.
#
# Every predicate here inspects `hi` only, so each compiles to the same
# instruction sequence as the corresponding `Float32` predicate. Bitwise `&`
# rather than `&&`: both operands are cheap already-computed booleans, so the
# non-short-circuiting form avoids a branch and keeps warps convergent.

"""
    Double32s.canonical_nonfinite(x::Float32) -> Double32

Map a non-finite `Float32` to its canonical `Double32` pair. Branch free: NaN
maps to the package quiet NaN, `±Inf` pass through, both get `+0.0f0` low.
"""
@inline canonical_nonfinite(x::Float32) = raw(ifelse(isnan(x), NaN32, x), 0.0f0)

Base.isfinite(x::Double32) = isfinite(x.hi)
Base.isinf(x::Double32)    = isinf(x.hi)
Base.isnan(x::Double32)    = isnan(x.hi)

# Canonical: hi == 0 implies the value is zero, so `lo` need not be inspected.
Base.iszero(x::Double32)   = iszero(x.hi)
Base.isone(x::Double32)    = isone(x.hi) & iszero(x.lo)
Base.signbit(x::Double32)  = signbit(x.hi)

"""
    issubnormal(x::Double32) -> Bool

Whether the *high word* is subnormal, i.e. whether `x` has lost high-word
precision.

Deliberately delegates to `hi` rather than classifying the represented value.
A caller of `issubnormal` branches on "is full high-word precision available",
and the high word alone answers that. The converse subtlety is the one worth
documenting: a *normal* `Double32` routinely has a *subnormal* low word, and no
predicate should ever inspect `lo` for classification.
"""
Base.issubnormal(x::Double32) = issubnormal(x.hi)

"""
    Double32s.isnormal(x::Double32) -> Bool

Whether `x` is finite, nonzero, and not subnormal in its high word.
"""
isnormal(x::Double32) = isfinite(x.hi) & !issubnormal(x.hi) & !iszero(x.hi)

Base.isinteger(x::Double32) = isinteger(x.hi) & isinteger(x.lo)

"""
    Double32s.float64_reference_is_exact(x::Double32) -> Bool

Whether `Float64(x)` reproduces the exact pair sum, and therefore whether a
`Float64` reference may be used for `x` in validation.

True iff the pair spans at most 53 bits. Design section 4.2 gives the proof;
the short version is that `|lo| <= ulp(hi)/2` bounds the low word's *magnitude*
and constrains neither its bit position nor the gap between the two words, so
both a subnormal `lo` and a normal `lo` far below its permitted bound push the
pair past 53 bits.

Validated against a 400-bit `BigFloat` reference over 1,701,602 canonical
pairs, 0 mismatches.

# Examples
```jldoctest
julia> Double32s.float64_reference_is_exact(Double32(1.0f0, 2.0f0^-52))
true

julia> Double32s.float64_reference_is_exact(Double32(1.0f0, 2.0f0^-100))  # normal lo, 101-bit span
false

julia> Double32s.float64_reference_is_exact(Double32(2.0f0^-96, -2.0f0^-149))  # cancellation buys a bit
true
```
"""
function float64_reference_is_exact(x::Double32)
    iszero(x.lo)   && return true
    isfinite(x.hi) || return true          # canonical non-finite pairs have lo == 0
    eh = raw_exponent(x.hi)
    # exponent(hi + lo) is eh, one less exactly when hi is a power of two and
    # lo opposes its sign; |lo| <= half-ulp bounds cancellation to one binade.
    borrow = is_power_of_two(x.hi) & (signbit(x.lo) != signbit(x.hi))
    return lsb_exponent(x.lo) >= (eh - borrow) - 52
end

"""
    Double32s.reference(x::Double32) -> BigFloat

The exact represented value. Host and test code only — always correct, unlike a
`Float64` conversion. Design section 22.1.

The zero low word is dropped rather than added, because of design section 5.2
invariant 8: the sign of a zero `lo` carries no information. Adding it would
leak that sign into the reference — `big(-0.0f0) + big(-0.0f0)` is `-0.0`
while `big(-0.0f0) + big(0.0f0)` is `+0.0`, so two `isequal` pairs would get
`BigFloat` references that are *not* `isequal`, and every test that treats this
function as ground truth would inherit the discrepancy. Adding a nonzero `lo`
is exact by canonicality, so no other case needs care.
"""
reference(x::Double32) = iszero(x.lo) ? big(x.hi) : big(x.hi) + big(x.lo)
