# core/constants.jl — canonical exceptional values and precision constants.
#
# Design sections 6.5 and 11.2. Spelled with Base's own literals rather than
# bit patterns; lowercase unexported names inside the module are already
# unambiguous.

"""
    Double32s.POS_ZERO

Canonical `+0.0`, the pair `(+0.0f0, +0.0f0)`.
"""
const POS_ZERO = raw(+0.0f0, +0.0f0)

"""
    Double32s.NEG_ZERO

Canonical `-0.0`, the pair `(-0.0f0, +0.0f0)`.
"""
const NEG_ZERO = raw(-0.0f0, +0.0f0)

"""
    Double32s.POS_INF

Canonical `+Inf`, the pair `(+Inf32, +0.0f0)`.
"""
const POS_INF = raw(+Inf32, +0.0f0)

"""
    Double32s.NEG_INF

Canonical `-Inf`, the pair `(-Inf32, +0.0f0)`.
"""
const NEG_INF = raw(-Inf32, +0.0f0)

"""
    Double32s.D32_NAN

The package's canonical quiet NaN, the pair `(NaN32, +0.0f0)`.
`NaN32 === reinterpret(Float32, 0x7fc00000)`.
"""
const D32_NAN = raw(NaN32, +0.0f0)

# Shift used by the scaled overflow-rescue paths of design section 7.5.
#
# 64 is chosen so that a single constant serves both operands of `*` and both
# of `/`, and it is provably enough: `x.hi * y.hi` overflows only when
# `exponent(x.hi) + exponent(y.hi) >= 127`, and since neither exceeds 127 that
# forces both to be non-negative — so scaling each down by 2^64 leaves both
# high words normal. The same bound run backwards covers `/`.
const RESCUE_SHIFT = 64

# Exponent floor for the scaled division and reciprocal paths of section 8.4.
#
# `div_dd_` and `inv_dd_` form residuals of order `2^-48` relative to the
# numerator, and a further `2^-24` inside `two_prod_`. Both must stay above
# `2f0^-149`, so the numerator's exponent must stay above about `-101`; `-78`
# leaves the deepest residual at `2f0^-126`, comfortably normal, with room to
# spare on backends that flush subnormals.
const SCALE_FLOOR = -78

# Reciprocals only need scaling when `inv(y.hi)` itself would overflow, which
# happens for `exponent(y.hi) <= -128`.
const INV_SCALE_FLOOR = -120

# Underflow trigger for the multiplication guard (section 8.3).
#
# The multiplication kernel is silently degraded once the product exponent
# drops below about -100: the `two_prod_` residual needs `e_p >= -101` to stay
# representable (a nonzero residual of a 48-bit product is >= 2^(e_p - 47)),
# and the cross-term `fma` starts rounding at subnormal granularity 2^-149,
# which exceeds u^2 relative once `e_p < -101`. Measured, pre-guard: relative
# error 8.2e-8 (2.3e7 u^2, about 23 bits) for products near 2^-127.
#
# The trigger tests the *output* high word, which sits within a binade of the
# product, so the threshold carries a two-binade margin. Over-triggering costs
# one cold-path recomputation in a rare band; under-triggering is a silent
# wrong answer.
const MUL_TINY = 2f0^-98

"""
    Double32s.nominal_eps(::Type{Double32}) -> Double32

The nominal 48-bit unit in the last place, `2.0^-47`.

This is an algorithmic precision constant, **not** the distance to the next
representable pair. Design section 5.6.
"""
nominal_eps(::Type{Double32}) = ldexp(one(Double32), -47)

"""
    Double32s.unit_roundoff(::Type{Double32}) -> Double32

The nominal unit roundoff `2.0^-48`, i.e. half [`nominal_eps`](@ref).
"""
unit_roundoff(::Type{Double32}) = ldexp(one(Double32), -48)

Base.zero(::Type{Double32})    = POS_ZERO
Base.one(::Type{Double32})     = raw(1.0f0, 0.0f0)
Base.oneunit(::Type{Double32}) = one(Double32)
Base.typemin(::Type{Double32}) = NEG_INF
Base.typemax(::Type{Double32}) = POS_INF

Base.zero(::Double32) = zero(Double32)
Base.one(::Double32)  = one(Double32)

Base.floatmin(::Type{Double32}) = raw(floatmin(Float32), 0.0f0)

"""
    floatmax(::Type{Double32}) -> Double32

The largest finite `Double32`.

Derivation, since the constant is otherwise unauditable:
`ulp(floatmax(Float32)) == 2^104`, so the canonical bound is `|lo| <= 2^103`.
The significand of `floatmax(Float32)` (`0x7f7fffff`) is odd, so a low word of
exactly `2^103` is a tie that rounds *up* and overflows.
`prevfloat(ldexp(1.0f0, 103))` is therefore the largest admissible low word.
"""
Base.floatmax(::Type{Double32}) = raw(floatmax(Float32), prevfloat(ldexp(1.0f0, 103)))

Base.precision(::Type{Double32}; base::Integer=2) = _precision_in_base(48, base)

function _precision_in_base(bits::Int, base::Integer)
    base > 1 || throw(DomainError(base, "`base` must be greater than 1."))
    base == 2 && return bits
    return Int(floor(bits * log(2.0) / log(Float64(base))))
end

"""
    eps(::Type{Double32}) -> Double32

The nominal 48-bit unit in the last place, `2.0^-47`.

!!! warning
    `Double32` is a double-word format, not an IEEE format with a uniform
    significand lattice. This value is the algorithmic precision constant
    appropriate for tolerances and convergence tests; it is **not** in general
    the distance to the next representable pair, which can be as small as
    `2.0f0^-149` relative to a high word of `1.0f0`. `nextfloat` and
    `prevfloat` are deliberately not implemented.
"""
Base.eps(::Type{Double32}) = nominal_eps(Double32)

"""
    eps(x::Double32) -> Double32

The nominal ulp at `x`: `ldexp(eps(Double32), exponent(x))`.

Returns the smallest positive `Double32` for zero and `NaN` for a non-finite
argument, so the method is total where `exponent` is not.
"""
function Base.eps(x::Double32)
    isfinite(x.hi) || return D32_NAN
    iszero(x.hi)   && return raw(nextfloat(0.0f0), 0.0f0)
    return ldexp(nominal_eps(Double32), raw_exponent(x.hi))
end
