# math/powers.jl — hypot and integer powers.
#
# Integer `^` needs no method: Base's `power_by_squaring` reaches it through
# `*` and `one`. Only the pieces Base cannot derive live here.

"""
    hypot(x::Double32, y::Double32) -> Double32

`sqrt(x^2 + y^2)` computed with an exact power-of-two scaling, so that the
intermediate squares neither overflow nor underflow prematurely.

The scale factor is `2^-exponent(max(|x|,|y|))`, an exact power of two, so
scaling introduces no error of its own.
"""
function Base.hypot(x::Double32, y::Double32)
    (isinf(x.hi) | isinf(y.hi)) && return POS_INF     # IEEE: hypot(Inf, NaN) == Inf
    (isnan(x.hi) | isnan(y.hi)) && return D32_NAN
    ax, ay = abs(x), abs(y)
    big_hi = max(ax.hi, ay.hi)
    iszero(big_hi) && return POS_ZERO
    k  = -raw_exponent(big_hi)
    xs = ldexp(ax, k)
    ys = ldexp(ay, k)
    s  = add(DEFAULT_POLICY, mul(DEFAULT_POLICY, xs, xs), mul(DEFAULT_POLICY, ys, ys))
    return ldexp(sqrt_or_nan(s), -k)
end

Base.hypot(x::Double32, y::Real) = hypot(x, Double32(y))
Base.hypot(x::Real, y::Double32) = hypot(Double32(x), y)

# `literal_pow` is defined only for the two exponents that are hot enough to
# matter and that Base would otherwise route through `power_by_squaring`.
@inline Base.literal_pow(::typeof(^), x::Double32, ::Val{0}) = one(Double32)
@inline Base.literal_pow(::typeof(^), x::Double32, ::Val{1}) = x
@inline Base.literal_pow(::typeof(^), x::Double32, ::Val{2}) = x * x
@inline Base.literal_pow(::typeof(^), x::Double32, ::Val{3}) = (x * x) * x
@inline Base.literal_pow(::typeof(^), x::Double32, ::Val{-1}) = inv(x)

"""
    ^(x::Double32, n::Integer) -> Double32

Integer power.

A method is needed for the **negative** case and only for it. Base's
`power_by_squaring` refuses a negative exponent for any type it cannot see is a
float — `Double32(2.0f0)^-30` raised
`DomainError: Cannot raise an integer x to a negative power` until this existed.
Non-negative exponents still go straight to `power_by_squaring`.

`-n` is widened before negation so that `n == typemin(Int)` does not overflow
into itself.

# Examples
```jldoctest
julia> Double32(2.0f0)^10
1024.0

julia> Double32(2.0f0)^-30
9.31322574615478515625e-10
```
"""
function Base.:^(x::Double32, n::Integer)
    n >= 0 && return Base.power_by_squaring(x, n)
    return Base.power_by_squaring(inv(x), -widen(n))
end

"""
    ^(x::Double32, y::Double32) -> Double32

General power, as `exp(y * log(x))`.

Accuracy follows from the composition: `log` contributes its own error, and then
`exp` converts the **absolute** error of `y*log(x)` into relative error, so the
result degrades as `|y*log(x)|` grows — the same amplification documented on
[`exp10`](@ref). For an integer `y`, use the integer method above, which does
not go through `log` at all.

Follows IEEE `pow` on the special cases that have an unambiguous answer:
`x^0 == 1` for every `x` including `NaN`, `1^y == 1`, and a negative base is
`NaN` unless `y` is an integer.
"""
function Base.:^(x::Double32, y::Double32)
    iszero(y.hi) && iszero(y.lo) && return one(Double32)
    isone(x.hi) && iszero(x.lo) && return one(Double32)
    (isnan(x.hi) | isnan(y.hi)) && return D32_NAN
    if signbit(x.hi) && !iszero(x.hi)
        # A negative base is defined only for integer exponents.
        yr = round(y.hi)
        (yr == y.hi && iszero(y.lo) && isfinite(y.hi)) || return D32_NAN
        m = unsafe_trunc(Int, yr)
        p = abs(x)^y
        return isodd(m) ? -p : p
    end
    iszero(x.hi) && return signbit(y.hi) ? POS_INF : POS_ZERO
    return exp(y * log_or_nan(x))
end

# No `^(::Double32, ::Real)` or `^(::Real, ::Double32)` method. Both were
# written, and both were ambiguous — against `^(::T, ::Rational) where
# T<:AbstractFloat` and against `^(::Irrational{:e}, ::Number)` respectively.
# This is design section 11.3 rule 2 ("never bound at `<:Real`") generalized
# past `promote_rule`: Base's `^(x::Number, y::Number)` already promotes, and
# with the promotion lattice in place `Double32^Float32` reaches
# `Double32^Double32` on its own. The methods bought nothing and cost two
# ambiguities.
