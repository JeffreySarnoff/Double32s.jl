# math/reciprocal.jl — inv.
#
# Design section 8.5.

"""
    Double32s.reciprocal(policy, y::Double32) -> Double32

Layer-2 reciprocal.

`inv_dd_` forms `1 - q0*y`, whose residuals sit near `2^0` whatever the
operand's exponent, so unlike [`Double32s.divide`](@ref) this needs no scaling
for accuracy. It needs it only for *range*: `inv(y.hi)` overflows once
`exponent(y.hi) <= -128`, which loses the whole computation. Scaling only ever
goes up, and that direction is exact.

A large operand is left alone deliberately. The correction `q1` then underflows
to zero, but the result's low word would sit below `2f0^-149` anyway — that is
the documented precision taper of design section 5.5, not a defect.
"""
@inline function reciprocal(::ArithmeticPolicy, y::Double32)
    (isfinite(y.hi) & !iszero(y.hi)) || return canonical_nonfinite(inv(y.hi))
    k  = max(0, INV_SCALE_FLOOR - raw_exponent(y.hi))
    ys = scale2(y, k)                      # exact: scaling up never underflows
    h, l = inv_dd_(words(ys))
    isfinite(h) || return canonical_nonfinite(h)
    iszero(k) && return raw(h, l)          # the common case pays nothing
    return ldexp(raw(h, l), k)             # guarded: may overflow
end

"""
    inv(x::Double32) -> Double32

Reciprocal: one leading estimate plus one compensated correction.

The correction is **not** skipped when `x.lo == 0`. A one-word input such as
`3.0f0` still needs it to reach double-word precision; that skip is a tempting
"optimization" precisely because it looks like an algebraic identity.

Accuracy, measured against a 400-bit `BigFloat` reference over lattice-range
operands: worst observed error **1.16 nominal-48-bit ulps**, one faithfulness
violation in 319 structured operands — marginally past faithful, and a second
correction does not reach faithfulness for the analogous division kernel, so
the same GPU-driven single-correction decision applies (see
[`div_dd_refined_`](@ref)).

Note that `x / y` is **not** defined as `x * inv(y)`: that is one rounding
worse, and users reasonably assume `/` is the accurate operation.
"""
Base.inv(x::Double32) = reciprocal(DEFAULT_POLICY, x)
