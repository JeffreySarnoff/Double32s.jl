# math/sqrt.jl — square root.
#
# Design section 8.6.

"""
    Double32s.square_root(policy, x::Double32) -> Double32

Layer-2 square root. Non-throwing; see [`sqrt_or_nan`](@ref), which this is.
"""
@inline square_root(::ArithmeticPolicy, x::Double32) = sqrt_or_nan(x)

"""
    Double32s.sqrt_or_nan(x::Double32) -> Double32

Square root that returns `NaN` for a negative argument instead of throwing.

**This is the spelling device kernels should use.** `Base.sqrt` throws a
`DomainError` to match `sqrt(::Float32)`, and a `throw` in a device kernel is
not free: it forces the exception machinery into the generated code and can
abort the kernel.

The operand is always scaled to an even exponent near zero first, and that is
not an optimization to be removed. `sqrt_dd_` forms the residual `x - s0*s0`,
whose magnitude is about `|x| * 2^-48`; below `exponent(x.hi) ≈ -101` that
residual falls under `2f0^-149` and is silently lost even though the *result*
remains fully representable. Measured before scaling was added:
`sqrt(Double32(2.0165723f-36, 8.0f-44))` had relative error `2.4e-11`, about 35
correct bits instead of 48. An even shift keeps both the scaling and its
inverse exact.

Exceptional values follow IEEE and Base:

```text
sqrt(+0)   -> +0        sqrt(-0)   -> -0
sqrt(+Inf) -> +Inf      sqrt(x<0)  -> NaN
sqrt(NaN)  -> NaN
```
"""
function sqrt_or_nan(x::Double32)
    iszero(x.hi)   && return x               # ±0 -> ±0, sign preserved
    signbit(x.hi)  && return D32_NAN         # negative finite, -Inf, -NaN
    isfinite(x.hi) || return x               # +Inf -> +Inf, +NaN -> +NaN
    k  = raw_exponent(x.hi)
    k -= k & 1                               # even, so k >> 1 is exact
    xs = scale2(x, -k)                       # high word into [1, 4)
    h, l = sqrt_dd_(HILO(xs))
    isfinite(h) || return canonical_nonfinite(h)
    return ldexp(raw(h, l), k >> 1)
end

"""
    sqrt(x::Double32) -> Double32

Square root, one Newton correction on the `Float32` seed.

Throws a `DomainError` for a negative argument, matching `sqrt(::Float32)`.
Use [`Double32s.sqrt_or_nan`](@ref) in kernels, where a `throw` is expensive.

!!! note "Faithfulness: measured, and the single correction suffices"
    Design experiment 10 asked whether `sqrt` needs a second refinement. It
    does not: over lattice-range operands including the near-perfect-square
    adversaries of design section 23.8, the worst observed error is **0.75
    nominal-48-bit ulps with zero faithfulness violations** (400-bit `BigFloat`
    reference; `test/core/arithmetic.jl` carries the regression). The result is
    empirically faithful; no analytic bound is claimed. This mirrors the
    standard analysis shape — the Newton step for `sqrt` contracts error
    quadratically from a correctly rounded seed, unlike division's, which is
    why `sqrt` reaches faithfulness where `/` (4.00 ulps worst) does not.
"""
function Base.sqrt(x::Double32)
    signbit(x.hi) && !iszero(x.hi) && !isnan(x.hi) && throw(DomainError(x,
        "sqrt was called with a negative real argument; use sqrt(Complex(x))"))
    return sqrt_or_nan(x)
end
