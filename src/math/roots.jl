# math/roots.jl — cube root and reciprocal square root.
#
# Design section 20.2: a backend `Float32` result is a *seed*, never the answer.
# Both functions here take a correctly-rounded `Float32` root and spend one
# Newton correction in `Double32`, the same shape as `sqrt` and `inv`.

"""
    cbrt(x::Double32) -> Double32

Cube root, from a `Float32` seed and one Newton correction.

Defined for negative arguments — `cbrt` is odd, so the sign is split off and
restored, and no `DomainError` is possible. `cbrt(-0.0)` is `-0.0`.

The exponent is reduced to a multiple of three before the seed is taken, so the
seed is always formed on a high word in `[1, 8)` and the final `ldexp` by `k/3`
is exact. Without that reduction the `Float32` seed underflows to zero below
`2^-126` and the correction has nothing to correct.

!!! note "Accuracy: measured 0.894 nominal-48-bit ulps"
    Over 1,372 operands spanning `2^-40` to `2^40` including pairs perturbed by
    a few nominal eps, the worst observed error is **0.894 ulp48 with 14
    faithfulness violations (1.0%)** against a 400-bit `BigFloat` reference.
    That is comparable to `inv` (1.16) and better than `/` (4.00); one Newton
    step suffices and a second was not pursued. `test/core/elementary.jl`
    carries the regression.

# Examples
```jldoctest
julia> cbrt(Double32(27.0f0))
Double32(3.0f0, 0.0f0)

julia> cbrt(Double32(-8.0f0))
Double32(-2.0f0, 0.0f0)

julia> Float64(cbrt(Double32(2.0f0)))
1.2599210498948725
```
"""
function Base.cbrt(x::Double32)
    iszero(x.hi) && return x                    # ±0 -> ±0
    isfinite(x.hi) || return x                  # ±Inf, NaN pass through
    neg = signbit(x.hi)
    a = abs(x)
    k = raw_exponent(a.hi)
    k -= mod(k, 3)                              # multiple of 3, so k/3 is exact
    xs = scale2(a, -k)                          # high word into [1, 8)
    s0 = Double32(cbrt(xs.hi))
    # y = s + (x/s^2 - s)/3. One division, and the bracket is a small residual
    # so its own rounding enters the answer scaled by 1/3.
    y = s0 + (xs / (s0 * s0) - s0) / Double32(3.0f0)
    y = ldexp(y, div(k, 3))
    return neg ? -y : y
end

"""
    Double32s.rsqrt(x::Double32) -> Double32

Reciprocal square root, `1/sqrt(x)`.

**It is `sqrt(inv(x))`, and the order is the whole point.** Design section 20.2
lists reciprocal square root among the seed-and-refine candidates, so a
dedicated Newton kernel was written and measured against both compositions:

| implementation | worst ulp48 | not faithful |
|---|---|---|
| Newton, `y*(3 - x*y²)/2` | 3.611 | 444 / 1372 |
| Newton, residual form | 3.111 | 384 / 1372 |
| `inv(sqrt(x))` | 2.340 | 239 / 1372 |
| **`sqrt(inv(x))`** | **1.235** | **44 / 1372** |

The composition wins, and the *order* of the composition wins again, for a
reason worth keeping: `sqrt` **halves** the relative error of its argument,
while `inv` passes it through. Doing the reciprocal first puts `inv`'s 1.16 ulp
through that halving and adds `sqrt`'s own 0.75 — about 1.33, which is what is
observed. Doing it the other way adds the two errors instead.

So there is no dedicated kernel and no new code path: fewer operations, better
accuracy, and nothing extra to maintain.

`rsqrt(0)` is `+Inf`, `rsqrt(-0.0)` is `-Inf`, and a negative argument gives
`NaN` rather than throwing — this is a kernel-safe function.

# Examples
```jldoctest
julia> Double32s.rsqrt(Double32(4.0f0))
Double32(0.5f0, 0.0f0)

julia> Double32s.rsqrt(Double32(0.0f0))
Double32(Inf32, 0.0f0)

julia> isnan(Double32s.rsqrt(Double32(-1.0f0)))
true
```
"""
@inline rsqrt(x::Double32) = sqrt_or_nan(reciprocal(DEFAULT_POLICY, x))
