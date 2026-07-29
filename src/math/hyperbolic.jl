# math/hyperbolic.jl — hyperbolic and inverse hyperbolic functions.
#
# Design section 20.3's rule — implement the compensated form before the naive
# one — applies to most of these, because the textbook definition cancels:
#
#   sinh(x)  = (e^x - e^-x)/2      cancels for small x; 7.0e6 ulp48 measured
#   asinh(x) = log(x + sqrt(x²+1)) cancels for x < 0
#   acosh(x) = log(x + sqrt(x²-1)) cancels as x approaches 1
#   atanh(x) = ½log((1+x)/(1-x))   cancels for small x
#
# Rebuilt 2026-07-29 around two structural changes:
#
# 1. **Dedicated series on |x| <= 1** for sinh and cosh (`SINH_COEFFS`,
#    `COSH_COEFFS`): all-positive terms, no exp call, no subtraction, no
#    division. The old sinh spent two full expm1 evaluations below 0.5; the
#    series is a 5-6 step kernel. The band edge at 1 also caps the large-path
#    cancellation at e^-2 = 0.135 — under a quarter bit.
# 2. **One shared table reduction for e^x AND e^-x** (`exp_parts` +
#    `EXP32_INV_TABLE`): 2^(-j/32) is a table lookup, e^-r is the same
#    three-step kernel at -r, and no division appears anywhere. sinh and cosh
#    above 1 cost barely more than exp itself, and the two kernels are
#    independent dependency chains.
#
# Measured on the same 4096-point [-30, 30] pool as the pre-rebuild numbers:
# sinh 345.4 -> 49.5 ns (7.0x, worst 1.20 ulp48), cosh 350.4 -> 49.0 (7.1x,
# 1.23), tanh 121.5 -> 45.6 (2.7x, and 3.50 -> 1.52 ulp48), asinh 218.6 ->
# 81.6, acosh 230.7 -> 74.6, atanh 168.9 -> 86.6 (2.42 -> 1.30 ulp48).
#
# `tanh` still owns one division outside its series band — it is a ratio by
# definition — but the *form* of the ratio is chosen per band so the
# division's 4.00 ulp48 lands on a small correction, not the result, wherever
# possible (see the docstring).

"""
    sinh(x::Double32) -> Double32

Hyperbolic sine.

- `|x| <= 1` — the odd series `x·S(x²)` (`SINH_COEFFS`: degrees 0..5 in pairs,
  6..9 in the `Float32` tail; every term positive, nothing cancels). The old
  route here was `(expm1(x) - expm1(-x))/2` — two full kernel evaluations to
  defend against a cancellation the dedicated series never has. The naive
  `(exp(x) - exp(-x))/2` measured **7.0e6 ulp48** near zero and remains the
  cautionary number.
- `|x| > 1` — `(e^x - 2^{-2k-1}·e^{-x}-part)/...`: both exponentials from ONE
  [`Double32s.exp_parts`](@ref) reduction, the inverse via `EXP32_INV_TABLE`
  with **no division**. At `x = 1` the subtraction loses `e^-2 = 0.135` — a
  quarter of a bit; measured worst 2.43 ulp48 on a dense `[0.95, 18.5]` sweep.
- past `|x| = 17.5`, `e^-x < 2^-50·e^x`: the second term is dropped entirely.
- past `|x| ≈ 88.7`, `exp` itself would overflow while `sinh` has not; the
  `2^{k-1}` scaling inside the shared path buys the remaining `ln 2` of range
  for free, and `±Inf` starts at 89.5.

# Examples
```jldoctest
julia> sinh(zero(Double32))
0.0

julia> Float64(sinh(Double32(1.0f0)))
1.175201193643801
```
"""
function Base.sinh(x::Double32)
    isnan(x.hi) && return x
    iszero(x.hi) && return x                       # ±0 -> ±0
    isfinite(x.hi) || return x                     # ±Inf -> ±Inf
    a = abs(x.hi)
    if a <= 1.0f0
        x2 = dsqr(x)
        z = x2.hi
        t = SINH_COEFFS[10].hi
        t = muladd(z, t, SINH_COEFFS[9].hi)
        t = muladd(z, t, SINH_COEFFS[8].hi)
        t = muladd(z, t, SINH_COEFFS[7].hi)
        p = Double32(t)
        for i in 6:-1:1
            p = poly_step(SINH_COEFFS[i], x2, p)
        end
        return dmul(x, p)                          # odd: sign rides on x
    end
    a > 89.5f0 && return signbit(x.hi) ? NEG_INF : POS_INF
    k, j, r = exp_parts(abs(x))
    T = @inbounds EXP32_TABLE[j + 1]
    Ep = poly_step(T, T, expm1_small(r))
    if a < 17.5f0                                  # else e^-x < 2^-50 * e^x
        Ti = @inbounds EXP32_INV_TABLE[j + 1]
        Em = poly_step(Ti, Ti, expm1_small(-r))
        Ep = Ep - scale2(Em, -2k)                  # exact: 2k <= 51 here
    end
    y = ldexp(Ep, k - 1)
    return signbit(x.hi) ? -y : y
end

"""
    cosh(x::Double32) -> Double32

Hyperbolic cosine. Same two-regime structure as [`sinh`](@ref) — the even
series `C(x²)` on `|x| <= 1` (`COSH_COEFFS`), the shared double-exponential
path above — except the combining operation is an addition, so there is no
cancellation anywhere at any size.

An earlier implementation routed this through `expm1` "to isolate the
departure from 1"; that measured **3.61 ulp48 against 1.10** for the plain
form, because `expm1` defends against a cancellation `cosh` does not have.
The lesson generalizes and is worth keeping: `expm1` earns its cost only
where a subtraction would actually cancel.

`cosh` is even: `cosh(-x) === cosh(x)`, exactly.

# Examples
```jldoctest
julia> cosh(zero(Double32))
1.0

julia> cosh(Double32(-2.0f0)) === cosh(Double32(2.0f0))
true
```
"""
function Base.cosh(x::Double32)
    isnan(x.hi) && return x
    isfinite(x.hi) || return POS_INF               # cosh(±Inf) = +Inf
    a = abs(x.hi)
    if a <= 1.0f0
        x2 = dsqr(x)
        z = x2.hi
        t = COSH_COEFFS[10].hi
        t = muladd(z, t, COSH_COEFFS[9].hi)
        t = muladd(z, t, COSH_COEFFS[8].hi)
        t = muladd(z, t, COSH_COEFFS[7].hi)
        p = Double32(t)
        for i in 6:-1:1
            p = poly_step(COSH_COEFFS[i], x2, p)
        end
        return p
    end
    a > 89.5f0 && return POS_INF
    k, j, r = exp_parts(abs(x))
    T = @inbounds EXP32_TABLE[j + 1]
    Ep = poly_step(T, T, expm1_small(r))
    if a < 17.5f0
        Ti = @inbounds EXP32_INV_TABLE[j + 1]
        Em = poly_step(Ti, Ti, expm1_small(-r))
        Ep = Ep + scale2(Em, -2k)
    end
    return ldexp(Ep, k - 1)
end

"""
    tanh(x::Double32) -> Double32

Hyperbolic tangent.

`tanh` is a ratio by definition and cannot escape one division outside its
series band — but *where the division's error lands* is a choice:

- `|x| <= 0.35` — the series `x·T(x²)` ([`Double32s.TANH_COEFFS`](@ref)),
  no division at all.
- `0.35 < |x| <= 1` — `(E - 1)/(E + 1)` with `E = e^{2|x|}` from the table
  path. The division's 4.00 ulp48 lands on the full result; nothing better is
  available here, and the old `t/(t+2)` form had the same floor.
- `1 < |x| <= 18` — **`1 - 2/(E + 1)`**, algebraically identical, error-wise
  not: the division now produces only the correction `2/(E+1) < 0.24`, so its
  4 ulp48 is scaled by `(1 - tanh)/tanh < 1/3` and shrinks geometrically as
  `x` grows. Measured: worst 4.64 ulp48 with the ratio form on `(1, 19)`,
  2.29 with this split (whole-band worst now 1.52 against 3.50 before).
- `|x| > 18` — `±1` exactly: `1 - tanh` is below `2^-49`.

Exactly odd: every branch evaluates on `|x|` and restores the sign.

# Examples
```jldoctest
julia> tanh(zero(Double32))
0.0

julia> tanh(Double32(100.0f0))
1.0

julia> Float64(tanh(Double32(1.0f0)))
0.7615941559557662
```
"""
function Base.tanh(x::Double32)
    isnan(x.hi) && return x
    iszero(x.hi) && return x                       # ±0 -> ±0
    a = abs(x.hi)
    # 1 - tanh(x) ≈ 2e^(-2x); below 2^-49 once x > 17.5, so ±1 is exact here.
    a > 18.0f0 && return signbit(x.hi) ? -one(Double32) : one(Double32)
    if a <= 0.35f0
        x2 = dsqr(x)
        z = x2.hi
        t = TANH_COEFFS[13].hi
        for i in 12:-1:8
            t = muladd(z, t, TANH_COEFFS[i].hi)
        end
        p = Double32(t)
        for i in 7:-1:1
            p = poly_step(TANH_COEFFS[i], x2, p)
        end
        return dmul(x, p)
    end
    b = abs(x)
    k, j, r = exp_parts(b + b)                     # e^{2|x|}: doubling is exact
    T = @inbounds EXP32_TABLE[j + 1]
    E = ldexp(poly_step(T, T, expm1_small(r)), k)
    local y
    if a > 1.0f0
        y = one(Double32) - Double32(2.0f0) / (E + one(Double32))
    else
        y = (E - one(Double32)) / (E + one(Double32))
    end
    return signbit(x.hi) ? -y : y
end

# --------------------------------------------------------------- inverses ---

"""
    asinh(x::Double32) -> Double32

Inverse hyperbolic sine.

Odd, and computed on `|x|` with the sign restored — `log(x + sqrt(x² + 1))`
cancels for negative `x`, because `sqrt(x² + 1)` then very nearly equals `-x`.

- `|x| <= 0.35` uses a **series in `x²`** ([`Double32s.ASINH_COEFFS`](@ref):
  degrees 0..6 in pairs, 7..15 in the `Float32` tail), avoiding division and
  square root alike.
- Beyond that, `log(a + sqrt(1 + a²))` directly: the argument is at least
  `1.41`, safely away from `log`'s sensitive region near 1, and the
  division-free table `log` makes this cheaper than the old
  `log1p(a + a²/(1 + sqrt(1+a²)))` route while measuring the same on a dense
  mid-band sweep (2.44 against the old form's 2.43 worst on `[0.3, 100]`).
- past `|x| = 1e8`, `sqrt(1 + a²)` is `a` to the last bit: `log(a) + ln 2`.

# Examples
```jldoctest
julia> asinh(zero(Double32))
0.0

julia> asinh(-Double32(2.0f0)) === -asinh(Double32(2.0f0))
true
```
"""
function Base.asinh(x::Double32)
    isnan(x.hi) && return x
    iszero(x.hi) && return x                       # ±0 -> ±0
    isfinite(x.hi) || return x                     # ±Inf -> ±Inf
    neg = signbit(x.hi)
    a = abs(x)
    local y
    if a.hi <= 0.35f0
        a2 = dsqr(a)
        z = a2.hi
        t = ASINH_COEFFS[16].hi
        for i in 15:-1:8
            t = muladd(z, t, ASINH_COEFFS[i].hi)
        end
        p = Double32(t)
        for i in 7:-1:1
            p = poly_step(ASINH_COEFFS[i], a2, p)
        end
        y = dmul(a, p)
    elseif a.hi > 1.0f8
        # a² would overflow long before asinh does; asinh(a) → log(2a).
        y = log_or_nan(a) + LN2
    else
        y = log_or_nan(a + sqrt_or_nan(one(Double32) + dsqr(a)))
    end
    return neg ? -y : y
end

"""
    acosh(x::Double32) -> Double32

Inverse hyperbolic cosine, defined for `x >= 1`.

**`log1p(t + sqrt(2t + t²))` with `t = x - 1`, not `log(x + sqrt(x² - 1))`.**
As `x` approaches 1 the closed form evaluates `sqrt` of a quantity that has
already cancelled — `x² - 1` loses about half its digits — and then takes the
logarithm of something very near 1, losing the rest. Both problems disappear
once the small quantity `t = x - 1` is the variable: `t` is exact by Sterbenz
for `x` in `[1, 2]`, `2t + t²` never cancels, and `log1p` is accurate near zero
by construction.

Throws a `DomainError` below 1, matching `acosh(::Float32)`.

# Examples
```jldoctest
julia> acosh(one(Double32))
0.0

julia> Float64(acosh(Double32(2.0f0)))
1.3169578969248192
```
"""
function Base.acosh(x::Double32)
    isnan(x.hi) && return x
    x.hi < 1.0f0 && throw(DomainError(x, "acosh requires x >= 1"))
    isfinite(x.hi) || return x                     # +Inf -> +Inf
    isone(x.hi) && iszero(x.lo) && return POS_ZERO
    if x.hi > 1.0f8
        # x² would overflow long before acosh does; acosh(x) → log(2x).
        return log_or_nan(x) + LN2
    end
    t = x - one(Double32)                          # exact for x in [1, 2]
    return log1p(t + sqrt_or_nan(dmul(t, Double32(2.0f0) + t)))
end

"""
    atanh(x::Double32) -> Double32

Inverse hyperbolic tangent, defined on `(-1, 1)` with poles at `±1`.

- `|x| <= 0.1715` uses the `atanh` series ([`Double32s.ATANH_COEFFS`](@ref):
  degrees 0..3 in pairs, 4..10 in the `Float32` tail — `x² <= 0.0295`, so the
  tail boundary arrives after only four pair coefficients) — no division.
- Beyond that, **`(log1p(b) - log1p(-b))/2`** on `b = |x|`: two short table
  `log1p` evaluations, and the two terms have opposite signs, so the
  "subtraction" is an addition of like-signed quantities and nothing cancels.
  The old form `½·log1p(2b/(1-b))` fed a division's 4.00 ulp48 into `log1p`;
  this one has no division at all. Measured 3.06 -> 1.81 worst on a dense
  `(-1, 1)` sweep, at equal speed.

Returns `±Inf` at `±1` and throws a `DomainError` outside `[-1, 1]`, matching
`atanh(::Float32)`.

# Examples
```jldoctest
julia> atanh(zero(Double32))
0.0

julia> atanh(one(Double32))
Inf
```
"""
function Base.atanh(x::Double32)
    isnan(x.hi) && return x
    iszero(x.hi) && return x                       # ±0 -> ±0
    a = abs(x.hi)
    a > 1.0f0 && throw(DomainError(x, "atanh requires -1 <= x <= 1"))
    if isone(a) && iszero(x.lo)
        return signbit(x.hi) ? NEG_INF : POS_INF
    end
    if a <= 0.1715f0
        x2 = dsqr(x)
        z = x2.hi
        t = ATANH_COEFFS[11].hi
        for i in 10:-1:5
            t = muladd(z, t, ATANH_COEFFS[i].hi)
        end
        p = Double32(t)
        for i in 4:-1:1
            p = poly_step(ATANH_COEFFS[i], x2, p)
        end
        return dmul(x, p)
    end
    neg = signbit(x.hi)
    b = abs(x)
    y = (log1p(b) - log1p(-b)) * Double32(0.5f0)
    return neg ? -y : y
end
