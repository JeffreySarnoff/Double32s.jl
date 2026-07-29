# math/exp.jl — the exponential family.
#
# Design section 20.3, rebuilt around a 32-entry table reduction (2026-07-29).
# The original reduced by whole multiples of ln(2), leaving |r| <= ln(2)/2 and
# a 14-coefficient pair-arithmetic Horner — thirteen ~36-cycle serial steps.
# Reducing instead with k2 = round(x * 32/ln2) leaves |r| <= ln(2)/64 = 0.0108,
# where three pair steps plus a Float32 tail reach full precision:
#
#     exp(x) = 2^(k2 >> 5) * 2^(j/32) * e^r,     j = k2 & 31
#
# with 2^(j/32) a tabulated pair (`EXP32_TABLE`). Measured: 172.7 -> 43.8 ns
# per call (3.9x), worst error 0.858 -> 0.984 ulp48 on the same 4096-point
# pool. The inverse table (`EXP32_INV_TABLE`) additionally gives e^{-x} from
# the *same* reduction with no division, which is what rebuilt sinh and cosh
# (hyperbolic.jl) at 7x.
#
# `expm1` keeps a direct series on |x| <= ln 2 — wider than the old ln(2)/2
# band — and past it computes `exp(x) - 1` by subtraction, which by then
# cancels at most one bit. The old doubling-identity ladder
# (`expm1(2u) = expm1(u)*(expm1(u)+2)`, applied up to three times) measured
# 4.63 ulp48 worst; the wide band plus direct subtraction measures 1.66, and
# is one code path instead of four.
#
# Coefficients and tables come from `scripts/gen_coefficients.jl` (400-bit
# BigFloat, emitted as bit patterns).

"""
    Double32s.exp_parts(x::Double32) -> (k::Int, j::Int, r::Double32)

Table range reduction for the exponential family:
`x = (32k + j) * ln(2)/32 + r` with `j in 0..31` and `|r| <= ln(2)/64`, so
`exp(x) = 2^k * EXP32_TABLE[j+1] * e^r` and, from the same call,
`exp(-x) = 2^-k * EXP32_INV_TABLE[j+1] * e^-r`.

The subtraction is Cody–Waite through the scaled split `LN2_32_C1`/`LN2_32_C2`:
`k2 * LN2_32_C1` is **exact** in `Float32` because the split keeps 8
significand bits and `|k2| <= 4113`, so only the much smaller `C2` product is
rounded. `j = k2 & 31` and `k = k2 >> 5` agree with `fld`/`mod` for negative
`k2` because 32 is a power of two.
"""
@inline function exp_parts(x::Double32)
    k2f = round(x.hi * INVLN2_32)
    k2 = unsafe_trunc(Int, k2f)
    r = (x - k2f * LN2_32_C1) - k2f * LN2_32_C2
    return k2 >> 5, k2 & 31, r
end

"""
    Double32s.expm1_small(r::Double32) -> Double32

`exp(r) - 1` for `|r| <= ln(2)/64` (plus reduction slop): three dominant-add
pair steps over degrees 1..3 and a `Float32` muladd tail over degrees 4..7.
The tail's leading term is `r^4/4! <= 9.2e-9 < 2^-26` of the result, which is
the mixed-precision criterion of math/horner.jl; dominance in the pair steps
is `|r * p| / |c| <= 0.0109 * 3 < 0.04`. No range check, no argument checking:
the callers own both.
"""
@inline function expm1_small(r::Double32)
    z = r.hi
    t = muladd(z, muladd(z, muladd(z, EXPM1_COEFFS[7].hi, EXPM1_COEFFS[6].hi),
                         EXPM1_COEFFS[5].hi), EXPM1_COEFFS[4].hi)
    p = poly_step(EXPM1_COEFFS[3], r, Double32(t))
    p = poly_step(EXPM1_COEFFS[2], r, p)
    p = poly_step(EXPM1_COEFFS[1], r, p)
    return dmul(r, p)
end

# Shared tail of every exp-family entry point: fold the table entry and the
# kernel value into T * (1 + em1) = T + T*em1 — a single dominant-add step
# (|T*em1| <= 0.011|T|) — then scale. `ldexp` is the guarded one: k reaches
# -149, where the low word legitimately underflows.
@inline function exp_finish(k::Int, j::Int, em1::Double32)
    T = @inbounds EXP32_TABLE[j + 1]
    return ldexp(poly_step(T, T, em1), k)
end

"""
    exp(x::Double32) -> Double32

Exponential, by the 32-entry table reduction (`|r| <= ln(2)/64`) and a
three-step kernel.

Overflows to `Inf` above `x ≈ 88.72` and underflows to zero below `x ≈ -103.28`,
matching `Float32`'s range — `Double32` widens the significand, not the
exponent.

!!! note "Accuracy: measured"
    See `docs/Reports/accuracy_report.md` for the measured worst-case error in
    nominal-48-bit ulps, and `test/core/elementary.jl` for the regression that
    holds it.

# Examples
```jldoctest
julia> exp(zero(Double32))
1.0

julia> Float64(exp(Double32(1.0f0)))
2.718281828459041

julia> exp(Double32(-1000.0f0)), exp(Double32(1000.0f0))
(Double32(0.0f0, 0.0f0), Double32(Inf32, 0.0f0))
```
"""
function Base.exp(x::Double32)
    isnan(x.hi) && return x
    x.hi > 88.73f0 && return POS_INF
    x.hi < -103.3f0 && return POS_ZERO
    k, j, r = exp_parts(x)
    return exp_finish(k, j, expm1_small(r))
end

"""
    expm1(x::Double32) -> Double32

`exp(x) - 1`, accurate near zero where `exp(x) - 1` is not.

For `|x| <= ln 2` a direct series (`EXPM1_COEFFS`, degrees 0..8 in pairs,
9..15 in the `Float32` tail) is evaluated with **no reduction and no
subtraction of 1**. Past `ln 2`, `exp(x)` is at least 2 or at most 1/2, so
`exp(x) - 1` computed by subtraction cancels at most one bit and the table
path is used directly.

The previous implementation bridged the gap with the doubling identity
`expm1(2u) = expm1(u)·(expm1(u) + 2)` applied one to three times; each
application rounds, and the ladder measured **4.63 ulp48** worst. The wide
band plus direct subtraction measures **1.66** on the same pool and replaces
four code paths with two.

# Examples
```jldoctest
julia> expm1(zero(Double32))
0.0

julia> x = Double32(1.0f-7);

julia> Float64(expm1(x)) - Float64(exp(x) - 1)   # the subtraction loses digits
-3.2907060594240117e-16
```
"""
function Base.expm1(x::Double32)
    isnan(x.hi) && return x
    iszero(x.hi) && return x                       # ±0 -> ±0
    x.hi > 88.73f0 && return POS_INF
    x.hi < -103.3f0 && return -one(Double32)
    if abs(x.hi) <= 0.6931f0
        z = x.hi
        t = EXPM1_COEFFS[16].hi
        for i in 15:-1:10
            t = muladd(z, t, EXPM1_COEFFS[i].hi)
        end
        p = Double32(t)
        for i in 9:-1:1
            p = poly_step(EXPM1_COEFFS[i], x, p)
        end
        return dmul(x, p)
    end
    k, j, r = exp_parts(x)
    return exp_finish(k, j, expm1_small(r)) - one(Double32)
end

"""
    exp2(x::Double32) -> Double32

`2^x`. The reduction is exact here — `k2 = round(32x)` and `r0 = x - k2/32`
cost one exact product and one near-exact subtraction — so this is the table
path with `r = r0 * ln 2` as the only inexact reduction step.
"""
function Base.exp2(x::Double32)
    isnan(x.hi) && return x
    x.hi > 128.0f0 && return POS_INF
    x.hi < -149.0f0 && return POS_ZERO
    k2f = round(x.hi * 32.0f0)
    k2 = unsafe_trunc(Int, k2f)
    r0 = x - k2f * 0.03125f0                       # k2f/32 is exact
    r = dmul(r0, LN2)
    return exp_finish(k2 >> 5, k2 & 31, expm1_small(r))
end

"""
    exp10(x::Double32) -> Double32

`10^x`, reduced in `x` itself rather than by handing `x*ln(10)` to `exp`.

!!! warning "Why the obvious implementation is wrong"
    `exp(x * LN10)` looks correct and is not. `exp` turns an **absolute** error
    in its argument into a **relative** error in its result, and `x*ln(10)` has
    absolute error `|x|·ln(10)·2⁻⁴⁸` — growing with `x`. At `x ≈ 29.4` the
    argument is about 67.7, so the result carries `67.7·2⁻⁴⁸ ≈ 2⁻⁴²` of relative
    error. **Measured: 56.1 ulp48**, against 0.65 for `exp` itself.

    The fix is to reduce in `x`: `k2 = round(x·32·log2(10))` and
    `r0 = x - k2·log10(2)/32` through the Cody–Waite split
    `L10_32_C1`/`L10_32_C2`, giving `|r0·ln 10| <= ln(2)/64` — inside the
    kernel band, never growing with `x`. The same reasoning applies to any
    `exp`-like function reduced through a multiplication.
"""
function Base.exp10(x::Double32)
    isnan(x.hi) && return x
    x.hi > 38.6f0 && return POS_INF
    x.hi < -44.9f0 && return POS_ZERO
    k2f = round(x.hi * L10_32)
    k2 = unsafe_trunc(Int, k2f)
    r0 = (x - k2f * L10_32_C1) - k2f * L10_32_C2
    r = dmul(r0, LN10)
    return exp_finish(k2 >> 5, k2 & 31, expm1_small(r))
end
