# math/log.jl — the logarithm family.
#
# Design section 20.3, rebuilt division-free (2026-07-29). The original
# evaluated `log(m) = 2*atanh(s)` with `s = (m-1)/(m+1)` — one full pair
# division (this package's `/` carries 4.00 ulp48) plus a 10-step pair Horner.
# The rebuild removes the division entirely with a 23-bin table over the
# reduced mantissa `m in [sqrt2/2, sqrt2)`:
#
#     i = round(m * 32),   c = fl32(32/i),   r = m*c - 1,   |r| <= 0.0222
#     log(x) = k*ln(2) + (-log c) + log1p(r)
#
# `-log c` is tabulated per base (`LOG_TABLE_LN`/`_L2`/`_L10`), and `r` is
# computed **exactly** — see `mulm1` below, whose exactness is the load-bearing
# step of the whole scheme. `log1p(r)` is a 4-step pair Horner plus Float32
# tail (`LOG1P_COEFFS`).
#
# Measured: log 145.0 -> 41.0 ns (3.5x); worst error near 1 — where the old
# division error was amplified — 4.66 -> 1.59 ulp48 on a dense [0.70, 1.45]
# sweep. The bin containing m = 1 has c = 1 exactly, so r = m - 1 is exact and
# `log` keeps full relative accuracy as the result crosses zero.

"""
    Double32s.mulm1(m::Double32, c::Float32) -> Double32

`m*c - 1`, **exactly**, for `m*c in (0.978, 1.023)`. Both partial products are
split with `two_prod_` (each exact), `fl(m.hi*c) - 1` is Sterbenz-exact in
that interval, and the small pieces are combined with `two_sum_` chains whose
own roundings sit at `2^-54` absolute.

The care is not optional. Folding the low pieces with one fma —
`fma(m.lo, c, e1)` — rounds at `2^-48` **absolute** (the fma's result sits at
the `2^-24` scale), and the whole point of the table reduction is that `r`
then cancels down to `~2^-5`: against the smallest result a bin can produce
(`|log m| ≈ 0.0157` at the boundary of the bin containing 1) that is a
derived ~60 ulp48. The exact form measures 1.59 worst over a dense
`[0.70, 1.45]` sweep.
"""
@inline function mulm1(m::Double32, c::Float32)
    p1, e1 = two_prod_(m.hi, c)
    p2, e2 = two_prod_(m.lo, c)
    rh = p1 - 1.0f0                                # Sterbenz-exact
    l1, f1 = two_sum_(e1, p2)
    s, t = two_sum_(rh, l1)
    return raw(two_sum_(s, (t + f1) + e2)...)
end

"""
    Double32s.log1p_small(r::Double32) -> Double32

`log1p(r)` for `|r| <= 0.0222`, relative-accurate: `r * P(r)` with
`LOG1P_COEFFS` degrees 0..4 as dominant-add pair steps (dominance
`|r*p|/|c| <= 0.0222 * 1.3 < 0.03`) and degrees 5..9 as the `Float32` tail
(leading tail term `r^5/6 <= 8.9e-10 < 2^-26` of `P ≈ 1`).
"""
@inline function log1p_small(r::Double32)
    z = r.hi
    t = LOG1P_COEFFS[10].hi
    for i in 9:-1:6
        t = muladd(z, t, LOG1P_COEFFS[i].hi)
    end
    p = Double32(t)
    for i in 5:-1:1
        p = poly_step(LOG1P_COEFFS[i], r, p)
    end
    return dmul(r, p)
end

"""
    Double32s.log_reduce(x::Double32) -> (k::Int, m::Double32)

Split `x = 2^k * m` with `m` in `[sqrt(2)/2, sqrt(2))`. Centering on `sqrt(2)`
keeps every bin index in `23..45` and the residual bound symmetric. The
scaling is by a power of two and therefore exact.
"""
@inline function log_reduce(x::Double32)
    k = raw_exponent(x.hi)
    m = scale2(x, -k)                       # m.hi in [1, 2)
    if m.hi >= 1.4142135f0
        m = scale2(m, -1)
        k += 1
    end
    return k, m
end

# Shared table reduction: bin index (23..45 -> 1..23), exact residual.
@inline function log_core(x::Double32)
    k, m = log_reduce(x)
    i = unsafe_trunc(Int, round(m.hi * 32.0f0))
    idx = i - 22
    c = @inbounds LOG_TABLE_C[idx]
    return k, idx, mulm1(m, c)
end

"""
    log(x::Double32) -> Double32

Natural logarithm.

Throws a `DomainError` for a negative argument, matching `log(::Float32)`. Use
[`Double32s.log_or_nan`](@ref) in kernels, where a `throw` is expensive.

# Examples
```jldoctest
julia> log(one(Double32))
Double32(0.0f0, 0.0f0)

julia> Float64(log(Double32(2.0f0)))
0.6931471805599454

julia> log(Double32(0.0f0))
Double32(-Inf32, 0.0f0)
```
"""
function Base.log(x::Double32)
    signbit(x.hi) && !iszero(x.hi) && !isnan(x.hi) && throw(DomainError(x,
        "log was called with a negative real argument; use log(Complex(x))"))
    return log_or_nan(x)
end

"""
    Double32s.log_or_nan(x::Double32) -> Double32

Natural logarithm returning `NaN` for a negative argument instead of throwing.
The kernel-safe spelling, exactly as [`Double32s.sqrt_or_nan`](@ref) is for
`sqrt`.

```text
log(+0)   -> -Inf       log(-0)   -> -Inf
log(+Inf) -> +Inf       log(x<0)  -> NaN
log(1)    -> +0         log(NaN)  -> NaN
```
"""
function log_or_nan(x::Double32)
    isnan(x.hi) && return x
    iszero(x.hi) && return NEG_INF
    signbit(x.hi) && return D32_NAN
    isfinite(x.hi) || return x                      # +Inf
    isone(x.hi) && iszero(x.lo) && return POS_ZERO  # exact
    k, idx, r = log_core(x)
    return (kf_ln2(k) + @inbounds LOG_TABLE_LN[idx]) + log1p_small(r)
end

# k*ln(2) with the Cody–Waite split, so the integer part is exact and only the
# small remainder is rounded. For |k| <= 150 the first product is exact.
@inline kf_ln2(k::Int) = Float32(k) * LN2_C1 + Float32(k) * LN2_C2

"""
    log1p(x::Double32) -> Double32

`log(1 + x)`, accurate near zero where `log(1 + x)` is not.

Three regimes, all division-free:

- `|x| <= 0.0222` — the direct series ([`Double32s.log1p_small`](@ref)),
  never forming `1 + x` at all. Design section 20.3 asks for exactly this.
- `0.0222 < |x| <= 0.28` — the log table applied to `u = 1 + x` **without
  ever rounding `u` to a pair**: with `c` from the bin of `1 + x`,
  `r = (1+x)*c - 1 = (c - 1) + x*c`, where `c - 1` is exact (Sterbenz) and
  `x*c` is two exact `two_prod_` splits. Forming `1 + x` first and taking
  `log` rounds at `2^-49` absolute, which is up to 22 ulp48 relative to
  `log1p(0.0222)`; this path keeps the error at the series' own level
  (measured 1.52 worst over a dense `[-0.285, 0.285]` sweep, against 2.58
  for the previous 27-term series).
- past `0.28`, `log(1 + x)`: the rounding of `1 + x` is `2^-48` relative and
  `log` amplifies it by `1/((1+x)·log(1+x))`, bounded by about 3.2 there.

# Examples
```jldoctest
julia> log1p(zero(Double32))
Double32(0.0f0, 0.0f0)

julia> log1p(Double32(-1.0f0))
Double32(-Inf32, 0.0f0)
```
"""
function Base.log1p(x::Double32)
    isnan(x.hi) && return x
    x.hi < -1.0f0 && throw(DomainError(x, "log1p was called with an argument below -1"))
    x.hi == -1.0f0 && iszero(x.lo) && return NEG_INF
    isfinite(x.hi) || return x
    iszero(x.hi) && return x                       # ±0 -> ±0 (dmul loses the sign)
    a = abs(x.hi)
    a <= 0.0221f0 && return log1p_small(x)
    if a <= 0.28f0
        i = unsafe_trunc(Int, round(muladd(x.hi, 32.0f0, 32.0f0)))
        idx = i - 22
        c = @inbounds LOG_TABLE_C[idx]
        cm1 = c - 1.0f0                            # exact: c in (0.78, 1.4)
        p1, e1 = two_prod_(x.hi, c)
        p2, e2 = two_prod_(x.lo, c)
        s, t = two_sum_(cm1, p1)
        l1, f1 = two_sum_(e1, p2)
        s2, t2 = two_sum_(s, l1)
        r = raw(two_sum_(s2, ((t + f1) + t2) + e2)...)
        return (@inbounds LOG_TABLE_LN[idx]) + log1p_small(r)
    end
    return log_or_nan(one(Double32) + x)
end

"""
    log2(x::Double32) -> Double32

Base-2 logarithm. Exact for exact powers of two.

The exponent `k` of `x = 2^k · m` is an integer and is kept as one, with only
the mantissa's contribution carrying error: `k + (-log2 c) + log1p(r)/ln 2`.
The obvious `log(x)/LN2` folds `k·ln(2)` into the logarithm and then divides
the sum, spending rounding on a term that was exact.

# Examples
```jldoctest
julia> log2(Double32(1024.0f0))
Double32(10.0f0, 0.0f0)
```
"""
function Base.log2(x::Double32)
    signbit(x.hi) && !iszero(x.hi) && !isnan(x.hi) && throw(DomainError(x,
        "log2 was called with a negative real argument"))
    isnan(x.hi) && return x
    iszero(x.hi) && return NEG_INF
    isfinite(x.hi) || return x
    # A power of two is exact and common enough to be worth the check.
    iszero(x.lo) && is_power_of_two(x.hi) &&
        return Double32(Float32(raw_exponent(x.hi)))
    k, idx, r = log_core(x)
    return (Double32(Float32(k)) + @inbounds LOG_TABLE_L2[idx]) +
           dmul(log1p_small(r), INV_LN2)
end

"""
    log10(x::Double32) -> Double32

Base-10 logarithm, with the exponent kept exact exactly as in [`log2`](@ref):
`k·log10(2) + (-log10 c) + log1p(r)·log10(e)`.
"""
function Base.log10(x::Double32)
    signbit(x.hi) && !iszero(x.hi) && !isnan(x.hi) && throw(DomainError(x,
        "log10 was called with a negative real argument"))
    isnan(x.hi) && return x
    iszero(x.hi) && return NEG_INF
    isfinite(x.hi) || return x
    k, idx, r = log_core(x)
    return (Float32(k) * LOG10_2 + @inbounds LOG_TABLE_L10[idx]) +
           dmul(log1p_small(r), LOG10_E)
end
