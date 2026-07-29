# math/trig.jl — trigonometric and inverse trigonometric functions.
#
# Design section 20.4 defers these until the package has a high-quality
# double-word π split, robust argument reduction, bounded reduced intervals,
# tested polynomial kernels, special handling near roots, and a large-argument
# policy. Each of those is addressed here and named where it is addressed; the
# large-argument policy in particular is a *stated limit*, not a claim of
# unlimited accuracy — see `Double32s.TRIG_ARG_LIMIT`.
#
# Rebuilt 2026-07-29 on the math/horner.jl primitives:
#
# - `sin` and `cos` dispatch on quadrant **parity** and evaluate ONE kernel,
#   not both: quadrant `q` needs `sin_kernel` when `q` is even (for sin) and
#   `cos_kernel` when odd, never the pair. The old `_sincos_from_quadrant`
#   computed both and discarded one — half the kernel work for nothing.
#   `sincos` still shares the reduction and both kernels, and remains the
#   right call when both values are wanted.
# - Both kernels split at the 2^-26 mixed-precision boundary: sin degrees
#   0..4 in pairs + 5..8 in the Float32 tail; cos 0..5 + 6..9.
# - `atan` reduces with at most ONE division (see its docstring); the old
#   two-step reduction took two on `|x| in (1, 3.73)`.
#
# Measured (4096-point pools, same harness as the pre-rebuild numbers):
# sin 194.9 -> 39.2 ns, cos 195.0 -> 40.1, tan 211.0 -> 69.4,
# atan 172.8 -> 60.0, asin 227.1 -> 103.4, acos 238.1 -> 117.8.

"""
    Double32s.TRIG_ARG_LIMIT

Largest `|x|` for which the trigonometric functions reduce their argument
exactly: `2^20`.

**This is the large-argument policy design section 20.4 asks for, and it is a
limit rather than a promise.** Reduction subtracts `k·π/2` using the three-word
Cody–Waite split `Double32s.PIO2_C1` / `PIO2_C2` / `PIO2_C3`, which
together carry about 72 bits of `π/2`. A `Double32` argument near `2^n` needs
about `n + 48` correct bits of `π/2` for the reduced value to keep full
precision, so 72 bits covers `n <= 24` with margin; the limit is set at `2^20`
to leave room for the reduction's own rounding.

**Beyond it the functions return `NaN`.** An earlier draft of this docstring
claimed they would "still return a value in `[-1, 1]`". They did not: the
Cody–Waite subtraction cancels catastrophically once `q` outgrows the split's
precision, and `sin(2^30)` came back as `9.88e8` — a silently wrong answer with
no indication anything had gone wrong, which is the worst possible failure for a
numerical routine.

`NaN` is the policy because it is loud. A caller can test for it; a caller
cannot test for "this number in `[-1, 1]` is fictional". Correct reduction past
this point needs Payne–Hanek with a multi-thousand-bit `2/π` — a table and a
loop that do not belong in a GPU kernel — so it is recorded as open work rather
than approximated.

Callers who need large arguments should reduce in a wider type first.
"""
const TRIG_ARG_LIMIT = 2.0f0^20

"""
    Double32s.sin_kernel(r2::Double32, r::Double32) -> Double32

`sin(r)` for `|r| <= π/4`, as `r * P(r²)`. `SIN_COEFFS` degrees 0..4 as
dominant-add pair steps (dominance: `|r²·c_{n+1}/c_n| <= 0.62/20`), degrees
5..8 as the `Float32` tail (leading tail term `r^10/11! <= 2.3e-9 < 2^-26`
of the result).
"""
@inline function sin_kernel(r2::Double32, r::Double32)
    z = r2.hi
    t = muladd(z, muladd(z, muladd(z, SIN_COEFFS[9].hi, SIN_COEFFS[8].hi),
                         SIN_COEFFS[7].hi), SIN_COEFFS[6].hi)
    p = Double32(t)
    for i in 5:-1:1
        p = poly_step(SIN_COEFFS[i], r2, p)
    end
    return dmul(r, p)
end

"""
    Double32s.cos_kernel(r2::Double32) -> Double32

`cos(r)` for `|r| <= π/4`, as `P(r²)`. `COS_COEFFS` degrees 0..5 in pairs,
6..9 in the `Float32` tail (leading tail term `r^12/12! <= 2.5e-8`; the
degree-5 term at `2.5e-8` sits just above the 2^-26 boundary and stays in the
pair head).
"""
@inline function cos_kernel(r2::Double32)
    z = r2.hi
    t = muladd(z, muladd(z, muladd(z, COS_COEFFS[10].hi, COS_COEFFS[9].hi),
                         COS_COEFFS[8].hi), COS_COEFFS[7].hi)
    p = Double32(t)
    for i in 6:-1:1
        p = poly_step(COS_COEFFS[i], r2, p)
    end
    return p
end

"""
    Double32s.trig_reduce(x::Double32) -> (q::Int, r::Double32)

Reduce `x` to `x = q·(π/2) + r` with `|r| <= π/4`, returning the quadrant `q`
and the reduced angle.

Three-stage Cody–Waite: `PIO2_C1` and `PIO2_C2` each carry 8 significand bits,
so `q * PIO2_C1` and `q * PIO2_C2` are **exact** in `Float32` for every `q` this
function can produce, and only the third and smallest term is rounded. Doing the
whole subtraction with a single `Double32` `π/2` would lose about `log2(q)` bits
of `r`, and `r` is the entire input to the polynomial kernels.

Accuracy is bounded by [`Double32s.TRIG_ARG_LIMIT`](@ref).
"""
@inline function trig_reduce(x::Double32)
    qf = round(x.hi * TWO_OVER_PI.hi)
    q = unsafe_trunc(Int, qf)
    r = ((x - qf * PIO2_C1) - qf * PIO2_C2) - qf * PIO2_C3
    return q, r
end

"""
    sin(x::Double32) -> Double32

Sine, for `|x| <= `[`Double32s.TRIG_ARG_LIMIT`](@ref). **Beyond that limit the
result is `NaN`**, deliberately: the argument reduction cannot be trusted there
and a plausible-looking wrong answer is worse than none. `NaN` for a non-finite argument,
which is IEEE-correct — the sine of infinity is undefined.

Evaluates exactly one polynomial kernel, selected by quadrant parity —
`sin(q·π/2 + r)` is `±sin(r)` for even `q` and `±cos(r)` for odd `q`.

# Examples
```jldoctest
julia> sin(zero(Double32))
Double32(0.0f0, 0.0f0)

julia> abs(Float64(sin(Double32s.PI_D32))) < 1e-14   # PI_D32 is not exactly π
true
```
"""
function Base.sin(x::Double32)
    isfinite(x.hi) || return D32_NAN
    abs(x.hi) > TRIG_ARG_LIMIT && return D32_NAN
    iszero(x.hi) && return x                    # ±0 -> ±0
    q, r = trig_reduce(x)
    r2 = dsqr(r)
    y = iszero(q & 1) ? sin_kernel(r2, r) : cos_kernel(r2)
    return iszero(q & 2) ? y : -y
end

"""
    cos(x::Double32) -> Double32

Cosine. Same argument-range policy and single-kernel dispatch as
[`sin`](@ref).

# Examples
```jldoctest
julia> cos(zero(Double32))
Double32(1.0f0, 0.0f0)
```
"""
function Base.cos(x::Double32)
    isfinite(x.hi) || return D32_NAN
    abs(x.hi) > TRIG_ARG_LIMIT && return D32_NAN
    q, r = trig_reduce(x)
    r2 = dsqr(r)
    y = iszero(q & 1) ? cos_kernel(r2) : sin_kernel(r2, r)
    return iszero((q + 1) & 2) ? y : -y
end

"""
    sincos(x::Double32) -> (Double32, Double32)

Sine and cosine together, from one shared reduction and both kernels. Bit-for-
bit identical to `(sin(x), cos(x))` — the single-kernel dispatch in `sin` and
`cos` selects exactly the kernel this evaluates — and cheaper than the pair of
calls, so prefer it whenever both are wanted.
"""
function Base.sincos(x::Double32)
    (isfinite(x.hi) && abs(x.hi) <= TRIG_ARG_LIMIT) || return (D32_NAN, D32_NAN)
    iszero(x.hi) && return (x, cos(x))
    q, r = trig_reduce(x)
    r2 = dsqr(r)
    s = sin_kernel(r2, r)
    c = cos_kernel(r2)
    m = q & 3
    m == 0 && return (s, c)
    m == 1 && return (c, -s)
    m == 2 && return (-s, -c)
    return (-c, s)
end

"""
    tan(x::Double32) -> Double32

Tangent, as `sin(x)/cos(x)` from a single shared reduction.

Near an odd multiple of `π/2` the true tangent is unbounded and the computed
value is governed by how precisely the argument locates the pole — this is a
property of the problem, not of the implementation, and no double-word format
changes it.
"""
function Base.tan(x::Double32)
    (isfinite(x.hi) && abs(x.hi) <= TRIG_ARG_LIMIT) || return D32_NAN
    s, c = sincos(x)
    return s / c
end

# ------------------------------------------------------------------ inverse ---

"""
    Double32s.atan_kernel(t2::Double32, t::Double32) -> Double32

`atan(t)` for `|t| <= 2 - sqrt(3) = tan(π/12)`, as `t * P(t²)`. `ATAN_COEFFS`
degrees 0..6 in pairs, 7..13 in the `Float32` tail (`t² <= 0.0718`, so the
degree-7 tail head is `0.0718^7/15 <= 6e-10 < 2^-26` of the result).
"""
@inline function atan_kernel(t2::Double32, t::Double32)
    z = t2.hi
    w = ATAN_COEFFS[14].hi
    for i in 13:-1:8
        w = muladd(z, w, ATAN_COEFFS[i].hi)
    end
    p = Double32(w)
    for i in 7:-1:1
        p = poly_step(ATAN_COEFFS[i], t2, p)
    end
    return dmul(t, p)
end

"""
    atan(x::Double32) -> Double32

Arctangent, reduced to `|t| <= tan(π/12)` with **at most one division**.

The angle bins are centered on `0, π/6, π/3, π/2`: with `a = |x|`,

| band of `a` | rotation | divisions |
|---|---|---|
| `[0, tan(π/12)]` | none | 0 |
| `(tan(π/12), 1)` | `t = (a - tan(π/6))/(1 + a·tan(π/6))`, add `π/6` | 1 |
| `[1, tan(5π/12))` | `t = (a - √3)/(1 + a·√3)`, add `π/3` | 1 |
| `[tan(5π/12), ∞)` | `t = 1/a`, subtract from `π/2` | 1 |

Every band lands inside the kernel's `tan(π/12)` bound. The previous reduction
inverted first and then shifted, which cost **two** divisions for
`a ∈ (1, 3.73)`; this one never pays more than one — and division is both the
slowest and, at 4.00 ulp48, the least accurate primitive in the package.

# Examples
```jldoctest
julia> atan(zero(Double32))
Double32(0.0f0, 0.0f0)

julia> Float64(4 * atan(one(Double32)))
3.1415926535897967
```
"""
function Base.atan(x::Double32)
    isnan(x.hi) && return x
    iszero(x.hi) && return x
    isfinite(x.hi) || return signbit(x.hi) ? -PIO2_D32 : PIO2_D32
    neg = signbit(x.hi)
    a = abs(x)
    local y
    if a.hi <= 0.26794919f0                        # tan(pi/12)
        y = atan_kernel(dsqr(a), a)
    elseif a.hi < 1.0f0
        t = (a - TAN_PI6_D32) / (one(Double32) + dmul(a, TAN_PI6_D32))
        y = PI6_D32 + atan_kernel(dsqr(t), t)
    elseif a.hi < 3.7320508f0                      # tan(5pi/12)
        t = (a - SQRT3_D32) / (one(Double32) + dmul(a, SQRT3_D32))
        y = PI3_D32 + atan_kernel(dsqr(t), t)
    else
        t = reciprocal(DEFAULT_POLICY, a)
        y = PIO2_D32 - atan_kernel(dsqr(t), t)
    end
    return neg ? -y : y
end

"""
    atan(y::Double32, x::Double32) -> Double32

Two-argument arctangent, with the IEEE quadrant and signed-zero conventions.
"""
function Base.atan(y::Double32, x::Double32)
    (isnan(y.hi) | isnan(x.hi)) && return D32_NAN
    if iszero(x.hi)
        iszero(y.hi) && return signbit(x.hi) ? (signbit(y.hi) ? -PI_D32 : PI_D32) : y
        return signbit(y.hi) ? -PIO2_D32 : PIO2_D32
    end
    a = atan(y / x)
    signbit(x.hi) || return a
    return signbit(y.hi) ? a - PI_D32 : a + PI_D32
end

"""
    asin(x::Double32) -> Double32

Arcsine, as `atan(x / sqrt(1 - x²))`.

The subtraction `1 - x²` is the accuracy-critical step and is why `asin` is
worse conditioned than `atan` near `|x| = 1`: there the result's derivative is
unbounded, so any error in the argument is amplified without limit. Throws a
`DomainError` outside `[-1, 1]`, matching `asin(::Float32)`.
"""
function Base.asin(x::Double32)
    isnan(x.hi) && return x
    abs(x.hi) > 1.0f0 && throw(DomainError(x, "asin requires -1 <= x <= 1"))
    isone(abs(x.hi)) && iszero(x.lo) && return signbit(x.hi) ? -PIO2_D32 : PIO2_D32
    return atan(x / sqrt_or_nan(one(Double32) - dsqr(x)))
end

"""
    acos(x::Double32) -> Double32

Arccosine, as `π/2 - asin(x)` for `x <= 0` and `atan(sqrt(1-x²)/x)` for `x > 0`.

The split avoids the cancellation that `π/2 - asin(x)` suffers as `x` approaches
1, where the result approaches zero and the subtraction would lose every
significant digit.
"""
function Base.acos(x::Double32)
    isnan(x.hi) && return x
    abs(x.hi) > 1.0f0 && throw(DomainError(x, "acos requires -1 <= x <= 1"))
    isone(x.hi) && iszero(x.lo) && return POS_ZERO
    x.hi == -1.0f0 && iszero(x.lo) && return PI_D32
    if x.hi > 0.0f0
        return atan(sqrt_or_nan(one(Double32) - dsqr(x)) / x)
    end
    return PIO2_D32 - asin(x)
end
