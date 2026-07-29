# Shared operand generators and reference helpers.
#
# Design sections 22.5 (random testing), 22.6 (structured exhaustive subsets),
# 23.11 (wide-gap pairs), and 23.12 (signed zeros in the low word).
#
# The generators build the adversarial operands *directly*. Every defect this
# package has had lives in a region of near-zero measure under uniform
# sampling, so `rand(Double32)` alone is not an operand source.

using Double32s
using Double32s: raw, HILO, canonicalize, iscanonical, reference,
                 float64_reference_is_exact, lsb_exponent, is_power_of_two,
                 sqrt_or_nan, compensated_dot, compensated_sum,
                 compensated_mapreduce, norm2, compensated_norm2, split, unsplit,
                 SplitDouble32Array, StrictPolicy, FastPolicy, NativePolicy,
                 DEFAULT_POLICY, nominal_eps, unit_roundoff, isnormal,
                 has_gradual_underflow,
                 POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, D32_NAN, EFT
using Random
using Test

setprecision(BigFloat, 400)

const U  = 2.0^-24            # Float32 unit roundoff
const U2 = 2.0^-48            # nominal Double32 unit roundoff
const LOMIN = reinterpret(Float32, 0x00000001)   # 2^-149

"Exact represented value, always correct — unlike `Float64(x)`."
exact(x::Double32) = reference(x)

"Relative error of `got` against an exact `BigFloat`, as a `Float64`."
function relerr(got::Double32, e::BigFloat)
    iszero(e) && return iszero(got.hi) ? 0.0 : Inf
    return Float64(abs(reference(got) - e) / abs(e))
end

"""
Whether the exact result sits far enough inside `Float32` range that a
*relative* error bound is meaningful. Design section 9: near underflow and
overflow, use absolute error and separate status tests instead.
"""
inrelrange(e::BigFloat) = isfinite(e) && !iszero(e) &&
                          big(2.0)^-100 <= abs(e) <= big(2.0)^100

# --------------------------------------------------------------- operands ---

"""
    zero_surface() -> Vector{Double32}

Every pair with a zero high word, plus both low-word zero signs under a
representative set of nonzero high HILO.

This is design section 23.12 enumerated *by construction*. An earlier draft
listed five expressions and called them the whole failure surface; they collapse
to three distinct pairs, and the surface is not confined to zero high HILO at
all, because `-` flips the low word's sign bit whatever the high word is.
"""
function zero_surface()
    v = Double32[]
    for h in (0.0f0, -0.0f0), l in (0.0f0, -0.0f0)
        push!(v, raw(h, l))
    end
    for h in (1.0f0, -1.0f0, 1.5f0, -1.5f0, Inf32, -Inf32, NaN32, -NaN32),
        l in (0.0f0, -0.0f0)
        push!(v, raw(h, l))
    end
    return v
end

"""
    wide_gap_pairs() -> Vector{Double32}

Design section 23.11, both subclasses: subnormal low HILO under a not-tiny
high word, and *normal* low HILO far below their permitted bound. A generator
that produces only the first reproduces the mistake the design's first
correction made.
"""
function wide_gap_pairs()
    his = Float32[1.0f0, prevfloat(1.0f0), 2f0^-62, 2f0^-96, 2f0^-97, 2f0^-100,
                  floatmin(Float32), floatmax(Float32)]
    v = Double32[]
    for h in his, s in (1.0f0, -1.0f0)
        # subclass 1: subnormal low HILO
        for l in (LOMIN, 3LOMIN, prevfloat(floatmin(Float32)))
            push!(v, canonicalize(h, s * l))
        end
        # subclass 2: normal low HILO at every gap width the invariant allows
        for k in 0:8:120
            l = ldexp(1.0f0, exponent(h) - 24 - k)
            (isfinite(l) && !iszero(l)) && push!(v, canonicalize(h, s * l))
        end
    end
    return filter(x -> isfinite(x) && !iszero(x.lo), unique(v))
end

"""
    structured_pool() -> Vector{Double32}

Broad structured operands: powers of two and their neighbours, the range
extremes, and low HILO at every interesting distance below the half-ulp bound.
"""
function structured_pool()
    v = Double32[]
    for h in Float32[1, -1, 3, 0.5, 1.5, prevfloat(1f0), nextfloat(1f0),
                     2f0^-62, 2f0^30, 2f0^-30, floatmin(Float32), floatmax(Float32),
                     0f0, -0f0]
        push!(v, Double32(h))
        iszero(h) && continue
        for k in (0, 1, 24, 25, 40, 60, 100, 149), s in (1f0, -1f0)
            push!(v, canonicalize(h, s * ldexp(abs(h), -k - 24)))
        end
        for l in (LOMIN, -LOMIN, 3LOMIN)
            push!(v, canonicalize(h, l))
        end
    end
    return unique(v)
end

"""
    random_pool(n; seed) -> Vector{Double32}

Uniform bit patterns run through `canonicalize`, so the pool covers the whole
lattice rather than the `[0,1)` grid `rand(Double32)` draws from.
"""
function random_pool(n::Int; seed::Int = 20260728)
    rng = MersenneTwister(seed)
    v = Double32[]
    while length(v) < n
        x = canonicalize(reinterpret(Float32, rand(rng, UInt32)),
                         reinterpret(Float32, rand(rng, UInt32)))
        isfinite(x) && push!(v, x)
    end
    return v
end

const NONFINITE = Double32[POS_INF, NEG_INF, D32_NAN, raw(NaN32, -0.0f0), raw(-NaN32, 0.0f0)]

"""
    worst_relerr(op, refop, pool) -> (err, case)

Worst relative error of a binary operation over all ordered pairs from `pool`,
skipping results outside the meaningful relative-error range.
"""
function worst_relerr(op, refop, pool)
    w = 0.0
    wc = nothing
    for a in pool, b in pool
        (isfinite(a) & isfinite(b)) || continue
        g = op(a, b)
        isfinite(g) || continue
        e = refop(exact(a), exact(b))
        inrelrange(e) || continue
        r = relerr(g, e)
        if r > w
            w = r
            wc = (a, b, g, e)
        end
    end
    return w, wc
end

function worst_relerr1(op, refop, pool)
    w = 0.0
    wc = nothing
    for a in pool
        isfinite(a) || continue
        g = op(a)
        (g isa Double32 && isfinite(g)) || continue
        e = refop(exact(a))
        inrelrange(e) || continue
        r = relerr(g, e)
        if r > w
            w = r
            wc = (a, g, e)
        end
    end
    return w, wc
end
