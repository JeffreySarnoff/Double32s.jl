# reductions.jl — compensated reductions.
#
# Design section 19.
#
# Note what is deliberately *absent*: a `compensated_sum(A::Array{Float32})`.
# Base already spells that, and it works with no package code at all beyond
# `Double32(::Float32)` and `+`:
#
#     sum(Double32, A::AbstractArray{Float32})    # Double32 accumulator
#     mapreduce(Double32, +, A)                   # the same thing, spelled out
#
# Adding a package name for it would duplicate a Base spelling and split the
# API for no benefit. `sum(A::AbstractArray{Double32})` likewise comes free —
# not piracy, but not needed either, since Base's pairwise `mapreduce` already
# uses our `+`.
#
# `dot` is different: it lives in LinearAlgebra, and defining it for two
# Float32 arrays would be piracy. So it needs a package name.

"""
    Double32s.compensated_dot(x, y) -> Double32

Dot product accumulated in `Double32`.

For `Float32` inputs each product is formed *exactly* by `two_prod_` and the
exact pair is accumulated, so no information is lost before accumulation —
strictly more accurate than rounding each product to `Float32` first. The `raw`
call is justified because `two_prod_` returns a magnitude-ordered nonoverlapping
pair by construction.

"Exactly" carries `two_prod_`'s condition: the product residual is
representable only for products `≥ 2^-101` (and only under gradual underflow —
gate on [`has_gradual_underflow`](@ref)). Smaller products contribute their
correctly rounded value with the residual truncated at `2^-149` — the format's
underflow taper, harmless to a sum whose magnitude sits in the normal range.

When `LinearAlgebra` is loaded, an extension may add
`LinearAlgebra.dot(::AbstractArray{Double32}, ::AbstractArray{Double32})`
delegating here — legitimate, since the element type is ours.
"""
function compensated_dot(x::AbstractArray{Float32}, y::AbstractArray{Float32};
                         policy::ArithmeticPolicy = DEFAULT_POLICY)
    axes(x) == axes(y) || throw(DimensionMismatch(
        lazy"x has axes $(axes(x)) but y has axes $(axes(y))"))
    acc = zero(Double32)
    @inbounds for i in eachindex(x, y)
        p, e = two_prod_(x[i], y[i])
        acc = add(policy, acc, raw(p, e))
    end
    return acc
end

function compensated_dot(x::AbstractArray{Double32}, y::AbstractArray{Double32};
                         policy::ArithmeticPolicy = DEFAULT_POLICY)
    axes(x) == axes(y) || throw(DimensionMismatch(
        lazy"x has axes $(axes(x)) but y has axes $(axes(y))"))
    acc = zero(Double32)
    @inbounds for i in eachindex(x, y)
        acc = add(policy, acc, mul(policy, x[i], y[i]))
    end
    return acc
end

compensated_dot(x::AbstractArray{Float32}, y::AbstractArray{Double32}; kw...) =
    compensated_dot(map(Double32, x), y; kw...)
compensated_dot(x::AbstractArray{Double32}, y::AbstractArray{Float32}; kw...) =
    compensated_dot(x, map(Double32, y); kw...)

"""
    Double32s.compensated_sum(A; policy = DEFAULT_POLICY) -> Double32

Sequential `Double32`-accumulated sum with a *specified* summation order.

Prefer `sum(Double32, A)` unless you need the order pinned: Base's pairwise
reduction is more accurate and already uses our `+`. This exists for
reproducibility work, where a fixed tree matters more than the error bound.
"""
function compensated_sum(A; policy::ArithmeticPolicy = DEFAULT_POLICY)
    acc = zero(Double32)
    @inbounds for a in A
        acc = add(policy, acc, convert(Double32, a))
    end
    return acc
end

"""
    Double32s.compensated_mapreduce(f, op, A; init, policy) -> Double32

Sequential `Double32` map-reduce with a specified order, the general form of
[`compensated_sum`](@ref).
"""
function compensated_mapreduce(f, op, A;
                               init::Double32 = zero(Double32),
                               policy::ArithmeticPolicy = DEFAULT_POLICY)
    acc = init
    @inbounds for a in A
        acc = op(acc, convert(Double32, f(a)))
    end
    return acc
end

"""
    Double32s.norm2(x) -> Double32

Euclidean norm via [`compensated_dot`](@ref), with an exact power-of-two
scaling so the squares neither overflow nor underflow prematurely.
"""
function norm2(x::AbstractArray)
    isempty(x) && return zero(Double32)
    m = zero(Float32)
    @inbounds for a in x
        d = convert(Double32, a)
        isnan(d.hi) && return D32_NAN
        isinf(d.hi) && return POS_INF
        m = max(m, abs(d.hi))
    end
    iszero(m) && return zero(Double32)
    # `raw_exponent`, not `Base.exponent`: `m` is finite and nonzero by the two
    # returns above, so the domain check is dead code that would nonetheless
    # leave a `throw` in the generated code. See `Double32s.raw_exponent`.
    k  = -raw_exponent(m)
    xs = [ldexp(convert(Double32, a), k) for a in x]
    return ldexp(sqrt_or_nan(compensated_dot(xs, xs)), -k)
end
