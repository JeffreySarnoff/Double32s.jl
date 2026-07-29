# gpu/reduce.jl — reductions, on AcceleratedKernels.
#
# Design sections 14.4/14.5 and roadmap Phase 2. These were originally a
# hand-written multipass tree in a `KernelAbstractions` extension. They now
# delegate to `AcceleratedKernels.mapreduce`, which is a hard dependency, and
# the delegation was made only after checking that it keeps the two properties
# the hand-written version existed for:
#
#   1. **Cross-backend bit-identity.** `AK.mapreduce(Double32, +, A)` returns
#      the *same bits* on `CPU()` and on `CUDABackend()` — and agreed
#      bit-for-bit with the hand-written tree it replaces.
#   2. **Determinism for a fixed block size.** Block sizes 64 and 256 give the
#      same answer; 1024 gives a different one. That is exactly the caveat the
#      hand-written tree had for `groupsize`: pin the block size and the tree
#      shape is pinned with it.
#
# Speed is a wash — 50 M `Float32` on an RTX 5090: AK 0.439 ms against 0.408 ms
# for the hand-written tree, about 7%. Paying that to delete a multipass driver,
# two stage kernels, and their barrier-scoping hazards is the right trade; those
# hazards are design section 14.3, and every one of them was a live bug at some
# point in this file's history.
#
# AK provides no GEMM, so `gpu/linalg.jl` still writes a
# `KernelAbstractions.@kernel` directly — which is why `KernelAbstractions` is a
# direct dependency rather than reached transitively through AK.
#
# **Naming.** These were `ak_sum`, `ak_dot`, … The prefix is gone: what can be
# a method of the Base or LinearAlgebra generic **is one** — with an owned type
# (`Double32`) somewhere in the signature, so nothing is pirated and no
# `Float32`-only call changes meaning. `sum(Double32, A)` and
# `dot(Double32, x, y)` spell "accumulate in Double32" for any input eltype;
# `sum(A)`/`dot(x, y)` on `Double32` data take these paths by dispatch. Only
# `norm2` keeps a package-local name — `LinearAlgebra` has no `norm2`.

# The shared core. No owned-type constraint here, so it stays internal: the
# public methods below each carry `Double32` in their signatures.
function _mapreduce_impl(f, op, A::AbstractArray, neutral::T; kwargs...) where {T}
    isempty(A) && return neutral
    return AK.mapreduce(f, op, A, KA.get_backend(A);
                        init = neutral, neutral = neutral, kwargs...)
end

"""
    mapreduce(f, op, A::AbstractArray, neutral::Double32; kwargs...) -> Double32

Parallel reduction of `op(f(A[1]), f(A[2]), …)` on whatever backend owns `A`,
returning a host scalar. Wrapper over `AcceleratedKernels.mapreduce`.

`neutral` must be the identity of `op` and **fixes the accumulator type**. It
is positional rather than a keyword because AK's own default,
`neutral_element(op, eltype(src))`, is the wrong type for this package's
headline pattern: reducing a `Vector{Float32}` into a `Double32` accumulator
needs `zero(Double32)`, not `zero(Float32)`. Requiring it makes the accumulator
type impossible to get wrong — the same rule design section 14.4 states as
advice, expressed as a signature. It is also **not** `Base`'s `init`: AK seeds
every block with it, so a non-identity value would be added once per block.

**Determinism.** For a fixed `block_size` and length the tree shape is fixed, so
the result is reproducible across runs *and across backends* — a CPU and a CUDA
reduction of the same array return the same bits, which the test suite asserts.
It is **not** the same answer as `Base`'s pairwise tree; `Double32` addition is
not associative, so those differ in the last places. Neither is wrong; only one
is reproducible.

Extra keywords pass through to `AcceleratedKernels.mapreduce` — `block_size`,
`temp`, `switch_below`, `dims`.

The backend is always `KernelAbstractions.get_backend(A)` — the backend that
owns the memory is the only one that can run the kernel, so it is derived,
never a parameter.
"""
Base.mapreduce(f, op, A::AbstractArray, neutral::Double32; kwargs...) =
    _mapreduce_impl(f, op, A, neutral; kwargs...)

# A Float32 accumulator over Double32 data (the norm2 scale pass is the
# in-package example). Owned through the eltype; a Float32-over-Float32
# method would pirate Base.
Base.mapreduce(f, op, A::AbstractArray{Double32}, neutral::Float32; kwargs...) =
    _mapreduce_impl(f, op, A, neutral; kwargs...)

"""
    reduce(op, A::AbstractArray, neutral::Double32; kwargs...) -> Double32

[`mapreduce`](@ref mapreduce(::Any, ::Any, ::AbstractArray, ::Double32)) with
`f = identity`. `eltype(A)` must equal `typeof(neutral)`.
"""
Base.reduce(op, A::AbstractArray, neutral::Double32; kwargs...) =
    _mapreduce_impl(identity, op, A, neutral; kwargs...)

"""
    sum(Double32, A::DenseArray; policy = DEFAULT_POLICY, kwargs...) -> Double32
    sum(A::AbstractArray{Double32}; policy = DEFAULT_POLICY, kwargs...) -> Double32

Parallel sum in `Double32`, on whatever backend owns `A`.

The first form is the accurate-accumulation pattern: `A` may hold `Float32` —
an element converts exactly and for free — and the reduction tree is the
reproducible AcceleratedKernels one. The second form routes every `sum` of
`Double32` data through the same tree by dispatch.

`policy` selects the addition kernel: `StrictPolicy()` (the default) gives the
`3u²` bound of design section 8.1; `FastPolicy()` gives roughly `Float32`-level
error per step for about 8 flops instead of 20.

Measured on 50 M `Float32` on an RTX 5090: 0.439 ms against 0.146 ms for a
plain `Float32` `sum` — the ~3× is the arithmetic cost of 48-bit accumulation,
not a launch artifact.

The first method is bounded at `DenseArray`, not `AbstractArray`, to stay
unambiguous against packages that specialize `sum(f, a)` on their own array
types (`StaticArrays` was the live collision:
`typeintersect(StaticArray, DenseArray)` is `Union{}`, so the narrower bound
removes the crossing). A view or wrapper falls back to `Base`'s generic
`sum(Double32, A)` — same values, `Base`'s tree.

# Examples
```jldoctest
julia> sum(Double32, Float32[1, 2, 3, 4])
1e+01
```
"""
Base.sum(::Type{Double32}, A::DenseArray;
         policy::ArithmeticPolicy = DEFAULT_POLICY, kwargs...) =
    _mapreduce_impl(Double32, PolicyOp(add, policy), A, zero(Double32); kwargs...)

Base.sum(A::AbstractArray{Double32};
         policy::ArithmeticPolicy = DEFAULT_POLICY, kwargs...) =
    _mapreduce_impl(identity, PolicyOp(add, policy), A, zero(Double32); kwargs...)

# ------------------------------------------------------------------- dot ---

"""
    Double32s.ExactProduct()

Elementwise product for the `Double32` [`dot`](@ref) methods. On `Float32`
operands it returns the **exact** product as a canonical `Double32` —
`two_prod_` gives `(p, e)` with `|e| ≤ ½ulp(p)`, which is the canonical
invariant, so `raw` is justified and no normalization runs. A singleton, so it
costs no kernel argument.
"""
struct ExactProduct end

@inline function (::ExactProduct)(a::Float32, b::Float32)
    p, e = two_prod_(a, b)
    return raw(p, e)
end
@inline (::ExactProduct)(a::Double32, b::Double32) = a * b
@inline (p::ExactProduct)(a, b) = p(convert(Double32, a), convert(Double32, b))

# The reduction core takes one array; a dot product needs two. Reducing over an
# index *range* and reading both inside `f` keeps a single reduction
# implementation instead of a second hand-written kernel, and the range is
# isbits — indices are computed in registers, so the reduction's memory
# traffic is exactly x and y.
struct ZipProduct{X,Y} <: Function
    x::X
    y::Y
end
@inline (z::ZipProduct)(i::Integer) = @inbounds ExactProduct()(z.x[i], z.y[i])

"""
    LinearAlgebra.dot(Double32, x, y; policy = DEFAULT_POLICY, kwargs...) -> Double32

Dot product accumulated in `Double32`, for inputs of **any** element type, on
whatever backend owns `x`. The first argument is the accumulator type, exactly
as in `sum(Double32, A)` — a `dot(::Vector{Float32}, ::Vector{Float32})` method
returning `Double32` would change the meaning of every existing `Float32` dot,
so the widening must be asked for.

When `x` and `y` hold `Float32`, **each product is formed exactly** as a
`Double32` by [`EFT.two_prod_`](@ref) before it is accumulated, so nothing is
lost before the summation itself — strictly better than rounding each product to
`Float32` first, and better than any `Float32` `dot`. This is the parallel
counterpart of [`Double32s.compensated_dot`](@ref), which is sequential and host
only.

It is also the faster spelling on device: on 50 M `Float32` pairs on an RTX
5090 it beats `sum(x .* y)` by 1.5× (0.418 ms against 0.630 on 50 M pairs), because the fused reduction never
materializes the product array that the broadcast form writes and reads back.

Products below `2⁻¹⁰¹` lose their residual to subnormal underflow; see
[`EFT.two_prod_`](@ref) and gate on
[`Double32s.has_gradual_underflow`](@ref).
"""
function LinearAlgebra.dot(::Type{Double32}, x::AbstractArray, y::AbstractArray;
                           policy::ArithmeticPolicy = DEFAULT_POLICY, kwargs...)
    axes(x) == axes(y) || throw(DimensionMismatch(
        lazy"x has axes $(axes(x)) but y has axes $(axes(y))"))
    isempty(x) && return zero(Double32)
    # Reduce over an index *range*, not an index array: a range is isbits, so
    # the kernel computes indices in registers and the reduction's memory
    # traffic is exactly x and y. The first implementation allocated an Int32
    # index array and filled it from the host — measured 53.8 ms for a 50 M
    # dot on an RTX 5090, all of it the host-side iota and its bus transfer.
    # The backend must be passed explicitly here: derived from the range it
    # would be CPU regardless of where x lives.
    z = zero(Double32)
    return AK.mapreduce(ZipProduct(x, y), PolicyOp(add, policy),
                        Int32(1):Int32(length(x)), KA.get_backend(x);
                        init = z, neutral = z, kwargs...)
end

# ----------------------------------------------------------------- norm2 ---

# Magnitude of the leading word, for the scale pass. Reduced in `Float32`: the
# scale only needs an exponent, so carrying a `Double32` through this pass would
# double the traffic to no purpose.
struct LeadMagnitude end
@inline (::LeadMagnitude)(x::Double32) = abs(x.hi)
@inline (::LeadMagnitude)(x::Float32) = abs(x)
@inline (m::LeadMagnitude)(x) = m(convert(Double32, x))

# Square after an exact power-of-two rescale, so the square lands near 1.
struct ScaledSquare
    scale::Float32
end
@inline function (s::ScaledSquare)(x)
    t = convert(Double32, x) * s.scale
    return t * t
end

"""
    Double32s.norm2(x; kwargs...) -> Double32

Euclidean norm on whatever backend owns `x`, with an exact power-of-two
prescaling so the squares neither overflow nor underflow. Accepts `Float32` or
`Double32` elements. (`LinearAlgebra` has no `norm2` generic to extend, so the
name is package-local.)

The scale is `2^-k` with `k` taken from the largest `|x[i]|`, which costs one
extra reduction pass to find. Scaling by a power of two is exact, so the only
error is the summation and the final `sqrt`.
"""
function norm2(x::AbstractArray; policy::ArithmeticPolicy = DEFAULT_POLICY,
               kwargs...)
    isempty(x) && return zero(Double32)
    m = _mapreduce_impl(LeadMagnitude(), max, x, 0.0f0; kwargs...)
    iszero(m) && return zero(Double32)
    isfinite(m) || return Double32(m)              # Inf or NaN propagates
    k = raw_exponent(m)
    scale = ldexp(1.0f0, -k)
    ss = _mapreduce_impl(ScaledSquare(scale), PolicyOp(add, policy), x,
                         zero(Double32); kwargs...)
    return ldexp(sqrt_or_nan(ss), k)
end
