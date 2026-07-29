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

"""
    Double32s.ak_mapreduce(f, op, A, neutral; kwargs...) -> Double32

Parallel reduction of `op(f(A[1]), f(A[2]), …)` on whatever backend owns `A`,
returning a host scalar. Wrapper over `AcceleratedKernels.mapreduce`.

`neutral` must be the identity of `op` and **fixes the accumulator type**. It is
positional rather than a keyword because AK's own default,
`neutral_element(op, eltype(src))`, is the wrong type for this package's
headline pattern: reducing a `Vector{Float32}` into a `Double32` accumulator
needs `zero(Double32)`, not `zero(Float32)`. Requiring it makes the accumulator
type impossible to get wrong — the same rule design section 14.4 states as
advice, expressed as a signature.

**Determinism.** For a fixed `block_size` and length the tree shape is fixed, so
the result is reproducible across runs *and across backends* — a CPU and a CUDA
reduction of the same array return the same bits, which the test suite asserts.
It is **not** the same answer as `sum(A)`, whose tree GPUArrays chooses;
`Double32` addition is not associative, so those differ in the last places.
Neither is wrong; only one is reproducible.

Extra keywords pass through to `AcceleratedKernels.mapreduce` — `block_size`,
`temp`, `switch_below`, `dims`.

The backend is always `KernelAbstractions.get_backend(A)` — the backend that
owns the memory is the only one that can run the kernel, so it is derived,
never a parameter. (An earlier draft exposed it as a keyword; the only thing
that keyword could express was a backend mismatched to the data, which no
backend can execute.)
"""
function ak_mapreduce(f, op, A::AbstractArray, neutral::T; kwargs...) where {T}
    isempty(A) && return neutral
    return AK.mapreduce(f, op, A, KA.get_backend(A);
                        init = neutral, neutral = neutral, kwargs...)
end

"""
    Double32s.ak_reduce(op, A, neutral; kwargs...) -> Double32

[`ak_mapreduce`](@ref) with `f = identity`. `eltype(A)` must equal
`typeof(neutral)`.
"""
ak_reduce(op, A::AbstractArray, neutral; kwargs...) =
    ak_mapreduce(identity, op, A, neutral; kwargs...)

"""
    Double32s.ak_sum(A; policy = DEFAULT_POLICY, kwargs...) -> Double32

Parallel sum of `A` in `Double32`. `A` may hold `Double32` or `Float32` — a
`Float32` element converts exactly and for free, so this is the
accurate-accumulation pattern with a reproducible tree.

`policy` selects the addition kernel: `StrictPolicy()` (the default) gives the
`3u²` bound of design section 8.1; `FastPolicy()` gives roughly `Float32`-level
error per step for about 8 flops instead of 20.

Measured on 50 M `Float32` on an RTX 5090: 0.439 ms against 0.417 ms for
`sum(Double32, A)`, and 0.146 ms for a plain `Float32` `sum` — the ~3× is the
arithmetic cost of 48-bit accumulation, not a launch artifact. Choose `ak_sum`
for a reproducible tree or a non-default policy; `sum` for brevity.

# Examples
```jldoctest
julia> Double32s.ak_sum(Float32[1, 2, 3, 4])
Double32(10.0f0, 0.0f0)
```
"""
ak_sum(A::AbstractArray; policy::ArithmeticPolicy = DEFAULT_POLICY, kwargs...) =
    ak_mapreduce(Double32, PolicyOp(add, policy), A, zero(Double32); kwargs...)

# ------------------------------------------------------------------- dot ---

"""
    Double32s.ExactProduct()

Elementwise product for [`Double32s.ak_dot`](@ref). On `Float32` operands it
returns the **exact** product as a canonical `Double32` — `two_prod_` gives
`(p, e)` with `|e| ≤ ½ulp(p)`, which is the canonical invariant, so `raw` is
justified and no normalization runs. A singleton, so it costs no kernel
argument.
"""
struct ExactProduct end

@inline function (::ExactProduct)(a::Float32, b::Float32)
    p, e = two_prod_(a, b)
    return raw(p, e)
end
@inline (::ExactProduct)(a::Double32, b::Double32) = a * b
@inline (p::ExactProduct)(a, b) = p(convert(Double32, a), convert(Double32, b))

# `ak_mapreduce` reduces one array; a dot product needs two. Reducing over an
# index array and reading both inside `f` keeps a single reduction
# implementation instead of a second hand-written kernel. The index array lives
# on the same backend as the data, so nothing touches host memory.
struct ZipProduct{X,Y} <: Function
    x::X
    y::Y
end
@inline (z::ZipProduct)(i::Integer) = @inbounds ExactProduct()(z.x[i], z.y[i])

"""
    Double32s.ak_dot(x, y; policy = DEFAULT_POLICY, kwargs...) -> Double32

Dot product accumulated in `Double32`.

When `x` and `y` hold `Float32`, **each product is formed exactly** as a
`Double32` by [`EFT.two_prod_`](@ref) before it is accumulated, so nothing is
lost before the summation itself — strictly better than rounding each product to
`Float32` first, and better than any `Float32` `dot`. This is the device
counterpart of [`Double32s.compensated_dot`](@ref), which is sequential and host
only.

It is also the faster spelling: on 50 M `Float32` pairs on an RTX 5090 it beats
`sum(x .* y)` by about 1.5×, because the fused reduction never materializes the
product array that the broadcast form writes and reads back.

Products below `2⁻¹⁰¹` lose their residual to subnormal underflow; see
[`EFT.two_prod_`](@ref) and gate on
[`Double32s.has_gradual_underflow`](@ref).
"""
function ak_dot(x::AbstractArray, y::AbstractArray;
                policy::ArithmeticPolicy = DEFAULT_POLICY, kwargs...)
    axes(x) == axes(y) || throw(DimensionMismatch(
        lazy"x has axes $(axes(x)) but y has axes $(axes(y))"))
    isempty(x) && return zero(Double32)
    # The index array must live where the data lives; the reduction then
    # derives the same backend from it.
    idx = KA.allocate(KA.get_backend(x), Int32, length(x))
    copyto!(idx, Int32.(1:length(x)))
    return ak_mapreduce(ZipProduct(x, y), PolicyOp(add, policy), idx,
                        zero(Double32); kwargs...)
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
    Double32s.ak_norm2(x; kwargs...) -> Double32

Euclidean norm on whatever backend owns `x`, with an exact power-of-two
prescaling so the squares neither overflow nor underflow.

The scale is `2^-k` with `k` taken from the largest `|x[i]|`, which costs one
extra reduction pass to find. Scaling by a power of two is exact, so the only
error is the summation and the final `sqrt`.
"""
function ak_norm2(x::AbstractArray; policy::ArithmeticPolicy = DEFAULT_POLICY,
                  kwargs...)
    isempty(x) && return zero(Double32)
    m = ak_mapreduce(LeadMagnitude(), max, x, 0.0f0; kwargs...)
    iszero(m) && return zero(Double32)
    isfinite(m) || return Double32(m)              # Inf or NaN propagates
    k = raw_exponent(m)
    scale = ldexp(1.0f0, -k)
    ss = ak_mapreduce(ScaledSquare(scale), PolicyOp(add, policy), x,
                      zero(Double32); kwargs...)
    return ldexp(sqrt_or_nan(ss), k)
end
