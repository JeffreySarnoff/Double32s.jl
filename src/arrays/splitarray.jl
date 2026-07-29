# arrays/splitarray.jl — the structure-of-arrays representation.
#
# Design sections 13.2–13.4. Array-of-structures (`Array{Double32}`,
# `CuArray{Double32}`, `ROCArray{Double32}`) is the primary user representation
# and needs no package code — `Double32` is an 8-byte `isbits` struct, so it
# lands in device memory unchanged. This wrapper is for kernels that want
# independent coalesced streams over the two words.

"""
    SplitDouble32Array(hi::AbstractArray{Float32,N}, lo::AbstractArray{Float32,N})

A `Double32` array stored as two `Float32` arrays — structure of arrays.

Use it when a kernel reads only high words during a seed phase, when vendor
`Float32` code produces a high approximation followed by a correction, when a
reduction stores high and low partials separately, or when benchmarks show
better memory behavior than an array of `Double32`.

Both backing arrays must have identical axes. Mutating one component alone
through the public API is not possible; the fields remain accessible for expert
use, with the canonical invariant then the caller's responsibility.

The backing arrays may live on a device. `Adapt.adapt` recurses into both, so a
`SplitDouble32Array{2,CuArray{Float32,2},CuArray{Float32,2}}` passed to a kernel
arrives as a `SplitDouble32Array` of device arrays; see
[`Double32s.allocate_split`](@ref) to build one for a `KernelAbstractions`
backend. Adaptation deliberately bypasses the axes check — inside a kernel that
check would compile to a `DimensionMismatch` throw and drag the exception
machinery in with it.

# Examples
```jldoctest
julia> A = Double32s.SplitDouble32Array([1.0f0, 2.0f0], [0.0f0, 0.0f0]);

julia> A[2]
2.0

julia> A[1] = Double32(1.0f0) / 3;  A[1]
0.333333333333333
```
"""
struct SplitDouble32Array{N,AH<:AbstractArray{Float32,N},
                            AL<:AbstractArray{Float32,N}} <: AbstractArray{Double32,N}
    hi::AH
    lo::AL

    # Unchecked: for `Adapt.adapt_structure` and for any path that has already
    # established matching axes. Same contract as `Raw()` on the scalar.
    function SplitDouble32Array(::Raw, hi::AH, lo::AL) where {N,
            AH<:AbstractArray{Float32,N}, AL<:AbstractArray{Float32,N}}
        return new{N,AH,AL}(hi, lo)
    end

    function SplitDouble32Array(hi::AH, lo::AL) where {N,
            AH<:AbstractArray{Float32,N}, AL<:AbstractArray{Float32,N}}
        axes(hi) == axes(lo) || throw(DimensionMismatch(
            lazy"hi has axes $(axes(hi)) but lo has axes $(axes(lo))"))
        return new{N,AH,AL}(hi, lo)
    end
end

Base.size(A::SplitDouble32Array) = size(A.hi)
Base.axes(A::SplitDouble32Array) = axes(A.hi)

# Correct because both backing arrays are indexed identically; declaring it
# lets Base's generic eachindex, copyto!, and reduction fallbacks take their
# fast paths.
Base.IndexStyle(::Type{<:SplitDouble32Array}) = IndexLinear()

# `@propagate_inbounds` rather than `@inline` is the load-bearing detail: it
# forwards the caller's `@inbounds` context to the two inner accesses. Without
# it an `@inbounds` loop still pays two bounds checks per element, and inside a
# GPU kernel those drag in the exception machinery.
#
# `raw` in getindex is justified: elements were canonical when stored, and
# `setindex!` is the only way to store them.
Base.@propagate_inbounds function Base.getindex(A::SplitDouble32Array, i::Int)
    return raw(A.hi[i], A.lo[i])
end

Base.@propagate_inbounds function Base.setindex!(A::SplitDouble32Array, v, i::Int)
    x = convert(Double32, v)
    A.hi[i] = x.hi
    A.lo[i] = x.lo
    return A
end

Base.similar(A::SplitDouble32Array, ::Type{Double32}, dims::Dims) =
    SplitDouble32Array(similar(A.hi, Float32, dims), similar(A.lo, Float32, dims))
Base.similar(A::SplitDouble32Array) = similar(A, Double32, size(A))

# Recurse into both words. Uses the unchecked constructor: the axes matched on
# the host, `adapt` preserves shape, and the checking constructor would leave a
# `throw` in device code.
function Adapt.adapt_structure(to, A::SplitDouble32Array)
    return SplitDouble32Array(Raw(), Adapt.adapt(to, A.hi), Adapt.adapt(to, A.lo))
end

"""
    Double32s.split(A::AbstractArray{Double32}) -> SplitDouble32Array

Convert array-of-structures to structure-of-arrays.

Backend generic: the two word arrays are produced by broadcasting, so a
`CuArray{Double32}` splits into two `CuArray{Float32}` on the device with no
host round trip and no scalar indexing.
"""
split(A::AbstractArray{Double32}) = SplitDouble32Array(Raw(), HI.(A), LO.(A))

"""
    Double32s.unsplit(A::SplitDouble32Array) -> AbstractArray{Double32}

Convert structure-of-arrays back to array-of-structures, on whichever backend
holds the words. Elements are already canonical, so no normalization runs.
"""
unsplit(A::SplitDouble32Array) = raw.(A.hi, A.lo)

"""
    Double32s.split!(S::SplitDouble32Array, A::AbstractArray{Double32}) -> S

In-place [`Double32s.split`](@ref): write the words of `A` into the existing
backing arrays of `S`.

For a conversion inside a loop this is the form to use — `split` allocates two
arrays per call, `split!` allocates nothing. Written as two in-place broadcasts,
so it is one fused kernel per word on any backend and needs no GPU package
loaded.

# Examples
```jldoctest
julia> A = Double32[1, 2] ./ 3;

julia> S = Double32s.allocate_split_host(size(A));

julia> Double32s.split!(S, A); S[1] === A[1]
true
```
"""
function split!(S::SplitDouble32Array, A::AbstractArray{Double32})
    axes(S) == axes(A) || throw(DimensionMismatch(
        lazy"destination has axes $(axes(S)) but source has axes $(axes(A))"))
    S.hi .= HI.(A)
    S.lo .= LO.(A)
    return S
end

"""
    Double32s.unsplit!(A::AbstractArray{Double32}, S::SplitDouble32Array) -> A

In-place [`Double32s.unsplit`](@ref). Allocation free; see
[`Double32s.split!`](@ref).
"""
function unsplit!(A::AbstractArray{Double32}, S::SplitDouble32Array)
    axes(A) == axes(S) || throw(DimensionMismatch(
        lazy"destination has axes $(axes(A)) but source has axes $(axes(S))"))
    A .= raw.(S.hi, S.lo)
    return A
end

"""
    Double32s.allocate_split_host(dims) -> SplitDouble32Array

Allocate a host-memory [`SplitDouble32Array`](@ref) with uninitialized words.
The backend-generic form is [`Double32s.allocate_split`](@ref), which needs
`KernelAbstractions` loaded.
"""
allocate_split_host(dims::Dims) =
    SplitDouble32Array(Raw(), Array{Float32}(undef, dims), Array{Float32}(undef, dims))
allocate_split_host(dims::Integer...) = allocate_split_host(dims)
