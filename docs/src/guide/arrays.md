# [Arrays, on CPU and GPU](@id arrays-guide)

```@meta
CurrentModule = Double32s
DocTestSetup = quote
    using Double32s
end
```

## Array-of-structs is the default

`Array{Double32}` stores each element as its 8-byte pair, inline. It is the
right representation for elementwise work on both CPU and GPU: one load per
element, ordinary broadcast fusion, ordinary `similar`/`copyto!`/views.

```jldoctest arrays
julia> v = Double32[1, 2, 3];

julia> v .+= one(Double32) ./ 3;

julia> Float64.(v)
3-element Vector{Float64}:
 1.3333333333333321
 2.3333333333333357
 3.3333333333333357
```

On the GPU this is `CuArray{Double32}` / `ROCArray{Double32}` — `Double32` is
`isbits`, so device arrays hold it inline and kernels receive it in
registers:

```julia
using CUDA
A = CuArray(rand(Double32, 10^6))
B = A .* A .- Double32(0.5f0)              # fused device kernel
```

## Reductions: use Base's spelling

The important patterns need **no package function** — once the scalar methods
exist, Base's `mapreduce` machinery does the rest, on CPU and through
GPUArrays on device:

```jldoctest arrays
julia> A32 = Float32[1f8; fill(1.0f0, 1000)];   # big value then many small ones

julia> Float32(sum(A32))                        # Float32 accumulator: wrong tail
1.0000097f8

julia> Float64(sum(Double32, A32))              # Double32 accumulator: exact
1.00001e8
```

| task | spelling | notes |
|---|---|---|
| sum `Float32` data accurately | `sum(Double32, A)` | the headline pattern |
| sum a `Double32` array | `sum(A)` | Base pairwise reduction |
| min/max/extrema | `minimum(A)`, … | via the package's `min`/`max` |
| sort | `sort!(A)` | via `isless`; total order is tested exhaustively |
| general map-accumulate | `mapreduce(f, +, A)` | promote inside `f` |

Package names exist only where Base has no spelling:

```jldoctest arrays
julia> x32 = Float32[1, 2, 3]; y32 = Float32[4, 5, 6];

julia> Double32s.compensated_dot(x32, y32)      # each product formed EXACTLY
Double32(32.0f0, 0.0f0)
```

[`Double32s.compensated_dot`](@ref) forms every `Float32×Float32` product
exactly as a pair (`two_prod_`) *before* accumulating, so no information is
lost until the accumulation itself — strictly better than rounding each
product to `Float32` first. [`Double32s.compensated_norm2`](@ref) adds an exact
power-of-two scaling so squares neither overflow nor underflow:

```julia
julia> Double32s.compensated_norm2(Double32[1f30, 1f30])   # Float32 squares would overflow
Double32(1.4142136f30, 9.023091f22)
```

[`Double32s.compensated_sum`](@ref) and
[`Double32s.compensated_mapreduce`](@ref) exist for one reason: they pin a
*sequential* order for reproducibility work. Prefer `sum(Double32, A)`
otherwise — it is the parallel AcceleratedKernels tree, reproducible per
backend for a pinned block size.

## Struct-of-arrays: [`SplitDouble32Array`](@ref)

Some device kernels want the two words as **independent coalesced streams**:

- a seed phase reads only high words (`Float32` speed, half the bandwidth);
- a vendor `Float32` routine produces the high approximation, a correction
  pass fills in the low words;
- a reduction keeps high and low partials in separate scratch arrays.

```jldoctest arrays
julia> v = [Double32(1.0f0) / 3, Double32(2.0f0)];

julia> S = Double32s.split(v);                  # two Float32 arrays

julia> S.hi, S.lo
(Float32[0.33333334, 2.0], Float32[-9.934108f-9, 0.0])

julia> S[1] === v[1]                            # still an AbstractArray{Double32}
true

julia> S[2] = Double32(5.0f0, 2.0f0^-28); S.hi[2]
5.0f0

julia> Double32s.unsplit(S) == [v[1], Double32(5.0f0, 2.0f0^-28)]
true
```

`SplitDouble32Array` is a full `AbstractArray{Double32}`: indexing
reconstructs the pair, `setindex!` stores both words, and both backing arrays
must share axes — the constructor enforces it. Its indexing methods propagate
`@inbounds`, which matters inside GPU kernels where a bounds check drags in
exception machinery.

!!! warning "Mutating one component"
    The public API never mutates one word alone. The `hi`/`lo` fields stay
    reachable for expert seed/correct kernels — with the canonical invariant
    then the caller's responsibility.

### On the GPU

Three ways in, all device resident.

**Split what you already have.** `split` and `unsplit` are broadcasts, so they
stay on whatever backend owns the data — no host round trip, no scalar
indexing:

```julia
using CUDA
A = CuArray(rand(Double32, 10^6))

S = Double32s.split(A)                      # two CuArray{Float32}
S.hi                                        # Float32-only pass reads this alone
A2 = Double32s.unsplit(S)                   # back to CuArray{Double32}
```

**Allocate for a backend.** [`Double32s.allocate_split`](@ref) needs
`KernelAbstractions` loaded (`CUDA` and `AMDGPU` both bring it):

```julia
using KernelAbstractions, CUDA
S = Double32s.allocate_split(CUDABackend(), 10^6)
KernelAbstractions.get_backend(S)           # CUDABackend()
```

**Wrap arrays you built yourself**, for interop with a vendor routine that
produced the two streams separately:

```julia
S = Double32s.SplitDouble32Array(CuArray(h_hi), CuArray(h_lo))
seed = some_float32_vendor_pass(S.hi)       # Float32-only first pass
# ... correction kernel then consumes S.hi, S.lo, writes a Double32 result
```

Either way it goes into kernels directly: `Adapt` recurses into both words, so
the wrapper is rebuilt around device arrays automatically.

```julia
@kernel function scale!(S, f)
    i = @index(Global)
    @inbounds S[i] = S[i] * f
end

scale!(CUDABackend())(S, Double32(2.0f0); ndrange = length(S))
```

!!! note "Why adaptation skips the axes check"
    The checking constructor is the right thing on the host. Inside a kernel it
    compiles to a `DimensionMismatch` throw, and that drags in the exception
    machinery for a condition that was already established before the launch.
    `Adapt.adapt_structure` therefore uses the unchecked path — the same
    contract as `raw` on the scalar.

## Choosing between them

| situation | representation |
|---|---|
| elementwise math, broadcast | `Array{Double32}` / `CuArray{Double32}` |
| mixed with `Float32`-only passes | `SplitDouble32Array` |
| reduction scratch/partials | split arrays of partial words |
| interop with vendor `Float32` routines | `SplitDouble32Array` |
| benchmarks disagree with this table | believe the benchmarks |

The design's measured recommendation (section 13): AoS as the primary user
representation, SoA where a kernel genuinely consumes the words separately.
