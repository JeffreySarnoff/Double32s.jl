```@meta
CurrentModule = Double32s
```

# Double32s.jl

*Nearly `Float64` precision from pairs of `Float32`s — built for GPUs.*

`Double32` is a double-word floating-point number: the unevaluated sum of two
`Float32` words, carrying **about 48 significant bits** (~14 decimal digits)
across the `Float32` exponent range. On hardware where `Float64` throughput is
a small fraction of `Float32` throughput — most consumer and many datacenter
GPUs — `Double32` recovers most of the precision of `Float64` while running
entirely on `Float32` execution units.

```jldoctest
julia> using Double32s

julia> x = Double32(1.0f0) / 3
Double32(0.33333334f0, -9.934108f-9)

julia> Float64(x)
0.33333333333333304
```

The package exports exactly one name, `Double32`. Everything else that is
supported API is `public` and reached as `Double32s.name` — see
[Public API](@ref api-surface).

!!! note "Module `Double32s`, type `Double32`"
    A Julia module cannot bind its own name to a type: inside
    `module Double32`, the name `Double32` *is* the module. So the package and
    module are `Double32s`, and the scalar is `Double32`.

## Why this exists

| | `Float32` | `Double32` | `Float64` on GPU |
|---|---|---|---|
| significant bits | 24 | ~48 nominal | 53 |
| decimal digits | ~7 | ~14 | ~16 |
| exponent range | ±38 | ±38 | ±308 |
| GPU execution units | native | native `Float32` + FMA | often 1/32–1/64 rate |
| memory per element | 4 B | 8 B | 8 B |

The headline use is accumulating `Float32` data at ~48-bit precision with no
package-specific API at all:

```julia
sum(Double32, A)               # A::AbstractArray{Float32} — Base spells it
```

Every scalar operation is allocation-free, type-stable, and branch-free on the
hot path — a single output check routes to a cold guarded path — and the
kernels are written so the same code compiles for device execution.

## What it is not

`Double32` is **not** an IEEE format with a uniform significand lattice, and
this documentation never pretends otherwise:

- the distance to the next representable value is *not* `eps(Double32)`; it
  can be as small as `2f0^-149` beside a high word of `1.0f0`. `nextfloat`
  and `prevfloat` are deliberately not implemented;
- `Float64(x)` is **not** exact for every value — see
  [`Double32s.float64_reference_is_exact`](@ref);
- precision tapers from 48 bits toward 24 as results approach the underflow
  boundary — see [Accuracy Model](@ref accuracy-model);
- division is accurate to a measured 4.0 nominal ulps, not faithfully
  rounded: measured, documented, and chosen deliberately over a 2× slower
  kernel that still would not be faithful.

Every accuracy claim here carries the measurement that backs it; the design
documents under `docs/Design/` record the full audit trail, including the
defects found on the way.

## Installation, and what you pay for

```julia
pkg> add Double32s
```

That is the whole install. `Double32s` depends on `AcceleratedKernels`,
`KernelAbstractions`, `Adapt`, and `Random` — a host-only user still pulls **no
vendor stack**, and the whole parallel layer works on `CPU()` out of the
box. GPU support arrives as **package extensions**, activated by whatever you
load:

```julia
pkg> add CUDA        # or AMDGPU, or KernelAbstractions alone
```

| you load | what activates |
|---|---|
| nothing | scalar, arrays, elementary functions, **the whole parallel layer on `CPU()`**, capability probe |
| `CUDA` | the same routines on `CUDABackend()`, plus `device`, `capability`, `math_mode` in [`capability_report`](@ref) |
| `AMDGPU` | the same on `ROCBackend()`, plus `device`, `gcn_arch` |

## Status

The scalar core is complete, measured, and tested (≈2.5 million assertions).
The parallel layer is built on `AcceleratedKernels`; device arithmetic is
asserted bit-identical to the host, and the capability probe runs on the backend
being questioned. Reductions are reproducible per backend but **not** across
backends — see the GPU guide. Warp-shuffle reductions,
deterministic fixed-tree reduction, and tiled matrix kernels remain roadmap.

## Contents

- [Getting Started](@ref getting-started) — construct, convert, compute.
- [Using Double32 on a GPU](@ref gpu-guide) — the rules that make kernels
  correct and fast, and the patterns that already work.
- [Arrays, on CPU and GPU](@ref arrays-guide) — array-of-structs vs
  struct-of-arrays, reductions, dot products.
- [Accuracy Model](@ref accuracy-model) — what 48 nominal bits means, where
  it tapers, and the measured error of every operation.
- [Public API](@ref api-surface) and [Internals](@ref internals).
