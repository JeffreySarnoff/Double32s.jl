# kernels.jl — the portable execution layer's public surface.
#
# Design section 14 and roadmap Phase 2. Every function here is *declared* in
# the root module and *implemented* in `Double32sKernelAbstractionsExt`, so the
# names, their docstrings, and their contracts live with the package while the
# dependency stays weak. Calling one without `KernelAbstractions` loaded raises
# a `MethodError` naming the function, which is the clearest possible signal.
#
# What is deliberately *not* here: elementwise arithmetic. `dest .= f.(A, B)`
# already compiles to one fused device kernel through GPUArrays and is the
# recommended spelling. `ak_map!` exists for backends that have
# KernelAbstractions but not GPUArrays broadcast, and for callers that need
# explicit control of the workgroup size.

"""
    Double32s.DEFAULT_GROUPSIZE

Workgroup size used by the `ka_*` kernels when none is given: `256`.

Chosen, not inherited. It must be a power of two for the tree reduction, and
256 is the smallest power of two that satisfies all four constraints at once:

- it is a whole number of 32-lane NVIDIA warps **and** of 64-lane AMD
  wavefronts, so neither vendor runs a partially-populated wave;
- `256 × sizeof(Double32)` is 2 KiB of local memory per workgroup, against
  48–228 KiB of shared memory per SM on CUDA and 64 KiB of LDS per CU on
  AMD — small enough that local memory never limits occupancy;
- the reduction tree is `log2(256) = 8` barriers deep, and each stage consumes
  `2 × 256` elements, so a 100-million-element array finishes in 4 passes;
- it is large enough to amortize the launch, and small enough that a short
  array still fills several workgroups.

**Measured** on an RTX 5090, `ak_sum` over 50 M `Float32`:

| W | 128 | **256** | 512 | 1024 |
|---|---|---|---|---|
| ms | 0.413 | **0.395** | 0.407 | 0.406 |

256 wins, and the whole spread is 4.5% — the reduction is dominated by the
arithmetic of `Double32` addition, not by launch shape, so this is a default
worth having and not worth agonizing over. CUDA's `launch_configuration`
suggests 768 instead; see [`Double32s.launch_config`](@ref) for why the
measurement is preferred.

Override per call with the `groupsize` keyword when a measurement disagrees.
"""
const DEFAULT_GROUPSIZE = 256

"""
    Double32s.ak_map!(f, dest, a)              -> dest
    Double32s.ak_map!(f, dest, a, b)           -> dest
    Double32s.ak_map!(f, dest, a, b, c)        -> dest

Elementwise `dest[i] = f(a[i], …)` through `AcceleratedKernels.foreachindex`.

**Prefer broadcast.** `dest .= f.(a, b)` compiles to one fused device kernel
through GPUArrays, handles mixed shapes and scalars, and is the spelling this
package's documentation uses everywhere; measured on an RTX 5090 over 50 M
elements the two are a tie. `ak_map!` exists for backends that provide
`KernelAbstractions` but not GPUArrays broadcast, and for callers who want to
pass AK's tuning keywords (`block_size`, `max_tasks`, `min_elems`) explicitly.

Arities are fixed at one, two, and three rather than variadic: a variadic
kernel would splat a tuple of device arrays inside the body, which is the
construct most likely to defeat inference on a backend compiler, and these three
cover every use in this package.

All arrays must share `axes`, and `dest` must be distinct from the inputs.
"""
function ak_map!(f, dest::AbstractArray, a::AbstractArray;
                 backend = KA.get_backend(dest), kwargs...)
    _same_axes(dest, a)
    isempty(dest) && return dest
    AK.foreachindex(dest, backend; kwargs...) do i
        @inbounds dest[i] = f(a[i])
    end
    return dest
end

function ak_map!(f, dest::AbstractArray, a::AbstractArray, b::AbstractArray;
                 backend = KA.get_backend(dest), kwargs...)
    _same_axes(dest, a, b)
    isempty(dest) && return dest
    AK.foreachindex(dest, backend; kwargs...) do i
        @inbounds dest[i] = f(a[i], b[i])
    end
    return dest
end

function ak_map!(f, dest::AbstractArray, a::AbstractArray, b::AbstractArray,
                 c::AbstractArray; backend = KA.get_backend(dest), kwargs...)
    _same_axes(dest, a, b, c)
    isempty(dest) && return dest
    AK.foreachindex(dest, backend; kwargs...) do i
        @inbounds dest[i] = f(a[i], b[i], c[i])
    end
    return dest
end

@inline function _same_axes(dest, srcs...)
    for s in srcs
        axes(s) == axes(dest) || throw(DimensionMismatch(
            lazy"destination has axes $(axes(dest)) but a source has axes $(axes(s))"))
    end
    return nothing
end

"""
    Double32s.launch_config(backend; kind = :reduction) -> NamedTuple

The workgroup size this package uses on `backend`, and what the vendor's
occupancy heuristic suggests instead.

Fields: `groupsize` (what is used), `suggested` (the occupancy API's answer, or
`nothing`), `kind`, and any vendor detail. Design section 15.3 asked for
occupancy-informed launch tuning and called its candidate numbers "hypotheses,
not fixed constants" — this is where the hypothesis is tested.

**The measured answer and the heuristic disagree, and the measurement wins.**
On an RTX 5090, CUDA's `launch_configuration` suggests 768 threads for the
reduction stage; measured times over 50 M elements are 128 → 0.413 ms,
**256 → 0.395 ms**, 512 → 0.407 ms, 1024 → 0.406 ms. The heuristic maximizes
resident warps to hide memory latency, but this kernel is arithmetic bound on
`Double32` addition — extra resident warps buy nothing, while a larger block
deepens the intra-block tree by `log2(W)` barriers. Both numbers are reported
so the disagreement stays visible instead of being buried in a constant.

Requires `KernelAbstractions`; vendor detail requires `CUDA` or `AMDGPU`.
"""
function launch_config end

"""
    Double32s.MATMUL_TILE

Tile edge for [`Double32s.ak_matmul!`](@ref): `16`.

`16 × 16` is 256 work items — the same [`DEFAULT_GROUPSIZE`](@ref) the reductions
use, and for the same reasons: whole warps on NVIDIA, whole wavefronts on AMD.
Local memory is two `Float32` tiles plus one `Double32` accumulator tile,
`16² × (4 + 4 + 8)` = 4 KiB per workgroup, which leaves occupancy unconstrained
on both vendors.
"""
const MATMUL_TILE = 16

"""
    Double32s.wavefront_size(backend) -> Int

Lanes per hardware execution unit on `backend`: a CUDA warp, an AMD wavefront,
or `1` for `CPU()`.

Design section 16.2 is explicit that **64 must not be hardcoded** for AMD — RDNA
parts run wave32, CDNA parts wave64, and the same device can differ by
compilation mode. The number is therefore always queried, never assumed, and
this is the one place to query it.

The package's own kernels do not branch on it: [`DEFAULT_GROUPSIZE`](@ref) is
256, a whole multiple of 32 **and** 64, so no partially populated wave occurs on
either vendor at the default. It is exposed because a caller choosing a
non-default `groupsize` needs it to keep that property.

Requires `KernelAbstractions`; the vendor answers require `CUDA` or `AMDGPU`.
"""
function wavefront_size end

"""
    Double32s.kernel_report(backend) -> Vector{NamedTuple}

Register, local-memory, and occupancy figures for representative `Double32`
kernels on `backend` — design section 15.1's "PTX and register-count test
hooks", and the Phase 3 occupancy report.

One row per operation (`+`, `*`, `/`, `fma`, `sqrt_or_nan`, and the reduction
stage), each with `registers`, `local_bytes`, `shared_bytes`, `max_threads`, and
`occupancy` at the package's default workgroup size. The Phase 3 gate item "no
spills or occupancy collapse" is checked against exactly this: `local_bytes` is
spill space, so a nonzero figure that grows with a code change is the warning.

!!! note "CUDA only, and the reason is not neglect"
    There is no `ROCBackend` method. AMDGPU.jl exposes no per-kernel register or
    VGPR count — `CUDA.registers`/`CUDA.memory` have no counterpart, and the
    only route is manual inspection via `AMDGPU.@device_code_gcn`. Rather than
    ship an AMD method that returns invented or empty rows, the method is absent
    and a `MethodError` names both the function and the backend. Design section
    16's "VGPR/LDS analysis" is therefore an open item, not a delivered one.

!!! warning "Read these in a normal build"
    `Pkg.test()` runs Julia with `--check-bounds=yes`, which disables every
    `@inbounds` in the package. Kernels compiled under it are not the kernels a
    user gets: measured on the same machine, `reduce_stage` reports **0 bytes**
    of local memory in a normal build and **32 bytes** under
    `--check-bounds=yes`. The test suite therefore asserts only generous
    ceilings; the figures worth quoting come from
    `docs/Reports/gpu_report.jl`, which runs normally.

Requires `CUDA` (implemented in the CUDA extension).
"""
function kernel_report end

