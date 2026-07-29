# [Using Double32 on a GPU](@id gpu-guide)

```@meta
CurrentModule = Double32s
DocTestSetup = quote
    using Double32s
end
```

This is the page the package exists for. `Double32` is designed so that the
**same scalar methods** that run on the host compile cleanly into GPU kernels:
every hot path is branch-free (one output check per operation), allocation-free,
type-stable, and built exclusively from `Float32` adds, multiplies, and
**genuine `fma`** — the operations every GPU executes at full rate.

!!! note "What works today vs. what is roadmap"
    Everything on this page runs today. The parallel layer is built on
    **`AcceleratedKernels`**, a hard dependency, with `KernelAbstractions`
    alongside it for the one kernel AK has no equivalent for (the tiled
    matmul). **`CUDA` and `AMDGPU` stay weak dependencies** — a host-only user
    pulls no vendor stack. The device paths in the test suite run on real
    hardware when a device is present and are skipped, not faked, when it is
    not.

    Warp-shuffle reduction was implemented and measured at **0.48–0.56×** the
    tree's speed, so it does not ship — see design section 15.2 for why an
    expensive operator inverts that usual advice.

## Installing: you choose the backend

```julia
pkg> add Double32s          # AcceleratedKernels, KernelAbstractions, Adapt, Random

pkg> add CUDA               # activates Double32sCUDAExt
pkg> add AMDGPU             # activates Double32sAMDGPUExt
```

| you load | you get |
|---|---|
| nothing | the scalar, arrays, **all `ak_*` parallel routines on `CPU()`**, the capability probe |
| `CUDA` | the same routines on `CUDABackend()`, plus `device`, `capability`, `math_mode` in the report |
| `AMDGPU` | the same on `ROCBackend()`, plus `device` and `gcn_arch` |

The parallel routines are always present because `AcceleratedKernels` is a hard
dependency — loading a vendor package adds a *backend*, not the functions
themselves. Only the vendor-specific reporting lives in an extension.

## Why a pair of Float32s beats Float64 on a GPU

On most consumer GPUs (and many datacenter parts), `Float64` arithmetic runs
at 1/32 to 1/64 the rate of `Float32`. A `Double32` addition costs about 20
`Float32` operations and a multiplication about 9 — so even at 20× the
operation count, `Double32` matches or beats native `Float64` throughput on
such hardware, while using the same 8 bytes per element.

The arithmetic relies on two hardware guarantees, both universal on modern
GPUs:

- IEEE round-to-nearest `Float32` add/mul/div/sqrt;
- a single-rounding fused multiply-add (`fma.rn.f32` on PTX,
  `v_fma_f32` on GCN).

## The element type is device-ready

`Double32` is an 8-byte immutable `isbits` struct of two `Float32`s. That
means:

- `CuArray{Double32}` / `ROCArray{Double32}` store elements inline — no
  boxing, no pointers, one 64-bit load per element;
- it passes into kernels by value, lives in registers, and participates in
  broadcast fusion like any hardware float.

```julia
using CUDA, Double32s

# host: build in Double32, or convert Float32 data exactly
h = rand(Double32, 1_000_000)
A = CuArray(h)                        # CuArray{Double32}: inline, 8 B/element

B = similar(A)
c = Double32(1.0f0) / 3

B .= muladd.(A, A, c)                 # one fused kernel from the scalar methods
```

Broadcasting works because `Double32 <: Number` — it is a broadcast scalar,
and the broadcast machinery compiles the package's scalar `+`, `*`, `muladd`
into the kernel body.

!!! tip "Device results are bit-identical to the host"
    Every hot path is branch-free `Float32` add/mul/`fma`, so there is nothing
    that may legitimately differ between backends. The test suite asserts
    exactly that: `+`, `-`, `*`, `/`, and `sqrt_or_nan` on a `CuArray{Double32}`
    return the same bits as the host, `===`, including at the underflow and
    overflow guard bands.

    The one exception is **reductions**. `+` on `Double32` is not associative
    and a device reduction tree is not the host's, so `sum` may differ in the
    last place or two. Compare a reduction against an exact reference, never
    against the host's reduction.

## The headline pattern: Float32 data, Double32 accumulator

Most GPU data is `Float32`. The accuracy problem is the *accumulation*, and
Base already spells the fix:

```julia
using CUDA, Double32s

data = CUDA.rand(Float32, 100_000_000)

s = sum(Double32, data)               # Double32 accumulator over Float32 data
Float64(s)                            # ~14 correct digits
```

Each element converts exactly (`Double32(x::Float32)` is `(x, 0)` — two
registers, zero work), and the reduction tree adds in ~48-bit precision. On
100 million elements, a plain `Float32` sum can lose 4–6 digits to
accumulated rounding; the `Double32` accumulator loses essentially none
(measured on host: worst 0.07 nominal ulps vs 3.6 million for the naive sum —
see the accuracy report).

The same shape works for any mapreduce:

```julia
mapreduce(x -> Double32(x)^2, +, data)      # sum of squares, 48-bit carry
```

### The portable alternative: `ak_sum`

Always available — `AcceleratedKernels` is a hard dependency:

```julia
Double32s.ak_sum(data)
Double32s.ak_sum(data; policy = Double32s.FastPolicy())
Double32s.ak_sum(data; block_size = 512)
```

What it gives you that `sum` does not:

- **a reproducible tree, per backend.** For a fixed `block_size` and length the
  tree is fixed, so repeated runs on the same backend return the same bits.
- **a policy.** `FastPolicy()` is about 8 flops per step instead of 20.
- **backends without GPUArrays**, where `sum` has no fast path at all.

!!! warning "Reductions are not bit-identical across backends"
    A CPU reduction and a CUDA reduction of the same array return **different
    bits**. `AcceleratedKernels` chooses its own tree per backend — the CPU path
    is task-parallel, the GPU path block-parallel — and `Double32` addition is
    not associative.

    An earlier version of this package shipped a hand-written fixed-tree
    reduction that *was* bit-identical across backends. Moving the reductions
    onto `AcceleratedKernels` traded that away, and the trade is recorded rather
    than glossed: see design section 19.6.

    `ak_matmul` is unaffected — it is still a hand-written kernel with a fixed
    tile loop, and **is** bit-identical across backends.

Speed is a wash: 50 M elements on an RTX 5090, `ak_sum` 0.439 ms against
`sum(Double32, A)` 0.417 ms.

### Reproducibility: three modes, no new API

Reduction order changes finite-precision results, so "which answer" is a real
choice. Design section 19.6 asks for three modes; all three are existing
spellings, and no keyword was added because a keyword that forwarded to `sum`
would be surface area for no capability.

| mode | spelling | guarantee |
|---|---|---|
| tuned | `sum(A)`, `sum(Double32, A)` | accurate; tree chosen by GPUArrays per device |
| deterministic, per backend | `Double32s.ak_sum(A; block_size = 256)` | same bits on every run of that backend |
| cross-backend | `Double32s.ak_matmul(A, B; tile = 16)` | same bits on **every** backend |

The second is asserted by repetition in the test suite, on `randn` data — note
that testing it on something like `1/k` would be vacuous, because that sum is
*exact* in `Double32` and every tree then agrees.

Cross-backend bit-identity is available only where the package still writes its
own kernel, which today means `ak_matmul`. Reductions gave it up when they moved
onto `AcceleratedKernels`; if you need a cross-backend reproducible sum, do it
with `ak_matmul`-style fixed structure or compare against an exact reference
instead.

### Matrix multiply with exact products

```julia
C = Double32s.ak_matmul(A, B)      # A, B :: CuArray{Float32}
```

Both element types are supported: `Float32` operands get **exact** products
(`two_prod_`) before accumulation; `Double32` operands use split-channel
products. On 1024³, normwise error is **1.0e-13 against 4.4e-7** for `Float32`
BLAS on the same data.

Measured on an RTX 5090: 0.33 ms (`Float32` operands) and 0.42 ms (`Double32`
operands) at 1024³, against 0.048 ms for cuBLAS SGEMM — **7–9× the time for
about seven more decimal digits.** For host matrices, `A * B` on
`Matrix{Double32}` uses a cache-aware compensated `mul!` that is ~28× the
generic fallback's speed at n = 64.

### Inspecting the device

```julia
Double32s.capability_report(CUDABackend())   # probe + device + math mode
Double32s.launch_config(CUDABackend())       # workgroup size used, and what CUDA suggests
Double32s.kernel_report(CUDABackend())       # registers, spill space, occupancy
```

`kernel_report` is the one to reach for when a kernel of yours gets slow:
`local_bytes` is spill space, and a jump there is usually the cause. Read it
from a normal session — `Pkg.test()` runs under `--check-bounds=yes`, which
disables `@inbounds` and inflates both registers and spills.
`docs/Reports/gpu_report.jl` regenerates the full table.

### `ak_dot` is both more accurate *and* faster

```julia
Double32s.ak_dot(x, y)          # x, y :: CuArray{Float32}
```

Every `Float32` product is formed **exactly** as a `Double32` before it is
accumulated, so nothing is lost before the summation. It is also faster
than `sum(x .* y)` — 1.52x, 0.414 ms vs 0.630 ms on 50 M pairs — because the
fused kernel never materializes the product array. This is the one place in this
guide where the accurate spelling is also the fast one.

`Double32s.ak_norm2(x)` follows the same shape, with an exact power-of-two
prescale so squares neither overflow nor underflow.

## Rules for writing device code with Double32

These rules are what the package's own kernels follow, and they are the
difference between "compiles and is right" and subtle device-only wrongness.

### 1. Use `sqrt_or_nan`, never `sqrt`, in kernels

`Base.sqrt(::Double32)` throws `DomainError` for negative arguments, matching
`sqrt(::Float32)`. A `throw` inside a device kernel drags in the exception
machinery and can abort the kernel. [`Double32s.sqrt_or_nan`](@ref) is the
same operation returning NaN instead:

```julia
f(x) = Double32s.sqrt_or_nan(x)       # device-safe
B .= f.(A)
```

### 2. Never `@fastmath`, and know your `muladd` from your `fma`

The arithmetic is *made of* rounding-error bookkeeping. `@fastmath` licenses
the compiler to reassociate it away, silently. Related: `Base.muladd` merely
*permits* fusion while `Base.fma` *requires* it — this package builds on
explicit `fma` internally, and `muladd(::Double32, ...)` is safe because it
routes to the same compensated kernel. But never "optimize" package internals
from `fma` to `muladd`.

### 3. Resolve policies on the host, pass them positionally

Operations take an [`ArithmeticPolicy`](@ref) as a compile-time singleton
first argument. `Base.:+` etc. use `StrictPolicy()`; reductions may opt into
`FastPolicy()`:

```julia
acc = Double32s.add(Double32s.FastPolicy(), acc, x)   # 8 flops vs 20, Float32-level bound
```

Never test a policy at runtime inside a kernel, and never pass one as a
keyword — keywords don't participate in dispatch. Resolve user-facing options
to singletons at the launch boundary, then pass them as positional arguments.

### 4. Probe the device — don't assume

Two hardware properties carry the whole error analysis, and they fail
differently:

- **a genuine fused multiply-add.** This is not negotiable. Every error-free
  transformation on a product goes through `two_prod_`, whose residual is
  *identically zero* when the multiply and the add round separately. An unfused
  backend does not give you a slightly worse `Double32`; it gives you a
  `Float32` wearing two words, silently.
- **gradual underflow.** Weaker: flush-to-zero does not void the arithmetic, it
  moves the boundary below which the exactness claims hold.

[`Double32s.capability_report`](@ref) answers both, for the host or for any
backend:

```jldoctest
julia> r = Double32s.capability_report();

julia> r.fused_multiply_add, r.gradual_underflow, r.usable
(true, true, true)
```

The probe is six independent *lanes* (see [`Double32s.probe_lane`](@ref)).
Lanes 1–5 exercise subnormal add, multiply, `fma`, an EFT residual, and sign;
lane 6 is a `two_prod_` whose residual is *normal*, so it isolates fusion from
flush-to-zero. A backend that passes 6 and fails 1–5 has FTZ. One that fails
both has no fused multiply-add and cannot host this package at all — which is
why `usable` tracks `fused_multiply_add` and not gradual underflow.

With a GPU package loaded, hand it a backend and the probe runs as a real
kernel compiled by that backend's pipeline:

```julia
julia> using CUDA, Double32s

julia> Double32s.capability_report(CUDABackend())
(target = "CUDABackend on NVIDIA GeForce RTX 5090",
 gradual_underflow = true, fused_multiply_add = true,
 lanes = (true, true, true, true, true, true),
 functional = true, usable = true,
 device = "NVIDIA GeForce RTX 5090", capability = v"12.0.0",
 math_mode = CUDACore.DEFAULT_MATH)
```

Passing an array instead of a backend probes through broadcast rather than a
kernel launch — useful, and worth doing as well as the kernel route, since a
backend may compile the two paths under different inlining.

!!! note "CUDA's math mode is context, not a verdict"
    `math_mode` appears in the report because it is task-local and invisible in
    your source. But measured on an RTX 5090 (sm_120, CUDA 13.3),
    `CUDA.math_mode!(CUDA.FAST_MATH)` changed *nothing* this package depends
    on: all six lanes still pass, `Float32` division and `sqrt` return the same
    bits as `PEDANTIC_MATH`, and `Double32` `/` and `sqrt` hold their relative
    error at 8.3e-16 and 1.0e-15 across all three modes.

    CUDA.jl's math mode selects approximate intrinsics where Julia asks for
    them; it does not retroactively relax arithmetic written without
    `@fastmath`. The thing that *does* destroy an error-free transformation is
    a literal `@fastmath` or `CUDA.@fastmath` at the call site — see rule 2 —
    and no report field can see that from outside the expression.

The result is deliberately not cached: `set_zero_subnormals` is a per-thread
switch and `math_mode` is per-task. Probe once at launch-configuration time and
cache it yourself, per device.

### 5. Exceptional values cost nothing on the hot path

There is no operand pre-classification anywhere: every kernel runs
branch-free and a **single check of the output high word** routes to a cold
`@noinline` guarded path. NaN and ±Inf propagate through the same code the
finite values take, so warps stay convergent except in the rare guard bands
(overflow rescue, deep-underflow rescue).

## Memory layout: one array or two

`Array{Double32}` (array-of-structs) is the default and right answer for
elementwise work — one coalesced 64-bit load per element.

For kernels that want the words in **separate streams** — a `Float32` seed
pass that reads only high words, vendor `Float32` routines followed by a
correction pass, split reduction partials — use
[`SplitDouble32Array`](@ref). It is backend generic in both directions:

```julia
using CUDA, Double32s

A = CuArray(rand(Double32, 10^6))
S = Double32s.split(A)                # two CuArray{Float32}, no host round trip
S.hi                                  # seed pass reads this alone
A2 = Double32s.unsplit(S)             # back to CuArray{Double32}
```

`split` and `unsplit` are written as broadcasts, so they never index an element
on the host and work on any backend without a package-side special case.

To allocate one directly for a backend, without naming that backend's array
type:

```julia
using KernelAbstractions, CUDA

S = Double32s.allocate_split(CUDABackend(), 10^6)
KernelAbstractions.get_backend(S)     # CUDABackend()
```

And it passes into kernels: `Adapt` recurses into both words, so a
`SplitDouble32Array` of `CuArray`s arrives inside a kernel as a
`SplitDouble32Array` of device arrays.

```julia
@kernel function scale!(S, f)
    i = @index(Global)
    @inbounds S[i] = S[i] * f
end

scale!(CUDABackend())(S, Double32(2.0f0); ndrange = length(S))
```

Adaptation deliberately bypasses the axes-checking constructor: inside a kernel
that check compiles to a `DimensionMismatch` throw and drags the exception
machinery in with it.

See [Arrays, on CPU and GPU](@ref arrays-guide) for the full discussion.

## What the roadmap adds

In design-document order (phases 3–5): CUDA warp-shuffle and AMDGPU wavefront
reductions; tiled matrix kernels with `Float32×Float32 → Double32` accumulation.
The scalar semantics documented here are frozen ahead of all of it — device
results match the host reference bit-for-bit wherever the capability probe says
the hardware can, and the test suite checks it.
