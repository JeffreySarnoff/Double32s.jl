# [Advanced Uses](@id examples-advanced)

These examples use the compensated kernels, the linear-algebra layer, the
structure-of-arrays representation, and the GPU-portable `ak_*` routines.
Everything except the final GPU section runs on a plain CPU and is checked by
the documentation build.

## Compensated dot products of Float32 data

`Double32s.compensated_dot` takes plain `Float32` vectors, forms every product
**exactly** (an `fma`-based error-free transformation), and accumulates in the
pair format:

```jldoctest advanced
julia> using Double32s, LinearAlgebra

julia> v = Float32.(1 ./ (1:100_000));

julia> dot(v, v)                       # Float32 BLAS: ~6 digits
1.6449082f0

julia> Float64(Double32s.compensated_dot(v, v))
1.6449240820936808
```

The reference value is `1.6449240820945124` — the compensated result carries
~12 correct digits from the same 4-byte inputs. `LinearAlgebra.dot` on
`Double32` vectors uses the same compensated kernel:

```jldoctest advanced
julia> x = Double32.(Float32.(1:5)) ./ 7;

julia> y = Double32.(ones(Float32, 5));

julia> Float64(dot(x, y))              # 15/7
2.142857142857139
```

## Dense linear algebra at ~48 bits

`*` and the five `LinearAlgebra.mul!` forms (`A*B`, `A*x`, `A'x`, `A'B`,
`AB'`) run split-channel compensated kernels — Float32 SIMD channels with a
Kahan-compensated low word. A Hilbert-matrix square, which is hostile to
`Float32`, against a 256-bit `BigFloat` reference:

```jldoctest advanced
julia> n = 6; H = [Double32(1.0f0) / (i + j - 1) for i in 1:n, j in 1:n];

julia> P = H * H;

julia> Float64(P[1, 1])                # true value 1.4913888888888889…
1.491388888888892
```

Normwise, the full product is at `~3e-15` relative error — against `~5e-8`
for the same product in `Float32` BLAS. The error stays flat as `n` grows
(measured through `n = 256` in the [benchmark report](../reports/benchmarks.md)).

## Structure of arrays: `split` and `unsplit`

GPUs and SIMD units prefer two contiguous `Float32` planes over an array of
structs. `split` produces exactly that view of the data, `unsplit` reassembles
it, and both round-trip bit-for-bit:

```jldoctest advanced
julia> a = Double32.(Float32[1, 3, 7]) ./ 3;

julia> S = Double32s.split(a);

julia> S.hi, S.lo
(Float32[0.33333334, 1.0, 2.3333333], Float32[-9.934108f-9, 0.0, 7.947286f-8])

julia> all(Double32s.unsplit(S) .=== a)
true
```

`Double32s.allocate_split(backend, n)` allocates the split form directly on
any `KernelAbstractions` backend, host included.

## Portable parallel routines: the `ak_*` family

The `ak_*` routines run on any `KernelAbstractions` backend through
AcceleratedKernels — the **same call** works on `CPU()` with nothing
installed, and on `CUDABackend()` or `ROCBackend()` once the vendor package
is loaded:

```jldoctest advanced
julia> v = Float32.(1 ./ (1:100_000));

julia> Float64(Double32s.ak_sum(v))            # Float32 in, Double32 out
12.09014619539721

julia> Float64(Double32s.ak_dot(v, v))         # products formed exactly
1.6449240820936808
```

Reductions are reproducible run-to-run on a given backend, but **not**
bit-identical across backends — the reduction trees differ. See the GPU
guide for what is and is not promised.

## On a GPU

The same code, moved to a device (not executed here; requires CUDA and a
GPU — measured numbers are in `docs/Reports/gpu_report.md`):

```julia
using Double32s, CUDA, KernelAbstractions

# Broadcast scalar kernels: device arithmetic is asserted bit-identical
# to the host in the test suite.
dx = CuArray(Double32.(Float32.(1:1_000_000) ./ 3))
dy = exp.(log.(dx))                    # elementary functions in-kernel

# Reductions and matmul on the device:
s = Double32s.ak_sum(dx)               # Double32 result
C = Double32s.ak_matmul(dA, dB)        # split-channel tiled kernel;
                                       # CPU and CUDA agree bit-for-bit

# Capability probe runs ON the backend it reports about:
Double32s.capability_report(CUDABackend())
```

At `1024³`, the device matmul measures `~0.4 ms` for `Double32×Double32` with
`~1e-13` normwise error — against `~6e-7` for `Float32` cuBLAS on the same
data (RTX 5090; `gpu_report.md`).
