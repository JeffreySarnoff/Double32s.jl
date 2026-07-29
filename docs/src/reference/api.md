# [Public API](@id api-surface)

```@meta
CurrentModule = Double32s
```

One exported name; everything else is `public` and reached as
`Double32s.name`. Nothing outside this page (and Base methods on `Double32`)
is supported API.

## The module

```@docs
Double32s
```

## The type

```@docs
Double32
```

## Accessors and normalization

```@docs
hi
lo
words
canonicalize
```

## Predicates and bit-level helpers

```@docs
isnormal
float64_reference_is_exact
lsb_exponent
raw_exponent
sqrt_nonneg
```

## [Backend capability probes](@id capability-probes)

The probe characterizes an execution target — the host thread, a device behind
an array, or a `KernelAbstractions.Backend` — by running arithmetic and reading
the answers, never by consulting a table of hardware names.

```@docs
capability_report
has_fused_multiply_add
has_gradual_underflow
probe_lane
probe_results
probe_operands
build_report
PROBE_LANES
UNDERFLOW_LANES
NO_LANES
```

## Precision constants

```@docs
nominal_eps
unit_roundoff
```

## Policies

```@docs
ArithmeticPolicy
StrictPolicy
NativePolicy
FastPolicy
DEFAULT_POLICY
```

## Kernel-safe operations

```@docs
sqrt_or_nan
```

## Reductions

```@docs
compensated_dot
compensated_sum
compensated_mapreduce
compensated_norm2
```

## Structure-of-arrays

```@docs
SplitDouble32Array
Double32s.split
unsplit
split!
unsplit!
allocate_split
allocate_split_host
```

## [The portable execution layer](@id ka-layer)

Built on `AcceleratedKernels` and `KernelAbstractions`, both hard
dependencies: everything here runs on `CPU()` out of the box, and on
`CUDABackend()`/`ROCBackend()` once the vendor package is loaded. What can be
a method of the `Base` or `LinearAlgebra` generic **is** one, with `Double32`
somewhere in the signature: `sum(Double32, A)` and `dot(Double32, x, y)`
spell "accumulate in `Double32`" for any input element type, and `sum`,
`mapreduce`, `reduce`, `map!`, and `dot` on `Double32` data dispatch here on
their own. Only `norm2`, `matmul`, and `matmul!` are package-local names —
no generic exists to extend.

```@docs
Base.sum(::Type{Double32}, ::DenseArray)
LinearAlgebra.dot(::Type{Double32}, ::AbstractArray, ::AbstractArray)
LinearAlgebra.dot(::StridedVector{Double32}, ::StridedVector{Double32})
norm2
Base.mapreduce(::Any, ::Any, ::AbstractArray, ::Double32)
Base.reduce(::Any, ::AbstractArray, ::Double32)
Base.map!(::Any, ::AbstractArray{Double32}, ::AbstractArray)
DEFAULT_GROUPSIZE
PolicyOp
matmul
matmul!
MATMUL_TILE
MATMUL_BLOCK
launch_config
kernel_report
wavefront_size
```

## Elementary functions

Roadmap Phase 6. Measured worst-case errors are in
`docs/Reports/accuracy_report.md`; none of these is faithfully rounded and none
is claimed to be.

```@docs
rsqrt
log_or_nan
TRIG_ARG_LIMIT
```

The rest are `Base` methods on `Double32` and carry their reasoning in their
own docstrings: `cbrt`, `exp`, `expm1`, `exp2`, `exp10`, `log`, `log1p`,
`log2`, `log10`, `sin`, `cos`, `sincos`, `tan`, `asin`, `acos`, `atan`,
`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, and `^`.

## Named values

```@docs
POS_ZERO
NEG_ZERO
POS_INF
NEG_INF
D32_NAN
```

## The EFT layer

Error-free transformations, in their own namespace:

```@docs
EFT
EFT.two_sum
EFT.two_sum_
EFT.two_hilo_sum
EFT.two_hilo_sum_
EFT.two_hilo_sum_checked
EFT.two_diff
EFT.two_diff_
EFT.two_prod
EFT.two_prod_
EFT.nonfinite_pair
```
