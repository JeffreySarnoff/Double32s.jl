# Double32s

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JeffreySarnoff.github.io/Double32s.jl/dev/)
[![Build Status](https://github.com/JeffreySarnoff/Double32s.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JeffreySarnoff/Double32s.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JeffreySarnoff/Double32s.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JeffreySarnoff/Double32s.jl)

A GPU-oriented double-word float: `Double32` carries about 48 significant bits
as the unevaluated sum of two `Float32` words, while keeping the `Float32`
exponent range.

## This package is not yet registered

### Installation

`Double32s.jl` is not registered in the Julia General registry, so install it directly from GitHub.

From the Julia REPL, press `]` to enter package mode:

```julia
(@v1.x) pkg> add https://github.com/JeffreySarnoff/Double32s.jl
```

Or equivalently:

```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/JeffreySarnoff/Double32s.jl")
```

### Using it in a project environment (recommended)

```bash
mkdir myproject && cd myproject
julia --project=.
```

```julia
(myproject) pkg> add https://github.com/JeffreySarnoff/Double32s.jl
```

```julia
(myproject) pkg> instantiate
```

### Development mode

To clone the source locally and edit it:

```julia
(@v1.x) pkg> dev https://github.com/JeffreySarnoff/Double32s.jl
```

The repo lands in `~/.julia/dev/Double32s/`. Pair with [Revise.jl](https://github.com/timholy/Revise.jl) for live reloading.

### Usage

```julia
using Double32s

# Add a minimal example here.
```

### Updating

```julia
(@v1.x) pkg> update Double32s
```

### Requirements

- Julia 1.12 or later
- Git available on your `PATH`

----
----

```julia
julia> using Double32s

julia> x = Double32(1.0f0) / 3
Double32(0.33333334f0, -9.934108f-9)

julia> Float64(x)
0.33333333333333304

julia> sum(Double32, rand(Float32, 10^6))    # Float32 data, Double32 accumulator
```

Every scalar operation is allocation-free, type-stable, and branch-free on the
hot path, with a single output check routing to a cold guarded path.

> **The module is `Double32s`; the type is `Double32`.** A Julia module cannot
> also bind its own name to a type — inside `module Double32`, `using Double32`
> imports the *module*, and the scalar would be unreachable as a bare name.

## Status

The scalar core is complete and measured. GPU execution (KernelAbstractions,
CUDA, AMDGPU) is not yet implemented — see
[`docs/Checkpoint.md`](docs/Checkpoint.md) for exactly what is done, what is
not, and which analysis questions remain open.

## Accuracy

Measured against a 400-bit `BigFloat` reference over random and adversarial
operand pools, with the underflow/overflow zones excluded per the design's
error model. `u² = 2^-48`.

| Operation | worst relative error |
|---|---|
| `+`, `-` | 1.25 u² |
| `*` | 3.4 u² |
| `/` | 4.7 u² |
| `sqrt` | 1.2 u² |
| `inv` | 2.4 u² |
| `fma` | 2.0 u² (as `abserr / max(\|xy\|,\|z\|)`) |

`Double32` is **not** an IEEE format with a uniform significand lattice. The
distance to the next representable pair is not `eps(Double32)` and can be as
small as `2f0^-149` beside a high word of `1.0f0`; `nextfloat` and `prevfloat`
are deliberately not implemented. Two consequences worth knowing:

- `Float64(x)` is **not** exact for every pair. Use `Double32s.reference(x)`
  (a `BigFloat`) when exactness matters, or gate on
  `Double32s.float64_reference_is_exact(x)`.
- comparison against `Float64` is done exactly in the pair domain rather than by
  promotion, so `Double32(1.0f0, 2f0^-149) == 1.0` is correctly `false`.

## API

One exported name, `Double32`. Everything else is `public` and reached as
`Double32s.name`:

```
hi, lo, words, canonicalize, isnormal, reference, float64_reference_is_exact
StrictPolicy, NativePolicy, FastPolicy, DEFAULT_POLICY
add, sub, mul, divide, reciprocal, square_root, sqrt_or_nan, compensated_fma
nominal_eps, unit_roundoff
SplitDouble32Array, split, unsplit
compensated_dot, compensated_sum, compensated_mapreduce
EFT.two_sum, EFT.two_prod, …
```

Use `Double32s.sqrt_or_nan` rather than `Base.sqrt` inside kernels: `sqrt`
throws a `DomainError` for negative arguments to match `sqrt(::Float32)`, and a
`throw` forces the exception machinery into device code.

## Design

[`docs/Design/DesignDouble32.md`](docs/Design/DesignDouble32.md) is the full
design; [`docs/Design/UseFableHere.md`](docs/Design/UseFableHere.md) ranks the
parts whose failure mode is a wrong number rather than an error.

Both documents record defects that were found, including three in the design's
own earlier corrections. Every numerical claim in them carries the size of the
sweep that produced it — a claim without one is a claim that has not been run.
