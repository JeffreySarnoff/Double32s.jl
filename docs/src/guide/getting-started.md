# [Getting Started](@id getting-started)

```@meta
CurrentModule = Double32s
DocTestSetup = quote
    using Double32s
end
```

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/JeffreySarnoff/Double32s.jl")
using Double32s
```

`Double32` is the only exported name.

## Constructing values

From the types you already have:

```jldoctest
julia> Double32(1.0f0)                    # Float32: exact, free
Double32(1.0f0, 0.0f0)

julia> Double32(0.1)                      # Float64: rounded into ~48 bits
Double32(0.1f0, -1.4901161f-9)

julia> Double32(2^24 + 1)                 # integers wider than Float32 keep their bits
Double32(1.6777216f7, 1.0f0)

julia> Double32(1//3)                     # host-only: Rational via BigFloat
Double32(0.33333334f0, -9.934108f-9)
```

The two-argument constructor takes a pair of `Float32` words and
canonicalizes them — any pair, in any order, overlapping or not:

```jldoctest
julia> Double32(2.0f0^-30, 1.0f0)         # words out of order: fixed
Double32(1.0f0, 9.313226f-10)

julia> Double32(1.0f0, 0.5f0)             # overlapping words: renormalized
Double32(1.5f0, 0.0f0)
```

`show` prints the exact pair, and the printed form reconstructs the value —
that is Base's round-trip contract for two-argument `show`.

## Arithmetic

`Double32 <: AbstractFloat`, so it works everywhere a float works: arithmetic,
comparisons, `Dict` keys, sorting, broadcasting, `Complex{Double32}`,
generic linear algebra.

```jldoctest
julia> a = Double32(1.0f0) / 3;

julia> a * 3 == 1                         # rounds to exactly one
true

julia> a + a + a == 1                     # the sum keeps its ~48-bit residue
false

julia> Float64(a + a + a)
0.9999999999999991

julia> sqrt(Double32(2.0f0))^2
Double32(2.0f0, 0.0f0)

julia> fma(Double32(2.0f0), Double32(3.0f0), one(Double32))   # compensated, not hardware-fused
Double32(7.0f0, 0.0f0)
```

Mixed expressions promote the way you would expect — and one case deserves
attention:

```jldoctest
julia> one(Double32) + 1                  # integers promote to Double32
Double32(2.0f0, 0.0f0)

julia> one(Double32) + 1.0f0              # Float32 promotes to Double32
Double32(2.0f0, 0.0f0)

julia> one(Double32) + 1.0                # Float64 WINS: wider in both directions
2.0
```

`promote_type(Double32, Float64) == Float64` because `Float64` has both a
wider significand (53 vs ~48 bits) and a wider exponent range — it loses the
least. Comparisons against `Float64` do **not** go through that lossy
promotion; they are computed exactly in the pair domain:

```jldoctest
julia> x = Double32(1.0f0, 2.0f0^-149);   # one quantum above 1

julia> Float64(x) == 1.0                  # the conversion is lossy here...
true

julia> x == 1.0                           # ...the comparison is not
false

julia> x > 1.0
true
```

## Conversions out

```jldoctest
julia> x = Double32(1.0f0) / 3;

julia> Float32(x)                         # the high word: nearest Float32
0.33333334f0

julia> Float64(x)                         # within one Float64 rounding
0.33333333333333304

julia> BigFloat(x) == Double32s.reference(x)   # always exact
true
```

For validation work, prefer `Double32s.reference(x)` (a `BigFloat`) or gate a
fast `Float64` reference on [`Double32s.float64_reference_is_exact`](@ref) —
`Float64(x)` alone silently loses low-word bits for wide-gap pairs.

## Constants and properties

```jldoctest
julia> precision(Double32), precision(Double32; base = 10)
(48, 14)

julia> eps(Double32)                      # nominal 48-bit ulp — see the warning below
Double32(7.1054274f-15, 0.0f0)

julia> floatmax(Double32)
Double32(3.4028235f38, 1.0141204f31)

julia> typemax(Double32), typemin(Double32)
(Double32(Inf32, 0.0f0), Double32(-Inf32, 0.0f0))
```

!!! warning "`eps` is nominal, not adjacency"
    `eps(Double32) == 2.0^-47` is the right constant for tolerances and
    convergence tests, and it is **not** the distance to the next
    representable value, which can be as small as `2f0^-149` next to `1.0`.
    `nextfloat`/`prevfloat` are deliberately not defined; code that needs
    adjacency semantics on this format needs to know it is not a lattice.
    See [Accuracy Model](@ref accuracy-model).

## Random numbers

`rand(Double32)` draws uniformly from a true 48-bit grid in `[0, 1)` — the
low word is genuinely random, not the zero that a naive
`Double32(rand(Float32))` would produce:

```julia
julia> using Random

julia> v = rand(MersenneTwister(1), Double32, 3)
3-element Vector{Double32}:
 Double32(0.23603334f0, -6.352898f-9)
 Double32(0.34651703f0, 4.2467468f-9)
 Double32(0.31270003f0, 1.1071297f-8)
```
