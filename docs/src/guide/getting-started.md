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

Besides `Double32`, `using Double32s` exports the word accessors `HI`, `LO`,
`HILO`, plus `canonicalize`, `isnormal`, the named arithmetic (`add`, `sub`,
`mul`, `divide`, `reciprocal`, `rsqrt`), the compensated reductions, and
`norm2`, `matmul`, `matmul!`. See [Public API](@ref api-surface) for the full
list and for the two names that collide with other packages.

## Constructing values

From the types you already have:

```jldoctest
julia> Double32(1.0f0)                    # Float32: exact, free
1.0

julia> Double32(0.1)                      # Float64: rounded into ~48 bits
0.1

julia> Double32(2^24 + 1)                 # integers wider than Float32 keep their bits
16777217.0

julia> Double32(1//3)                     # host-only: Rational via BigFloat
0.333333333333333
```

The two-argument constructor takes a pair of `Float32` words and
canonicalizes them — any pair, in any order, overlapping or not:

```jldoctest
julia> Double32(2.0f0^-30, 1.0f0)         # words out of order: fixed
1.0000000009313226

julia> Double32(1.0f0, 0.5f0)             # overlapping words: renormalized
1.5
```

## Display and parsing

A `Double32` displays as a decimal: the shortest string that reconstructs the
identical value under `parse(Double32, s)`. The round trip is verified for that
value before the string is printed, rather than inferred from a digit count.

There is only one rendering, so every display path agrees — the REPL, arrays,
tuples, `Dict` values, `string`, and `$` interpolation:

```jldoctest
julia> x = sqrt(Double32(5))
2.23606797749979

julia> (x, x)
(2.23606797749979, 2.23606797749979)

julia> "the root is $x"
"the root is 2.23606797749979"

julia> parse(Double32, repr(x)) === x     # exact reconstruction
true
```

Use [`Double32s.HILO`](@ref) to see the representation behind the value:

```jldoctest
julia> Double32s.HILO(sqrt(Double32(5)))
(2.236068f0, -3.283041f-8)
```

!!! note "Reconstruction is `parse`, not `eval`"
    `eval(Meta.parse(repr(x)))` returns a `Float64`, because a bare decimal is
    `Float64` literal syntax and nothing about a type changes what the parser
    makes of that token. `parse(Double32, repr(x))` returns the `Double32`.
    `BigFloat` behaves the same way: `repr(big"0.1")` also evaluates to a
    `Float64`, and also reconstructs exactly through `parse`.

A decimal may be long. `Double32(1.0f-7)` prints 39 significant digits, because
the pair lattice is fine enough that no shorter decimal identifies that exact
value — the digits are required, not decorative.

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
2.0

julia> fma(Double32(2.0f0), Double32(3.0f0), one(Double32))   # compensated, not hardware-fused
7.0
```

Mixed expressions promote as the type hierarchy implies, with one case worth
stating explicitly:

```jldoctest
julia> one(Double32) + 1                  # integers promote to Double32
2.0

julia> one(Double32) + 1.0f0              # Float32 promotes to Double32
2.0

julia> one(Double32) + 1.0                # Float64 wins: wider in both directions
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

## Converting to other types

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
7.105427357601001858711242675781e-15

julia> floatmax(Double32)
3.40282356779733e38

julia> typemax(Double32), typemin(Double32)
(Inf, -Inf)
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

```jldoctest
julia> using Random

julia> v = rand(MersenneTwister(1), Double32, 3)
3-element Vector{Double32}:
 0.567905329600336
 0.968355522079257
 0.68204096259782

julia> Double32s.HILO(v[1])              # the low word is not zero
(0.5679053f0, 2.2784235f-8)
```
