# [Basic Uses](@id examples-basic)

Everything on this page runs on a plain CPU with nothing installed beyond
`Double32s` itself. Every output shown is checked by the documentation build.

## Fourteen digits from Float32 words

A `Double32` is the unevaluated sum of two `Float32` words. Arithmetic keeps
about 48 significant bits — ~14 decimal digits — where `Float32` keeps ~7:

```jldoctest basic
julia> using Double32s

julia> x = Double32(1.0f0) / 3
Double32(0.33333334f0, -9.934108f-9)

julia> Float64(x)               # ~14 correct digits
0.33333333333333304

julia> Float64(1.0f0 / 3.0f0)   # Float32 alone: ~7
0.3333333432674408
```

The two words are visible in the printed form: the high word is the nearest
`Float32`, the low word carries what the high word could not.

```jldoctest basic
julia> Float64(sqrt(Double32(2.0f0)))
1.4142135623730958

julia> sqrt(2.0)                # Float64 reference
1.4142135623730951
```

## The headline use: accumulate Float32 data exactly

Summing `Float32` data in `Float32` loses digits to accumulated rounding.
Summing it through `Double32` needs **no package-specific API** — `Base`
already spells it — and returns the sum of the stored values essentially
exactly:

```jldoctest basic
julia> A = fill(0.1f0, 10_000);

julia> sum(A)                   # Float32 accumulation: 4 digits survive
999.9998f0

julia> Float64(sum(Double32, A))
1000.0000149011612
```

That `…0149` is not an error — `0.1f0` is not exactly `1/10`, and
`1000.0000149011612` is the true sum of ten thousand copies of the value
actually stored. `Double32` reports the data as it is; `Float32` accumulation
buries it in rounding noise.

The same holds for ill-behaved series:

```jldoctest basic
julia> v = Float32.(1 ./ (1:100_000));

julia> sum(v)                   # Float32: 6 digits
12.090148f0

julia> Float64(sum(Double32, v))
12.09014619539721
```

The reference value to 17 digits is `12.090146195397210`: every printed digit
of the `Double32` sum is correct.

## Elementary functions at full width

The whole elementary family — `exp`, `log`, trigonometric, hyperbolic, and
their inverses — is implemented directly on the pair format and measured
against a 400-bit reference (see the [accuracy report](../reports/accuracy.md)):

```jldoctest basic
julia> Float64(sin(Double32(1.0f0)))
0.8414709848078967

julia> sin(1.0)                 # Float64 reference
0.8414709848078965

julia> Float64(exp(Double32(1.0f0)))
2.718281828459041

julia> Float64(4 * atan(one(Double32)))
3.1415926535897967
```

`expm1` and `log1p` exist for the same reason they exist for `Float64`: near
zero, `exp(x) - 1` and `log(1 + x)` destroy exactly the digits being asked
for.

```jldoctest basic
julia> x = Double32(1.0f-7);

julia> Float64(expm1(x))        # correct: 1e-7 + 5e-15 - ...
1.0000000616860965e-7

julia> expm1(x) === exp(x) - one(Double32)   # the naive spelling differs
false
```

## Integer powers and exact operations

Operations that should be exact, are, and stay inside the type:

```jldoctest basic
julia> Double32(2.0f0)^10
Double32(1024.0f0, 0.0f0)

julia> log2(Double32(1024.0f0))
Double32(10.0f0, 0.0f0)

julia> Double32(1.0f0) + 0.5f0          # Float32 promotes to Double32
Double32(1.5f0, 0.0f0)
```

`Double32` widens the significand, **not** the exponent: overflow happens at
`Float32`'s `~3.4e38`, and `exp` saturates accordingly:

```jldoctest basic
julia> exp(Double32(1000.0f0)), exp(Double32(-1000.0f0))
(Double32(Inf32, 0.0f0), Double32(0.0f0, 0.0f0))
```
