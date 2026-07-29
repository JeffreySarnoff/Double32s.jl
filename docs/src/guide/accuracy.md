# [Accuracy Model](@id accuracy-model)

```@meta
CurrentModule = Double32s
DocTestSetup = quote
    using Double32s
end
```

Every number here was measured against a 400-bit `BigFloat` reference, and
the regressions live in the test suite. `docs/Reports/accuracies.jl`
regenerates the full report on your machine.

## The represented value

A finite `Double32` is the **exact real sum** `hi + lo` of its two `Float32`
words, canonicalized so `hi` is the nearest `Float32` to that sum and
`|lo| ≤ ½ulp(hi)`. Nominal precision is 48 bits (`precision(Double32) == 48`),
unit roundoff `u² = 2⁻⁴⁸ ≈ 3.6e-15` relative.

Two consequences to internalize:

**It is not a lattice.** Canonicality bounds `lo` from above only, so
`(1.0f0, 2f0^-149)` is a legitimate value — the representable set is
nonuniform, `eps` is a nominal constant rather than a neighbor distance, and
`nextfloat`/`prevfloat` deliberately do not exist.

**`Float64` conversion is not always exact.** A canonical pair can span up to
150 bits; `Float64(x)` is exact iff the pair spans at most 53:

```jldoctest
julia> x = Double32(1.0f0, 2.0f0^-149);

julia> Double32s.float64_reference_is_exact(x)
false

julia> Double32s.reference(x) == BigFloat(x)     # BigFloat is always exact
true
```

Use `Double32s.reference` for validation; gate any fast `Float64` reference
on the predicate. (The predicate is span-based and was validated over
1,701,602 canonical pairs, 0 mismatches.)

## Measured operation accuracy

Errors in **nominal 48-bit ulps of the true value**
(`ulp48(V) = 2^(⌊log2|V|⌋−47)`); "faithful" means error < 1 ulp48, i.e. one
of the two neighboring 48-bit model values.

| operation | worst measured | faithful? | notes |
|---|---|---|---|
| `+`, `-` | 2.0 u² relative | — | proven ≤ 3u²/(1−4u) (machine-checked, see below) |
| `*` | 3.4 u² relative | — | includes over/underflow band guards |
| `/` | **4.00 ulp48** | no (1.56% of cases ≥ 1) | decision documented below |
| `inv` | 1.16 ulp48 | 1 violation / 319 | |
| `sqrt` | **0.75 ulp48** | **yes, 0 violations** | incl. near-perfect-square adversaries |
| `fma`/`muladd` | 2.0000 u²·M | — | derived bound `2u²(1+80u)·M`, tight |

Three of these deserve their stories:

**Addition is proven, not just measured.** The kernel is AccurateDWPlusDW:
Joldeş, Muller & Popescu, *ACM TOMS* 44(2), Article 15res (2017),
Algorithm 6 with the bound `3u²/(1−4u)` from Theorem 3.1 (both p. 15res:7,
valid for `p ≥ 3`, so at this precision); machine-checked in Coq and proven
asymptotically optimal by Muller & Rideau, *ACM TOMS* 48(1), Article 9
(2022), §2.2 Property 2.1 — where the same algorithm is renumbered
Algorithm 4. DOIs `10.1145/3121432`, `10.1145/3484514`.

**Division is deliberately not faithful.** A second Newton correction is
implemented and measured (2.30 ulp48 worst) — it *still* isn't faithful, and
it doubles the cost of every device kernel. So every policy uses one
correction and the 4.0-ulp48 bound is documented instead of wished away.
Faithful double-word division requires a different algorithm, not a repeated
correction.

**`fma` is bounded absolutely, of necessity.** Under near-total cancellation
the true result of `x*y + z` can be `2^-173` relative to unit operands —
below anything a 48-bit format can express relatively. The bound
`2u²(1+80u)·max(|x·y|, |z|)` is derived in the source, measured at exactly
`2.0000u²`, and the test asserts the derived constant, not a tolerance.

## Where precision tapers

The low word lives in the same `Float32` exponent range as the high word, so
as results shrink toward the underflow boundary the low word runs out of
room:

| result magnitude | usable precision |
|---|---|
| `≥ 2^-102` | full ~48 bits |
| `2^-126` … `2^-102` | tapers 48 → 24 bits (low word goes subnormal) |
| `2^-149` … `2^-126` | 24 bits and falling (high word subnormal, `lo == 0`) |

This is the *format*, not the kernels: the operations were hardened so they
deliver the nearest **representable** pair through this region (division and
`sqrt` scale their operands; multiplication has an underflow-band rescue that
recovered 23 silently-lost bits during development). Accuracy statements
below `2^-100` are absolute, not relative, by design.

All subnormal-range behavior additionally assumes **gradual underflow**;
check the executing backend:

```jldoctest
julia> Double32s.has_gradual_underflow()
true
```

## Exceptional values

IEEE semantics throughout, verified against `Float32` behavior exhaustively
over the classification cross-product: NaN propagates (including through
`min`/`max`), infinities saturate with correct signs, signed zeros follow
IEEE through `+`, `-`, `*` — and overflow rescue paths return `Inf` only when
the true result genuinely rounds to `Inf`:

```jldoctest
julia> x = Double32(2.0f0^64, -2.0f0^40);       # x*x overflows its leading term...

julia> x * x                                     # ...but the true square is finite
3.402823263561204650859e+38
```

## Comparing against other types

- `==`/`<`/`isless` against `Float64` are computed **exactly in the pair
  domain**, never through the lossy conversion (918 boundary cases, 0
  mismatches);
- `Float32`, `Float16`, and integers convert exactly, so promotion is safe
  there;
- `hash` follows the `Float64` value, so `Dict` and `Set` behave consistently
  across types; `isequal`/`isless` form a total order tested over the entire
  signed-zero surface.
