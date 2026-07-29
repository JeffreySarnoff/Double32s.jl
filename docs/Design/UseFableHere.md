# Where to spend the strongest reasoning

**Companion to:** `DesignDouble32.md`
**Purpose:** the parts of the design whose failure mode is a wrong number rather
than an error, ranked by how hard they are to get right and how unlikely a test
suite is to notice.

**Status of this revision (2026-07-28 20:01 EDT): nothing in this document is
open.** Tier 0 and Tiers 1–3 are all closed, each with a measurement and a
regression in `test/core/invariants.jl`. Two defects were found while closing
Tier 3, and one claim in `DesignDouble32.md` §14.3 turned out to be folklore and
has been replaced with what KA's source actually does — see 3.1.

Every Tier 1 item was independently re-derived and checked numerically against a
400-bit `BigFloat` reference on Julia 1.12. **Both Tier 1 "fixes" were themselves
wrong**, and a third defect of the same family was found in a section the
previous revision did not flag at all. All three are corrected in
`DesignDouble32.md`; the corrections carry their operand counts so the next
reader can tell verification from assertion.

The common shape of everything below: an invariant is *established* in one
section and *relied upon* in another, and violating it produces a plausible
result. Reviewing a plausible-looking derivation is exactly what has now failed
four times — re-derive these independently rather than checking them.

---

## Tier 0 — the corrections were wrong; now corrected and verified

These were listed in the previous revision as *closed*. They were not. This is
the single most important entry in the document: a defect marked fixed is worse
than a defect marked open, because the wrong claim now has an argument standing
behind it.

### 0.1 The `Float64`-exactness condition was wrong in **both** directions

**Was claimed** (§4.2, §5.1, §22.1, §23.11):

```text
exact  ⟺  lo == 0, or lo is normal, or exponent(hi) ≤ -97
```

**Both halves fail.**

*A normal `lo` does not imply exactness.* Canonicality bounds `exponent(lo)`
from above (`el ≤ eh - 24`) and never from below, so a canonical pair may carry
an arbitrarily wide gap of implicit zeros. `two_sum(2f0^-62, floatmin(Float32))`
returns a canonical pair with a **normal** low word spanning 65 bits.
`two_sum(1.0f0, 2f0^-100)` spans 101. Both convert inexactly.

The proof sketch that supported this contained a flipped inequality: from
`el ≤ eh - 24` it concluded the lowest bit of `lo` lies at
`2^(el-23) ≥ 2^(eh-47)`. That subtraction gives `el - 23 ≤ eh - 47` — an *upper*
bound on the bit position, which constrains nothing.

*The `-97` boundary is not tight.* `(2f0^-96, -2f0^-149)` is canonical and
converts **exactly**: `hi` is a power of two and `lo` opposes its sign, so
cancellation drops the sum one binade and buys a 53rd bit.

**Measured:** 31,030 disagreements between that condition and reality over a
1,701,602-pair structured sweep.

**Correct condition** — a statement about the *span* of the pair, not the class
of either word:

```text
exact  ⟺  lo == 0,  or  lsb_exponent(lo) ≥ exponent(hi + lo) - 52
```

where `exponent(hi + lo)` is `exponent(hi)`, less one exactly when `hi` is a
power of two and `signbit(lo) ≠ signbit(hi)`. Proof in §4.2, closed form in
§22.1. **Validated: 1,701,602 canonical pairs, 0 mismatches.** Dropping the
cancellation term gives a conservative variant that can only under-report
exactness — the safe direction for a reference gate.

**Coverage — closed.** The remaining question was not the condition but whether
the generator actually produces the subclass the wrong predicate waved through:
wide-gap pairs with *normal* low words. It does. `wide_gap_pairs()` yields 172
pairs, **96 with normal low words, spanning 24 to 144 bits**, of which **64 are
`Float64`-inexact** — and the retired predicate would have called all 64 exact.
The random pool supplies 15,358 more independently.

Coverage claims decay silently, so this is now an assertion rather than a
finding: `test/core/invariants.jl` requires the pool to contain the subclass
*and* requires the retired predicate to still be visibly wrong on it. A
generator edit that drops the subclass fails the suite instead of quietly
making every exactness test vacuous.

### 0.2 The zero rule was scoped to the wrong operand

**Was claimed** (§8.8, §8.10, §23.12): invariant 6 relaxes to "when `hi` is
zero, `lo` is a signed zero whose sign is unspecified", and therefore a
total-order predicate "may inspect `lo` only when `hi` is nonzero".

**Why it is false:** `-` flips the sign bit of `lo` unconditionally. A negative
zero low word occurs under *every* high word, not only a zero one:

```text
-one(Double32)   = (-1.0f0, -0.0f0)      Double32(-1.0f0) = (-1.0f0, +0.0f0)
-D32_NAN         = (NaN32,  -0.0f0)      D32_NAN          = (NaN32,  +0.0f0)
```

Each row is two bit-distinct pairs holding one value. Under the guarded-on-`hi`
predicates, `isequal(-one(Double32), Double32(-1.0f0))` is `false` while `==`
and `hash` both say they are identical — so `Dict` lookup misses (measured:
`haskey` returns `false`), `unique` keeps duplicates, `issorted` disagrees with
`==`, and `isequal(D32_NAN, -D32_NAN)` is `false` where
`isequal(NaN32, -NaN32)` is `true`.

**The trichotomy property test does not catch this.** Exactly one of
`isless(x,y)`, `isless(y,x)`, `isequal(x,y)` still holds. Over the 576-pair
enumeration of the full sign-of-zero surface: **0 trichotomy violations, 20
semantic mismatches.** The previous revision's claim that the §22.3 properties
"would have caught" this family of defect is true of the first failure and false
of the second.

**Correct rule:** *the sign of a zero low word is never information, at any high
word.* Compare low words with `==` and `<`, never `isequal`/`isless`:

```julia
Base.isequal(x::Double32, y::Double32) = isequal(x.hi, y.hi) & (x.lo == y.lo)
Base.isless(x::Double32, y::Double32)  = isless(x.hi, y.hi) |
                                         (isequal(x.hi, y.hi) & (x.lo < y.lo))
```

The `iszero(x.hi)` guards vanish — fewer operations *and* correct. **Validated:
576 pairs, 0 trichotomy violations, 0 semantic mismatches.** Promoted to
invariant 8 of §5.2 so it binds construction, not just comparison.

**The rest of the surface — audited, and two leaks found.** The open question
was whether any *other* method distinguishes low-word zero signs. Every method
reachable from a `Double32` was enumerated and probed over the signed-zero
surface, on the 36 `isequal` pairs it contains:

| method | before | after |
|---|---|---|
| `hash`, `Float64`, `Float32`, `Dict`, `Set`, `unique`, `sort`, `Base.Order.Forward`/`Reverse` | 0 | 0 |
| `show` / `repr` / `string` | **18 leaked** | 0 |
| `reference` (the `BigFloat` ground truth) | **2 leaked** | 0 |
| `objectid` | 11 differ | 11 differ — *correct*, see below |

- **`show` printed the raw sign**, making `repr` a function of the bit pattern
  rather than of the value: `isequal(a, b)` while `repr(a) != repr(b)`. Fixed —
  a zero low word now always prints `0.0f0`. The sign of a zero *high* word is
  untouched, because there it is the value's own sign, so `NEG_ZERO` still
  round-trips. Nothing diagnostic is hidden: a `-0.0` low word is never a defect.
- **`reference(x) = big(x.hi) + big(x.lo)` leaked the sign into the ground
  truth.** `big(-0.0f0) + big(-0.0f0)` is `-0.0` while `big(-0.0f0) + big(0.0f0)`
  is `+0.0`, so two `isequal` pairs got `BigFloat` references that were not
  `isequal` — and this is the function every accuracy sweep in the package
  treats as truth. Fixed by dropping a zero low word rather than adding it.
  This is the more dangerous of the two: a leak in the *reference* is a leak
  that can only ever be found by looking for it.
- **`objectid` is left alone deliberately.** It distinguishes `0.0` from `-0.0`
  in Base too (verified), and `IdDict` is identity-based by contract. Making it
  value-based would be the defect.

**Serialization needs no change, because a normal form already exists.**
`canonicalize` emits `+0.0f0` for every zero low word, so `isequal` pairs
canonicalize to *identical bits* — verified, 0 negative-zero low words emitted
over the surface. Byte-wise comparison, content hashing, and serialization
should canonicalize first; that is now stated on `canonicalize`.

Normalizing at construction was **rejected on performance grounds**: `raw` is
the hot path and negation is `raw(-hi, -lo)`, so clearing the sign there adds a
select to every negation to fix something no value-based operation can observe.

Standing regression: `test/core/invariants.jl` re-runs the whole audit,
including the two named cases, so a new method that leaks fails the suite.

### 0.3 `min`/`max` swallow NaN — previously unflagged

Not mentioned in the previous revision at all, and the same family: a predicate
written from `<` alone, where `<` cannot see what the semantics require.

```julia
Base.min(x, y) = ifelse(isnan(x) | (y < x), y, x)      # WRONG
```

`min(Double32(1), D32_NAN)` returns `1`, because `isnan(x)` tests the wrong
operand and `y < x` is `false` when `y` is NaN. In a reduction this turns a
poisoned accumulator clean. Separately, `min(0.0, -0.0)` returns `+0.0` where
Base returns `-0.0`. **13 disagreements with `Base.min`/`Base.max` over the 49
combinations of `±1.0f0`, `±0.0f0`, `±Inf32`, `NaN32`.** Corrected form in
§8.11, verified 0 mismatches.

---

## Tier 1 — verified correct as written; no change needed

Recorded so the next reviewer does not re-litigate them.

### 1.1 The exact mixed `Double32`/`Float64` comparison (§8.10)

The previous revision marked this "a sketch — Tier 1 material itself". It is
correct. The `==` method holds at every boundary, and ordering needs no special
cases at all:

```julia
Base.:(<)(x::Double32, y::Float64) = Float64(x.lo) < y - Float64(x.hi)
Base.:(<)(y::Float64, x::Double32) = y - Float64(x.hi) < Float64(x.lo)
```

`y - hi` is exact by Sterbenz in the only regime where its exact value matters,
and faithful rounding preserves the answer outside it; overflow is impossible
because `floatmax(Float32) ≪ ulp(floatmax(Float64))`; the `Inf - Inf → NaN →
false` cases are the IEEE-correct ones. **918 operand combinations spanning
`±0.0`, `±Inf`, `NaN`, `floatmax`, `2.0^-1074`, and both subnormal and normal
low words: 0 mismatches for `==`, `<`, and `>`.** The "Note Well" caveat has
been removed from §8.10 and the ordering methods added.

### 1.2 `floatmax(Double32)` (§11.2) — previously Tier 2.4

Flagged as "exactly the shape of argument that is off by one bit half the time".
Re-derived; every step holds. `ulp(floatmax(Float32)) == 2^104`; half-ulp bound
`2^103`; significand of `0x7f7fffff` is odd; `two_sum(floatmax, 2^103)` gives an
infinite high word while `two_sum(floatmax, prevfloat(2^103))` returns the pair
unchanged. `V ≈ 3.4028235677973306e38`. §28 experiment 11 is closed.

---

## Tier 2 — resolved

All five items below are now closed in the implementation, each with the
resolution written into the relevant docstring and a regression in the test
suite. One constraint ran through every decision: **this code runs in GPU
kernels**, so a resolution that costs hot-path operations, registers, or warp
divergence had to buy something the format can actually deliver.

### 2.1 The unchecked `two_hilo_sum_` preconditions — resolved by citation

The `|hi| ≥ |c|` justification stays refuted (834 violations in 40,000
structured additions, including `hi = -2^-24, c = -3·2^-25`). The general proof
the previous revision asked for exists in the literature: `add_dd_` is,
operation for operation, **AccurateDWPlusDW** — Algorithm 6 of Joldeş, Muller &
Popescu (ACM TOMS 2017) — with relative error `≤ 3u²/(1-4u)`, and that proof
was **machine-checked in Coq/Flocq by Muller & Rideau (ACM TOMS 2022)**,
generic in the precision (hence valid at p = 24) and covering the Fast2Sum
steps, so the algorithm's correctness never rested on the naive precondition.
`sub_dd_` inherits the bound by exact negation.

Standing evidence stays live: the `two_sum_` differential (0 disagreements over
1.54M additions) remains in `test/core/eft.jl` as a regression, and the other
`two_hilo_sum_` call sites each carry a `|correction| ≤ u|leading|` magnitude
argument in a comment. Measured worst addition error 2.0u², under the 3u²
ceiling.

### 2.2 The multiplication overflow boundary — proof written, plus two bugs

The trigger and scale-factor proofs now live in the `mul_guarded` and
`div_guarded` docstrings. The load-bearing steps: non-finite output forces
`es = e_x + e_y ≥ 126` and tiny output forces `es ≤ -97` (disjoint, so one
signed `2^±64` shift serves both); scaled operands stay normal in both
directions; and the final exact rescale **maps the Float32 round-up-to-binade
boundary precisely onto the `2^128 - 2^103` overflow threshold, ties included**
(both `1.0` and `2^128` have even significands), so the rescue never
misclassifies the boundary.

Writing the proof surfaced two real defects, both fixed and regression-tested:

- **division had no rescue at all and returned NaN** — not even `Inf` — for
  representable quotients: `Double32(2f0^127) / Double32(0.5f0, 2f0^-25)` has
  true quotient `floatmax + 2^80`, and the overflowed `q0` poisoned the
  residual with `Inf - Inf`. Fixed with a `2^-64` numerator rescue; still-
  overflowing retries saturate to a *signed* infinity (the NaN would otherwise
  reappear there).
- **the multiplication guard triggered only on non-finite output**, leaving the
  underflow band `[2^-126, 2^-98]` silently degraded — measured 8.2e-8 relative
  error (2.3e7 u², ~23 bits) where `two_prod_` residuals die at `2^-149`. The
  output check is now a range check (`MUL_TINY = 2^-98`, one abs + two
  compares, still a single branch, hot-path `zero_sign` select deleted in
  exchange: measured `*` cost 4.65 ns/op vs 5.02 before). After the scaled
  retry the band holds `4u²·|V| + 2^-149` — the representation optimum, since
  at `exponent(hi) = -126` the canonical bound `½ulp(hi) = 2^-150` admits no
  low word at all.

### 2.3 `two_prod_` residual underflow — neutralized per consumer, probed per backend

The hazard table now lives in the `two_prod_` docstring: `*` covered by the
2.2 range guard; `/`, `inv`, `sqrt` covered by their operand scaling; `fma`
bound stated with an explicit validity window (`M ≥ 2^-77`); `compensated_dot`
documents the `≥ 2^-101` product condition. The backend-conditional part is
now testable: `has_gradual_underflow()` runs the §5.9 probe list — subnormal
add / mul / fma results, a `two_prod_` case with subnormal exact residual,
signed-subnormal survival.

Two probe implementation traps, both hit and both now documented in the
source: expected values must be `reinterpret` constants (under FTZ the probe
would otherwise compute its own expectations with the faulty arithmetic), and
operands must load from a runtime array — with literal operands **the compiler
constant-folds the entire probe to `true`**, which is how the first version
failed its own `set_zero_subnormals` test. Pure `Float32`, zero allocations,
so the same body can later run inside a KA kernel with operands as arguments.

### 2.4 One-correction faithfulness for `/` and `sqrt` — measured, decided

Lattice-based measurement (error in nominal-48-bit ulps against a 400-bit
`BigFloat` reference, restricted to `|V| ≥ 2^-100` where the 48-bit lattice
exists; 101,761 quotients):

| kernel | worst (ulp48) | not faithful |
|---|---|---|
| `div_dd_` (one correction) | 4.00 | 1.56% |
| `div_dd_refined_` (two) | 2.30 | 0.66% |
| `inv` (one) | 1.16 | 1 case / 319 |
| `sqrt` (one) | **0.75** | **0** |

Decisions, GPU-weighted:

- **`sqrt` is empirically faithful with one correction** — including the
  near-perfect-square adversaries of §23.8. Experiment 10's sqrt half: closed,
  no second refinement.
- **division keeps one correction for every policy.** The refinement doubles
  kernel cost and register pressure *and still is not faithful* — §8.4's
  contingency assumed it would close the gap, and it does not. Faithful DW
  division needs a different algorithm, not a repeated correction. The measured
  4.00-ulp48 bound is documented on `div_dd_`; `div_dd_refined_` stays
  implemented, measured, and regression-tested for a future host-only policy.
- One earlier misstep worth recording: the first faithfulness sweep reported
  identical 7.11-ulp worsts for both kernels — because it included quotients
  near `2^-105`, where `ulp48 = 2^-153` is **unrepresentable** and no algorithm
  at any cost can be faithful. Faithfulness claims are meaningful only where
  the lattice exists; the sweep now filters accordingly.

### 2.5 Compensated-FMA term ordering — bound derived

Derived in the `fma_dd_` docstring: for `M = max(|xh·yh|, |zh|) ∈ [2^-77, 2^126]`,

```text
|fma_dd_(x,y,z) − (x·y + z)| ≤ 2u²·(1 + 80u)·M
```

The proof telescopes the exact 2Sum/2Prod chain — every term is accounted for
exactly except the `xl·yl` rounding (`u³M`) — leaving the final `l + low`
recombination as the **only** O(u²) rounding: `≤ 2u²M(1+16u)`; the six `low`
accumulations contribute `≤ 126u³M`. Measured worst: `2.0000u²·M`, so the bound
is tight to under one percent, and the test asserts the proof's constant
(`≤ 2.01u²`), not a tolerance. The ordering question closes with it: once the
exact-telescope structure is fixed, ordering moves only the O(u³) terms; the
`2u²` floor is irreducible without a three-word accumulator. The bound is
absolute in `M` by necessity — under near-total cancellation the true result
can sit at `2^-173` relative to unit operands, below anything any 48-bit
format can express relatively.

---

## Tier 3 — resolved

All five closed. Where a decision was required it was made on performance,
clarity, and throughput, and the reasoning is recorded at the decision site
rather than only here.

### 3.1 Barrier convergence under `unsafe_indices` (§14.3–14.4) — the design's justification was wrong

Read from `KernelAbstractions/src/macros.jl` rather than assumed, because the
natural assumption is the false one.

With `unsafe_indices=false` (the **default**), `transform_gpu!` emits
`__active_lane__ = __validindex(__ctx__)` and then calls `split(emit_gpu, body)`.
`split` cuts the body **at each `@synchronize`**; `emit_gpu` wraps only the
statements *between* barriers in `if __active_lane__ … end`, and the barrier is
emitted between segments, outside the conditional.

**So the default is already barrier-convergent by construction.** The design
implied that KA's bounds handling places a barrier in divergent control flow and
that `unsafe_indices=true` is the rescue. It is not. What the annotation
actually does is skip *both* the `__validindex` computation and the splitting,
emitting the body raw — so every lane, in range or not, runs every statement.

**Decision — it is a throughput choice, not a correctness one:**

- **`unsafe_indices=true` only when both hold:** the kernel contains
  `@synchronize`, **and** the launcher guarantees `ndrange` is an exact multiple
  of the workgroup size, making `__active_lane__` statically true so the
  per-segment branch is pure overhead. Such a kernel derives indices from
  `@index(Group, …)`/`@index(Local, …)` only — never `@index(Global)` — and
  guards every *data* access against the true length, which it needs regardless
  because the data length is not the ndrange.
- **Barrier-free kernels always take the default.** With no `@synchronize` there
  is no splitting to skip, so the annotation buys the removal of one masked test
  and gives up KA's entire index-validity guarantee. Bad trade at any size.

§14.3 and §14.4 are rewritten to say this. The package's only kernel today — the
§5.9 capability probe — is barrier-free and correctly takes the default; its
source now carries the reason, so the next author does not re-derive it from the
same wrong assumption.

### 3.2 Promotion-lattice transitivity (§11.3) — verified, 0 failures

Checked over 18 types: the five float types, six integer types including
`BigInt`, `Rational{Int}`, `Rational{BigInt}`, and four `Complex`
instantiations.

| property | result |
|---|---|
| symmetry | 324 pairs, 0 failures |
| **associativity** | **5,832 triples, 0 failures** |
| no `Any` fallback | 324 pairs, 0 occurrences |
| `Test.detect_ambiguities(Double32s)` | 0 |

Two results are worth naming because both are the right answer arriving for
free from rules the package never wrote. `Complex` derives entirely from the
real rules — `Complex{Float32}` gives `Complex{Double32}`, `Complex{Float64}`
correctly stays `ComplexF64`. And `BigInt`/`Rational{BigInt}` reach **`BigFloat`,
not `Double32`**, through Base's own rules overriding the `<:Integer` case —
the least-loss answer, since neither fits the `Float32` exponent range. This is
§11.3 rule 2 paying off: a bound narrower than `<:Real` leaves Base's
better-informed rules in place.

### 3.3 Weak-versus-strong normalization (§5.3) — decided as a boundary rule

The design left this "decidable per kernel, but each decision needs writing
down". A per-kernel decision is the kind that gets made once, written nowhere,
and then relied on by a caller who never read it — so the decision is a single
boundary rule instead:

> **No weak pair crosses a function boundary.** Weak pairs exist only between
> statements inside one kernel body.

**This is also the faster rule, which is why it needs no hedging.** Every
kernel's final operation is already a `two_sum_`/`two_hilo_sum_`, whose output
is by construction `(fl(a+b), residual)` — strongly normalized at zero cost. The
alternative, permitting weak returns and renormalizing at the public boundary,
would add a full `two_sum_` (six dependent `Float32` operations) to *every*
operation in the package to buy nothing any kernel wants. Because the rule
holds, every wrapper constructs with `raw` and pays nothing.

**Verified: 2,424,867 results across all eleven tuple kernels — `add_dd_`,
`add_dd_fast_`, `sub_dd_`, `sub_dd_fast_`, `mul_dd_`, `mul_dd_fast_`,
`div_dd_`, `div_dd_refined_`, `inv_dd_`, `sqrt_dd_`, `fma_dd_` — 0 weakly
normalized.** Now a regression, so a future kernel returning a weak pair fails
rather than silently widening the contract.

### 3.4 Canonicalization when `hi` is subnormal — verified where results land

The claim is structural, not a soft statement about precision taper: when
`exponent(hi) ≤ -126` the canonical bound `|lo| ≤ ½ulp(hi)` is at most `2⁻¹⁵⁰`,
and no nonzero `Float32` is that small, so `lo` must be exactly zero.

The verification detail that matters is *where the results come from*. Placing
operands in the band proves nothing about kernels that produce values there, so
the sweep drives `+`, `-`, `*`, `/`, `inv`, `sqrt`, negation, `abs`, and
`canonicalize` over a pool built to land results in the band. **140,542 results
actually reached it: 0 with a nonzero low word, 0 non-canonical anywhere in the
sweep.** Recorded in §5.3 as a corollary and regression-tested.

### 3.5 Remaining methods distinguishing low-word zero signs — audited

Folded into Tier 0.2 above, where the two leaks found (`show` and `reference`)
are described with their fixes.

---

## Tier 4 — what is genuinely still open

Not defects; work that has not been done.

- **Roadmap phases 3–5** (§14.5 onward): warp-shuffle and wavefront reductions,
  deterministic fixed-tree reduction, tiled matrix kernels. The first KA
  reduction kernel written under the §14.3 rule above will be the first real
  test of that rule.
- **The JMP/Muller–Rideau citation should be page-referenced** when the PDFs are
  at hand. The result is not in doubt; the pointer is imprecise.
- **`Double32sBFloat16sExt` and `Double32sDoubleFloatsExt`** (§11.3 rule 4) are
  specified and unwritten.
- **A three-argument `show(io, ::MIME"text/plain", x)`** with a decimal
  rendering and a round-trip promise (§11.5). Deliberately deferred: the pair
  form is honest about the representation, and a decimal form that does not
  round-trip would be worse than none.

---

## How to work these

1. **Re-derive, do not review.** Two Tier 1 defects and then both of their
   *corrections* survived several readings because the stated reasoning was
   plausible. An independent derivation that reaches the same answer is
   evidence; agreement with the written argument is not.
2. **Say which direction is proved.** Tier 0.1 and Tier 2.1 are the same
   mistake at different sites: a *sufficient* condition written down as though
   it were the *necessary* one. If only one direction holds, write only that
   one.
3. **Construct the adversarial operands directly, and cover both subclasses.**
   Every item above lives in a region of near-zero measure under uniform
   sampling. §23.11 and §23.12 now enumerate by construction rather than by
   example — the previous revision's five hand-picked expressions collapsed to
   three distinct pairs and missed the failing class entirely.
4. **Reference against `BigFloat`, never `Float64`,** unless the exactness
   predicate of §22.1 has been asserted for the operand.
5. **Prefer a stated rule to a local patch, and scope the rule to the operation
   that causes it.** Defect 0.2 existed because a relaxation was scoped to the
   *high* word when the operation that forces it — sign-bit flipping — touches
   the *low* word regardless.
6. **Report the operand count.** Every numerical claim added in this revision
   carries the size of the sweep that produced it. A claim without one is a
   claim that has not been run.
7. **Read the dependency's source before asserting what it does.** Tier 3.1 and
   the retracted `CUDA.FAST_MATH` claim in `Checkpoint.md` are the same mistake:
   a plausible account of another package's behavior, written down as fact,
   never run. Both survived because they were plausible. `KernelAbstractions`
   and `CUDA` are ninety seconds of `grep` away.
8. **Verify where the results land, not where the operands sit.** Tier 3.4 is
   about outputs in the subnormal band; a pool of subnormal *inputs* would have
   passed while testing almost nothing. The same error voided the first
   faithfulness sweep in Tier 2.4.
9. **Turn a coverage claim into an assertion.** Tier 0.1's open item was
   coverage, and coverage decays silently — every test still passes as the pool
   stops being adversarial. The regression now asserts both that the pool
   contains the hard subclass and that the *retired* predicate is still visibly
   wrong on it.
