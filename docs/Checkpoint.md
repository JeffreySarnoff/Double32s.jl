# Checkpoint

Restart point for the `Double32s.jl` implementation. Read this before
re-deriving anything: everything listed as **done** has been run and measured.

**Last updated:** 2026-07-28, after resolving all five Tier 2 open questions —
suite green at **2,508,781+ assertions, 0 failures** (`Pkg.test()`). See
"Tier 2 — ALL RESOLVED" below.

---

## Decisions already made — do not re-litigate

| Decision | Resolution | Why |
|---|---|---|
| Package/module name | **`Double32s`** (`Project.toml` `name = "Double32s"`) | Verified: inside `module Double32`, `using Double32` imports the *module*, and the exported type is shadowed. Design section 1.3. |
| Type name | `Double32`, the only exported name | |
| Repository directory | stays `Double32.jl` | Design section 1.3 blesses this explicitly. |
| UUID | unchanged (`0bebc928-…`) | Package was never registered. |
| Dependencies | `Random` (stdlib) only | `KernelAbstractions`/`Adapt` are roadmap Phase 2; nothing in the scalar core needs them. |
| `Base.convert` | **not** defined | Base routes `convert` to the constructors. Design section 6.4. |
| `nextfloat`/`prevfloat` | **not** defined | Deliberate: absence is informative. Design section 11.2. |
| `minimum`/`maximum` | **not** defined | `mapreduce` over `min`/`max` already works. |
| `compensated_sum(::Array{Float32})` | **not** the headline API | `sum(Double32, A)` already spells it. Design section 19.2. |

---

## Design-document corrections (complete)

`docs/Design/UseFableHere.md` and `docs/Design/DesignDouble32.md` were both
revised. Every numerical claim in them now carries the size of the sweep that
produced it. The three defects found and fixed:

1. **§4.2 `Float64`-exactness condition was wrong in both directions.**
   `lo is normal ⟹ exact` is false (wide-gap pairs); the `-97` boundary is not
   tight (cancellation at a power-of-two `hi`). 31,030 disagreements over a
   1,701,602-pair sweep. Correct condition is span-based and now lives in
   `float64_reference_is_exact`, validated 0 mismatches over 1,701,602 pairs.
2. **§8.10 zero rule was scoped to the wrong operand.** `-` flips `lo`'s sign
   bit under *any* high word. Fix: compare low words with `==`/`<`, never
   `isequal`/`isless`. Trichotomy does **not** catch this — 0 trichotomy
   violations, 20 semantic mismatches over the 576-pair surface.
3. **§8.11 `min`/`max` swallowed NaN** and could not see signed zero. 13 of 49
   combinations disagreed with Base.

Verified-correct-as-written, no change needed: the mixed `Double32`/`Float64`
`==` (918 cases, 0 mismatches) and `floatmax(Double32)`.

---

## Implementation status

### Done and measured — `src/`

| File | Contents |
|---|---|
| `src/Double32s.jl` | root module, `export`/`public` surface |
| `src/core/type.jl` | `Raw`, `Double32`, `raw`, `hi`, `lo`, `words` |
| `src/core/bits.jl` | `lsb_exponent`, `is_power_of_two` |
| `src/core/eft.jl` | submodule `EFT`: `two_sum(_)`, `two_hilo_sum(_)`, `two_hilo_sum_checked`, `two_diff(_)`, `two_prod(_)` |
| `src/core/policy.jl` | `ArithmeticPolicy`, `Strict`/`Native`/`FastPolicy`, `DEFAULT_POLICY` |
| `src/core/normalize.jl` | `canonicalize`, `iscanonical` |
| `src/core/constants.jl` | `POS_ZERO`…`D32_NAN`, `zero`/`one`/`typemin`/`typemax`/`floatmin`/`floatmax`/`precision`/`eps`, `nominal_eps`, `unit_roundoff` |
| `src/core/special.jl` | `canonical_nonfinite`, all predicates, `float64_reference_is_exact`, `reference` |
| `src/core/convert.jl` | constructors in, `Float32`/`Float64`/`BigFloat`/integer out, `widen` |
| `src/core/promote.jl` | `promote_rule` |
| `src/core/compare.jl` | `==`,`<`,`<=`,`isequal`,`isless`, mixed `Float64`, `min`/`max`/`minmax`, `hash` |
| `src/core/arithmetic.jl` | layers 1–3: `add_dd_`…`fma_dd_`, policy dispatch, `Base` operators, sign ops, `ldexp`/`scale2`/`exponent`/`frexp` |
| `src/math/reciprocal.jl` | `reciprocal`, `inv` |
| `src/math/sqrt.jl` | `sqrt_or_nan`, `sqrt` |
| `src/math/powers.jl` | `hypot`, `literal_pow` |
| `src/core/show.jl` | two-argument `show` (round-trips) |
| `src/core/random.jl` | `rand` from a true 48-bit grid |
| `src/arrays/splitarray.jl` | `SplitDouble32Array`, `split`, `unsplit` |
| `src/reductions.jl` | `compensated_dot`, `compensated_sum`, `compensated_mapreduce`, `norm2` |

### Measured accuracy (400-bit `BigFloat` reference, `u² = 2^-48`)

Underflow/overflow zones excluded per design section 9; operands both random
(690-element pool, all pairs) and structured (176-element adversarial pool).

| Op | random | structured |
|---|---|---|
| `+` | 1.14 u² | 1.25 u² |
| `-` | 1.24 u² | 2.00 u² |
| `*` | 3.35 u² | 1.50 u² |
| `/` | 4.73 u² | 4.00 u² |
| `sqrt` | 1.14 u² | 1.18 u² |
| `inv` | 2.44 u² | 1.00 u² |
| `fma` | — | **2.00 u²** measured as `abserr / max(\|xy\|,\|z\|)` |

`div_dd_refined_` (the optional second Newton step) measures 2.25 u² against
`div_dd_`'s 4.00 u². It is implemented but **not** selected by any policy yet —
that is a live decision, see Open below.

The `fma` row is the one to read carefully. Its *relative* error is unbounded,
and correctly so: under near-total cancellation the exact result can be `2^-173`
relative to operands of size 1, far below anything a 48-bit format expresses as
a relative quantity. Design section 9 says to use absolute error there, and the
2.00 u² figure is `abserr / max(|xy|,|z|)`, which is the meaningful bound.
`test/core/arithmetic.jl` asserts that metric, not the relative one.

### Implementation defects found by the sweeps and the suite, all fixed

Keep these; they are the non-obvious part of the work. Each has a standing
regression test.

1. **`fma` under cancellation.** The design's sketch accumulates correction
   terms with plain `+`. Measured relative error 1.67 on
   `fma((-1,-2^-24), (3-2^-23, 2^-24), (3, -5.33e-15))` — the answer *is* z's
   low word and a Float32 add rounds it away. Fix: split *both* cross terms
   with `two_prod_` and distill with `two_sum_` chains, keeping the double-word
   structure through the final combine.
2. **`/` lost ~13 bits for tiny numerators.** `div_dd_` forms `x - q0*y`,
   magnitude `|x|·2^-48`; below `exponent(x.hi) ≈ -101` that falls under
   `2f0^-149` and is silently lost. Measured `6.0e-10` relative error. Fix:
   scale the numerator **up** to `SCALE_FLOOR = -78`.
3. **…but scaling a large numerator *down* is a different bug.** The obvious
   symmetric "scale both operands to a unit binade" destroys a subnormal low
   word and breaks `x / one(Double32) === x` for `Double32(3.0f0, 2f0^-149)`.
   Scale-up-only is exact; the direction is load-bearing.
4. **`sqrt` had defect 2**, measured `2.4e-11`. Fix: always scale to an even
   exponent near zero (harmless there — a lost subnormal low word under a large
   high word is ≥100 binades below nominal precision).
5. **`inv` needed scaling for range, not accuracy.** `inv_dd_`'s residuals sit
   near `2^0` whatever the exponent; only `inv(y.hi)` overflowing at
   `exponent(y.hi) <= -128` loses the computation.
6. **IEEE signed-zero results were lost.** A pair kernel ending in `s = p + t`
   returns `+0` for `(-0) + (-0)` and for `0 * -0`. Fix: `zero_sign(h, ref)` in
   the layer-2 methods. The `iszero(reference)` conjunct in it is load-bearing —
   without it, an exact pair-level cancellation in `fma` gets replaced by the
   nonzero high-word `fma`, measured at `7.9e-8` relative.
7. **`add_nonfinite` silently returned a finite `floatmax`.** Recomputing from
   the high words is right for a non-finite *operand* but wrong for overflow
   caused by the *low* words: `floatmax(Double32) + Double32(2f0^80)` overflows
   the pair while `x.hi + y.hi` is exactly `floatmax(Float32)`.
8. **`ldexp` could produce a non-canonical pair.** Scaling down can round the
   low word into the subnormals and land it exactly on the half-ulp boundary,
   where canonicality depends on the parity of the high word's significand.
   Observed at `2.6295364f-31` with `lo = -2f0^-126`. Fix: renormalize.
9. **`iscanonical` and `Float64` mishandled the sign-preserved zero.**
   `two_sum_(-0.0f0, 0.0f0)` is `+0.0f0`, so `NEG_ZERO` failed its own
   invariant check, and `Float64(NEG_ZERO)` gave `+0.0` — which broke the
   `isequal ⟹ equal hash` contract against `(-0.0f0, -0.0f0)`.
10. **Two method ambiguities** (`Aqua`-style, caught by `detect_ambiguities`):
    a `T<:Integer` conversion against `Bool(::Real)`, and a `Double32(::Real)`
    catch-all against Base's `(::Type{T})(::Rational)`.

Do not "optimize" the scaling out of `/`, `inv`, or `sqrt` without re-running
the random sweep; the failure is silent and invisible to structured operands.

### Test suite — `test/`

| File | Covers |
|---|---|
| `test/setup.jl` | operand generators: `zero_surface`, `wide_gap_pairs`, `structured_pool`, `random_pool`, `worst_relerr` |
| `test/core/type.jl` | layout, constructors, canonicalization, conversions |
| `test/core/eft.jl` | EFT exactness, the false `two_hilo_sum_` precondition, `two_prod_` underflow, the Tier 2.1 differential regression |
| `test/core/exactness.jl` | `lsb_exponent`, `float64_reference_is_exact` incl. both directions the design first got wrong |
| `test/core/compare.jl` | the zero rule, trichotomy, Dict/unique, mixed `Float64`, `min`/`max`, hash |
| `test/core/arithmetic.jl` | canonicality, accuracy bounds, non-finite tables, sign ops, `ldexp`/`frexp`, policies, `sqrt`, `hypot` |
| `test/core/interface.jl` | constants, `floatmax` derivation, `eps`, predicates, promotion, `show` round-trip, `rand`, generic Base algorithms, API surface |
| `test/arrays.jl` | `SplitDouble32Array`, `compensated_dot`/`sum`/`mapreduce`, `norm2` |

The generators build adversarial operands **by construction**; `rand(Double32)`
alone is not an operand source, since every defect this package has had lives in
a region of near-zero measure under uniform sampling.

### Codegen properties verified

- **Allocation-free:** `@allocated` is 0 after warmup for `+`, `-`, `*`, `/`,
  `sqrt`, `inv`, `fma`, `abs`, and for `sum(::Vector{Double32})`.
- **Type-stable:** `Base.return_types` is `[Double32]` for every arithmetic
  operation and for `sum`; `@inferred` is asserted across the public surface.
- **No ambiguities:** `Test.detect_ambiguities(Double32s)` is empty, asserted in
  the suite.

---

## Not yet done

| Item | Roadmap phase | Note |
|---|---|---|
| `docs/` (Documenter) | Phase 1 | docstrings are written; the site is not |
| `benchmark/scalar_cpu.jl` | Phase 1 | design section 8.3 wants it from the day both `*` kernels exist |
| KernelAbstractions layer (`src/kernels/`) | Phase 2 | needs `KernelAbstractions`, `Adapt` deps + `get_backend`/`@adapt_structure` on `SplitDouble32Array` |
| CUDA extension | Phase 3 | no GPU available in this environment to validate |
| AMDGPU extension | Phase 4 | same |
| Matrix kernels, `LinearAlgebra` ext | Phase 5 | |
| Elementary functions | Phase 6 | |
| `BFloat16s`, `DoubleFloats` extensions | any | one `promote_rule` each |
| `floor`/`ceil`/`trunc`/`round`/`rem`/`mod` | follow-on | design section 11.1 lists them as follow-on |
| `parse`/`tryparse` | follow-on | host only, route through `BigFloat` |

## Tier 2 — ALL RESOLVED (this session)

Full write-up in `UseFableHere.md` §Tier 2; resolutions live in the docstrings
they concern, each with a regression test. Every decision was weighted by the
GPU mandate: hot paths stay single-branch and register-lean.

- **2.1 resolved by citation.** `add_dd_` is AccurateDWPlusDW (JMP, ACM TOMS
  2017), bound `3u²/(1-4u)`, machine-checked in Coq by Muller & Rideau (ACM
  TOMS 2022), precision-generic, Fast2Sum steps included. `sub_dd_` inherits by
  exact negation. Differential regression stays in `test/core/eft.jl`.
- **2.2 proofs written; two bugs found and fixed.** (a) Division had *no*
  overflow rescue and returned **NaN** for representable quotients
  (`2f0^127 / (0.5f0 + 2f0^-25)` → true `floatmax + 2^80`); now `div_guarded`
  with a signed-Inf saturation fallback. (b) The mul guard now triggers on a
  *range* check (`MUL_TINY = 2f0^-98`): the band `[2^-126, 2^-98]` was silently
  ~23 bits degraded (8.2e-8 rel); after the `2^±64` scaled retry it holds
  `4u²|V| + 2^-149` — the representation optimum (at `exponent(hi) = -126` the
  format admits no low word: `½ulp = 2^-150 < 2^-149`). Boundary-mapping proof
  (round-up-to-binade ⟷ `2^128 - 2^103`, RNE ties included) in `mul_guarded`'s
  docstring. Hot `*` got *faster* (zero_sign select removed): 4.65 vs 5.02 ns.
- **2.3 resolved.** Per-consumer neutralization table in `two_prod_`'s
  docstring; `has_gradual_underflow()` probes §5.9 (public, pure `Float32`,
  0 alloc, kernel-portable). Two probe traps documented in source: expected
  values must be `reinterpret` constants, and operands must load from a runtime
  array or **the compiler constant-folds the probe to `true`** (first version
  failed its own FTZ test exactly this way).
- **2.4 measured and decided.** Lattice faithfulness (ulp48, 101,761 quotients):
  div single 4.00 worst/1.56% violations; refined 2.30/0.66% — *refinement does
  not reach faithfulness either*, so it cannot justify 2× kernel cost on GPU:
  **single correction everywhere**, refined retained under test. `sqrt` is
  faithful (0.75 worst, 0 violations, incl. §23.8 adversaries) — experiment 10
  closed both halves. `inv`: 1.16 worst, 1/319. Caveat that burned the first
  sweep: faithfulness is meaningless below `|V| ≈ 2^-100` where `ulp48` is
  unrepresentable — filter before claiming.
- **2.5 bound derived.** `|fma_dd_ − exact| ≤ 2u²(1+80u)·max(|xh·yh|, |zh|)`
  for `M ∈ [2^-77, 2^126]`; measured 2.0000u²M (tight to <1%); test asserts
  2.01u². Ordering closed: only the final `l + low` rounding is O(u²);
  everything else telescopes exactly.

Post-resolution state: **suite green** (2.51M assertions), accuracy sweeps
unchanged, perf within noise of baseline (`+` 6.97/6.99, `*` 4.65/5.02,
`/` 15.43/15.55 ns/op), 0 allocations including the probe, guarded paths
`@inferred`-clean.

New/changed since the previous checkpoint: `src/core/capability.jl` (new),
`MUL_TINY`/probe constants in `constants.jl`, guard rewrites in
`arithmetic.jl`, docstring resolutions in `eft.jl`/`reciprocal.jl`/`sqrt.jl`/
`reductions.jl`, new testsets "multiplication guard bands", "division overflow
rescue", "faithfulness against the 48-bit lattice", "capability probe".

## Remaining open questions

None at Tier 2. What remains is roadmap work (table above), plus two watch
items: the JMP/Muller-Rideau citation should be page-referenced when the PDFs
are at hand, and `has_gradual_underflow` needs its KA-kernel variant when
Phase 2 lands.

---

## Reports and documentation (this session, after Tier 2)

- **`docs/Reports/{benchmarks,accuracies}.jl`** rewritten from their
  DoubleFloats.jl originals for Double32s: one precision section, baselines
  Float32/Float64/BigFloat, only the implemented surface. Both run clean and
  wrote `benchmark_report.md` / `accuracy_report.md`. The accuracy script
  measures in ulp48 with adversarial pools and charges errors only where the
  48-bit lattice exists — both lessons from this development cycle.
- **Documenter site** builds clean (`docs/make.jl`, Documenter 1.x,
  `checkdocs = :exports`, all doctests pass): index + four guides
  (getting-started, **gpu** — the centerpiece, arrays incl. GPU SoA patterns,
  accuracy model) + API/Internals references pulling the proof-carrying
  docstrings. GPU examples are labeled generic-GPUArrays-path vs roadmap;
  nothing is claimed validated on device.
- Doctest sweep fixed five stale outputs in source docstrings (the `/3` low
  word is `-9.934108f-9`, not `...107`; `eps` shows `7.1054274f-15`;
  `sqrt(2)^2` is `(2.0, 0.0)`) — and one genuinely instructive surprise:
  `a * 3 == 1` is `true` while `a + a + a == 1` is `false` for `a = 1/3`;
  the getting-started guide now teaches exactly that.
- `.github/workflows/CI.yml` doctest job renamed `Double32` → `Double32s`;
  `docs/.gitignore` covers `build/` and the docs `Manifest.toml`.
- Final state: `Pkg.test()` green at **2,508,785 assertions**.

---

## GPU dependencies as weakdeps + extensions (this session)

**Request.** Add `KernelAbstractions.jl`, `CUDA.jl`, `AMDGPU.jl` to
`Project.toml` — first as `[deps]`, then revised mid-turn to the extension +
weakdeps approach. Doing it as hard deps was flagged as wrong for a numeric
scalar package (every host-only user would pull both vendor stacks); the
revision confirmed that.

**Shape adopted.**

| | |
|---|---|
| `[deps]` | `Adapt`, `Random` — that is all a host-only user pays |
| `[weakdeps]` | `AMDGPU`, `CUDA`, `KernelAbstractions` |
| `[extensions]` | `Double32sKernelAbstractionsExt = "KernelAbstractions"`; the CUDA and AMDGPU extensions are **two-package triggers** (`["CUDA", "KernelAbstractions"]`) because both need the KA layer they build on |
| test target | adds `TOML`, `Adapt`, and all three GPU packages |

`Adapt` earns hard-dep status: it is dependency-free, and
`adapt_structure(SplitDouble32Array)` is what lets the SoA wrapper cross into a
kernel at all.

**Division of labor — deliberately, extensions hold only what abstraction
cannot reach.** Most of what looked like it needed a vendor extension turned
out to be generic:

- `split`/`unsplit` rewritten as **broadcasts** (`hi.(A)`, `raw.(A.hi, A.lo)`),
  so a `CuArray{Double32}` splits on-device with no host round trip and no
  scalar indexing — and no GPU dependency in `src/`;
- the capability probe grew an `AbstractArray` prototype method that copies
  operands to whatever backend owns the array and broadcasts — again no GPU
  dependency;
- `Double32` is `isbits`/8 B, so `CuArray{Double32}` and `ROCArray{Double32}`
  need no package code whatsoever.

What is left in the extensions is real: KA supplies `get_backend`,
`allocate_split`, and a `@kernel` probe launch (**not** redundant with the
broadcast route — a backend may compile broadcast and hand-written kernels
under different inlining, and users' kernels take the second path); CUDA and
AMDGPU supply device identity and compilation context.

**Probe restructured into lanes.** `probe_lane(code, a, b)` is one pure
`Float32` function, no allocation, no `throw`, shared verbatim by the host
loop, the broadcast route, and the KA kernel — so the three cannot drift.
`PROBE_LANES` is now **6**: lanes 1–5 are the old FTZ list; **lane 6 is new** —
a `two_prod_` whose residual `2⁻⁴⁶` is *normal*, which isolates missing `fma`
fusion from flush-to-zero. That distinction is load-bearing: FTZ narrows the
exactness claims, an unfused `fma` voids them entirely (the residual is
identically zero, so the type degrades to `Float32` wearing two words with no
symptom). Hence `capability_report(...).usable` tracks `fused_multiply_add`,
not gradual underflow.

New surface: `capability_report`, `has_fused_multiply_add`, `probe_lane`,
`probe_results`, `probe_operands`, `build_report`, `PROBE_LANES`,
`UNDERFLOW_LANES`, `NO_LANES`, `allocate_split`. Extension hooks `_describe`
and `_vendor_fields` let a backend append report fields without redefining the
report. Host `probe_results()` stays 0-alloc and `@inferred` — required
`ntuple(f, Val(PROBE_LANES))`; `ntuple(f, ::Int)` would have allocated.

**Measured on the hardware present (RTX 5090, sm_120, CUDA 13.3):**

- all six lanes pass on device; `capability_report(CUDABackend())` reports
  `gradual_underflow = true`, `fused_multiply_add = true`;
- `+`, `-`, `*`, `/`, `sqrt_or_nan` on `CuArray{Double32}` are **bit-identical
  to the host** (`===`), including at the guard bands — now asserted in the
  suite. Reductions are excluded and compared against an exact reference
  instead: `+` is not associative and the device tree is not the host's;
- `SplitDouble32Array` round-trips on device and is mutable from inside a
  `@kernel` after `Adapt`.

**A folklore claim I wrote and then had to retract.** The first CUDA extension
asserted that `CUDA.FAST_MATH` enables FTZ and contraction and is "fatal" to
the error bounds. Measured: `math_mode!(FAST_MATH)` changed *nothing* — all six
lanes still pass, raw `Float32` `/` and `sqrt` return the same bits as
`PEDANTIC_MATH` and as the host, and `Double32` `/` and `sqrt` hold 8.3e-16 and
1.0e-15 relative error across all three modes. CUDA.jl's math mode selects
approximate intrinsics where Julia asks for them; it does not retroactively
relax arithmetic written without `@fastmath`. The extension and the GPU guide
now report `math_mode` as *context, not verdict*, and name the actual hazard: a
literal `@fastmath`/`CUDA.@fastmath` at the call site, whose LLVM `reassoc`
flag cancels `(a - (s - v))` to zero and which no report field can detect.

**Other corrections made along the way.** `SplitDouble32Array` gained an
unchecked `Raw()` constructor because `adapt_structure` through the checking
one leaves a `DimensionMismatch` throw inside every kernel. A non-functional
backend (AMDGPU with no ROCm hardware — the state on this machine) must not
throw out of a diagnostic call: `capability_report` degrades to
`functional = false` with `NO_LANES`, while `probe_results` still throws a
clear `ArgumentError`.

**Docs.** GPU guide rewritten — the "roadmap, not in CI" note is gone, replaced
by an install/extension table, the bit-identical-to-host guarantee with its
reduction exception, the two-property probe discussion, and working
`allocate_split`/`Adapt`/`@kernel` examples. Arrays guide gained three
device-resident SoA construction routes. `index.md` gained the installation
section. `api.md` gained a capability-probes section. Docs build exit 0, all
doctests pass (gpu.md needed a `DocTestSetup` — it had never contained a
doctest before).

**Final state:** `Pkg.test()` green at **2,508,854 assertions**, CUDA device
tests executed (not skipped), AMDGPU device tests skipped with the absent-device
report asserted instead. Nothing committed.

---

## Closing every open item in UseFableHere.md — 2026-07-28 20:01 EDT

*(From this entry onward, checkpoint sections carry date **and** time of day.)*

**Request.** Resolve everything still unresolved in `UseFableHere.md`; where a
decision is required, determine the best one weighing performance, clarity,
throughput, efficiency — and follow it.

Six items were open: the two "still to verify independently" notes under Tier 0,
and all five Tier 3 entries. All are now closed, each with a measurement and a
regression in the new `test/core/invariants.jl`. **Two defects and one folklore
claim in the design document were found in the process.**

### Defects found

**`reference` leaked the low word's zero sign into the ground truth.**
`reference(x) = big(x.hi) + big(x.lo)`, and `big(-0.0f0) + big(-0.0f0)` is
`-0.0` while `big(-0.0f0) + big(0.0f0)` is `+0.0` — so two `isequal` pairs got
`BigFloat` references that were not `isequal`. This is the function every
accuracy sweep in the package treats as truth. Fixed by dropping a zero low word
rather than adding it. The dangerous one of the two: a leak in the *reference*
can only ever be found by looking for it.

**`show` printed the raw sign**, making `repr` a function of the bit pattern
rather than the value — `isequal(a, b)` while `repr(a) != repr(b)`, the same
shape as the §8.10 comparison bug. A zero low word now always prints `0.0f0`;
a zero *high* word keeps its sign, so `NEG_ZERO` still round-trips.

Audit that found them: every method reachable from a `Double32`, probed over the
36 `isequal` pairs of the signed-zero surface. `show`/`repr`/`string` leaked 18,
`reference` leaked 2; `hash`, `Float64`, `Float32`, `Dict`, `Set`, `unique`,
`sort`, `Base.Order.Forward`/`Reverse` leaked 0. `objectid` distinguishes and is
left alone deliberately — Base does the same for `±0.0`, and `IdDict` is
identity-based by contract.

### Folklore corrected in DesignDouble32.md §14.3–14.4

The design said KA's automatic bounds handling puts a barrier in divergent
control flow and that `unsafe_indices=true` is the rescue. Read from
`KernelAbstractions/src/macros.jl`: with the **default**, `transform_gpu!` emits
`__active_lane__ = __validindex(__ctx__)` and calls `split(emit_gpu, body)`;
`split` cuts the body **at each `@synchronize`**, and `emit_gpu` wraps only the
inter-barrier *segments* in `if __active_lane__`. The barrier sits between
segments, outside the conditional. **The default is already barrier-convergent.**

What the annotation really does is skip both `__validindex` and the splitting,
so every lane runs every statement. **Decision, recorded in §14.3:** it is a
throughput choice, not a correctness one — `unsafe_indices=true` only when the
kernel has `@synchronize` *and* the launcher guarantees `ndrange` is an exact
multiple of the workgroup size (making `__active_lane__` statically true); such
a kernel must then use `@index(Group)`/`@index(Local)` only, never
`@index(Global)`. Barrier-free kernels always take the default. The probe kernel
now carries the reason in its source.

Same mistake as the retracted `CUDA.FAST_MATH` claim earlier this session:
a plausible account of another package's behavior, written as fact, never run.
Added as working rule 7 in `UseFableHere.md`.

### Decisions made

**Weak vs strong normalization (§5.3)** — the design left this "decidable per
kernel, each decision needs writing down". Replaced with one boundary rule:
**no weak pair crosses a function boundary.** A per-kernel decision is the kind
made once, written nowhere, then relied on by a caller who never read it. The
rule is also the *faster* option, which is why it needs no hedging: every
kernel's final operation is already a `two_sum_`/`two_hilo_sum_`, so strong
normalization is free, while permitting weak returns would add a full `two_sum_`
(six dependent ops) to every operation at the public boundary. Because the rule
holds, every wrapper uses `raw` and pays nothing. Verified: **2,424,867 results
across all eleven `*_dd_` kernels, 0 weak.**

**Zero-sign normalization at construction — rejected.** `raw` is the hot path
and negation is `raw(-hi, -lo)`; clearing the sign there adds a select to every
negation to fix something no value-based operation can observe. `canonicalize`
already emits `+0.0f0` for a zero low word, so a byte-level normal form exists
for serialization and content hashing — verified byte-identical for every
`isequal` pair — and that is now documented on `canonicalize`.

### Verified, no change needed

- **Tier 0.1 coverage.** `wide_gap_pairs()` yields 172 pairs, 96 with normal low
  words spanning 24–144 bits, 64 of them `Float64`-inexact — every one waved
  through by the retired predicate; 15,358 more from the random pool. Coverage
  claims decay silently, so the regression now asserts both that the pool
  contains the subclass and that the retired predicate is still visibly wrong
  on it (working rule 9).
- **Tier 3.2 promotion lattice.** 18 types: symmetry 324 pairs 0 failures;
  **associativity 5,832 triples 0 failures**; no `Any` fallback; 0 ambiguities.
  `Complex` derives free from the real rules; `BigInt`/`Rational{BigInt}` reach
  `BigFloat` via Base's own rules — the least-loss answer, and §11.3 rule 2
  (never bound wider than `<:Integer`) is why that is safe.
- **Tier 3.4 subnormal-band canonicalization.** The structural claim: at
  `exponent(hi) ≤ -126` the half-ulp bound is `≤ 2⁻¹⁵⁰` and no nonzero `Float32`
  is that small. Verified where results *land*, not where operands sit —
  **140,542 results reached the band, 0 with a nonzero low word, 0
  non-canonical.** Placing operands there would have proved nothing (working
  rule 8; same error voided the first Tier 2.4 sweep).

### State

- New `test/core/invariants.jl` (7,030 assertions, 2.0 s) carries all five
  sweeps as regressions.
- `Pkg.test()` green at **2,515,884 assertions**, 25.0 s. CUDA device tests ran;
  AMDGPU skipped with the absent-device report asserted.
- Docs build exit 0, all doctests pass (the new `show` docstring's examples
  included).
- `UseFableHere.md`: Tier 3 is now "resolved"; a new **Tier 4** lists what is
  genuinely still open — roadmap phases 3–5, the page-reference for the
  JMP/Muller–Rideau citation, the BFloat16s/DoubleFloats extensions, and the
  deferred three-argument `show`. Working rules 7–9 added.
- Nothing committed.

---

## Phase 2 — KA portability — 2026-07-28 20:39 EDT

Roadmap Phase 2 (design section 24) implemented. All six deliverables and all
four validation-gate items are done. **Two defects were found: one in the
design's own sample kernel, and one in the shipped `sqrt_or_nan`.**

### The design's §14.4 reduction kernel does not compile

The sketched loop was `stride = W >>> 1; while stride > 0 … @synchronize() … end`
with the comment "uniform trip count, so every work item reaches every barrier".
The trip-count reasoning is correct and the kernel still does not run. KA's
`split` cuts the body at every `@synchronize` and recurses into scope constructs
with **hard scoping** — it `deepcopy`s the environment; `macros.jl` carries a
`TODO` saying soft scoping is what is wanted — so a local assigned before a
barrier is gone after it.

The failure does not look like a scoping error:

```text
MethodError: no method matching isless(::Int64, ::typeof(stride))
```

`stride` was not merely lost — the name resolved to **`Base.stride`**, the
function. Renaming turns it into an honest `UndefVarError`, which is how the
cause becomes visible. Fix: iterate a compile-time tuple,
`for s in halves(Val(W))`, so every stride is a literal and nothing crosses a
barrier. §14.4 rewritten with the working kernel and the failure preserved.

### `sqrt_or_nan` — the documented device-safe operation — contained `throw`

Found by adding the Phase 1 `@code_llvm` gate. `Base.exponent` raises
`DomainError` for `0`/`Inf`/`NaN` and `Base.sqrt` for negatives; every call site
excludes those first, but the compiler cannot prove it, so the exception path
survived into the generated code for `/`, `inv`, `norm2`, **and `sqrt_or_nan`**.
Added `Double32s.raw_exponent` (bit extraction, verified against
`Base.exponent` over **13,756,239 sampled `Float32` values, 0 mismatches**) and
`Double32s.sqrt_nonneg` (`Base.sqrt_llvm`, NaN not `DomainError`).

Measured on both sides rather than assumed:

- **host: no change.** A/B of `/` and `sqrt_or_nan` against the `Base.exponent`
  versions gave 0.98× and 0.93× — inside the noise. An earlier reading suggested
  a 2× win; that was harness drift between benchmark runs, not this change, and
  is not claimed.
- **device: real.** RTX 5090, `CUDA.registers` on an elementwise kernel —
  `/` 35 → **33** registers and 96 → **88** B local; `sqrt_or_nan` 30 → **26**
  registers and 56 → **32** B local. Local memory is spill space, so the 43% cut
  on `sqrt_or_nan` is the number that matters. Categorically, some backends
  decline to compile a kernel containing exception machinery at all.

Every device-safe operation now shows **0 throw references, no `alloca`, no
`double`** in optimized LLVM.

### Decisions

**Names and placement.** Public names and contracts live in `src/kernels.jl`;
only implementations live in `ext/Double32sKernelAbstractionsExt/` (now a
directory: `arrays.jl`, `capability.jl`, `elementwise.jl`, `reduce.jl`). A
reader looking up what `ak_sum` promises should not have to know it is
extension-provided. The `ka_` prefix signals "launches kernels, needs
KernelAbstractions" and keeps these distinct from `Base.sum`, which remains
correct and recommended.

**`split!`/`unsplit!` are in-place broadcasts in `src`, not KA kernels.**
Broadcast already *is* a fused elementwise kernel on every backend, so a
hand-written KA version would add a dependency to reimplement what exists.
Allocation free, verified.

**`DEFAULT_GROUPSIZE = 256`, chosen not inherited.** Measured, `ak_sum` over
50 M `Float32` on an RTX 5090: W=64 0.502 ms, 128 0.467, **256 0.442**, 512
0.472, 1024 0.451. 256 also divides both the 32-lane NVIDIA warp and the 64-lane
AMD wavefront, and 2 KiB of local memory per workgroup never limits occupancy.
The whole spread is 13% — this kernel is arithmetic bound on `Double32`
addition, not launch-shape bound.

**Design §28 experiment 1 closed: `@localmem Double32`, not two `Float32`
arrays.** Both implemented and timed; the gap is ≤4% and inside noise. The
reason the layout does not matter is in the absolute numbers — a `Float32` `sum`
of the same array takes 0.161 ms against 0.43 ms here, so bank conflicts are not
the constraint. With performance tied, the tiebreakers are the stated ones: the
scalar API, one array to size, and a reduction body of `op(s[lid], s[lid+s])`
instead of four word loads and two stores.

**`unsafe_indices=true` on the two stage kernels**, justified by the launcher
per the §14.3 rule settled earlier today: they have barriers *and* `_launch`
always passes `ndrange = groups * W`, an exact multiple of the workgroup size,
so `__active_lane__` is statically true. They therefore use `@index(Group)` and
`@index(Local)` only, never `@index(Global)`.

**`PolicyOp` is a zero-size binary callable.** A reduction needs `op(a, b)` but
every accuracy-selecting operation takes the policy first. A closure would work
on the host and is the wrong thing to hand a kernel; `PolicyOp` has two
singleton fields, so `sizeof` is 0 and the reduction specializes completely.

**Phase 1 gate amended.** "No calls to package functions in `+` and `*`" would
have failed the design's own architecture — §7.5 and §8 deliberately route each
guard through a `@noinline` call so the hot path stays branch-free. `+` calls
`add_nonfinite`, `*` calls `mul_guarded`; satisfying the literal gate would mean
inlining the cold path into the hot one. The gate now asserts what was intended:
every call is a hardware intrinsic or *the* one documented cold guard.

### Delivered

| deliverable | where |
|---|---|
| KA elementwise kernels | `ak_map!`, arities 1–3 |
| AoS ↔ SoA conversion | `split!`/`unsplit!` (src), `allocate_split` (ext) |
| KA reduction + dot baseline | `ak_mapreduce`, `ak_reduce`, `ak_sum`, `ak_dot`, `ak_norm2` |
| KA CPU tests, GPU broadcast tests | `test/kernels.jl` |
| device capability tests | `test/extensions.jl` (earlier today) |

### Validation gate

- **"identical or acceptably bounded scalar vs KA CPU"** — every `ka_*` checked
  against a 400-bit `BigFloat` reference at sizes forcing 1, 2, and 3 passes and
  at partially-filled final workgroups.
- **"CUDA/AMDGPU KA paths compile"** — CUDA device tests run; AMDGPU skipped
  (no ROCm hardware) with the absent-device report asserted instead.
- **"no scalar GPU indexing"** — device tests run under
  `CUDA.allowscalar(false)`, so a scalar read throws rather than working slowly.
- **"convergent barrier review completed"** — see §14.4 above; nothing crosses a
  barrier, by construction.

**Stronger than the gate asked:** `ak_sum`, `ak_dot`, and `ak_norm2` are
**bit-identical between CPU() and CUDABackend()** at every size and groupsize
tested — `===`, not "acceptably bounded".

### State

- `Pkg.test()` green at **2,516,056 assertions**, 37.3 s.
- Docs build exit 0, all doctests pass.
- New: `src/kernels.jl`, `test/kernels.jl`, `test/gates.jl`,
  `ext/Double32sKernelAbstractionsExt/{arrays,capability,elementwise,reduce}.jl`.
  `InteractiveUtils` added to the test target.
- **Outstanding from the Phase 1 gate:** `AllocCheck`, `JET.@test_opt`, and
  `Aqua.test_all` are still not wired in. `test/gates.jl` covers their intent
  with cheaper stand-ins (`@allocated == 0`, `@inferred`/`Base.return_types`,
  `Test.detect_ambiguities`) — stand-ins, not the tools. Recorded as open in the
  design rather than claimed closed.
- Nothing committed.

---

## Correction to the Phase 2 entry — 2026-07-28 20:52 EDT

Asked whether the two defects above were still present, I checked instead of
answering from memory. **Defect 1 was gone; defect 2 was not.**

**Defect 2 was only partly fixed.** The `raw_exponent` sweep patched `hypot`
(`src/math/powers.jl`) and missed **`norm2`**, which lives in
`src/reductions.jl` — same defect, different file, and the Phase 2 gate list did
not cover `norm2`, so nothing caught it. The Phase 2 entry's claim is accurate
only for the operations it enumerates; it should not have been read as a claim
about the package.

**A third defect surfaced from the same audit.** `significand(::Double32)`
threw `DomainError` for `0`, `-0`, `Inf`, `-Inf`, and `NaN`, where
`Base.significand(::Float32)` returns the argument unchanged — five Base-parity
disagreements in a public method, inherited from `Base.exponent` exactly like
the others.

Every remaining live `Base.exponent` call site was then enumerated and fixed:

| site | before | now |
|---|---|---|
| `norm2` (`reductions.jl`) | `Base.exponent` | `raw_exponent` |
| `significand` | `Base.exponent`, threw | guard + `raw_exponent`, Base parity |
| `frexp` | `Base.exponent` | `raw_exponent` |
| `eps(::Double32)` | `Base.exponent` | `raw_exponent` |
| `float64_reference_is_exact` | `Base.exponent` | `raw_exponent` |
| `Base.exponent(::Double32)` | `Base.exponent` | **unchanged, deliberately** |

`Base.exponent` keeps its `DomainError`: throwing *is* its contract, and a
`Double32` method returning a number where `Float32` throws would be the odd one
out. It is now the counterexample in the gate — a test that can never fail is
not a test, so `gates.jl` asserts that `exponent` and `Base.sqrt` **do** contain
a throw while everything else contains none.

Verified after the fix: `norm2`, `significand`, `frexp`, `eps`, and
`float64_reference_is_exact` all at **0 throw references**; `exponent` at 2 and
`sqrt` at 1, as intended.

The gate now covers host routines too, not only kernel entry points, which is
the specific gap that let this through.

**State:** `Pkg.test()` green at **2,516,104 assertions**; docs build exit 0,
doctests pass (including the new `significand` examples). Nothing committed.

---

## Phase 3 — CUDA optimization — 2026-07-28 21:07 EDT

Roadmap Phase 3 (design section 24) implemented. The headline outcome is a
**negative result honestly arrived at**: no warp-shuffle reduction and no direct
CUDA kernel ship, because neither is faster. Design sections 15.2 and 15.5 both
require exactly that conclusion when the measurement says so, and the section
that mattered most was §15.2's last clause: "fall back to KA if no benefit is
measured."

### Three candidates implemented, measured, rejected

RTX 5090, `ak_sum` over 50 M `Float32`, against the KA local-memory tree. All
three produce **bit-identical results** to the KA path, so semantic parity was
never the obstacle — throughput was.

| candidate | speedup vs KA |
|---|---|
| full warp-shuffle reduction, exactly as §15.2 sketches | **0.48–0.56×** |
| hybrid: shared-memory tree to 32 lanes, then shuffle | 0.99–1.10× |
| direct `CUDA.@cuda` kernel, same tree as KA | 1.5% (after the fix below) |

**Why warp shuffle loses here, which is the part worth keeping.** A shuffle
round makes *every* lane apply the operator; a tree round halves the live lanes.
Trading five barriers for five extra *1-flop* `Float32` operations is a good
deal; trading them for five extra *20-flop* `Double32` additions is not. The
familiar "shuffle beats shared memory" result is a claim about cheap operators,
and this format's operator is not cheap. The obvious rescue — shuffle only where
the operator *is* cheap — was tested too: on a `Float32` `max` reduction the
hybrid reaches 1.07× at the default workgroup size. Not material, and it would
buy a second reduction path to maintain for one internal use. Rejected.

### Defect found: the KA driver over-synchronized

The direct CUDA kernel's apparent 8% edge was **not** a property of the kernel.
Both ran identical device code; the entire difference was host-side. The KA
driver called `KernelAbstractions.synchronize` after *every* reduction pass.
Successive launches on one backend are queue-ordered, and the CPU backend is
synchronous outright (`synchronize(::CPU) = nothing`), so those syncs bought
nothing.

Removing them took `ak_sum` from **0.444 ms to 0.395 ms** and closed the gap to
1.5% — inside noise. `ak_map!` lost its trailing sync for the same reason, which
also makes it match broadcast semantics. The one remaining sync is before the
final host `copyto!`, where it is load-bearing.

Two generalizations, recorded in §15.5:

1. Before accepting that a vendor kernel beats the portable one, check whether
   the portable **driver** is what is being measured.
2. A portable fix outranks a vendor-specific duplicate of equal speed — every
   backend gets it, including ones this project cannot test, and there is one
   implementation to maintain instead of two.

Post-fix, `ak_sum` is about **5% faster than GPUArrays' tuned `sum`** (0.395 vs
0.417 ms), and `ak_dot` is **1.52× faster** than `sum(x .* y)` while also being
exact in the products. The KA baseline is not a compromise position.

### Defect found: the report harness under-warmed

The first `gpu_report.md` recorded W=128 at 0.955 ms against 0.440 ms measured
minutes earlier — the first configuration in a sweep was absorbing warmup. One
warmup call is not enough. The harness now does three warmups and takes the best
of three timed batches; the sweep is now smooth (0.413 / 0.395 / 0.407 / 0.406)
and the artifact is gone. Nothing that had already been published was wrong, but
it nearly was.

### Defect found: a test encoded a number from a different codegen mode

`test/kernels.jl` asserted `reduce_stage.local_bytes == 0`, taken from a normal
build. It failed at 32. Cause: **`Pkg.test()` runs under `--check-bounds=yes`,
which disables every `@inbounds` in the package** — so kernels compiled during
testing are not the kernels a user gets. The test now asserts generous ceilings
that hold in either mode, and the exact figures live in
`docs/Reports/gpu_report.md`, which runs normally. Recorded in §15.3 and in the
`kernel_report` docstring, because it is a trap for anyone reading register
counts out of a test run.

### Decisions

**No reduction-strategy selector.** A `:tree`/`:warp` keyword was designed and
then dropped: with `:warp` never faster, the option would be surface area, test
time, and a decision the user cannot make well. The deterministic path is the
only path, which satisfies the gate's "deterministic fallback available" more
simply than a selector would.

**`launch_config` reports both numbers.** CUDA's `launch_configuration` suggests
768 threads; measurement says 256 (0.395 vs 0.407/0.406 at 512/1024, and 768 is
not a power of two so it cannot drive the tree at all). Rather than hard-code
256 and hide the disagreement, the function returns `groupsize` **and**
`suggested`. The heuristic maximizes resident warps to hide memory latency; this
kernel is arithmetic bound, so extra warps buy nothing while a bigger block adds
`log2(W)` barriers. Whole spread is 4.5% — launch shape is not where the time
goes.

**Capability matrix over guesswork.** `capability_report(CUDABackend())` gained
`warpsize`, `max_threads_per_block`, `shared_memory_per_block`, and
`multiprocessors` — measured from the device, never inferred from its name.

### Delivered

| deliverable | outcome |
|---|---|
| CUDA extension | extended: capability matrix, launch policy, kernel report |
| launch policy | `launch_config`, measured against the occupancy heuristic |
| warp-shuffle reduction | implemented, measured at 0.48–0.56×, **rejected per §15.2** |
| occupancy and register reports | `kernel_report` + `docs/Reports/gpu_report.jl` |
| direct CUDA kernels | measured at 1.5%, **rejected per §15.5** |
| CUDA capability matrix | in `capability_report` and the generated report |

### Validation gate

- **semantic parity with KA** — all three candidates were bit-identical to the
  KA path before being rejected on speed.
- **measured speedup** — achieved, but from the portable sync fix (+11%), not
  from vendor code.
- **no spills or occupancy collapse** — occupancy 1.00 for `+`, `*`, `/`,
  `sqrt_or_nan`, and the reduction stage; 0.83 for `fma` (45 registers, which
  buys the fused three-operand kernel). Asserted with ceilings in
  `test/kernels.jl`.
- **deterministic fallback available** — the tree is the only path; verified by
  running the same reduction nine times and against the CPU backend, all `===`.

### State

- `Pkg.test()` green at **2,516,161 assertions**, 40.4 s.
- Docs build exit 0, doctests pass.
- New: `docs/Reports/gpu_report.jl` + `gpu_report.md` (own environment — `CUDA`
  is a weakdep, so the root project cannot `using CUDA`), `docs/Reports/.gitignore`.
- Design §15.2, §15.3, §15.5 rewritten with the measurements and the rejections.
- Nothing committed.

### Defect confirmation

All defects found in this session and the previous two are fixed and verified:

| defect | status |
|---|---|
| §14.4 sample kernel does not compile (loop-carried var across barrier) | fixed; shipped kernels use the tuple form; verified |
| `throw` in device-safe ops via `Base.exponent`/`Base.sqrt` | fixed; **0 throw refs** in `+ - * / inv fma muladd sqrt_or_nan < isless probe_lane` |
| `norm2` missed by the first `raw_exponent` sweep | fixed; 0 throw refs; now covered by `test/gates.jl` |
| `significand(::Double32)` threw where Base returns the argument | fixed; Base parity at all five of `±0`, `±Inf`, `NaN` |
| KA driver over-synchronized | fixed; 0.444 → 0.395 ms |
| report harness under-warmed | fixed; three warmups, best of three |
| test encoded a `--check-bounds=no` number | fixed; ceilings that hold in both modes |

---

## Phase 4 — AMDGPU optimization — 2026-07-28 21:23 EDT

Roadmap Phase 4 implemented, with one deliverable **not** delivered and said so
plainly rather than faked.

### The constraint that shaped everything

**This machine has no ROCm hardware** — `AMDGPU.functional()` is `false`. Design
section 16.5 applies the CUDA gate to AMD kernels: *measured* benefit. With no
measurement possible, shipping vendor kernels would invert the gate rather than
satisfy it. So what shipped is what could be verified without a device, and the
rest is recorded as open.

### Delivered and verified

- **AMD capability matrix** — `device`, `gcn_arch`, `wavefrontsize`,
  `max_threads_per_block`, `shared_memory_per_block`, `multiprocessors`, read
  through `AMDGPU.HIP.properties`.
- **`Double32s.wavefront_size(backend)`** — design section 16.2 is explicit that
  64 must not be hardcoded (RDNA runs wave32, CDNA wave64, and the same part can
  differ by compilation mode). The width is queried; with no device it answers
  64, the conservative direction. Implemented for `ROCBackend`, `CUDABackend`
  (32, also queried), and `CPU()` (1 — a CPU work item is a loop iteration, not
  a lane).
- **`launch_config(::ROCBackend)`** — workgroup size, wavefront width, and
  whether they divide evenly.
- **Every no-device path**, asserted: absent fields are `nothing` rather than
  guessed from the backend's name, the report degrades instead of throwing, and
  `probe_results` still raises a clear `ArgumentError`.

No package kernel branches on wavefront width, because `DEFAULT_GROUPSIZE = 256`
is a whole multiple of 32 **and** 64 — asserted, so a future default that breaks
the property fails the suite.

### Not delivered, and why

**`kernel_report(::ROCBackend)` does not exist.** AMDGPU.jl exposes no
per-kernel register or VGPR count — there is no counterpart to `CUDA.registers`
or `CUDA.memory`, and the only route is manual inspection via
`AMDGPU.@device_code_gcn`. This was checked, not assumed. So the Phase 4 gate's
"VGPR/LDS analysis" is **open**. A method returning empty rows would have looked
like delivery; a `MethodError` naming the function and the backend does not, and
the test asserts that `MethodError` so the absence is deliberate and visible.

**`launch_config(::ROCBackend).suggested` is `nothing`.**
`AMDGPU.launch_configuration` needs a compiled kernel, which needs a device.
An honestly absent field beats compiling untested kernel code inside a
diagnostic call that must not throw.

**No wavefront-shuffle reduction.** There is a structural argument that it would
lose — Phase 3 measured the analogous CUDA path at 0.48–0.56×, and the cause (a
shuffle round makes every lane apply the operator; a tree round halves them) is
not NVIDIA-specific and is *worse* on a 64-lane wavefront. But that is an
argument, not a measurement, and §16.5 asks for a measurement. Recorded in
§16.5 as needing revisiting by someone with CDNA or RDNA hardware.

### Verification discipline applied

Before writing anything, every AMDGPU API the extension already used blind was
checked to exist and to accept the argument types passed: `HIP.name`,
`HIP.gcn_arch`, `HIP.wavefrontsize`, `HIP.properties` (all `(::HIPDevice)`), and
the `props.maxThreadsPerBlock` / `sharedMemPerBlock` / `multiProcessorCount`
accessors. All present. This is working rule 7 from `UseFableHere.md` — read the
dependency's source before asserting what it does — applied preventively rather
than after a failure, which is the first time in this project it has gone that
way round.

The extension header states in the file itself that the with-device paths are
unverified, so the next reader does not have to infer it from the absence of a
CI badge.

### State

- `Pkg.test()` green at **2,516,189 assertions**, 39.9 s. AMD device tests
  skipped (not faked); the absent-device contract is asserted in 17 checks.
- Docs build exit 0, doctests pass.
- Design §16.5 rewritten; new §16.6 records exactly what Phase 4 could and could
  not deliver.
- Nothing committed.

### Defect confirmation

**No new defects found in Phase 4.** Every previously found defect remains fixed
and was re-verified this session:

| defect | status |
|---|---|
| §14.4 sample kernel does not compile | fixed; shipped kernels use the tuple form |
| `throw` in device-safe ops | fixed; 0 throw refs across all 11 |
| `norm2` missed by the first sweep | fixed; 0 throw refs |
| `significand` Base-parity throw | fixed; parity at all five special values |
| KA driver over-synchronized | fixed; 0.444 → 0.395 ms |
| report harness under-warmed | fixed; three warmups, best of three |
| test encoded a `--check-bounds=no` number | fixed; mode-independent ceilings |

---

## Phase 5 — Reductions and linear algebra — 2026-07-28 21:33 EDT

Roadmap Phase 5 implemented. One new kernel, one API decision made by *not*
adding API, and **one defect found in my own Phase 2 documentation**.

### Defect found: my §14.4 explanation named the wrong backend

Phase 2 recorded that KA's `split` cuts the kernel body at every
`@synchronize`, and that this is why a loop-carried local cannot cross a
barrier. True — but incomplete in a way that matters, and writing the matrix
kernel exposed it.

`transform_gpu!` takes the `unsafe_indices` parameter and, when it is true,
emits the body **raw with no splitting at all**. `transform_cpu!` takes no such
parameter and *always* calls `split(emit_cpu, …)`, because it has to turn work
items into loops. Both stage kernels use `unsafe_indices=true`. So for exactly
those kernels the scoping restriction is imposed by the **CPU** backend, not the
GPU one — the opposite of what a reader would infer from what I wrote.

The rule is identical either way and the code was always correct; the
*explanation* was wrong, and a reader would have concluded that
`unsafe_indices=true` relaxes the restriction. It does not. Corrected in §14.3
and in the `reduce.jl` header.

The corrected section now states the restriction once, with its three practical
consequences — all three hit while writing the matmul:

- loop bounds must be inlined or `@uniform`, never a local computed above the loop;
- values derived from `@index` must be recomputed per barrier-separated segment
  (`@index` calls *are* re-emitted per segment, so calling it again is free of
  the problem while caching its result is not);
- an accumulator that must survive the loop belongs in `@localmem`. **The scalar
  `@private x = …` form does not survive, despite its docstring saying it may** —
  `split` resets `private_allocations` per segment, so the initializer is
  re-emitted rather than carried; and if that initializer references an `@index`
  result it is hoisted out of workitem scope and fails to compile outright.

### Delivered: the tiled matrix kernel

`Double32s.ak_matmul!` / `ak_matmul`, `Float32 × Float32 → Double32`, tile 16
(= 256 work items, the same reasoning as `DEFAULT_GROUPSIZE`; 4 KiB of local
memory per workgroup, so occupancy is unconstrained on both vendors).

Each product is formed exactly by `two_prod_` before it enters the accumulator.
Measured against a `Float64` reference:

| size | `ak_matmul` | `Float32` `*` | slowdown | D32 rel err | F32 rel err |
|-----:|---:|---:|---:|---:|---:|
| 256³ | 0.19 ms | 0.015 ms | 13× | 1.33e-14 | 9.77e-07 |
| 512³ | 1.24 ms | 0.036 ms | 34× | 1.69e-14 | 7.72e-07 |
| 1024³ | 9.22 ms | 0.073 ms | 126× | 2.33e-14 | 6.97e-07 |

About four and a half decimal orders of accuracy, for two orders of time. The
slowdown grows with size because this is a *correct* tiled kernel, not a tuned
one — no register blocking, no double buffering, no vectorized loads — while
cuBLAS is all three. Design §19.5 says up front that this will not match vendor
SGEMM, and no vendor library accumulates `Double32`, so there is nothing faster
to call. Bit-identical between CPU and CUDA for a fixed tile.

`Double32 × Double32` is **rejected with an `ArgumentError`**, not silently
approximated by the high words. §19.5 lists it as later work, and a wrong answer
that looks like an answer is the worse failure.

### Decision: §19.6's three modes need no new API

The design asks for `deterministic=false` (tuned), `deterministic=true` (fixed
KA tree), and an optional `:cross_backend` mode. All three already exist as
spellings:

| mode | spelling | guarantee |
|---|---|---|
| tuned | `sum(A)` | accurate; tree chosen per device |
| deterministic | `ak_sum(A)` | same bits every run of this backend |
| cross-backend | `ak_sum(A; groupsize = 256)` | same bits on **every** backend |

Adding a `deterministic` keyword that forwarded to `Base.sum` would be surface
area for zero capability, and would make `ak_sum` — whose prefix means
"KernelAbstractions" — sometimes not use KernelAbstractions. Rejected on
clarity. What was added instead is the *verification*: the guarantees are now
asserted rather than described, including CPU-vs-CUDA `===` at a pinned
groupsize.

### Already delivered earlier

Production `compensated_sum`, `compensated_dot`, `norm2` (host, Phase 1) and
their device counterparts `ak_sum`, `ak_dot`, `ak_norm2` (Phase 2). The
"Float32-input exact-product dot" deliverable is `ak_dot` + `ExactProduct`,
shipped in Phase 2 and 1.52× faster than `sum(x .* y)` as well as exact.

### State

- `Pkg.test()` green at **2,516,231 assertions**, 44.4 s.
- Docs build exit 0, doctests pass.
- New: `ext/Double32sKernelAbstractionsExt/linalg.jl`; `ak_matmul` section in
  `gpu_report.jl`/`gpu_report.md`; reproducibility and matmul sections in the
  GPU guide; §14.3 corrected; §19.5/§19.6 outcomes recorded.
- Nothing committed.

### Defect confirmation

| defect | status |
|---|---|
| §14.4 sample kernel does not compile | fixed |
| `throw` in device-safe ops | fixed; 0 throw refs across all 11 |
| `norm2` missed by the first sweep | fixed |
| `significand` Base-parity throw | fixed |
| KA driver over-synchronized | fixed; 0.444 → 0.395 ms |
| report harness under-warmed | fixed |
| test encoded a `--check-bounds=no` number | fixed |
| **§14.4 explanation named the wrong backend** | **fixed this session**; §14.3 and `reduce.jl` corrected |

All eight verified after the fix: suite green, 0 throw references in every
device-safe operation, shipped kernels use the compile-time-tuple form, and
`ak_matmul` is bit-identical across backends.

---

## Phase 6 — Elementary functions — 2026-07-28 21:56 EDT

Roadmap Phase 6 implemented: **all** function families, trigonometry included.
Sixteen new functions, a coefficient generator, and **four defects found and
fixed — every one of them found by measurement, not by reading the code.**

### Delivered

| family | functions |
|---|---|
| roots (§20.2) | `cbrt`, `rsqrt` |
| exponential (§20.3) | `exp`, `expm1`, `exp2`, `exp10` |
| logarithm (§20.3) | `log`, `log_or_nan`, `log1p`, `log2`, `log10` |
| trigonometric (§20.4) | `sin`, `cos`, `sincos`, `tan` |
| inverse trig | `atan`, `atan(y,x)`, `asin`, `acos` |
| power | `^(::Double32, ::Integer)`, `^(::Double32, ::Double32)` |

Plus `scripts/gen_coefficients.jl`, which generates `src/math/coefficients.jl`
from 400-bit `BigFloat`. Every constant is emitted as a `reinterpret` bit
pattern, never a decimal literal — the same rule the capability probe follows,
because a coefficient one ulp off is a defect no test of the *generator* would
catch.

Measured worst-case errors, nominal-48-bit ulps (full table in
`docs/Reports/accuracy_report.md`): `exp` 0.650, `log2` 0.796, `log` 0.838,
`exp10` 0.868, `cbrt` 0.894, `exp2` 0.960, `sin` 1.081, `log10` 1.237, `rsqrt`
1.235, `log1p` 1.349, `cos` 1.638, `atan` 1.664, `expm1` 1.851, `acos` 2.030,
`asin` 2.409, `tan` 2.952.

### Defect 1: `exp10` — 56.1 ulp48

`exp(x * LN10)` looks correct and is not. **`exp` converts an *absolute*
argument error into a *relative* result error**, and `x·ln(10)` has absolute
error growing with `x`. At `x ≈ 29.4` the argument is 67.7, so the result
carried `67.7·2⁻⁴⁸ ≈ 2⁻⁴²` — measured **56.127 ulp48** against 0.650 for `exp`
itself.

Fixed by reducing in `x`: `k = round(x·log2(10))`, `r = x - k·log10(2)` through
a Cody–Waite split, so `|r| <= 0.1505` and `r·ln(10)` lands inside the `expm1`
kernel's own band. **56.127 → 0.868.** Any `exp`-like function reduced through
a multiplication has this latent in it.

### Defect 2: `log1p` was division-limited exactly where it exists to help

The textbook `2·atanh(x/(2+x))` needs a division, and this package's `/` carries
4.00 ulp48 which passes undiminished into the result. Measured **4.063 ulp48**
near zero — `log1p`'s entire reason for existing is accuracy near zero.

Fixed with a direct 27-term series in `x`, no division anywhere, for
`|x| <= 0.28`. **4.063 → 1.349.**

### Defect 3: `log2`/`log10` spent division error on an exact term

`log(x)/LN2` folds the exact integer `k·ln(2)` into the logarithm and then
divides the sum. Keeping `k` as its own term and multiplying by `INV_LN2`
instead: **log2 1.699 → 0.796, log10 1.934 → 1.237.**

### Defect 4: trigonometry returned plausible garbage past its limit

The worst of the four. The first implementation reduced whatever it was given
and documented that accuracy degraded beyond `2^20` — my docstring said results
would "still return a value in `[-1, 1]`".

**They did not. `sin(2^30)` returned 9.88e8**, and `sin(2^40)` returned `Inf`.
The Cody–Waite subtraction cancels catastrophically once `q` outgrows the
split's precision. A number outside `[-1, 1]` with nothing signalling a problem
is the worst failure available to a numerical routine.

Fixed by making the policy **`NaN` beyond `TRIG_ARG_LIMIT = 2^20`**. `NaN` is
loud: a caller can test for it and cannot test for "this plausible number is
fictional". Correct reduction past that point needs Payne–Hanek with a
multi-thousand-bit `2/π` — a table and a loop that do not belong in a GPU
kernel — recorded as open rather than approximated.

### Also fixed: `Double32(2.0f0)^-30` threw `DomainError`

Base's `power_by_squaring` rejects a negative exponent for any type it cannot
see is a float. Added `^(::Double32, ::Integer)` handling the negative case via
`inv`, widening before negation so `typemin(Int)` does not overflow into itself.
Also added `^(::Double32, ::Double32)` now that `exp` and `log` exist.

**And a lesson re-learned immediately:** `^(::Double32, ::Real)` and
`^(::Real, ::Double32)` were written alongside them and produced **two
ambiguities** against `^(::T, ::Rational)` and `^(::Irrational{:ℯ}, ::Number)`.
This is design §11.3 rule 2 — never bound at `<:Real` — generalized past
`promote_rule`. Both methods were deleted: Base's `^(::Number, ::Number)`
already promotes, so with the lattice in place `Double32^Float32` reaches
`Double32^Double32` unaided. They bought nothing and cost two ambiguities.

### Decision: no dedicated `rsqrt` kernel

§20.2 lists reciprocal square root among the seed-and-refine candidates, so a
Newton kernel was written and measured against both compositions:

| implementation | worst ulp48 | unfaithful |
|---|---|---|
| Newton, `y·(3 - x·y²)/2` | 3.611 | 444/1372 |
| Newton, residual form | 3.111 | 384/1372 |
| `inv(sqrt(x))` | 2.340 | 239/1372 |
| **`sqrt(inv(x))`** | **1.235** | **44/1372** |

The composition wins, and the *order* wins again: `sqrt` **halves** the relative
error of its argument while `inv` passes it through, so doing the reciprocal
first puts `inv`'s 1.16 through that halving and adds `sqrt`'s 0.75. Fewer
operations, better accuracy, no new code path.

### State

- `Pkg.test()` green at **2,517,100 assertions**, 44.7 s; new
  `test/core/elementary.jl` with per-function accuracy ceilings so a regression
  fails here rather than in a report nobody reruns.
- Docs build exit 0, all doctests pass.
- `docs/Reports/accuracy_report.md` gained an elementary-function section.
- Design §20.3 and §20.4 record the four defects and the trig policy.
- Nothing committed.

### Defect confirmation — all twelve

| # | defect | status |
|---|---|---|
| 1 | §14.4 sample kernel does not compile | fixed |
| 2 | `throw` in device-safe ops | fixed; 0 throw refs |
| 3 | `norm2` missed by the first sweep | fixed |
| 4 | `significand` Base-parity throw | fixed |
| 5 | KA driver over-synchronized | fixed; 0.444 → 0.395 ms |
| 6 | report harness under-warmed | fixed |
| 7 | test encoded a `--check-bounds=no` number | fixed |
| 8 | §14.4 explanation named the wrong backend | fixed |
| 9 | `exp10` 56.1 ulp48 | **fixed**; 0.868 |
| 10 | `log1p` division-limited | **fixed**; 1.349 |
| 11 | `log2`/`log10` division on an exact term | **fixed**; 0.796 / 1.237 |
| 12 | trig returned garbage past its limit | **fixed**; NaN policy |
| 13 | `^` with a negative integer threw | **fixed** |
| 14 | `^(::Real)` methods ambiguous | **fixed**; deleted |

All verified after fixing: suite green, 0 ambiguities, 0 throw references in
every device-safe operation, and each accuracy fix asserted with a ceiling in
`test/core/elementary.jl`.

---

## Hyperbolic functions and report overhaul — 2026-07-28 22:37 EDT

### `src/math/hyperbolic.jl`

`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`. Two new coefficient sets
(`TANH_COEFFS`, `ASINH_COEFFS`) generated by `scripts/gen_coefficients.jl` —
the `tanh` set by dividing the `sinh(x)/x` series by the `cosh(x)` series in
`BigFloat`, a three-line recurrence, rather than transcribing Bernoulli numbers.

Measured worst error, nominal-48-bit ulps: `cosh` 0.869, `acosh` 0.935,
`asinh` 1.361, `sinh` 1.522, `atanh` 1.954, `tanh` 4.174.

**The finding worth keeping: `expm1` is not always the right tool.** The header
of the file originally asserted that every hyperbolic function needs the
compensated form. Measurement disagreed:

| function | naive form | `expm1` form | shipped |
|---|---|---|---|
| `sinh`, small `x` | **7.0e6** ulp48 | 1.49 | `expm1` below 0.5 |
| `sinh`, `\|x\| >= 0.5` | 1.49 | 3.58 | **`exp`** |
| `cosh`, everywhere | **1.098** | 3.614 | **`exp`** |

`cosh` adds two positive quantities and never cancels, so routing it through
`expm1` (1.851 ulp48) instead of `exp` (0.650) just accumulates more error for a
defence it does not need. `sinh` needs `expm1` only below 0.5, where the
subtraction actually cancels — and there the naive form is catastrophic, not
merely worse. **`expm1` earns its cost exactly where a subtraction would cancel
and nowhere else.** Both the file header and the docstrings now say this, with
the numbers.

`tanh` is a ratio by definition and inherits `/`'s 4.00 ulp48 floor outside its
series band — the same ceiling `tan`, `asin`, and `acos` sit under.

**Defect found in testing: `tanh` was not exactly odd.** `expm1(2x)` for
negative `x` is not bit-symmetric with the positive case, so
`tanh(-x) !== -tanh(x)` in the last place. Now evaluated on `|x|` with the sign
restored, which makes oddness structural rather than approximate.

**A test premise that was wrong.** The regression comparing `sinh` against the
naive form initially ran at `1e-8 … 1e-14` and failed with `bad == good`. Below
about `1e-7` the cubic term of `sinh` sits under `2^-48` relative, so both forms
return exactly `x` and are equally optimal — the cancellation only shows where
the answer still has structure to lose. The range is now `1e-2 … 1e-6`, where
the naive form is 8× to 424,000× worse.

### Report overhaul

Requested: drop the `BigFloat`, `Big/D32`, and `D32/F32` columns, and cover
every function including reductions and matrix multiplies.

`benchmarks.jl` now emits `Double32 | Float32 | Float64 | D32/F64` and gained
three sections — Elementary Functions, Trigonometric Functions, Hyperbolic
Functions — plus `^`, `compensated_sum`, `norm2`, `compensated_mapreduce`, and a
matrix-multiply table at n = 4…64. Each function is timed on a sample mapped
into its own domain, so nothing is measured on inputs it would reject, and
`tanh` is listed twice (inside and outside its series band) to show what the
division costs.

`accuracies.jl` gained the six hyperbolic functions, a reductions table, a
dot/norm table, and a matrix-multiply table. Its stale "elementary functions do
not exist yet" section is gone.

**Two numbers in the first draft were not credible, and both were wrong for
instructive reasons.**

1. **A column of exact zeros** for `sum(Double32, A)`. It is real: when the
   elements are `Float32` with narrowly-spread exponents, the *exact* sum needs
   only about 24 bits plus that span — roughly 44 bits for `1/k` over 1e5 terms
   — which fits inside `Double32`'s 48. Those sums are not accurate, they are
   **exact**. A row with exponents spread over `2^-60..2^60` was added so the
   table also shows the format's true error (1.16e-14), and the explanation is
   printed with the table rather than left for the reader to doubt.
2. **BLAS `Float32` matmul at 2.0e-2.** That was an elementwise *relative* error
   on `randn` data, where cancellation makes some entries of `C` near zero — it
   measured the conditioning of those entries, not the multiply. Switched to
   normwise `max|C - Cref| / max|Cref|`: **`Double32` 7.46e-15 against `Float32`
   3.94e-07.**

GPU timings and the tiled `ak_matmul` stay in `gpu_report.md`, which has its own
environment — the root project cannot `using CUDA`, since it is a weak
dependency. Both reports now say so explicitly.

### State

- `Pkg.test()` green at **2,517,557 assertions**, 45.3 s.
- Docs build exit 0, all doctests pass.
- New: `src/math/hyperbolic.jl`, hyperbolic testset in `test/core/elementary.jl`.
- Nothing committed.

### Defect confirmation

Two new defects this session, both fixed and verified: **`tanh` not exactly
odd** (now structural — asserted for three arguments) and **the non-credible
matmul metric** (now normwise). The `cosh`/`sinh` formulation changes were
corrections to my own reasoning caught by measurement before shipping, not
defects in released behaviour. All fourteen earlier defects remain fixed.

---

## AcceleratedKernels reorganization + optimized matmul — 2026-07-28 23:00 EDT

### Dependency restructure

`AcceleratedKernels` is now a **hard dependency**, with `KernelAbstractions`
alongside it. `Double32sKernelAbstractionsExt` is deleted; its four files moved
to `src/gpu/`. `CUDA` and `AMDGPU` remain weak — a host-only user still pulls no
vendor stack, and every `ak_*` routine now works on `CPU()` with no extension at
all.

| | before | after |
|---|---|---|
| `[deps]` | Adapt, Random | **AcceleratedKernels, KernelAbstractions,** Adapt, Random |
| `[weakdeps]` | CUDA, AMDGPU, KernelAbstractions | CUDA, AMDGPU |
| extensions | 3 | 2 |

Reductions delegate to `AK.mapreduce`; `ak_map!` to `AK.foreachindex`; the
capability probe likewise. `KernelAbstractions` stays a direct dependency for
one reason: **AK has no GEMM**, so `gpu/linalg.jl` still writes an `@kernel`.

`ka_*` → `ak_*` throughout, since the prefix named the engine and the engine
changed.

### The trade this made, stated rather than buried

**Cross-backend bit-identity for reductions is withdrawn.** AK picks its own
tree per backend — task-parallel on CPU, block-parallel on GPU — and `Double32`
addition is not associative, so a CPU `ak_sum` and a CUDA `ak_sum` return
different bits.

| mode | status |
|---|---|
| tuned (`sum`) | unchanged |
| deterministic per backend (`ak_sum(A; block_size = 256)`) | holds |
| cross-backend, reductions | **withdrawn** |
| cross-backend, `ak_matmul` | holds — still a hand-written kernel |

**A verification trap caught in the act.** My first evidence that AK preserved
cross-backend identity was a test on `1/k` data that passed — and was vacuous.
When elements are `Float32` with narrowly-spread exponents the *exact* sum fits
in 48 bits, so **every** tree returns the same answer. Re-tested on `randn` and
on exponents spread over `2^-40..2^40`, CPU and CUDA disagree. Design §19.6,
the GPU guide, `index.md`, and the regression all now say so, and the
determinism test uses hostile data.

### Matrix multiply: both element types, and two optimizations

`ak_matmul!` now accepts **`Double32 × Double32`** as well as
`Float32 × Float32` (the latter keeps its exact `two_prod_` products; the former
uses the pair multiply, since those operands have already spent their bits).
Rejecting `Double32` operands, as the previous version did, is gone.

Two optimizations, one kept and one rejected:

**Kept — private accumulators.** Design §14.3 says a plain local does not
survive a `@synchronize`, which is why the first version parked the accumulator
in shared memory and paid a read-modify-write per tile. The *array* form
`@private T (n,)` — a `Scratchpad`, unlike the scalar form — **does** survive;
verified directly before relying on it. 1024³ went from **9.22 ms to 5.62 ms**.

**Rejected — register blocking.** Each work item computing an `R×R` block is the
textbook fix for a shared-memory-bound GEMM. Measured at 1024³: `block=1`
**5.62 ms**, `block=2` **16.08 ms** — three times *slower*. The reason is worth
keeping: this is not register blocking at all. KA's `@private` maps to per-thread
**local** memory, which on CUDA is global-backed spill space, and the computed
index `ri*R + rj + 1` blocks promotion into registers. So `block > 1` multiplies
local-memory traffic instead of removing shared-memory traffic. Default is
`MATMUL_BLOCK = 1`; the keyword stays tunable and the negative result is
documented with it.

**Also kept — a bounds-free path.** Square problems that are an exact multiple
of the output block compile with `Val(false)` for the range checks and the
branches vanish.

Device figures after the change: 256³ 0.11 ms, 512³ 0.72 ms (`F32→D32`);
`D32×D32` about 2.4× that, as expected from the heavier tiles and multiply.
Normwise error 3.5e-14 against 8.1e-07 for BLAS `Float32`.

### Report changes (requested)

- Benchmark report: **`Float32` columns removed** as well as the earlier
  `BigFloat`/`Big-D32`/`D32-F32` — now `operation | Double32 | Float64 | D32/F64`.
- Accuracy report: the Elementary-functions paragraph now says what the
  "operands" column counts (arguments that produced a comparable result, after
  discarding those outside the measurable range).
- GPU report: matmul table now covers both element types and uses normwise error.

### State

- `Pkg.test()` green at **2,517,556 assertions**, 42.5 s.
- Docs build exit 0, doctests pass. All three reports regenerated.
- Nothing committed.

---

## Matrix-multiply review and optimization — 2026-07-28 23:29 EDT

Requested: review D32×D32 square matmul at n = 8, 15, 32, 64; improve
performance and throughput; consider different algorithms and organizations.
The answer turned out to be one algorithm applied twice — **split-channel
compensated accumulation** — and it rewrote both the host and the device story.

### The algorithm

Instead of pair arithmetic per element pair (a ~10-flop `*` then a ~20-flop
`+`, one long dependency chain), each product contributes:

- its exact high part, by `two_prod_(a.hi, b.hi)`;
- its three cross products, fused into two `muladd`s;
- accumulation by TwoSum on a high channel, with everything small collecting in
  a low channel; the weak pair is renormalized once per output (host) or once
  per tile (device — the §5.3 rule permits weak pairs inside one kernel body).

~14–17 flops per element pair, a short dependency chain, and — on the host —
the channels are plain `Float32` vectors, so the inner loop vectorizes.

Four host variants were measured before shipping. The plain low channel drifts
linearly in K (1.4e-13 at n = 128); **Kahan compensation on the low channel
holds it flat at 2.8e-14** for ~30% more time, and that variant shipped.

### Host: `LinearAlgebra.mul!` specialization (new `src/arrays/matmul.jl`)

`LinearAlgebra` (stdlib) added as a dep. jki loop order, per-column channel
buffers, `@simd` inner loop, full `α`/`β` support, non-finite gating (a
poisoned low channel must not turn Inf into NaN), wrappers fall back to the
generic path.

| n | generic before | now | speedup | error before | after |
|---|---|---|---|---|---|
| 8 | 3.24 µs | 0.30 µs | ~11× | 1.6e-15 | 5.4e-15 |
| 15 | 22.1 µs | 2.70 µs | ~8× | 3.3e-15 | 1.1e-14 |
| 32 | 211.7 µs | 7.52 µs | ~28× | 4.8e-15 | 1.4e-14 |
| 64 | 1701 µs | 59.9 µs | ~28× | 9.2e-15 | 2.0e-14 |

Error is ~2× the pair chain's and flat in n — inside the accuracy model, stated
in the docstring with the table. Benchmark report: n=64 D32/F64 ratio fell from
**176× to 9.6×**.

### Device: the same inner op in `matmul_tiled!` — and it uncovered the real bottleneck

`acc_step` replaces `op(acc, mul_tile(a, b))`; weak pairs refold once per tile;
non-finite gated at the store; `policy` keyword removed (the split-channel
accumulation *is* the algorithm — a knob selecting a slower equal-accuracy path
is surface area without a decision). RTX 5090, 1024³:

| path | before | after | speedup |
|---|---|---|---|
| F32→D32 | 5.62 ms | **0.33 ms** | 17× |
| D32×D32 | 22.1 ms | **0.42 ms** | 53× |
| cuBLAS SGEMM | 0.048 ms | — | now only 7–9× ahead |

Normwise error 1.0–1.9e-13 at 1024³ (≈ 28–53 nominal ulps, K = 1024) against
4.4–6.4e-07 for `Float32` BLAS. Verified: bit-identical CPU ↔ CUDA for both
element types (fixed fold schedule), BigFloat-checked at n = 8…96, device
F64-checked at 256…1024.

The 17× on the *unchanged math* path says the old inner op — pair arithmetic
through a function-valued kernel argument — was codegen-crippled, not merely
flop-heavy. Two follow-on corrections from re-measuring under the new op:

- **`MATMUL_BLOCK` note rewritten**: the old "block=2 is 3× slower" was a
  property of the old inner op; it now measures ±5%. Default stays 1; both
  measurements recorded.
- `D32×D32` operands are no longer 2.4× the F32 path; they are ~1.3×.

### State

- Suite green at **2,517,571** (new host-mul! testset: sizes 8/15/32/64, α/β,
  exact small-integer case, Inf/NaN gating, wrapper fallback with a
  *tolerance* — the wrapper takes a different algorithm, so `==` was the wrong
  premise and was fixed when it failed).
- All three reports regenerated; docs build exit 0. Nothing committed.

---

## Consolidated state snapshot — 2026-07-29 00:52 EDT

Everything else is chronological; this section is the restart point and is
**updated in place** when the state changes (entries below it may postdate its
original writing). A new session can start here and consult the chronological
entries only when it needs the reasoning behind a decision.

### What the package is now

All six roadmap phases (design §24) are **complete**, plus work beyond them:

| layer | contents | status |
|---|---|---|
| scalar core | `+ - * /`, `inv`, `sqrt`, `fma`/`muladd`, comparisons, conversions, promotion | done; every device-safe op 0 throw refs, 0 alloc, no `double` in LLVM |
| elementary | `cbrt`, `rsqrt`; `exp` `expm1` `exp2` `exp10`; `log` `log_or_nan` `log1p` `log2` `log10`; `sin` `cos` `sincos` `tan` (NaN past `TRIG_ARG_LIMIT = 2^20`); `atan(±)` `asin` `acos`; `sinh` `cosh` `tanh` `asinh` `acosh` `atanh`; `^` (Integer and Double32) | done; **rebuilt 2026-07-29** (design §20.6): table reductions, dominant-add Horner (`src/math/horner.jl`), Float32 tails — 2–7× per function, `log`/`atanh` division-free; ulp48 tables in `accuracy_report.md`, ceilings in `test/core/elementary.jl` |
| coefficients | `scripts/gen_coefficients.jl` → `src/math/coefficients.jl`, 400-bit BigFloat, `reinterpret` bit patterns only | regenerate with `julia scripts/gen_coefficients.jl` |
| parallel | `ak_sum` `ak_dot` `ak_norm2` `ak_reduce` `ak_mapreduce` `ak_map!` on AcceleratedKernels; capability probe via `AK.foreachindex` | done; reproducible per backend, **not** across backends |
| matmul + linear algebra | device `ak_matmul!`/`ak_matmul` (split-channel tiled kernel, both element types, cross-backend bit-identical); host `LinearAlgebra.mul!` for `A*B`, `A*x`, `A'x`, `A'B`, `AB'` and `LinearAlgebra.dot` — all split-channel | done; 0.33/0.42 ms at 1024³ device; host speedups 21–62× over generic, verified against pre-change baselines |
| arrays | `SplitDouble32Array`, `split`/`unsplit`(+`!`), `allocate_split`, Adapt through both words | done |
| probes | 6-lane capability probe (FTZ vs unfused-fma separated), `capability_report`, `kernel_report` (CUDA), `launch_config`, `wavefront_size` | done; AMD `kernel_report` deliberately absent (no VGPR API) |

### Dependency shape

- `[deps]`: AcceleratedKernels, Adapt, KernelAbstractions, **LinearAlgebra**, Random
- `[weakdeps]`: CUDA, AMDGPU → `Double32sCUDAExt`, `Double32sAMDGPUExt` (vendor
  reporting only; all compute is in `src/`)
- KA is direct because AK has no GEMM; LinearAlgebra for the host `mul!`.

### Key numbers (RTX 5090, this machine)

- scalar: `+` ≈7 ns, `*` ≈5, `/` ≈15; `/` worst 4.00 ulp48 (documented floor
  for everything still ending in a division: `tan` 2.01, `asin` 1.95,
  `acos` 2.38 on the regression pools)
- elementary (rebuilt, ns/call before → after, 4096-point pools): exp
  172.7→43.8, log 145.0→41.0, sin 194.9→39.2, sinh 345.4→49.5, cosh
  350.4→49.0, atan 172.8→60.0; accuracy ceilings all hold, expm1 4.63→1.66,
  tanh 3.50→1.52, atanh 2.42→1.30 ulp48
- reductions: `ak_sum` 50M F32 ≈0.44 ms; `ak_dot` 1.5× faster than `sum(x.*y)`
  and exact in products
- matmul 1024³: F32→D32 0.33 ms, D32×D32 0.42 ms, cuBLAS SGEMM 0.048 ms,
  error ~1e-13 normwise vs ~6e-7 for F32 BLAS
- host, n=64 (before → after): `A*B` 1701 → 59.9 µs; `A'B` 3210 → 60.5;
  `AB'` 1696 → 58.0; `A*x` 18.4 → 0.89; `dot` 10⁵ 712 → 141 µs. Error
  ≤ 2.1e-14 normwise, flat in n. n=32 ratios vs `Float64` BLAS:
  `A*x` 6.0×, `A'B` 8.2×, `AB'` 8.5×, `A*B` 11.0×, `dot(10⁴)` 20.7×
- suite: **2,517,816 assertions**, ~51 s; docs build exit 0, doctests green

### Regeneration commands

- tests: `julia --project=. -e 'using Pkg; Pkg.test()'`
- docs: `julia --project=docs docs/make.jl`
- reports: `julia docs/Reports/benchmarks.jl`, `…/accuracies.jl` (root project);
  `julia docs/Reports/gpu_report.jl` (own env; delete its `Manifest.toml` after
  changing root deps, it re-creates itself)
- coefficients: `julia scripts/gen_coefficients.jl`

### Open items (genuinely open, not defects)

- AMD with-device paths unverified (no ROCm hardware); §16.6 records the gap;
  `kernel_report(::ROCBackend)` intentionally a MethodError
- Payne–Hanek trig reduction past `2^20` — policy is NaN, recorded in §20.4
- `AllocCheck` / `JET` / `Aqua` not wired in; `test/gates.jl` carries stand-ins
- JMP/Muller–Rideau citation lacks page numbers
- `Double32sBFloat16sExt`, `Double32sDoubleFloatsExt` (§11.3 rule 4) unwritten
- three-argument decimal `show` deferred (§11.5)
- cross-backend bit-identical *reductions* withdrawn with the AK move (§19.6);
  `ak_matmul` is the remaining cross-backend-reproducible routine
- `lu` and `\` still run the generic fallbacks — the rank-1 update rewrites
  each element as a canonical pair through the stored matrix, so the
  split-channel trick needs a compensated-LU algorithm (active submatrix held
  in channel form across pivoting); real work, recorded as open
- device throughput of the rebuilt elementary functions unmeasured: the
  exp/log tables are dynamically indexed tuples, which compile on GPU but may
  spill to local memory (§20.6 records the trade)

### Git state

First commit landed 2026-07-29 00:57 EDT at user request: `e92076e`
("Complete Double32 package: scalar core, elementary functions, GPU layer",
76 files, 19,721 insertions — the entire package as of the elementary-function
rebuild). Not pushed; no remote interaction has been requested.

---

## Split-channel approach extended to the other matrix operations — 2026-07-28 23:52 EDT

Requested: apply the matmul speedup to other matrix operations and verify the
speedup. Baselines were measured **before** the new methods existed, with the
identical harness re-run after — same seeds, same sizes.

### New methods (all in `src/arrays/matmul.jl`)

`LinearAlgebra.mul!` for `A*x`, `A'x`, `A'B`, `AB'` (adjoint ≡ transpose:
`Double32` is real), and `LinearAlgebra.dot(x, y)` for `Double32` vectors. All
run the same split-channel compensated accumulation as `A*B`; all carry full
`α`/`β`, dimension checks, and the non-finite gate.

### Verified speedups (same machine, before → after)

| op | n | before | after | speedup |
|---|---|---|---|---|
| `A*x` | 8 | 0.29 µs | 0.06 µs | 4.8× |
| `A*x` | 64 | 18.4 µs | 0.89 µs | 21× |
| `A*x` | 256 | 300 µs | 13.0 µs | 23× |
| `A'B` | 64 | 3210 µs | 60.5 µs | **53×** |
| `A'B` | 256 | 209 ms | 3.36 ms | **62×** |
| `AB'` | 64 | 1696 µs | 58.0 µs | 29× |
| `AB'` | 256 | 107 ms | 3.39 ms | 32× |
| `dot` | 10³ | 7.11 µs | 1.41 µs | 5× |
| `dot` | 10⁵ | 712 µs | 141 µs | 5× |

Accuracy at every size 8/15/32/64 verified against a 400-bit reference:
2.7e-15 – 2.1e-14 normwise, same model as `A*B`, flat in n.

### A measured reversal inside `A'B`

The "natural" implementation — dot form, `C[i,j] = P[:,i] ⋅ B[:,j]`, both
columns unit stride — was implemented first and **lost at every size**: its
TwoSum chain is a loop-carried dependency, so it cannot vectorize. Materializing
the transpose (an O(MK) copy) and reusing the `@simd` axpy kernel won 2× at
n = 8 and 7× at n = 256 (0.39 vs 0.79 µs; 3.36 vs 23.5 ms). The copy is the
packing trade BLAS makes, and both measurements are recorded in the method's
docstring. The *vector* adjoint case keeps the dot form — there the copy would
cost as much as the whole computation.

`dot`'s 5× (not 20×) is the same chain: a single dot cannot SIMD across `k`.
Sequential flops are the ceiling there.

### Not extended, and why

`lu` and `\` still run the generic fallbacks. Their O(n³) core is a rank-1
update *through the stored matrix* — each element is read, updated, and
rewritten as a canonical pair every step, so the channel trick would need the
whole active submatrix held in channel form across pivot selection. That is a
real algorithm (compensated LU), not an afternoon's specialization; left open.

### State

- Suite green at **2,517,604** (new testset: correctness at 8/15/32/64 for all
  five ops, `transpose ≡ adjoint` bit-for-bit, α/β, exact integer cases,
  Inf/NaN gating, shape errors, 0 ambiguities).
- Benchmark report regenerated: linear-algebra section now measures `A*x`,
  `A'B`, `AB'`, `dot` — n=32 ratios vs `Float64` BLAS now 6.0×/8.2×/8.5×/20.7×,
  and the stale "generic fallbacks" prose corrected.
- Nothing committed.

---

## Elementary functions rebuilt: tables, mixed precision, division placement — 2026-07-29 00:50 EDT

Requested: examine the exp, log, trig, inverse trig, hyperbolic, and inverse
hyperbolic families; improve performance and throughput of each group;
consider different algorithms and other organizations of the calculation.

### What was measured first

Per-call latency was dominated by the *serial dependency chain* of 10–26 full
pair Horner steps (~60 cycles each through the policy `+`/`*`), plus a
4.00-ulp48 division inside every reduction that used one (`log`'s
`(m-1)/(m+1)`, `atan`'s two-step reduction, `tanh`'s ratio, `atanh`'s
`2b/(1-b)`, `sinh`'s two `expm1` calls below 0.5).

### What changed (all in `src/math/`, primitives in new `horner.jl`)

1. **Dominant-add Horner step** `poly_step` (exact leading product, cross
   terms in two fmas, FastTwoSum against the coefficient — valid because every
   kernel's coefficient dominates its product term; each call site states the
   ratio) and unguarded `dmul`/`dsqr` for range-safe kernel interiors.
   `dsqr` keeps the `lo²` term: cos on a dense sweep, 1.78 → 1.56 ulp48.
2. **Mixed-precision tails**: each kernel's coefficients split where the
   remaining tail drops below `2^-26` of the result; the tail runs in plain
   `Float32` muladds (< `2^-49` relative contribution).
3. **Table reductions**: `exp` via 32-entry `2^(j/32)` (`|r| <= ln2/64`,
   3-step kernel); the inverse table `2^(-j/32)` gives `e^{-x}` from the SAME
   reduction with no division — that is what rebuilt sinh/cosh at 7×. `log`
   via 23 bins `(c = fl32(32/i), -log c)`: the division is gone entirely, and
   `r = m*c - 1` is computed **exactly** (`mulm1`; the tempting one-fma fold
   rounds at 2^-48 absolute ≈ derived ~60 ulp48 after the near-1 cancellation).
   `log1p` mid-band applies the table to `u = 1+x` *without rounding u*.
4. **Division placement**: `atan` re-binned to rotations at `{0, π/6, π/3,
   π/2}` — at most ONE division (old path paid two on `(1, 3.73)`). `tanh`
   past 1 uses `1 - 2/(E+1)` so the division yields a small correction, not
   the result (4.64 → 2.29 on `(1,19)`). `atanh` mid-band is now
   `(log1p(b) - log1p(-b))/2` — no division at all. `sin`/`cos` dispatch on
   quadrant parity and evaluate ONE kernel (old `_sincos_from_quadrant`
   computed both and discarded one); `sincos` asserted bitwise `(sin, cos)`.
   `expm1`'s doubling-identity ladder (worst 4.63) replaced by a series band
   widened to `ln 2` + direct subtraction past it (1.66). sinh/cosh gained
   dedicated all-positive series on `|x| <= 1` (`SINH_COEFFS`/`COSH_COEFFS`).
5. **Generator** (`scripts/gen_coefficients.jl`): emits the exp/log tables,
   `EXPM1_COEFFS` now 16 (band ln 2), `LOG1P_COEFFS` now 10 (band 0.0222,
   shared by the whole log family), sinh/cosh series, `TAN_PI6_D32`,
   `PI6_D32`, `PI3_D32`; zero values normalized to +0.

### Measured (same 4096-point pools/harness, ns/call before → after)

exp 172.7→43.8 · expm1 173.6→48.5 · exp2 167.8→36.9 · exp10 183.4→46.7 ·
log 145.0→41.0 · log1p 158.7→59.3 · log2 146.4→40.3 · log10 147.2→42.8 ·
sin 194.9→39.2 · cos 195.0→40.1 · tan 211.0→69.4 · atan 172.8→60.0 ·
asin 227.1→103.4 · acos 238.1→117.8 · sinh 345.4→49.5 · cosh 350.4→49.0 ·
tanh 121.5→45.6 · asinh 218.6→81.6 · acosh 230.7→74.6 · atanh 168.9→86.6.

Accuracy: every ceiling in `test/core/elementary.jl` holds; changed ones
moved down — expm1 4.63→1.66, tanh 3.50→1.52 (ceiling tightened 5.0→3.0),
atanh 2.42→1.30, log near 1 (old division amplification) 4.66→1.59 dense.
Slight regressions, measured and inside ceilings: cos wide pool 1.44→1.84,
sinh just past the band 1.57→2.43 dense, asinh 0.81→1.64 (identical to the
old formula's 2.43 on the dense mid-band stress that matters).

### Defects found and fixed during the work

- `expm1_small` tail sliced `EXPM1_COEFFS[5:8]` instead of `[4:7]` — caught
  immediately by exp(1) at 1e-11; fixed, verified 0.912 on the ceiling pool.
- `expm1(-0)` and `log1p(-0)` lost the zero sign through unguarded `dmul`
  (the old policy `mul` guard had preserved it); early returns added, signed
  zeros now asserted in the suite.
- `exp(log(x))` identity tolerance was a flat 1e-13 — pool luck; replaced by
  the derived bound `(4 + 1.5|log x|)·2^-48` (exp converts absolute argument
  error to relative result error; the exp10 lesson in reverse).

### State

- Suite green: **2,517,816 assertions** (~51 s), incl. new sincos-bitwise and
  signed-zero regressions; docs build exit 0, doctests green (values updated
  where the last digits legitimately moved).
- Reports regenerated: elementary D32/F64 ratios now 9–19× (were 60–90×);
  accuracy table worst 2.80 (`tan`), most ≈1. Design §20.6 records the
  rebuild; §20.3/20.4 annotated.
- Open: device throughput of the rebuilt functions unmeasured — dynamic table
  indexing compiles on GPU but may spill to local memory; recorded in §20.6.
- Nothing committed.
