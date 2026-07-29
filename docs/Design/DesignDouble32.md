# Design for a GPU-Oriented Double32 Julia Package

**Document:** `Double32design.md`  
**Status:** Architectural design and implementation guidance; not an implementation  
**Target Julia:** Julia 1.12 or later  
**Primary execution layer:** `KernelAbstractions.jl`  
**Optional optimized backends:** `CUDA.jl`, `AMDGPU.jl`  
**Design date:** 2026-07-28

### How to read this document

Julia code blocks are **normative guidance**: they show the shape, annotations,
and dispatch structure the implementation should have, and they are written to
compile as-is modulo the helpers defined nearby. `text` blocks are **algorithm
sketches**: operation order matters, spelling does not.

Section 12, “Julia implementation practice,” collects the language-level rules —
inference, inlining, allocation, FP contraction, dispatch, module hygiene,
testing — that every other section assumes. Read it before writing code against
sections 6 through 8.

Three principles run through the design, and they do not trade against each
other here:

1. **Clarity comes from dispatch and small named functions**, not from comments
   on wide inline expressions. A double-word kernel that names its intermediate
   pairs is easier to audit *and* easier for LLVM to schedule than one that
   does not.
2. **Performance comes from types**, not from flags. Anything that could be a
   runtime `Bool` inside a kernel is instead a singleton type or a `Val`
   resolved on the host.
3. **Correctness comes from structure**, not from care. The canonical invariant
   is established in exactly one place, and the raw path that bypasses it is
   unexported, tagged, and audited.

---

## 1. Executive summary

This document specifies a GPU-oriented Julia package for double-word arithmetic built from two `Float32` components:

```julia
value(x) = Float64(x.hi) + Float64(x.lo)
```

For finite canonical values, the two components are magnitude ordered and nonoverlapping, normally satisfying

```text
|lo| ≤ ½ ulp(hi).
```

The intended arithmetic delivers approximately 48 bits of significand information through compensated `Float32` operations while retaining the exponent range—and the underflow and overflow limitations—of `Float32`.

The design is based on the architecture and arithmetic patterns in `DoubleFloats.jl`, especially its pair representation, `TwoSum`, `FastTwoSum`, FMA-based `TwoProd`, normalized constructors, and branch-free finite arithmetic kernels with guarded exceptional-value fallbacks. The public `DoubleFloats.jl` repository already defines

```julia
const Double32 = DoubleFloat{Float32}
```

so the proposed package is not justified merely by adding a `Float32` double-word type. Its purpose is instead to provide a deliberately small, GPU-compatible, performance-portable implementation with:

- a compact scalar core suitable for device compilation;
- `KernelAbstractions.jl` as the primary kernel and reduction layer;
- no mandatory CUDA, ROCm, generic linear-algebra, quad-precision, or special-function dependency;
- optional `CUDA.jl` and `AMDGPU.jl` extensions for backend-specific launch tuning, subgroup reductions, inspection, and carefully justified native kernels;
- explicit separation between strict numerical semantics and optional faster algorithms;
- deterministic multipass reductions when a fixed reduction tree is requested;
- rigorous high-precision validation on the CPU and cross-backend device testing.

### 1.1 Principal architectural recommendation

Use a concrete, immutable, `isbits` scalar containing exactly two `Float32` fields. Keep all elementary scalar operations in backend-neutral Julia and ensure they compile unchanged in CPU, KA, CUDA, and AMDGPU device code. Use `KernelAbstractions.jl` for elementwise kernels, conversions, reductions, dot products, and the initial matrix kernels. Add backend-specific extension modules only for capabilities that KA intentionally does not abstract or for which direct backend primitives measurably improve performance.

The first release should prioritize:

1. representation and invariants;
2. constructors and conversions;
3. error-free transforms;
4. `+`, `-`, `*`, `/`, `inv`, `sqrt`, and compensated `muladd`;
5. comparisons and predicates;
6. KA elementwise kernels;
7. KA sum and dot-product reductions;
8. CUDA and AMDGPU launch-policy and subgroup-reduction extensions;
9. exhaustive and randomized validation.

Transcendental functions, atomics, high-performance matrix multiplication, parsing, and complete `AbstractFloat` lattice operations should be deferred until the scalar core and reductions are stable.

### 1.2 Julia engineering contract

Every method that can appear in device code shall satisfy all of the following,
and each is mechanically checkable (see section 12):

| Property | Check |
|---|---|
| type stable, concrete return | `Test.@inferred`, `JET.@test_opt` |
| allocation free | `@allocated == 0`, `AllocCheck.@check_allocs` |
| no dynamic dispatch, no runtime intrinsics | `JET.@test_call`, device compilation |
| no `Float64` in a `Float32` kernel | LLVM IR / PTX scan for `double`, `f64` |
| no `@fastmath`, no reassociation | grep, plus PTX check for unwanted `fma.rn.f32` |
| exceptions only on unreachable-in-practice paths | `@code_llvm` for `jl_throw`; `@inbounds` in kernels |
| small enough to inline | `@code_llvm` instruction count, register report |

These are not aspirations to be checked at the end. They are cheap to test from
the first commit and expensive to retrofit, so Phase 1 (section 24) gates on
them.

### 1.3 Naming constraint that must be resolved

A Julia package whose root module is named `Double32` cannot also define a root binding named `Double32` for the scalar type. The module already occupies that binding.

There are two viable designs:

| Package/module | Scalar type | User spelling | Assessment |
|---|---|---|---|
| `Double32` | `Float32x2` or `DF32` | `Double32.Float32x2` | Preserves the requested package name but not the intended type name |
| `Double32s` | `Double32` | `Double32s.Double32` | Preserves the intended type name and is the recommended Julia API |

**Recommendation:** register and implement the package as `Double32s.jl`, exporting `Double32`. The repository or design project may retain the working name “Double32.jl,” but the Julia project/module should be `Double32s` unless the public scalar is renamed to `Float32x2`.

The rest of this document uses the intended scalar name `Double32` and assumes the recommended root module `Double32s`.

#### Export hygiene is part of the same constraint

A name that is exported enters the namespace of every user who writes
`using Double32s`, and a collision there is a `UndefVarError`/ambiguity at their
call site, not ours. Three names in early drafts of this design are unsafe:

| Draft export | Problem | Resolution |
|---|---|---|
| `normalize` | collides with `LinearAlgebra.normalize`, which almost every numerical user has loaded | rename to `canonicalize` |
| `hi`, `lo` | extremely common local-variable and field names; also shadowed constantly in user code | do not export; provide as documented module-qualified accessors |
| `components` | vague, and no Base precedent | do not export; provide `words(x)` returning a `Tuple{Float32,Float32}` |

Julia 1.11 added the `public` keyword, which marks a name as documented API
without injecting it into the user's namespace. Target 1.12 makes this the
correct tool:

```julia
export Double32

public hi, lo, words, canonicalize
public StrictPolicy, NativePolicy, FastPolicy
public compensated_sum, compensated_dot
public nominal_eps, unit_roundoff
```

One exported name is the goal. Everything else is reached as
`Double32s.canonicalize`, which reads well and never collides.

---

## 2. Scope and source basis

### 2.1 Source basis

The original draft of this design was prepared without access to the local
working copy at

```text
/home/jas/github/DoubleFloats.jl
```

and used the public `JuliaMath/DoubleFloats.jl` revision cited in the
bibliography. This revision was checked against the local working copy
(branch `upd`, `Project.toml` version 1.11.1). The structural claims below
hold in both:

- `src/Double.jl` defines `struct DoubleFloat{T} <: MultipartFloat` with two
  `T` fields, a normalizing inner constructor `DoubleFloat(hi::T, lo::T)` that
  calls `two_sum`, and a non-normalizing inner constructor
  `DoubleFloat{T}(hi::T, lo::T)` that does not;
- `src/math/errorfree.jl` centralizes `two_sum`, `two_diff`, `two_hilo_sum`,
  and FMA-based `two_prod`, with guarded entry points and `_`-suffixed
  branch-free variants;
- the guarded `two_sum` short-circuits on a non-finite sum through a
  `@noinline zero_error_result` helper — the cold-path-out-of-line pattern this
  design generalizes in section 12.3;
- `src/math/ops/op_dddd_dd.jl` implements the pair kernels on
  `Tuple{T,T}` rather than on the struct, with `_`-suffixed branch-free
  duplicates for the hot path — the tuple-core pattern this design adopts in
  section 12.7.

Where this document says **Observed**, it means observed in both sources unless
stated otherwise.

### 2.2 Source classifications

This document uses the following classifications:

- **Observed:** behavior found in the inspected source.
- **Established:** a standard property of error-free transformations or floating-point arithmetic under stated assumptions.
- **Proposed:** a design decision for this package.
- **Experimental:** a decision that must be confirmed by generated-code inspection or benchmarks.

### 2.3 Current package observations

At the design date, the inspected repositories reported:

| Package | Observed version or branch state | Design role |
|---|---:|---|
| `DoubleFloats.jl` | 1.11.2 | Numerical and API reference |
| `KernelAbstractions.jl` | main reported 0.10.0-dev | Primary kernel abstraction |
| `GPUArrays.jl` | 11.5.9 | Indirect array ecosystem; not required directly |
| `CUDA.jl` | 6.2.1 | Optional NVIDIA extension |
| `AMDGPU.jl` | 2.7.0 | Optional AMD extension |

Compatibility bounds shall be finalized against released versions used by CI rather than copied blindly from these snapshots.

---

## 3. Goals and non-goals

### 3.1 Goals

The package shall:

1. represent a double-word number as two `Float32` values;
2. be an immutable `isbits` scalar suitable for GPU registers and dense GPU arrays;
3. provide approximately 48-bit normal-range arithmetic using `Float32` hardware;
4. preserve a documented canonical representation after every public operation;
5. use explicit FMA where required by the numerical algorithm;
6. avoid allocation, dynamic dispatch, exceptions, and unsupported runtime services in device code;
7. compile through `KernelAbstractions.jl` on CPU, CUDA, and AMD GPU backends;
8. permit vendor-specific acceleration without making vendor packages mandatory;
9. expose reproducible reduction modes with a fixed tree;
10. distinguish numerical guarantees from assumptions about subnormals, FMA, and compiler flags;
11. use high-precision host references for validation;
12. remain small enough that device compilation and method specialization are predictable.

### 3.2 Non-goals for the initial release

The initial release shall not attempt to provide:

- a correctly rounded 48-bit IEEE-like format;
- a wider exponent range than `Float32`;
- complete transcendental or special-function coverage;
- direct support from cuBLAS, rocBLAS, or vendor tensor-core libraries;
- atomic `Double32` arithmetic;
- arbitrary-precision fallback in device code;
- transparent use of `Float64` in GPU kernels;
- a promise that fast-math compilation preserves error-free transforms;
- bitwise-identical results across different reduction trees;
- a complete replacement for `DoubleFloats.jl` on the CPU.

---

## 4. Lessons from `DoubleFloats.jl`

### 4.1 Techniques to retain

#### Pair representation

**Observed:** `DoubleFloats.jl` stores `hi` and `lo` in an immutable parametric struct and describes them as magnitude-ordered, nonoverlapping floating-point values. The normalizing constructor calls `two_sum`, while a typed inner constructor can preserve an already prepared pair.

**Retain:** use two concrete `Float32` fields and separate normalizing construction from a private raw constructor.

#### Error-free transformations

**Observed:** the source centralizes `two_sum`, `two_diff`, `two_hilo_sum`, and FMA-based `two_prod`. It also distinguishes guarded transforms from branch-free internal variants.

**Retain:** make EFTs a small internal layer with:

- strict guarded entry points;
- branch-free finite kernels;
- clear preconditions for `FastTwoSum`;
- no `@fastmath`;
- tests that compare exact real sums/products.

#### Arithmetic wrappers

**Observed:** finite public operations call branch-free arithmetic and fall back to guarded paths if the result becomes non-finite. Non-finite operands are handled separately.

**Retain conceptually:** use one common finite hot path and one exceptional path. On a GPU, avoid repeated exceptional branches inside every EFT.

#### FMA-based product residual

**Observed:** `two_prod(a,b)` computes

```julia
p = a * b
e = fma(a, b, -p)
```

for finite products.

**Retain:** this is the core primitive for multiplication, exact `Float32` products in dot products, and residual-based refinement.

### 4.2 Techniques to adapt

#### Canonical non-finite values

`DoubleFloats.jl` commonly stores non-finite values with both words non-finite. For a GPU-oriented scalar, the proposed canonical form is:

```text
+Inf  → (+Inf32, +0.0f0)
-Inf  → (-Inf32, +0.0f0)
NaN   → (canonical_qNaN32, +0.0f0)
```

This permits every classification predicate to inspect `hi` only and prevents unnecessary `Inf - Inf` or `NaN` operations on `lo`.

Note that Julia already spells the canonical quiet NaN: `NaN32` is
`reinterpret(Float32, 0x7fc00000)`. Write `NaN32`, not the bit pattern —
the constant is clearer and const-folds identically.

#### Promotion

The inspected promotion rules promote `DoubleFloat{Float32}` with any IEEE float to `DoubleFloat{Float32}`, including `Float64`. That can discard `Float64` precision.

The new package should instead use:

```text
promote(Double32, Float16)   → Double32
promote(Double32, BFloat16)  → Double32
promote(Double32, Float32)   → Double32
promote(Double32, Float64)   → Float64
promote(Double32, BigFloat)  → BigFloat
```

because `Float64` carries both a wider significand (53 bits against a nominal
48) and a wider exponent range, so it loses less of either operand than
`Double32` would lose of the `Float64` one.

#### Correction: `Float64` conversion is *not* exact for every canonical pair

Earlier drafts of this document justified the rule above by claiming that every
finite `Double32` pair converts exactly to `Float64`, on the grounds that the
pair carries at most 48 significant bits. **That claim is false**, and the
reason is worth understanding because the same mistake is available in several
other places.

The invariant `|lo| ≤ ½ulp(hi)` bounds the low word's *magnitude*. It bounds
neither the low word's *bit position* nor the *gap* between the two words. Two
independent mechanisms push a canonical pair past 53 bits:

- a **subnormal `lo`** always has its least significant bit at `2^-149`, however
  small its magnitude;
- a **normal `lo` far below its permitted bound** — canonicality constrains
  `exponent(lo)` only from above (`el ≤ eh - 24`), never from below, so the pair
  may carry an arbitrarily wide gap of implicit zero bits.

| `hi` | `lo` | `lo` class | canonical | `Float64(hi)+Float64(lo)` exact |
|---|---|---|:---:|:---:|
| `1.0f0` | `2^-149` | subnormal | yes | **no** |
| `2^-96` | `+2^-149` | subnormal | yes | **no** |
| `2^-96` | `-2^-149` | subnormal | yes | **yes** |
| `2^-97` | `2^-149` | subnormal | yes | yes |
| `2^-62` | `2^-126` | **normal** | yes | **no** |
| `1.0f0` | `2^-100` | **normal** | yes | **no** |
| `1.0f0` | `2^-52` | normal | yes | yes |

Every row is reachable by an ordinary operation: each is exactly what
`two_sum(hi, lo)` returns for those arguments. Section 5.6 already notes that
`(1.0f0, 2⁻¹⁴⁹)` is a legitimate nonoverlapping pair.

**An earlier revision of this section got the condition wrong in both
directions**, and the two errors are instructive:

- it asserted `lo is normal ⟹ exact`. Rows five and six above refute it. The
  proof sketch it rested on contained a flipped inequality — from `el ≤ eh - 24`
  it concluded that the lowest bit of `lo` lies at `2^(el-23) ≥ 2^(eh-47)`,
  when `el ≤ eh - 24` gives `el - 23 ≤ eh - 47`, an upper bound on the bit
  position and therefore no constraint at all on the span.
- it asserted `exponent(hi) ≤ -97` as the sharp subnormal boundary. Row three
  refutes it: when `hi` is a power of two and `lo` opposes its sign, cancellation
  drops the sum one binade and buys one more bit.

Measured over 1,701,602 structured canonical pairs, that condition disagreed
with reality on 31,030 of them.

The correct condition is a statement about the *span* of the pair, not about the
class of either word:

```text
Float64(x.hi) + Float64(x.lo) is exact
    ⟺  lo == 0,  or  lsb_exponent(lo) ≥ exponent(hi + lo) - 52
```

*Proof.* `Float32 ⊆ Float64`, so both words convert exactly and only the sum can
round. Because `|lo| ≤ ½ulp(hi)`, every set bit of `lo` lies strictly below the
least significant bit of `hi`; the two words therefore do not interact, and
`lsb(hi + lo) = lsb(lo)` whether `lo` adds or borrows. The sum is representable
in `Float64` iff its leading bit and its lowest set bit span at most 53
positions, which is the stated inequality. Finally `exponent(hi + lo)` is
`exponent(hi)`, except that it is one less exactly when `hi` is a power of two
and `signbit(lo) ≠ signbit(hi)`; `|lo| ≤ ½ulp(hi)` bounds the cancellation to
that single binade. If `hi` is subnormal then `½ulp(hi) < 2^-149` forces
`lo == 0`, so the subnormal-`hi` case is vacuous. ∎

As code — see section 22.1 for where this is consumed:

```julia
"""
    lsb_exponent(z::Float32) -> Int

Binary exponent of the lowest set significand bit of a nonzero `z`.
"""
function lsb_exponent(z::Float32)
    u = reinterpret(UInt32, z)
    m = u & 0x007fffff
    e = Int(u >> 23) & 0xff
    return e == 0 ? -149 + trailing_zeros(m) :
                    (e - 150) + trailing_zeros(m | 0x00800000)
end
```

Three consequences, each addressed where it arises:

1. the promotion rationale is rephrased above and no longer rests on exactness;
2. mixed `Double32`/`Float64` comparison must not be performed by promotion —
   see section 8.10;
3. `Float64` is not a valid exact reference for validation over the whole
   canonical set — see section 22.1.

`hash` is unaffected: `hash(x) = hash(Float64(x))` may now map two distinct
`Double32` values to one hash, but hash collisions are permitted. The
requirement Julia imposes — `isequal(x,y) ⟹ hash(x) == hash(y)` — still holds,
because equal values round identically.

A useful conservative variant, when a caller wants a cheap gate rather than the
sharp condition: dropping the cancellation refinement and testing
`lsb_exponent(lo) ≥ exponent(hi) - 52` is never *unsafe* — it can only
under-report exactness, never claim it wrongly. Verified over the same
1,701,602 pairs.

#### Predicate semantics

A normal `Double32` value may have a subnormal `lo` word. Therefore `issubnormal(x)` should classify the represented value, not ask whether either component is subnormal.

Recommended semantics, written as they should appear in the source. Every one
of these inspects `hi` only, so each compiles to the same instruction sequence
as the corresponding `Float32` predicate:

```julia
Base.isfinite(x::Double32)    = isfinite(x.hi)
Base.isinf(x::Double32)       = isinf(x.hi)
Base.isnan(x::Double32)       = isnan(x.hi)
Base.iszero(x::Double32)      = iszero(x.hi)   # canonical: hi == 0 ⟹ value == 0
Base.isone(x::Double32)       = isone(x.hi) & iszero(x.lo)
Base.signbit(x::Double32)     = signbit(x.hi)

Base.issubnormal(x::Double32) = issubnormal(x.hi)
isnormal(x::Double32)         = isfinite(x.hi) & !issubnormal(x.hi) & !iszero(x.hi)
```

Two deliberate choices here differ from the earlier draft and are worth stating
plainly.

**`issubnormal` delegates to the high word.** Classifying the *represented
value* by comparing `abs(x) < Double32(floatmin(Float32))` is both slower (a
pair comparison plus a construction) and less useful: a value whose high word is
normal is a value with full high-word precision available, which is what callers
of `issubnormal` actually branch on. The high word alone answers that. The
subtlety worth documenting is the converse one — a *normal* `Double32` routinely
has a *subnormal `lo`*, and no predicate should ever inspect `lo` for
classification.

**Bitwise `&` rather than `&&`.** Both operands are cheap, already-computed
booleans, so the non-short-circuiting form avoids a branch. This matters inside
kernels, where a predicate that lowers to a `select` instead of a jump keeps
warps convergent. Use `&&`/`||` only when the right operand is expensive or
unsafe to evaluate. See section 12.4.

#### Hashing

For finite values, hash the `Float64` conversion rather than independently hashing the fields. This preserves Julia’s expectation that numerically equal values across types have compatible hashes:

```julia
Base.hash(x::Double32, h::UInt) = hash(Float64(x), h)
```

This single line discharges the `hash`/`isequal` contract. Julia requires only
one direction — `isequal(x, y) ⟹ hash(x) == hash(y)` — and that holds because
equal values round to the same `Float64`. Equality with `Float32`, `Float64`,
`Int`, and `Rational` inherits Base's cross-type consistency for free. It is a
host method; hashing is not a device operation.

Note what this line does **not** claim. Since the conversion is lossy for pairs
with a subnormal low word and `exponent(hi) ≥ -96` (see the correction in
section 4.2), two distinct `Double32` values can share a hash. That is a
collision, which is permitted and costs only a bucket comparison; it is not a
correctness defect. Do not "fix" it by hashing the two words independently —
that would break cross-type consistency, which is the property that actually
matters.

The one case needing care is NaN: `Float64(NaN-valued Double32)` must produce a
NaN, which Base hashes to a single value regardless of payload. Since the
package canonicalizes NaN payloads anyway (section 6.5), this is consistent.
Signed zero is likewise handled by Base, which hashes `-0.0` and `0.0` alike.

### 4.3 Techniques to replace

#### Broad mandatory dependency graph

The inspected package loads generic linear algebra, generic Schur decomposition, Quadmath, and SpecialFunctions in its main module. This is appropriate for a CPU extended-precision package but not for a compact GPU scalar package.

Replace this with:

- direct dependencies: `KernelAbstractions`, `Adapt`;
- standard libraries only where needed;
- weak dependencies: `CUDA`, `AMDGPU`, `SpecialFunctions`, and optionally `ChainRulesCore`;
- extensions for all vendor-specific code.

#### CPU-oriented fallback through high precision

Any route through `BigFloat`, `Float128`, `Quadmath`, dynamic arrays, or generic factorizations is unsuitable for device code. Host reference implementations may use them in tests but never in scalar methods intended for kernels.

### 4.4 Techniques to omit initially

- string macros;
- rich decimal parsing;
- formatted printing beyond a minimal host `show`;
- generic matrix factorizations;
- special functions;
- random distributions beyond simple test generation;
- matrix functions;
- complex transcendental functions.

### 4.5 Techniques to defer

- `nextfloat` and `prevfloat`;
- complete `eps(x)` lattice semantics;
- correctly rounded elementary functions;
- atomic addition;
- production-quality matrix multiplication;
- automatic differentiation rules;
- serialization compatibility guarantees.

---

## 5. Numerical model

### 5.1 Represented value

For a finite pair `x = Double32(hi, lo)`:

```text
V(x) = hi + lo
```

where the right-hand side is an exact real sum, not a `Float32` evaluation.

A host `Float64` conversion computes this to within one `Float64` rounding:

```julia
Base.Float64(x::Double32) = Float64(x.hi) + Float64(x.lo)
```

and computes it *exactly* only when the pair spans at most 53 bits — that is,
when `x.lo` is zero or `lsb_exponent(x.lo) ≥ exponent(x.hi + x.lo) - 52`. It is
inexact whenever the two words sit far enough apart, which happens both when
`lo` is subnormal under a not-tiny `hi` (a subnormal places bits at `2^-149`
regardless of magnitude, so the pair can span 150 bits) **and** when `lo` is
normal but far below its permitted bound, since canonicality never constrains
`exponent(lo)` from below. Section 4.2 gives the condition and its proof;
sections 8.10 and 22.1 give the two places where the difference has
consequences.

For a reference conversion that is exact for every canonical pair, use
`BigFloat` with at least 160 bits, or `Rational{BigInt}`:

```julia
exact(x::Double32) = big(x.hi) + big(x.lo)     # host only, test code only
```

### 5.2 Canonical finite representation

A strongly normalized finite value shall satisfy:

1. `hi` is finite;
2. `lo` is finite;
3. `hi = round(Float32, V(x), RoundNearestTiesToEven)`;
4. `lo = V(x) - hi` exactly for the represented two-word value;
5. `|lo| ≤ ½ ulp(hi)` when `hi ≠ 0`;
6. `hi == ±0` implies `iszero(lo)`;
7. no two public constructors produce different canonical pairs for the same input pair sum;
8. **the sign of a zero `lo` is unspecified and carries no information, at every
   high word.** See section 8.8 for why it cannot be pinned down without paying
   for it on the hot sign operations, and section 8.10 for the rule that binds
   every predicate as a consequence.

Invariant 8 is deliberately stronger than the earlier "`hi == ±0` implies
`lo == +0`" it replaces. `-x` flips the sign bit of `lo` whether or not `hi` is
zero, so `-one(Double32)` is `(-1.0f0, -0.0f0)` while `Double32(-1.0f0)` is
`(-1.0f0, +0.0f0)`: two bit-distinct pairs, one value. Nothing may distinguish
them.

The normalizing constructor uses `TwoSum(hi, lo)`.

A private raw constructor may be used only when an internal algorithm has already established the invariant.

### 5.3 Strong and weak normalization

The package distinguishes these concepts internally:

- **Strongly normalized:** produced by `TwoSum` or `FastTwoSum` under a valid ordering; `hi` is the nearest `Float32` to the pair sum.
- **Weakly normalized:** `|lo|` is small relative to `hi`, but the pair may require one final renormalization.

**The rule, decided: no weak pair crosses a function boundary.** Weak pairs
exist only *between statements inside one kernel body*; every `*_dd_` tuple
kernel, and therefore every public operation, returns a strongly-normalized
pair. This is stated as a boundary rule rather than as a per-kernel judgement
because a per-kernel decision is exactly the kind of thing that is made once,
written nowhere, and then relied upon by a caller that never read it.

The decision is also the **faster** one, which is why it needs no hedging. Every
kernel's final operation is already a `two_sum_`/`two_hilo_sum_`, whose output
is by construction `(fl(a + b), residual)` — strongly normalized at no cost. The
alternative — permitting weak returns and renormalizing at the public boundary —
would add a full `two_sum_` (six dependent `Float32` operations) to *every*
operation in the package, to buy nothing that the kernels want. Because the rule
holds, every wrapper constructs with `raw` and pays zero.

**Verified: 2,424,867 kernel results across all eleven tuple kernels
(`add_dd_`, `add_dd_fast_`, `sub_dd_`, `sub_dd_fast_`, `mul_dd_`,
`mul_dd_fast_`, `div_dd_`, `div_dd_refined_`, `inv_dd_`, `sqrt_dd_`,
`fma_dd_`), 0 weakly-normalized outputs.** The sweep is a regression in
`test/core/invariants.jl`, so a future kernel that returns a weak pair fails
rather than silently widening the contract.

A corollary worth stating separately, because it is the case a reader is most
likely to doubt: **when `exponent(hi) ≤ -126` the low word must be exactly
zero.** The canonical bound `|lo| ≤ ½ulp(hi)` is then at most `2⁻¹⁵⁰`, and no
nonzero `Float32` is that small — so the "little or no room for a distinct low
correction" of section 5.5 is not a soft statement about precision taper but a
hard structural constraint. **Verified over 140,542 results that actually land
in that band** (not merely operands placed there): 0 with a nonzero low word,
0 non-canonical.

### 5.4 Nominal precision

Let the component precision be `p = 24`, and let

```text
u = 2⁻²⁴
```

be the `Float32` unit roundoff under round-to-nearest.

Away from underflow and overflow, standard double-word algorithms normally deliver errors of order `u²`, corresponding to approximately 48 significant bits. It is useful to define:

```text
p_nominal = 48
u_D32 ≈ 2⁻⁴⁸
```

This is an arithmetic accuracy model, not a claim that `Double32` is an IEEE format with a uniform 48-bit significand lattice.

### 5.5 Exponent range and precision taper

The high word is a `Float32`, so the type does not gain a wider exponent range.

- minimum positive representable value: `2⁻¹⁴⁹`;
- minimum positive normal value: `2⁻¹²⁶`;
- maximum high word: `(2 - 2⁻²³) × 2¹²⁷`;
- largest positive low correction that does not round the high word to infinity is just below `2¹⁰³`.

A useful maximum finite canonical pair is conceptually:

```julia
Double32(floatmax(Float32), prevfloat(ldexp(1.0f0, 103)))
```

subject to validation of the constructor and backend rounding mode.

Near the lower exponent range, `lo` becomes subnormal and then unavailable. Effective precision therefore tapers from roughly 48 bits toward ordinary `Float32` precision. When `hi` itself is subnormal, there is little or no room for a distinct low correction.

### 5.6 The representable set is not a conventional float lattice

A normalized pair can contain a very small `lo` relative to `hi`. For example, `(1.0f0, 2⁻¹⁴⁹)` is nonoverlapping. Consequently:

- the set of canonical pair sums is nonuniform;
- nominal precision is not the same as adjacent spacing;
- a conventional `eps(Double32) = 2⁻⁴⁷` is useful as an algorithmic precision constant but is not necessarily the distance to the mathematically next canonical pair;
- `nextfloat` and `prevfloat` require an explicit policy over the canonical pair set.

The first release should document `precision(Double32) == 48`, provide a named constant such as `nominal_eps(Double32)`, and defer `nextfloat`/`prevfloat` rather than implement misleading methods.

### 5.7 Rounding assumptions

Strict arithmetic requires:

- round-to-nearest, ties-to-even for `Float32` operations;
- correctly implemented `Float32` addition, subtraction, multiplication, division, and square root;
- a correctly rounded or IEEE-conforming fused multiply-add for `fma`;
- no algebraic reassociation;
- no contraction or simplification that destroys the explicit residual algorithm;
- gradual underflow for the strongest EFT claims.

The package must not apply `@fastmath` to strict EFT or arithmetic code. In
Julia this is a stronger guarantee than it looks: the compiler does **not**
contract `a * b + c` into an FMA on its own, so a plainly written error-free
transform is safe by default and only `@fastmath` (or an explicit `muladd`) can
break it. Section 12.5 states the full contraction rules.

### 5.8 Backend capability levels

Define host-selected, compile-time-specialized policies as empty singleton
types. They carry no data; their identity *is* the information, and Julia
specializes the whole call tree on them at zero runtime cost:

```julia
"""
    ArithmeticPolicy

Compile-time selector for the accuracy/performance contract of a `Double32`
operation. Instances are singletons and are always passed by value as the
first argument, so that dispatch — not a runtime flag — chooses the kernel.
"""
abstract type ArithmeticPolicy end

struct StrictPolicy <: ArithmeticPolicy end
struct NativePolicy <: ArithmeticPolicy end
struct FastPolicy   <: ArithmeticPolicy end

const DEFAULT_POLICY = StrictPolicy()
```

- `StrictPolicy`: preserve the documented operation order, use explicit FMA, guard exceptional values, and require capability tests for claims involving subnormals.
- `NativePolicy`: preserve the same public result contract where possible but permit backend-specific instruction selection and launch tuning.
- `FastPolicy`: explicitly relax error and exceptional-value guarantees; never become the implicit behavior of `+`, `*`, or `/`.

Operations take the policy as an ordinary first positional argument, mirroring
`rand(rng, …)` and `round(x, RoundNearest)` in spirit — the "how" is a value,
not a keyword:

```julia
@inline add(::StrictPolicy, x::Double32, y::Double32) = ...
@inline add(::FastPolicy,   x::Double32, y::Double32) = ...

Base.:+(x::Double32, y::Double32) = add(DEFAULT_POLICY, x, y)
```

Three rules follow from making the policy a type rather than a flag, and they
are what actually buys the performance:

1. **Never test a policy at runtime.** `if policy isa FastPolicy` inside a
   kernel is a code smell even though the compiler usually removes it; write a
   second method instead. Two three-line methods are clearer than one six-line
   method with a branch, and they cannot silently deoptimize.
2. **Never pass a policy as a keyword argument into device code.** Keyword
   arguments do not participate in dispatch, so a `policy=` keyword only reaches
   the specializer through constant propagation, which is not guaranteed.
   Accept keywords at the host API boundary, resolve them to singletons there,
   and pass positionally from then on.
3. **Never store a policy in a mutable or abstractly typed field.** A
   `ArithmeticPolicy`-typed field is a dynamic dispatch waiting to happen; use a
   type parameter (`struct Config{P<:ArithmeticPolicy}`) if a policy must be
   carried in a struct.

Runtime booleans should not be tested inside hot kernels. Host code should dispatch on policy types or `Val` parameters. Section 12.7 gives the general recipe for turning a user-facing keyword into a dispatch-visible type exactly once, at the launch boundary.

### 5.9 Subnormal and flush-to-zero behavior

Backends or compilation flags may flush subnormal inputs or results to zero. This invalidates some exactness results.

The package should include a device capability test suite that evaluates:

1. `minsubnormal + minsubnormal`;
2. `floatmin(Float32) * 0.5f0`;
3. `fma(floatmin(Float32), 0.5f0, -0.0f0)`;
4. `TwoSum` and `TwoProd` cases with subnormal residuals;
5. signed-zero preservation.

Results should be cached on the host per backend/device and used to qualify test expectations and documentation. Strict kernels must not silently claim full gradual-underflow accuracy on a backend that does not provide it.

---

## 6. Representation and constructors

### 6.1 Recommended scalar

Julia gives exactly one lever for enforcing a struct invariant: `new` is
reachable only from inner constructors. The idiomatic way to spend that lever is
to make the *unchecked* path the sole inner constructor, tag it so it cannot be
called by accident, and build every checked path as an outer constructor.

```julia
"""
    Raw()

Internal tag selecting the non-normalizing `Double32` constructor. Passing
`Raw()` asserts that the caller has already established the canonical
invariant of section 5.2. Not exported, not public API.
"""
struct Raw end

"""
    Double32(hi::Float32, lo::Float32) -> Double32

A double-word number represented as the unevaluated sum `hi + lo` of two
`Float32` words, canonicalized so that `hi == fl(hi + lo)`.
"""
struct Double32 <: AbstractFloat
    hi::Float32
    lo::Float32

    # The only inner constructor: no normalization, no checks, no branches.
    Double32(::Raw, hi::Float32, lo::Float32) = new(hi, lo)
end

"""
    raw(hi, lo) -> Double32

Construct without canonicalizing. Every call site must be able to justify the
invariant from the algorithm that produced `hi` and `lo`.
"""
@inline raw(hi::Float32, lo::Float32) = Double32(Raw(), hi, lo)

# The public two-word constructor: canonicalize exactly once, here.
@inline Double32(hi::Float32, lo::Float32) = canonicalize(hi, lo)
```

Four properties of this arrangement are worth naming, because each replaces a
weaker alternative that appeared in earlier drafts:

**Defining any inner constructor suppresses Julia's default ones.** With
`Double32(::Raw, ::Float32, ::Float32)` present, there is no automatically
generated `Double32(hi, lo)` that skips normalization. The invariant cannot be
bypassed by writing the obvious thing.

**A singleton tag beats `Val{:raw}`.** `Val{:raw}` is a `Symbol`-parameterized
type; it works, but `Raw` is one word, has no parameter to get wrong, cannot be
constructed accidentally by a user who happens to have a `:raw` symbol, and
reads as a noun at the call site. Both compile to nothing.

**`raw` is a plain function, not a constructor.** Constructors are public API by
convention in Julia; users reasonably expect any `Double32(...)` method to be
callable. Keeping the unchecked path behind a lowercase, unexported function
signals correctly and keeps `methods(Double32)` honest.

**`@inline` on `raw` is not optional.** It is a zero-cost wrapper only if it
inlines. Every wrapper of this shape in the package should be `@inline`; see
section 12.3.

The scalar should be `Base.@assume_effects`-free. Pure `Float32` arithmetic and
struct construction are already inferred as effect-free and constant-foldable by
Julia, so `zero(Double32)` and friends fold at compile time without help.
Reach for `@assume_effects` only in the bit-twiddling helpers of section 7.6,
and only with a test that pins the claim.

### 6.2 Why not parameterize the scalar

A general `DoubleFloat{T}` is elegant on the CPU, but this package has one purpose: a `Float32` pair optimized for GPU execution. A concrete type:

- reduces method count;
- shortens compile time;
- allows hard-coded bit operations;
- simplifies backend generated-code tests;
- avoids accidental `Float16` or `Float64` instantiations;
- permits more aggressive specialization.

Generic EFT helper functions may remain parameterized for tests, but the public scalar should be concrete.

This is a case where "generic Julia" and "good Julia" point in different
directions, so the split should be deliberate: **the numeric core is generic in
`T<:Base.IEEEFloat`, the type is not.** The error-free transforms of section 7
are written once with a type parameter, which lets the test suite validate them
against `Float64` (where a `BigFloat` reference is cheap and exhaustive
structured testing is easy) and reuse them at `Float32` in production. The
`Double32` struct and its arithmetic methods are concrete, which is what keeps
the method table, the specialization count, and the device-side compile time
small.

### 6.3 Size, alignment, and ABI

Expected properties, which should be asserted in the test suite rather than
assumed:

```julia
@test isbitstype(Double32)
@test sizeof(Double32) == 8
@test fieldoffset(Double32, 1) == 0
@test fieldoffset(Double32, 2) == 4
@test Base.datatype_alignment(Double32) == 4
```

The alignment result deserves comment, because it is the one people guess wrong.
Julia gives a two-`Float32` struct 4-byte alignment, not 8 — the alignment of a
struct is the maximum alignment of its fields, and Julia has no field-alignment
attribute to override it. This does *not* prevent 64-bit vector loads of array
elements: array buffers from `Array`, `CuArray`, and `ROCArray` are allocated
with alignment far above 8, so element `i` sits at a byte offset of `8i` and is
naturally 8-aligned. What it does mean is that the compiler must *prove* that
alignment to emit a paired load, and whether it does is a codegen question, not
a type question.

So: measure. CI should inspect

- LLVM IR for the load width and the `align` attribute on the load;
- PTX for CUDA — look for `ld.global.v2.f32` versus two `ld.global.f32`;
- GCN/LLVM output for AMDGPU — `global_load_dwordx2` versus two `dword` loads;
- whether field extraction after the load remains register-only;
- register counts, which pair loads can improve and can also worsen.

Do not add padding or bit-pack the pair into a `UInt64` for ordinary arithmetic. Bit-packing increases reinterpretation work and may inhibit scalar FMA scheduling.

If measurement shows that paired loads matter and are not being emitted, the
remedy is a *load helper* used inside kernels, not a change to the scalar type:

```julia
# Only if measurement justifies it. Keep `Double32` itself untouched.
@inline function load_pair(A::AbstractVector{Double32}, i::Integer)
    @inbounds v = reinterpret(NTuple{2,VecElement{Float32}}, A[i])
    return raw(v[1].value, v[2].value)
end
```

Reinterpretation at the load site is local and testable; a different in-memory
representation is neither.

### 6.4 Canonical constructors

Required host and device constructors:

```julia
Double32()
Double32(x::Float16)
Double32(x::Float32)
Double32(x::Double32)
Double32(hi::Float32, lo::Float32)
```

Host constructors may additionally support:

```julia
Double32(x::Float64)
Double32(x::Integer)
Double32(x::BigFloat)
Double32(x::Rational)
```

Device code should initially support only primitive conversions that compile without runtime services.

**Do not define `Base.convert` for these.** Base already provides
`convert(::Type{T}, x::Number) where {T<:Number} = T(x)`, so defining
constructors is sufficient and defining `convert` as well is redundant at best
and a source of ambiguities at worst. The Julia rule is: *define constructors,
get `convert` free.* The same applies to `Base.AbstractFloat(x::Double32)` and
`Base.float(x::Double32) = x`, both of which are one-liners worth having
explicitly.

#### From `Float32`

```julia
@inline Double32(x::Float32) = raw(x, ifelse(isfinite(x), 0.0f0, 0.0f0))
```

which is to say: `raw(x, 0.0f0)` is already correct for every `Float32`,
including `±Inf32` and NaN, provided NaN payload canonicalization is applied
where the package promises it. Prefer the honest form:

```julia
@inline Double32(x::Float32) = raw(x, 0.0f0)
@inline Double32(x::Float16) = Double32(Float32(x))   # exact, always
```

`Float16 → Float32` is exact, so it needs no residual. Both methods are branch
free and both fold to a constant when the argument is a literal.

#### From `Float64`

A finite `Float64` may exceed `Float32` range, so the high word can overflow
while the value is perfectly finite:

```julia
function Double32(x::Float64)
    hi = Float32(x)
    isfinite(hi) || return raw(hi, 0.0f0)     # ±Inf32 or NaN32: canonical, exact
    lo = Float32(x - Float64(hi))             # exact: |x - hi| ≤ ½ulp(hi)
    return raw(hi, lo)                        # already canonical by construction
end
```

Two details make this `raw` rather than `Double32(hi, lo)`. First, `Float32(x)`
rounds to nearest, so `hi` *is* the nearest `Float32` to the value — invariant 3
of section 5.2 holds by construction. Second, `x - Float64(hi)` is exact by the
Sterbenz-style argument, and rounding that residual to `Float32` cannot make
`|lo| > ½ulp(hi)`. Skipping the redundant `two_sum` here is a legitimate use of
the raw path, and the comment on the call site is what makes it auditable.

This rounds a 53-bit value into the approximately 48-bit pair, and is therefore
not injective — an intentional, documented loss.

#### From integers

Split on width rather than writing one clever generic method:

```julia
# Exact via Float64 for every integer that fits in 53 bits.
Double32(n::Union{Int8,UInt8,Int16,UInt16,Int32,UInt32}) = Double32(Float64(n))

# 64-bit and wider: Float64(n) may already round, so form the residual
# in the integer domain, where it is exact.
function Double32(n::Integer)                    # host only
    hi = Float32(n)
    isfinite(hi) || return raw(hi, 0.0f0)
    r  = n - oftype(n, hi)                       # exact when hi is integral
    return Double32(hi, Float32(r))
end
```

Device constructors should be limited to the first line until each integer width has generated-code and correctness tests. Avoid a `BigFloat` route in public hot methods; `DoubleFloats.jl` uses one in `DoubleFloat(x::T, y::T) where {T<:Integer}`, which is correct on the host and unusable in a kernel.

`Double32(x::Rational)`, `Double32(x::BigFloat)`, and `parse` are host-only and
should route through `BigFloat` with `setprecision(96)` or higher, then split.
Mark them clearly in the docstring as non-device methods.

### 6.5 Canonical exceptional values

Define them with Base's own spellings — `Inf32`, `NaN32`, and signed zero
literals — rather than bit patterns, and give them lowercase, unexported names
in the module's own namespace:

```julia
const POS_ZERO = raw(+0.0f0, +0.0f0)
const NEG_ZERO = raw(-0.0f0, +0.0f0)
const POS_INF  = raw(+Inf32,  +0.0f0)
const NEG_INF  = raw(-Inf32,  +0.0f0)
const D32_NAN  = raw(NaN32,   +0.0f0)   # NaN32 === reinterpret(Float32, 0x7fc00000)
```

Screaming-`D32_` prefixes are un-Julian: inside the module these are already
unambiguous, and outside it they are reached as `Double32s.POS_INF`. The one
exception is `D32_NAN`, since `NAN` alone reads poorly; `QUIET_NAN` is an
acceptable alternative.

For the non-finite path, a single branch-free canonicalizer keeps every guarded
kernel to one line:

```julia
"""
    canonical_nonfinite(x::Float32) -> Double32

Map a non-finite `Float32` to its canonical `Double32` pair. Branch free:
NaN maps to the package quiet NaN, ±Inf pass through, both get `+0.0f0` low.
"""
@inline canonical_nonfinite(x::Float32) = raw(ifelse(isnan(x), NaN32, x), 0.0f0)
```

NaN payload preservation is not an initial goal. All public operations may canonicalize any NaN to the package’s quiet NaN. Say so in the docstring of `Double32` itself, not only here — users discover semantics through `?Double32`.

---

## 7. Internal arithmetic primitives

### 7.0 Shape of the layer

The whole EFT layer follows one signature convention, and holding to it is what
keeps the pair kernels of section 8 readable:

```text
transform(a::T, b::T) -> Tuple{T,T}     # (rounded result, exact residual)
```

Tuples of a concrete `T` are returned in registers, cost nothing, and destructure
cleanly at the call site (`s, e = two_sum(a, b)`). Never return a `Double32`
from an EFT: the residual of a `Float32` transform is not a double-word value,
and wrapping it invites a spurious canonicalization.

Each transform gets exactly two methods:

| Method | Suffix | Branches | Where used |
|---|---|---|---|
| guarded | none | one non-finite check | constructors, tests, host code, cold paths |
| branch-free | `_` | none | inside pair kernels, which check once at the end |

The `_` suffix is borrowed directly from `DoubleFloats.jl`
(`two_sum` / `two_sum_`), and keeping the same convention makes the two
codebases comparable line by line. Both methods are `@inline` and generic in
`T<:Base.IEEEFloat` so that the test suite can validate them at `Float64`
against a `BigFloat` reference and production can call them at `Float32`.

### 7.1 `TwoSum`

```julia
"""
    two_sum_(a::T, b::T) -> (s::T, e::T)

Knuth's TwoSum, branch free. When `s` is finite, `s + e == a + b` exactly as
reals. When `s` is not finite, `e` is meaningless and the caller is
responsible for detecting that — see [`two_sum`](@ref).

Cost: six add/sub, no branches, no FMA.
"""
@inline function two_sum_(a::T, b::T) where {T<:Base.IEEEFloat}
    s  = a + b
    bb = s - a                       # the part of b actually absorbed by s
    e  = (a - (s - bb)) + (b - bb)   # what a lost, plus what b lost
    return s, e
end

"""
    two_sum(a::T, b::T) -> (s::T, e::T)

Guarded TwoSum. Identical to [`two_sum_`](@ref) for finite `s`; returns
`(s, zero(T))` when `s` overflows or an input is non-finite, where the error
formula would otherwise produce a spurious `NaN` from `Inf - Inf`.
"""
@inline function two_sum(a::T, b::T) where {T<:Base.IEEEFloat}
    s = a + b
    isfinite(s) || return nonfinite_pair(s)
    return two_sum_(a, b)
end

@noinline nonfinite_pair(s::T) where {T<:Base.IEEEFloat} = (s, zero(T))
```

**Established guarantee:** with IEEE round-to-nearest and finite `s`, `s + e` equals the exact real sum `a + b`.

**Why `@noinline` on the cold helper.** The non-finite return is taken
essentially never, but if it inlines, its constant and its return path
participate in register allocation and in the inliner's cost model for every
caller. Pushing it out of line keeps the hot body small enough that Julia's
inliner accepts `two_sum` itself into the pair kernels, which is the whole
point. `DoubleFloats.jl` already does exactly this with its
`@noinline zero_error_result`; the pattern generalizes and section 12.3 states
it as a rule.

**Why the guarded version recomputes `s`.** Writing `two_sum` as a check
followed by a call to `two_sum_` costs one extra addition in the cold path
only — LLVM CSEs the `a + b` in the hot path. The alternative (duplicating the
six-operation body) is faster to read wrong than to read right.

**GPU design:** call `two_sum_` inside pair kernels and check once at the end of the enclosing public operation, not once per transform.

### 7.2 `FastTwoSum`

```julia
"""
    two_hilo_sum_(a::T, b::T) -> (s::T, e::T)

Dekker's FastTwoSum. Three add/sub instead of six.

**Exactness condition.** `s + e == a + b` exactly, for finite `s`, iff
`fl(a + b) - a` is itself representable. `abs(a) ≥ abs(b)` is the familiar
*sufficient* condition, not a necessary one; `a == 0` and "`a + b` is exact"
are two further sufficient cases that call sites in section 8 actually rely on.
Stating the sufficient condition as though it were the requirement is what let
the section 8.1 justification stand unchallenged — see the note there.

The condition is *not* checked. Every call site must be justified by the
algorithm that produced `a` and `b`; see [`two_hilo_sum_checked`](@ref) for
constructor and test use.
"""
@inline function two_hilo_sum_(a::T, b::T) where {T<:Base.IEEEFloat}
    s = a + b
    e = b - (s - a)
    return s, e
end

"""
    two_hilo_sum_checked(a, b)

Ordering-checked FastTwoSum for host code and tests. Not for kernels.
"""
@inline function two_hilo_sum_checked(a::T, b::T) where {T<:Base.IEEEFloat}
    @boundscheck abs(a) ≥ abs(b) || throw(ArgumentError(
        lazy"FastTwoSum requires |a| ≥ |b|, got a=$a, b=$b"))
    return two_hilo_sum_(a, b)
end
```

The name follows `DoubleFloats.jl`'s `two_hilo_sum` rather than a transliterated
`fast_two_sum_ordered`: it says what the arguments *are* (a hi word and a lo
word) rather than what the caller promises, and it is the name a reader of both
packages will already know.

Two Julia-specific points in the checked variant. `@boundscheck` makes the check
elidable under `@inbounds` at the call site, which is the standard mechanism for
"validate on the host, trust in the kernel" — it is not restricted to array
indexing. And `lazy"..."` (Julia 1.8+, `LazyString`) defers the interpolation so
the error path allocates nothing unless it is actually taken, which keeps the
method `@allocated == 0` in the passing case.

For internal calls where the ordering follows from the algorithm, do not recheck it.

Approximate hot-path cost: three additions/subtractions.

### 7.3 `TwoDiff`

```julia
@inline function two_diff_(a::T, b::T) where {T<:Base.IEEEFloat}
    s  = a - b
    bb = s - a
    e  = (a - (s - bb)) - (b + bb)
    return s, e
end

@inline function two_diff(a::T, b::T) where {T<:Base.IEEEFloat}
    s = a - b
    isfinite(s) || return nonfinite_pair(s)
    return two_diff_(a, b)
end
```

Do **not** define `two_diff(a, b) = two_sum(a, -b)`. It is algebraically
identical and it is the kind of shortcut that costs nothing in accuracy and
everything in clarity when someone later needs to reason about signed zero:
`two_sum(a, -0.0)` and `two_diff(a, 0.0)` disagree on the sign of a zero result.
Write the symmetric transform out.

### 7.4 FMA-based `TwoProd`

```julia
"""
    two_prod_(a::T, b::T) -> (p::T, e::T)

FMA-based TwoProduct. `p + e == a * b` exactly, provided `p` is finite, `e` is
representable (no underflow below the least subnormal), and `fma` is a genuine
single-rounding fused multiply-add.
"""
@inline function two_prod_(a::T, b::T) where {T<:Base.IEEEFloat}
    p = a * b
    e = fma(a, b, -p)
    return p, e
end

@inline function two_prod(a::T, b::T) where {T<:Base.IEEEFloat}
    p = a * b
    isfinite(p) || return nonfinite_pair(p)
    return (p, fma(a, b, -p))
end
```

**`fma`, never `muladd`.** This is the single most important line-level rule in
the package. `Base.muladd` lowers to `llvm.fmuladd`, which *permits* but does not
require fusion; it may compute `a*b + c` with two roundings on one backend and
one rounding on another. `Base.fma` lowers to `llvm.fma` and always fuses. An
EFT built on `muladd` is silently wrong on whichever backend declines to fuse.
Section 12.5 gives the complete rule set, including why `p = a * b` on the line
above is *not* at risk of being contracted into the `fma` below it.

**Established under stated conditions:** when the exact residual is representable and FMA follows the required semantics, `p + e` equals the exact real product.

**Limit:** near underflow, the residual may be smaller than the least subnormal `Float32`. Under flush-to-zero, exactness may fail earlier.

This limit is confirmed and it is silent — `e` comes back a signed zero, not a
non-finite value, so no `isfinite` check anywhere in section 8 can see it:

```text
a                        b            p            e      p + e == a*b
2^-149 * 3               0.5f0        2^-149*1.5   -0.0f0   no   (err 7.0e-46)
prevfloat(1.0f0)         2f0^-140     7.17f-43     -0.0f0   no   (err 4.3e-50)
1.0000001f0              2f0^-145     2.2f-44      +0.0f0   no   (err 2.7e-51)
floatmin(Float32)        0.5f0        2^-127       +0.0f0   yes
```

Note the fourth row: the failure is not "whenever `p` is subnormal", it is
"whenever the exact residual falls below `2^-149`", which happens for some
subnormal `p` and not others. Any accuracy claim covering this region must be
derived per backend capability level (section 5.8), never asserted once — and
the sign of the returned zero is not a signal, per invariant 8 of section 5.2.

**Verify, do not assume, hardware FMA.** `Base.fma` on a CPU without FMA
instructions falls back to a correct but very slow software implementation, and
on some platforms historically to `libm`. Add a startup-free test — not a
runtime check — asserting that `@code_llvm fma(1.0f0, 2.0f0, 3.0f0)` contains
`llvm.fma.f32` and that the GPU backends emit `fma.rn.f32` (PTX) or
`v_fma_f32`/`v_fmac_f32` (GCN).

### 7.5 Scaled boundary transforms

A guarded multiplication path should detect exponent regions in which:

- `a*b` overflows although a correction could bring the full pair result back into range;
- the product is subnormal and the residual cannot be retained;
- cross terms overflow or underflow independently.

A scaled transform may use exact powers of two:

```text
a = a′ × 2ᵏ
b = b′ × 2ᵐ
a×b = (a′×b′) × 2ᵏ⁺ᵐ
```

Choose `k` and `m` from exponent bits so that the central product lies in a safe normal interval. Rescale both output words and renormalize.

The scaled path is a correctness boundary path, not the common GPU path. Its trigger and benefit must be validated against high precision.

### 7.6 Bit-level helpers

Implement `Float32` classification and exponent extraction with `reinterpret(UInt32, x)` where this improves generated code:

```julia
const SIGN_MASK = 0x8000_0000 % UInt32
const EXP_MASK  = 0x7f80_0000 % UInt32
const FRAC_MASK = 0x007f_ffff % UInt32
```

Use `0x8000_0000` with digit separators rather than `0x80000000` — the constants
are read far more often than they are written — and let the literals be `UInt32`
by construction rather than by annotation.

Three rules keep this section from metastasizing:

**Prefer Base, and prove the exception.** `signbit`, `exponent`, `significand`,
`ldexp`, `frexp`, `isnan`, and `isinf` all already lower to the obvious bit
operations. Reimplementing them buys nothing and loses the edge-case handling
Base has accumulated. Add a bit helper only with a comparative `@code_llvm` in
the commit message.

**Scale by construction, not by `2.0f0^k`.** Exact powers of two come from
`ldexp(1.0f0, k)` or from assembling the exponent field directly; never from
`Float32(2)^k`, which is a `pow` call unless the exponent is a literal, and
never from `exp2`. For the scaled boundary paths of section 7.5:

```julia
@inline pow2(k::Int) = reinterpret(Float32, (UInt32(k + 127) << 23))  # 
                                                     # valid for -126 ≤ k ≤ 127
```

with an `@assert` in the host-side test, not in the function.

**Keep helpers `@inline` and effect-clean.** `reinterpret` on isbits values is
already inferable and foldable. If a helper does *not* fold when you expect it
to, investigate before reaching for `Base.@assume_effects` — an incorrect
effects assertion is undefined behavior, not a hint, and it will be believed.

Do not duplicate Base predicates unless generated-code inspection demonstrates a benefit. Prefer Base semantics and use bit helpers only in tightly scoped internal code.

---

## 8. Arithmetic architecture

Each public operation follows:

1. execute a branch-free finite kernel;
2. test the final high word for non-finite output;
3. invoke a guarded or scaled fallback if required;
4. return a strongly normalized raw pair.

Note what is *not* in that list: a pre-classification of the operands. Testing
the output subsumes it. If either input is non-finite, the branch-free kernel
produces a non-finite high word, and the single output test catches it. This
halves the branch count relative to check-then-compute and it is what makes the
hot path one straight-line block ending in one predictable branch.

### 8.0 The three-layer shape of every operation

The package should be written in three layers, and the layer boundaries should
be visible in the file layout:

```julia
# Layer 1 — tuple kernels (core/arithmetic.jl). Generic, branch free, no type.
@inline function add_dd_(x::Tuple{T,T}, y::Tuple{T,T}) where {T<:Base.IEEEFloat}

# Layer 2 — policy dispatch (core/arithmetic.jl). Concrete, one branch.
@inline function add(::StrictPolicy, x::Double32, y::Double32)

# Layer 3 — Base integration (core/arithmetic.jl). One line each.
Base.:+(x::Double32, y::Double32) = add(DEFAULT_POLICY, x, y)
```

Layer 1 taking and returning `Tuple{T,T}` rather than `Double32` is the pattern
`DoubleFloats.jl` arrived at (`add_dddd_dd(x::Tuple{T,T}, y::Tuple{T,T})`), and
it is worth adopting for four reasons that are all Julia-specific:

1. **No constructor fights.** A tuple has no invariant, so an intermediate weak
   pair is expressible without either lying to `Double32` or paying for a
   canonicalization the algorithm will immediately undo.
2. **Reusable from SoA code.** The `SplitDouble32Array` kernels of section 13.2
   hold `hi` and `lo` in separate arrays and never materialize a `Double32`;
   they call the same layer-1 kernels.
3. **Testable at `Float64`.** The same source validates against `BigFloat` at
   double precision, where reference computation is cheap and the algorithm's
   error bounds are documented in the literature.
4. **Trivially inlinable.** Tuples of a concrete `T` are SROA'd away entirely;
   the emitted code is identical to hand-written scalar temporaries.

Layer 2 is where the struct, the policy, and the guard live. Layer 3 is one line
per Base function and contains no logic at all — if a `Base.:+` method has a
body longer than a single call, it is in the wrong layer.

#### What the single output check does and does not cover

It covers **classification** completely: every non-finite operand drives the
output high word non-finite, so a separate operand classification is redundant.

It does **not** cover the **overflow boundary for multiplication**. `xhi * yhi`
can round up to `Inf32` while negative cross terms bring the true pair product
back below `floatmax(Float32)`. The window is real but narrow: `Float32`
overflows to infinity at `2^128 - 2^103`, while the cross terms are of order
`2^-24` of the leading product, so the rescuable band has relative width about
`2^-24` just below the overflow threshold.

The consequence is a constraint on the *guarded* path, not on the hot path: for
`*`, `/`, and `sqrt`, the fallback must attempt the scaled recomputation of
section 7.5 before canonicalizing to `Inf32`. Writing the fallback as
`canonical_nonfinite(x.hi * y.hi)` — correct for `+` and `-`, where no such
rescue exists — would silently report `Inf` for a representable product.

Validate the boundary explicitly (section 23.4), not by sampling.

### 8.1 Addition

#### Accurate default

Algorithm 6 of Joldeş, Muller, and Popescu, relative error below `3u²`:

```julia
"""
    add_dd_(x, y) -> Tuple{T,T}

Accurate double-word addition. Branch free; the caller must check the result
for non-finiteness. Relative error < 3u² for finite results.
"""
@inline function add_dd_(x::Tuple{T,T}, y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    xhi, xlo = x
    yhi, ylo = y
    hi, lo   = two_sum_(xhi, yhi)
    thi, tlo = two_sum_(xlo, ylo)
    c        = lo + thi
    hi, lo   = two_hilo_sum_(hi, c)      # see the note below: |hi| ≥ |c| FAILS
    c        = tlo + lo
    hi, lo   = two_hilo_sum_(hi, c)      # ditto
    return hi, lo
end
```

**The earlier justification for these two calls was false.** It read
`|hi| ≥ |c| by construction of two_sum_`, on the grounds that `two_sum_` returns
`|e| ≤ ½ulp(s)` and the corrections stay below that bound. That reasoning
silently assumes `s ≠ 0`. Under high-word cancellation `hi` is exactly zero
while `c` is not, and the inequality is simply untrue. Measured on a structured
operand pool: **834 violations of `|hi| ≥ |c|` in 40,000 pair additions**, plus
violations with `hi ≠ 0` such as `hi = -2^-24`, `c = -3·2^-25`.

What actually saves the kernel is the *weaker* condition of section 7.2:
`two_hilo_sum_` is exact whenever `fl(a + b) - a` is representable. Every
observed violation falls into a case where it is:

- `hi == 0`: then `s = c` and `s - hi = c` exactly, so `e = 0` — correct;
- `hi ≠ 0` with `hi + c` exactly representable: `e = 0` again — correct.

Substituting `two_sum_` at both sites and comparing outputs bit-for-bit finds
**zero disagreements** across 40,000 structured and 1,500,000 random finite pair
additions, and end-to-end relative error stayed at `4.44e-15`, under the claimed
`3u² = 1.07e-14`.

That is evidence, not a proof. Two things follow, and both belong in the code:

1. Replace the false comment with the true one. The precondition being relied on
   is *representability of the Fast2Sum residual*, and the cancellation case is
   the one a reader must be pointed at, because it is the one that looks wrong.
2. The general bound is still open — see the Joldeş–Muller–Popescu argument for
   this exact operand ordering. Until it is reproduced, keep
   `two_hilo_sum_checked` wired into the host test build, and treat a
   disagreement against `two_sum_` as a correctness failure rather than a
   tolerance question. Where a proof is not available, use `two_sum_` and pay
   the three extra operations — an unproven `two_hilo_sum_` is a silent wrong
   answer, not a slow one.

Expected characteristics:

- high accuracy under cancellation;
- normalized output;
- roughly 18–22 scalar add/sub operations;
- several simultaneous temporaries;
- potentially significant register pressure.

#### Faster internal variant

For reductions and explicitly requested fast mode:

```julia
@inline function add_dd_fast_(x::Tuple{T,T}, y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    xhi, xlo = x
    yhi, ylo = y
    hi, e    = two_sum_(xhi, yhi)
    t        = (xlo + ylo) + e
    return two_hilo_sum_(hi, t)
end
```

This uses fewer operations and registers but has a weaker last-bit error bound.

#### Policy layer and the single guard

```julia
@inline function add(::StrictPolicy, x::Double32, y::Double32)
    hi, lo = add_dd_(words(x), words(y))
    isfinite(hi) || return add_nonfinite(x, y)
    return raw(hi, lo)
end

@inline function add(::FastPolicy, x::Double32, y::Double32)
    hi, lo = add_dd_fast_(words(x), words(y))
    isfinite(hi) || return add_nonfinite(x, y)
    return raw(hi, lo)
end

Base.:+(x::Double32, y::Double32) = add(DEFAULT_POLICY, x, y)
```

`words(x) = (x.hi, x.lo)` is the layer-1/layer-2 adapter and compiles away.
`raw` is justified here because the layer-1 kernels end in a `two_hilo_sum_`,
which establishes the canonical invariant by construction.

#### Non-finite behavior

The cold path is out of line, and it computes the IEEE-correct answer from the
high words alone:

```julia
@noinline function add_nonfinite(x::Double32, y::Double32)
    return canonical_nonfinite(x.hi + y.hi)
end
```

which yields exactly the required table, because `Float32` addition already
does:

```text
Inf + finite       → same infinity
+Inf + -Inf        → NaN
NaN + anything     → NaN
```

Return canonical non-finite pairs. Note that this path is also reached on
genuine overflow of two finite operands, where `x.hi + y.hi` is `±Inf32` — which
is the correct saturating answer.

### 8.2 Subtraction

Mirror addition using `two_diff_` for the high and low components. Preserve signed zero according to the `Float32` high-word operation and canonicalize `lo` to positive zero for a zero result.

Define `-` on its own kernel rather than as `x + (-y)`, for the signed-zero
reason given in section 7.3. Do, however, define the mixed-argument methods by
promotion rather than by duplication:

```julia
Base.:-(x::Double32, y::Double32) = sub(DEFAULT_POLICY, x, y)
Base.:-(x::Double32, y::Float32)  = sub(DEFAULT_POLICY, x, Double32(y))
Base.:-(x::Float32,  y::Double32) = sub(DEFAULT_POLICY, Double32(x), y)
```

Three explicit methods for the `Float32` mixed case are worth writing out
because `Double32(y::Float32)` is `raw(y, 0.0f0)` — free — and because relying
on `promote` here would route through the generic promotion machinery for an
operation this hot. For every *other* `Real`, do not write methods: Base's
promotion fallback plus the `promote_rule` of section 11.3 handles them
correctly, and each hand-written method is a new opportunity for an ambiguity.

### 8.3 Multiplication

#### Fast default candidate

A conventional GPU-friendly kernel is:

```julia
@inline function mul_dd_fast_(x::Tuple{T,T}, y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    xhi, xlo = x
    yhi, ylo = y
    p, e = two_prod_(xhi, yhi)
    c    = fma(xhi, ylo, xlo * yhi)     # both cross terms, one rounding saved
    t    = e + c
    return two_hilo_sum_(p, t)
end
```

This omits `x.lo*y.lo`, whose magnitude is normally of order `u²` relative to the leading product. It is efficient but can affect the final double-word bits.

#### Accurate default candidate

The recommended initial public `*` should include the low-low term:

```julia
@inline function mul_dd_(x::Tuple{T,T}, y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    xhi, xlo = x
    yhi, ylo = y
    p, e = two_prod_(xhi, yhi)
    c    = fma(xhi, ylo, xlo * yhi)
    c    = fma(xlo, ylo, c)
    t    = e + c
    return two_hilo_sum_(p, t)
end
```

Here `fma` is chosen for accuracy, not for exactness: each `fma` retains one
rounding that a separate multiply-and-add would lose. This is a legitimate use
of fusion and is *not* in tension with the "never `muladd` in an EFT" rule of
section 7.4 — the rule is about `muladd`'s optionality, and these are explicit
`fma` calls whose fusion is guaranteed.

This is still not a proof of correctly rounded `Double32` multiplication because residuals of cross products are not all retained. It is, however, a strong accuracy/performance compromise for GPUs.

#### Selection criterion

Implement both kernels internally. Make the more accurate kernel the public default unless benchmarks show a disproportionate regression. Publish measured error histograms and worst cases before changing the default.

Because the two kernels differ only by policy, the benchmark that decides this
is three lines, and it should live in `benchmark/scalar_cpu.jl` from the day
both kernels exist rather than being written once at decision time.

### 8.4 Division

Use one leading quotient and one compensated correction:

```julia
@inline function div_dd_(x::Tuple{T,T}, y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    q0     = x[1] / y[1]
    r      = sub_dd_(x, mul_dd_((q0, zero(T)), y))   # residual, EFT/FMA throughout
    q1     = (r[1] + r[2]) / y[1]
    return two_hilo_sum_(q0, q1)
end
```

An optional second refinement:

```julia
@inline function div_dd_refined_(x::Tuple{T,T}, y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    hi, lo = div_dd_(x, y)
    r2     = sub_dd_(x, mul_dd_((hi, lo), y))
    q2     = r2[1] / y[1]
    return two_hilo_sum_(hi, lo + q2)
end
```

which `StrictPolicy` may select if testing shows that one correction is not faithfully rounded in difficult cases. Selecting it is a one-line policy method, not a runtime flag — which is the payoff for the layering of section 8.0.

Special cases — zero divisor, infinities, NaNs, and overflow — need no pre-classification here either: every one of them drives `q0` non-finite or `hi` non-finite, and the single output check in the layer-2 method routes to

```julia
@noinline div_nonfinite(x::Double32, y::Double32) = canonical_nonfinite(x.hi / y.hi)
```

which reproduces IEEE `Float32` division semantics for `±0`, `±Inf`, and NaN
operands exactly. Verify this claim with an exhaustive test over the
classification cross-product (section 22.6 lists the structured subsets), rather than
reasoning about it once and trusting it.

### 8.5 Reciprocal

Implement `inv(y)` as a specialized division, saving the high-word divide's
operand setup:

```julia
@inline function inv_dd_(y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    q0 = inv(y[1])
    r  = sub_dd_((one(T), zero(T)), mul_dd_((q0, zero(T)), y))
    q1 = (r[1] + r[2]) / y[1]
    return two_hilo_sum_(q0, q1)
end

@inline Base.inv(x::Double32) = ...   # layer 2, as in section 8.1
Base.:/(x::Double32, y::Double32) = div(DEFAULT_POLICY, x, y)
```

Do not skip the correction merely because `y.lo == 0`; a one-word input such as `3.0f0` still requires the correction to produce double-word precision. This is a tempting "optimization" precisely because it looks like an algebraic identity, and it is the kind of thing a fast-path branch invites — another argument for keeping the kernels branch free by construction.

Also do not define `x / y` as `x * inv(y)`. It is one rounding worse and users
will reasonably assume `/` is the accurate operation.

### 8.6 Square root

For positive finite `x`:

```julia
@inline function sqrt_dd_(x::Tuple{T,T}) where {T<:Base.IEEEFloat}
    s0     = sqrt(x[1])
    r      = sub_dd_(x, mul_dd_((s0, zero(T)), (s0, zero(T))))
    s1     = (r[1] + r[2]) / (2 * s0)
    return two_hilo_sum_(s0, s1)
end
```

For very small or very large inputs, scale by an even power of two before the square root and reverse the scaling afterward — `ldexp(x, -2k)` before and `ldexp(result, k)` after, both exact.

Special cases:

```text
sqrt(+0)  → +0
sqrt(-0)  → -0, following Base/IEEE behavior
sqrt(+Inf)→ +Inf
sqrt(x<0) → NaN
sqrt(NaN) → NaN
```

Julia's `Base.sqrt(::Float32)` throws a `DomainError` for negative arguments
rather than returning NaN — `sqrt(-1.0f0)` is an error, `sqrt(complex(-1.0f0))`
is not. The `Double32` method must decide deliberately:

```julia
# Host-friendly, Base-consistent: throws, matching sqrt(::Float32).
function Base.sqrt(x::Double32)
    signbit(x.hi) && !iszero(x.hi) && throw(DomainError(x,
        "sqrt was called with a negative real argument; use sqrt(Complex(x))"))
    ...
end
```

A `throw` in a device kernel is not free: it forces the exception machinery into
the generated code and can abort the kernel. Provide the non-throwing spelling
as a separate function (`sqrt_or_nan`, selected by `FastPolicy` or called
directly from kernels) rather than making `Base.sqrt` silently non-Base-like.
Document which one kernels should use.

### 8.7 Fused multiply-add and `muladd`

A hardware-fused 48-bit `Double32` FMA is unavailable. The package can provide a compensated operation that computes all major pair terms and performs one final renormalization:

```text
x*y + z =
x.hi*y.hi
+ x.hi*y.lo
+ x.lo*y.hi
+ x.lo*y.lo
+ z.hi + z.lo
```

The implementation should use `TwoProd`, explicit FMA chains, and expansion-like summation.

Do not claim correct rounding or exact single-rounding semantics until proved. Document it as a high-accuracy compensated FMA.

Recommended API:

```julia
Base.fma(x::Double32, y::Double32, z::Double32) = compensated_fma(DEFAULT_POLICY, x, y, z)
Base.muladd(x::Double32, y::Double32, z::Double32) = fma(x, y, z)
```

They may share an implementation, but the docstring must state the achieved error
bound rather than imply hardware fusion.

Sharing an implementation is the right call here even though `fma` and `muladd`
differ for hardware floats. The reason is worth stating in the docstring:
`muladd`'s contract is "compute `x*y+z`, possibly with one fewer rounding," and
for `Double32` the compensated algorithm is both the most accurate *and* the
fastest way to do that, so there is no cheaper variant for `muladd` to select.
That is the opposite of the `Float32` situation and will surprise readers who
know the Base rule; say so.

### 8.8 Negation, absolute value, and sign transfer

All four sign operations are branch free and preserve the canonical invariant,
because negating both words of a canonical pair yields a canonical pair:

```julia
@inline Base.:-(x::Double32)              = raw(-x.hi, -x.lo)
@inline Base.abs(x::Double32)             = raw(abs(x.hi), flipsign(x.lo, x.hi))
@inline Base.flipsign(x::Double32, y::Real) = raw(flipsign(x.hi, y), flipsign(x.lo, y))
@inline Base.copysign(x::Double32, y::Real) = ifelse(signbit(x.hi) == signbit(y), x, -x)
@inline Base.abs2(x::Double32)            = x * x
```

`flipsign` and `copysign` on `Float32` are themselves sign-bit XORs in Base, so
these lower to two XORs each with no comparison and no branch. This is strictly
better than the earlier draft's `abs(x) = signbit(x.hi) ? -x : x`, which is a
select over an eight-byte struct that some backends will materialize as a
branch.

Two edge cases to document rather than to special-case:

**Zero low word sign.** `abs(Double32(-0.0f0))` produces `raw(0.0f0, -0.0f0)`,
whose low word is a negative zero. The represented value is `+0.0`, which is
correct, but the pair is not bit-identical to `POS_ZERO`. Rather than adding a
branch to fix this, relax invariant 6 of section 5.2 — but relax it *far enough*.

An earlier draft relaxed it only to "when `hi` is zero, `lo` is a signed zero
whose sign is unspecified", and that is not the actual reach of these four
methods. `-` flips the sign bit of `lo` unconditionally, so it produces a
negative zero low word under a perfectly ordinary high word:

```text
-one(Double32)     = (-1.0f0, -0.0f0)
Double32(-1.0f0)   = (-1.0f0, +0.0f0)      # constructor: raw(x, 0.0f0)

-Double32(1.5f0)   = (-1.5f0, -0.0f0)
Double32(-1.5f0)   = (-1.5f0, +0.0f0)

-D32_NAN           = (NaN32,  -0.0f0)
D32_NAN            = (NaN32,  +0.0f0)
```

Each row is two bit-distinct pairs holding one value, and the high word is
nonzero in every one of them. The correct relaxation is therefore invariant 8 of
section 5.2: *the sign of a zero `lo` is unspecified and carries no information,
at every high word.* The invariant test should assert `iszero(lo)`, never
`lo === 0.0f0`, and — this is the part the earlier draft got wrong — no
predicate may use `isequal` on the low words. See section 8.10.

**NaN sign.** `-D32_NAN` flips the sign bit of a quiet NaN. Canonical NaN
handling may preserve or discard the NaN sign, but it must do so consistently;
tests should assert `isnan(-x)` rather than any particular bit pattern.

### 8.9 Scaling by powers of two

```julia
Base.ldexp(x::Double32, n::Integer) = ldexp(x, Int(n))

function Base.ldexp(x::Double32, n::Int)
    hi = ldexp(x.hi, n)
    isfinite(hi) || return canonical_nonfinite(hi)
    return raw(hi, ldexp(x.lo, n))     # exact unless lo underflows
end
```

Funneling `n::Integer` to a single `n::Int` method is the standard Julia
idiom for integer-parameterized numeric functions: it gives one specialization
instead of one per integer type, which matters for both compile time and device
code size. Note that `ldexp` on the low word may underflow to zero while the
high word remains normal — a legitimate, documented precision taper, not an
error.

A guarded path is required when one component overflows or underflows before the pair result’s final status is known; `exponent`, `significand`, and `frexp` follow the same shape and the same funnel.

### 8.10 Comparisons

For canonical finite pairs, lexicographic comparison is valid, and it should be
written with non-short-circuiting operators so it lowers to selects:

```julia
@inline Base.:(==)(x::Double32, y::Double32) = (x.hi == y.hi) & (x.lo == y.lo)
@inline Base.:(<)(x::Double32, y::Double32)  = (x.hi < y.hi) | ((x.hi == y.hi) & (x.lo < y.lo))
@inline Base.:(<=)(x::Double32, y::Double32) = (x.hi < y.hi) | ((x.hi == y.hi) & (x.lo <= y.lo))

@inline Base.isequal(x::Double32, y::Double32) =
    isequal(x.hi, y.hi) & (x.lo == y.lo)

@inline Base.isless(x::Double32, y::Double32) =
    isless(x.hi, y.hi) | (isequal(x.hi, y.hi) & (x.lo < y.lo))
```

Each line is worth a sentence:

- `==` inherits NaN's unorderedness from `Float32` (`NaN32 == NaN32` is `false`,
  so the conjunction is `false`) and inherits `-0.0f0 == 0.0f0` being `true`.
  No special cases needed.
- `<` is `false` in both directions when either high word is NaN, which is the
  IEEE requirement. It needs no zero guard, because IEEE `<` already treats
  `-0.0f0` and `+0.0f0` as equal at both word positions.
- `isless` uses `isequal` on the **high** words, not `==`, so that `-0.0` sorts
  before `+0.0` and NaN sorts last — matching `isless` on `Float32` exactly, and
  therefore making `sort!` on a `Vector{Double32}` behave as users expect.
- `isequal` and `isless` use `==` and `<` on the **low** words, deliberately, not
  `isequal` and `isless`. See the rule below; it is not optional in either one.

#### The zero rule for total-order predicates

> **Rule.** The sign of a zero low word is never information. `isequal`,
> `isless`, and anything built on `Base.Order` must compare low words with `==`
> and `<`, which treat `-0.0f0` and `+0.0f0` as equal — never with `isequal` or
> `isless`, which do not.

This follows directly from invariant 8 of section 5.2 and its derivation in
section 8.8: `-`, `abs`, `flipsign`, and `copysign` all flip the sign bit of
`lo` unconditionally, so a negative zero low word occurs under *any* high word.

**Two successive drafts got this wrong, in two different ways.** They are worth
recording, because the second failure survived the property test written to
catch the first.

*First failure — guard on `isequal`, none on `isless`.* With
`x = abs(Double32(-0.0f0))` and `y = zero(Double32)`:

```text
x = (0.0f0, -0.0f0)      y = (0.0f0, +0.0f0)

unguarded isless(x, y) = isless(0.0f0, 0.0f0) | (isequal(0.0f0, 0.0f0) & isless(-0.0f0, 0.0f0))
                       = false | (true & true)
                       = true                      # and isequal(x, y) is also true
```

`isequal(x,y)` and `isless(x,y)` both true violates the strict weak ordering
`sort!` and `Base.Order` require, and the failure is a silently wrong sort.

*Second failure — guard on `hi` being nonzero, `isequal` still used on `lo`.*
The fix for the first failure was stated as "may inspect `lo` only when `hi` is
nonzero", which reads plausibly and is the wrong invariant. It leaves this:

```text
x = -one(Double32) = (-1.0f0, -0.0f0)     y = Double32(-1.0f0) = (-1.0f0, +0.0f0)

isequal(x, y) = isequal(-1.0f0, -1.0f0) & !iszero(-1.0f0) & isequal(-0.0f0, +0.0f0)
              = false                     # for two numerically equal values
isless(x, y)  = true                      # and they get ordered
```

This is not caught by trichotomy — exactly one of the three predicates still
holds, so `count(...) == 1` passes. What it breaks is everything downstream of
`isequal`: `Dict{Double32,V}` lookup misses (measured: `haskey` returns `false`
after storing the numerically identical key), `unique` keeps duplicates,
`issorted` disagrees with `==`, and `isequal(D32_NAN, -D32_NAN)` is `false`
where `isequal(NaN32, -NaN32)` is `true`. Over the 576-pair enumeration of the
full sign-of-zero surface, the guarded-on-`hi` form has **0 trichotomy
violations and 20 semantic mismatches**; the form above has 0 of each.

Both property tests must be present from the first commit — the first alone is
demonstrably insufficient:

```julia
# necessary, not sufficient: exactly one of the three holds, for every non-NaN pair
@test count((isless(x, y), isless(y, x), isequal(x, y))) == 1

# the one that actually pins the semantics: agree with the represented value,
# with the sign of a zero taken from the HIGH word only.
#   reference(x) = big(x.hi) + big(x.lo)   (section 22.1)
@test isequal(x, y) == (isnan(x.hi) ? isnan(y.hi) :
                        !isnan(y.hi) && reference(x) == reference(y) &&
                        signbit(x.hi) == signbit(y.hi))

@test isless(x, y) == (isnan(x.hi) ? false :
                       isnan(y.hi) ? true  :
                       reference(x) < reference(y) ||
                       (reference(x) == reference(y) &&
                        signbit(x.hi) && !signbit(y.hi)))
```

Note that the `iszero(x.hi)` guards disappear entirely under the corrected rule:
when `hi` is zero so is `lo`, and `==`/`<` on two zeros already give the right
answers. Fewer operations *and* correct.

The alternative fix — restoring a strict invariant by canonicalizing the zero
low word in `abs`, `-`, `flipsign`, and `copysign` — costs four selects on hot
sign operations, and, more importantly, leaves the same trap set for the next
predicate someone adds. Prefer the stated rule.

#### Mixed-type comparison must not go through promotion

`==` and `<` against a `Float64` cannot be implemented by promoting the
`Double32` operand: the conversion is lossy for pairs with a subnormal low word
(section 4.2), so promotion would report `Double32(1.0f0, 2f0^-149) == 1.0` as
`true` when the values differ.

Base sets the precedent here — `==(::Int64, ::Float64)` is deliberately exact
rather than promotion-based, for the same reason. Follow it:

```julia
# Exact: compare in the pair domain, never via Float64(x).
@inline function Base.:(==)(x::Double32, y::Float64)
    hi = Float64(x.hi)
    return (hi == y) & iszero(x.lo) | (Float64(x.lo) == y - hi) & isfinite(hi)
end

# Ordering needs no special cases at all: the non-finite answers fall out of
# IEEE arithmetic on `y - hi`, including the Inf - Inf -> NaN -> false cases.
@inline Base.:(<)(x::Double32, y::Float64) = Float64(x.lo) < y - Float64(x.hi)
@inline Base.:(<)(y::Float64, x::Double32) = y - Float64(x.hi) < Float64(x.lo)
```

**Why these are exact, not merely plausible.** Both words convert to `Float64`
exactly. The only rounding available is in `y - hi`, and it is harmless in each
of the two regimes:

- when `|y - hi| ≤ ½ulp(hi)`, the operands are within a factor of two of one
  another, so Sterbenz makes the subtraction exact and the comparison against
  `lo` is a comparison of exact quantities;
- when `|y - hi|` exceeds that, faithful rounding preserves both the sign and
  the fact that the magnitude exceeds `|lo| ≤ ½ulp(hi)`, so the comparison
  returns the same answer the exact difference would.

Overflow cannot occur: `|hi| ≤ floatmax(Float32)`, which is far below
`ulp(floatmax(Float64))`, so `y - hi` never leaves `Float64` range. When `hi` is
subnormal, invariant 5 forces `lo == 0` and the comparison degenerates to the
`Float32`/`Float64` one.

Verified over 918 operand combinations spanning `±0.0`, `±Inf`, `NaN`,
`floatmax`, `2.0^-1074`, `2f0^-96`/`2f0^-97` high words, and subnormal and
normal low words, against a 400-bit `BigFloat` reference: **0 mismatches for
`==`, `<`, and `>`.** This block is no longer a sketch.

Comparison against `Float32`, `Float16`, and integers *is* safe by promotion,
since every one of those converts to `Double32` exactly.

Do **not** define `!=`, `>`, or `>=`: Base derives all three from `==` and `<`,
and defining them by hand is three chances to get an inconsistency past the
test suite.

Tests must cover equality and ordering with `Float32`, `Float64`, and integers, including the subnormal-low-word and wide-gap pairs of section 23.11.

### 8.11 Min/max family

Implement Base-compatible `min`, `max`, and `minmax` only after NaN and signed-zero behavior is specified. Prefer delegating the selection to comparisons and returning an existing operand to avoid renormalization.

**The obvious formulation is wrong, in two independent ways.** An earlier draft
gave:

```julia
# WRONG — kept here as the failure it is
@inline Base.min(x::Double32, y::Double32) = ifelse(isnan(x) | (y < x), y, x)
@inline Base.max(x::Double32, y::Double32) = ifelse(isnan(x) | (x < y), y, x)
```

Checked against `Base.min`/`Base.max` on `Float32` (`math.jl`) over the 49
combinations of `±1.0f0`, `±0.0f0`, `±Inf32`, `NaN32`, this disagrees on **13**
of them:

- **NaN in either operand must propagate.** Base returns `NaN` for
  `min(1.0f0, NaN32)` *and* `min(NaN32, 1.0f0)`. The draft tests `isnan(x)` only,
  and `y < x` is `false` when `y` is NaN, so it returns the non-NaN operand in
  both directions. Silently swallowing a NaN in a reduction is the worst
  available failure: it turns a poisoned accumulator clean.
- **Signed zero must be ordered.** Base returns `-0.0f0` for both
  `min(-0.0f0, 0.0f0)` and `min(0.0f0, -0.0f0)`; `<` cannot see the difference,
  so the draft returns whichever operand came first.

Both fixes are branch free. Test the *other* operand for NaN, and add the
signed-zero tiebreak explicitly:

```julia
@inline function Base.min(x::Double32, y::Double32)
    lt = (y.hi < x.hi) | ((y.hi == x.hi) & (y.lo < x.lo))
    tie = iszero(x.hi) & iszero(y.hi) & signbit(y.hi) & !signbit(x.hi)
    return ifelse(isnan(y.hi) | lt | tie, y, x)
end

@inline function Base.max(x::Double32, y::Double32)
    lt  = (x.hi < y.hi) | ((x.hi == y.hi) & (x.lo < y.lo))
    tie = iszero(x.hi) & iszero(y.hi) & signbit(x.hi) & !signbit(y.hi)
    return ifelse(isnan(y.hi) | lt | tie, y, x)
end
```

`isnan(y.hi)` rather than `isnan(x.hi)`: when `y` is NaN the `ifelse` selects
`y`, and when `x` is NaN every one of `lt`, `tie` is `false`, so `x` is selected
— NaN propagates from either side with a single test. Verified: **0 mismatches**
against `Base.min`/`Base.max` over the same 49 combinations. `minmax` must be
defined as `(min(x,y), max(x,y))` using these, not by an independent comparison.

Do not implement `minimum`/`maximum`: they are `mapreduce` over `min`/`max` and
Base's implementations are already correct and already reach the GPU array
machinery once `min`/`max` exist. Adding methods there would be duplicated
effort and, for `AbstractArray{<:Real}` signatures, close to type piracy.

---

## 9. Accuracy requirements

The package shall state measured and, where available, proved guarantees separately.

| Operation | Initial target | Required qualification |
|---|---|---|
| constructor/renormalize | exact pair sum when finite | assumes gradual underflow for subnormal residuals |
| `TwoSum`/`TwoDiff` | exact decomposition | finite result, required rounding semantics |
| `TwoProd` | exact decomposition | residual representable; no FTZ violation |
| `+`, `-` | relative error of order `u²`; faithful target | exclude unavoidable underflow/overflow zones from relative bounds |
| `*` | small bounded relative error of order `u²` | default kernel and omitted/retained terms must be stated |
| `/`, `inv` | faithful target after correction | confirm with exhaustive structured testing |
| `sqrt` | faithful target after correction | scaled boundary path required |
| compensated `fma` | small error, preferably ≤ a few nominal ulps | do not call correctly rounded without proof |
| elementary functions | function-specific measured bound | no blanket guarantee |

Near zero, use absolute error. Near overflow and underflow, provide separate status and error tests.

### 9.1 Nominal ulp metric

For validation, define a nominal 48-bit ulp at a finite nonzero reference value `r`:

```text
ulp₄₈(r) = 2^(floor(log₂|r|) - 47).
```

This is a comparison metric, not necessarily adjacency in the pair representation.

### 9.2 Recommended result classifications

Validation reports should classify each result as:

- exact;
- correctly rounded to a selected 48-bit reference model;
- faithful (one of two nearest model values);
- within 1 nominal ulp;
- within 2 nominal ulps;
- larger finite error;
- incorrect exceptional status.

---

## 10. Julia package architecture

### 10.1 Recommended project layout

```text
Double32s.jl/
├── Project.toml
├── src/
│   ├── Double32s.jl
│   ├── core/
│   │   ├── type.jl          # Raw, Double32, raw, words, accessors
│   │   ├── bits.jl
│   │   ├── constants.jl
│   │   ├── eft.jl           # module EFT: two_sum, two_prod, …
│   │   ├── normalize.jl     # canonicalize
│   │   ├── special.jl
│   │   ├── convert.jl
│   │   ├── promote.jl
│   │   ├── compare.jl
│   │   └── arithmetic.jl    # layer 1 tuple kernels → layer 3 Base methods
│   ├── math/
│   │   ├── reciprocal.jl
│   │   ├── sqrt.jl
│   │   ├── hypot.jl
│   │   └── powers.jl
│   ├── arrays/
│   │   ├── splitarray.jl
│   │   └── adapt.jl
│   └── kernels/
│       ├── elementwise_ka.jl
│       ├── reduce_ka.jl
│       ├── dot_ka.jl
│       └── matmul_ka.jl
├── ext/
│   ├── Double32sCUDAExt.jl
│   ├── Double32sAMDGPUExt.jl
│   ├── Double32sBFloat16sExt.jl
│   ├── Double32sDoubleFloatsExt.jl
│   ├── Double32sLinearAlgebraExt.jl
│   └── Double32sSpecialFunctionsExt.jl
├── test/
│   ├── runtests.jl
│   ├── core/                # mirrors src/core one file per file
│   ├── reference/           # BigFloat / Arb high-precision references
│   ├── kernelabstractions/
│   ├── cuda/
│   ├── amdgpu/
│   ├── generated_code/      # LLVM / PTX / GCN assertions
│   └── quality/             # Aqua, JET, AllocCheck, ExplicitImports, doctests
├── benchmark/
│   ├── scalar_cpu.jl
│   ├── ka_cpu.jl
│   ├── ka_gpu.jl
│   ├── cuda_native.jl
│   ├── amdgpu_native.jl
│   ├── reductions.jl
│   └── matmul.jl
└── docs/
    └── src/
```

### 10.2 Dependency design

Provisional project structure:

```toml
[deps]
Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
PrecompileTools = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[weakdeps]
AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
DoubleFloats = "497a8b3b-efae-58df-a0af-a86822472b78"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[extensions]
Double32sAMDGPUExt = "AMDGPU"
Double32sBFloat16sExt = "BFloat16s"
Double32sCUDAExt = "CUDA"
Double32sDoubleFloatsExt = "DoubleFloats"
Double32sLinearAlgebraExt = "LinearAlgebra"
Double32sSpecialFunctionsExt = "SpecialFunctions"

[compat]
julia = "1.12"
Adapt = "4"
KernelAbstractions = "0.9.34, 0.10"
PrecompileTools = "1"
Random = "1"
AMDGPU = "2"
BFloat16s = "0.5"
CUDA = "6"
DoubleFloats = "1"
LinearAlgebra = "1"
SpecialFunctions = "2"
```

Three Julia-mechanical points about this file:

- **every weak dependency needs a `[compat]` entry**, exactly like a strong one;
  `Aqua.test_deps_compat` checks this and it is the most commonly missed line;
- **standard libraries are versioned too** in modern Julia — `Random = "1"`,
  `LinearAlgebra = "1"` — and omitting them fails the same check;
- **`LinearAlgebra` as a weak dependency** keeps `dot`/`norm` methods available
  when the user has loaded it without forcing the load otherwise. This costs
  nothing and follows the direction the ecosystem has taken.

Test-only dependencies (`Test`, `Aqua`, `JET`, `AllocCheck`,
`ExplicitImports`, `BenchmarkTools`, `Documenter`) belong in `test/Project.toml`
rather than in `[extras]`/`[targets]`; the separate-environment form is the
current convention and keeps heavy analysis tools out of the package's own
resolution.

The exact KA compatibility range must be tested against released versions. Do not depend directly on `GPUArrays.jl` unless the package eventually implements APIs that require its abstract array hierarchy. KA can obtain a backend from arrays supplied by CUDA or AMDGPU.

### 10.3 Public API

Recommended initial surface, applying the export hygiene rules of section 1.3:

```julia
export Double32                                   # the only exported name

public hi, lo, words, canonicalize                # accessors and normalization
public SplitDouble32Array
public StrictPolicy, NativePolicy, FastPolicy
public compensated_sum, compensated_dot, compensated_mapreduce
public nominal_eps, unit_roundoff
```

`public` (Julia 1.11+) marks a name as documented, supported API without
injecting it into the user's namespace. It is exactly the right tool for a
package whose useful names — `hi`, `lo`, `normalize`, `sum` — are all words
someone else is already using.

Do not export raw constructors or low-level EFT functions from the root module. Advanced users may access:

```julia
Double32s.EFT.two_sum
Double32s.EFT.two_prod
```

through a documented submodule. A submodule is the right container here because
it lets the EFT layer be `public` in its own namespace without any of it
reaching `using Double32s`, and because it makes the layering of section 8.0
visible in the module tree rather than only in the file tree.

#### Naming: prefer dispatch to prefixes

The names `d32sum`, `d32dot`, and `d32mapreduce` in earlier drafts encode the
element type into the function name. That is a C convention; in Julia the
element type is already in the argument, and the reason it *seemed* necessary is
worth separating into two different cases:

| Case | Verdict | Recommended spelling |
|---|---|---|
| `sum(A::AbstractArray{Double32})` | not piracy — we own `Double32` | overload `Base.sum` |
| `sum(A::AbstractArray{Float32})::Double32` | **type piracy** — Base owns both types | `sum(Double32, A)` |
| `dot(x::Array{Float32}, y::Array{Float32})::Double32` | **type piracy** | `compensated_dot(x, y)` |

The middle row is the important one. The headline use case — "sum `Float32`
data with a `Double32` accumulator" — needs no new name at all, because Base
already spells it:

```julia
sum(Double32, A)            # maps Double32 over A, accumulates in Double32
mapreduce(Double32, +, A)   # the same thing, spelled out
```

Once `Double32(::Float32)` and `+(::Double32, ::Double32)` exist, that works on
CPU and on GPU arrays with no package code whatsoever. Optimized paths attach by
adding methods whose signature mentions `Double32` — `sum(::Type{Double32},
::CuArray{Float32})` in the CUDA extension — which is dispatch, not piracy.

Only the genuinely new operations need new names, and they should be spelled as
what they do rather than as what they operate on: `compensated_sum`,
`compensated_dot`. Defer overloading `Base.sum` for `AbstractArray{Double32}`
until return type, allocation behavior, and backend dispatch are settled
(section 19.1), but plan for it — a `d32sum` that outlives the transition is a
permanent wart.

### 10.4 Internal API boundaries

- `core/eft.jl`: primitive `Float32` transformations only.
- `core/arithmetic.jl`: pair kernels without array or backend dependencies.
- `kernels/*_ka.jl`: KA kernel definitions and generic launch wrappers.
- extension modules: backend recognition, launch policy, profiling hooks, and native optimized kernels.
- no CUDA or AMDGPU symbol may appear in the root package source.

#### Method definition style

Define Base methods with qualified names rather than importing them:

```julia
Base.sqrt(x::Double32) = ...        # yes
import Base: sqrt; sqrt(x::Double32) = ...   # no
```

Qualified definition makes it obvious at every line which methods extend Base
and which are the package's own — the single most common source of accidental
piracy is a file that opened with `import Base: foo` two hundred lines earlier.
`ExplicitImports.jl` should run in CI to keep `using` lists minimal and to catch
names that arrive implicitly.

The include order in `src/Double32s.jl` should follow the dependency order of the
layers and be stated explicitly, not alphabetically:

```julia
module Double32s

using KernelAbstractions
using Adapt

include("core/type.jl")          # Raw, Double32, raw, words
include("core/bits.jl")
include("core/constants.jl")
include("core/eft.jl")           # module EFT
include("core/normalize.jl")     # canonicalize
include("core/special.jl")
include("core/arithmetic.jl")    # layers 1–3
include("core/compare.jl")
include("core/convert.jl")
include("core/promote.jl")
include("arrays/splitarray.jl")
include("arrays/adapt.jl")
include("kernels/elementwise_ka.jl")
include("kernels/reduce_ka.jl")
include("kernels/dot_ka.jl")

end # module
```

---

## 11. Julia numeric interface

### 11.1 Required Base methods

Initial release:

```text
zero, one, oneunit
typemin, typemax
floatmin, floatmax
precision
signbit, copysign, flipsign
iszero, isone, isfinite, isinf, isnan, isnormal, issubnormal
+, -, *, /, inv, sqrt, abs, abs2, fma, muladd
==, !=, <, <=, >, >=, isequal, isless
convert, promote_rule, widen
hash
show
broadcastable
```

Follow-on methods:

```text
significand, exponent, frexp, ldexp
floor, ceil, trunc, round
rem, mod, divrem
parse, tryparse
rand
nextfloat, prevfloat
```

Two entries in the first list should be struck rather than implemented.

**`broadcastable`** is unnecessary. `Double32 <: AbstractFloat <: Number`, and
Base already treats every `Number` as a zero-dimensional broadcast scalar. A
hand-written `Base.broadcastable(x::Double32) = Ref(x)` would be a redundant
method that can only diverge from Base's behavior later.

**`convert`** is unnecessary for the same reason given in section 6.4: Base's
`convert(::Type{T}, x::Number) where {T<:Number} = T(x)` already routes to the
constructors. Keep `promote_rule` and `widen`; drop `convert`.

Add instead:

```text
float, AbstractFloat        # float(x) === x; AbstractFloat(x) === x
eps                         # see below
Base.literal_pow            # only if benchmarks show x^2 is hot
```

### 11.2 Constants

Recommended semantics, spelled as methods:

```julia
Base.zero(::Type{Double32})     = POS_ZERO
Base.one(::Type{Double32})      = raw(1.0f0, 0.0f0)
Base.oneunit(::Type{Double32})  = one(Double32)
Base.typemin(::Type{Double32})  = NEG_INF
Base.typemax(::Type{Double32})  = POS_INF
Base.floatmin(::Type{Double32}) = raw(floatmin(Float32), 0.0f0)
Base.floatmax(::Type{Double32}) = raw(floatmax(Float32), prevfloat(ldexp(1.0f0, 103)))
Base.widen(::Type{Double32})    = Float64
Base.precision(::Type{Double32}; base::Integer=2) = _precision_in_base(48, base)
```

`floatmax` deserves its derivation in a comment rather than the phrase
"validated maximum canonical pair": `ulp(floatmax(Float32)) == 2^104`, so the
canonical bound is `|lo| ≤ 2^103`; the significand of `floatmax(Float32)` is
odd, so a low word of exactly `2^103` is a tie that rounds *up* and overflows.
`prevfloat(ldexp(1.0f0, 103))` is therefore the largest admissible low word.
Assert this in a test — `Double32(floatmax(Float32), ldexp(1.0f0, 103))` must be
infinite while `floatmax(Double32)` must be finite.

This derivation was flagged for independent re-derivation on the grounds that
it is the shape of argument that is off by one bit half the time. It has been
re-derived and checked numerically, and every step holds:
`ulp(floatmax(Float32)) == 2^104`; the half-ulp bound is `2^103`; the
significand of `floatmax(Float32)` is odd (`0x7f7fffff`); `two_sum` of
`floatmax(Float32)` with `ldexp(1.0f0, 103)` returns an infinite high word,
while `two_sum` with `prevfloat(ldexp(1.0f0, 103))` returns
`(floatmax(Float32), prevfloat(2^103))` unchanged, so the pair is canonical.
`V(floatmax(Double32)) ≈ 3.4028235677973306e38`. No change required.

`precision` takes a `base` keyword in Julia 1.8 and later. Define the method
with the keyword rather than relying on a Base internal to supply it, and pin
the behavior with a doctest:

```julia
julia> precision(Double32), precision(Double32; base=10)
(48, 14)
```

Provide, and mark `public`:

```julia
nominal_eps(::Type{Double32})   = ldexp(one(Double32), -47)
unit_roundoff(::Type{Double32}) = ldexp(one(Double32), -48)
```

Avoid claiming that `nominal_eps` is the true pair-lattice successor distance.

#### Revision: define `eps`, do not omit it

The earlier draft deferred `Base.eps` on the grounds that no single value is
truthful for a nonuniform pair lattice (section 5.6). That reasoning is sound
and the conclusion is wrong, because of how generic Julia code actually behaves:
`eps(T)` appears in the default tolerance of `isapprox`, in `rtoldefault`, in
iteration-stopping criteria throughout the ecosystem, and in most hand-written
convergence tests. Leaving it undefined does not make those callers careful; it
makes them throw `MethodError`, which pushes users toward `eps(Float32)` or
toward abandoning the type.

Define it with documented nominal semantics:

```julia
"""
    eps(::Type{Double32}) -> Double32

The nominal 48-bit unit in the last place, `2.0^-47`.

!!! warning
    `Double32` is a double-word format, not an IEEE format with a uniform
    significand lattice. This value is the algorithmic precision constant
    appropriate for tolerances and convergence tests; it is **not** in general
    the distance to the next representable pair, which can be as small as
    `2.0f0^-149` relative to a high word of `1.0f0`. See [`nextfloat`](@ref),
    which is deliberately not implemented.
"""
Base.eps(::Type{Double32}) = nominal_eps(Double32)
Base.eps(x::Double32) = ldexp(nominal_eps(Double32), exponent(x))
```

`nextfloat` and `prevfloat` stay deferred, and for the opposite reason: no
generic algorithm calls them speculatively, so their absence produces a clear
`MethodError` at a call site whose author genuinely needs adjacency semantics
and should be reading the documentation. Absence is informative there and
harmful for `eps`.

#### `rand`

`Random` reaches float types through a sampler for the `[0,1)` interval. Add
one method, following the pattern `BFloat16s.jl` uses:

```julia
function Random.rand(rng::AbstractRNG, ::Random.SamplerTrivial{Random.CloseOpen01{Double32}})
    u  = rand(rng, UInt64) >> 16              # 48 random bits
    a  = Float32((u >> 24) & 0x00ff_ffff)     # exact: 24 bits
    b  = Float32(u & 0x00ff_ffff)             # exact: 24 bits
    return canonicalize(ldexp(a, -24), ldexp(b, -48))
end
```

Every step is exact — 24-bit integers convert to `Float32` exactly and `ldexp`
by a power of two is exact — so the result is a uniform draw from a 48-bit grid
in `[0,1)`, which is precisely what `rand(Double32)` should mean. Test that the
low 24 bits are not systematically zero, the failure mode of the naive
`Double32(rand(Float32))`.

### 11.3 Promotion rules

Recommended:

| Other type | Promotion result |
|---|---|
| `Float16` | `Double32` |
| `BFloat16` | `Double32` |
| `Float32` | `Double32` |
| `Float64` | `Float64` |
| `BigFloat` | `BigFloat` |
| small/ordinary integer | `Double32`, following normal floating promotion behavior |
| `Complex{T}` | `Complex{promote_type(Double32,T)}` |
| `DoubleFloats.Double32` | extension-mediated conversion, no mandatory dependency |
| wider double-word type | wider type |

For `Float64`, the common type is chosen to lose the least: `Float64` has both a
wider significand (53 bits against a nominal 48) and a wider exponent range, so
promoting the other way would truncate a full `Float64` operand to `Float32`
range. Note that this is a least-loss argument, not a lossless one — the
`Double32 → Float64` conversion is itself inexact for pairs with a subnormal low
word (section 4.2), which is why `==` and `<` against `Float64` are implemented
exactly rather than by promotion (section 8.10).

As code, that is a short and complete list:

```julia
Base.promote_rule(::Type{Double32}, ::Type{Float16}) = Double32
Base.promote_rule(::Type{Double32}, ::Type{Float32}) = Double32
Base.promote_rule(::Type{Double32}, ::Type{Float64}) = Float64
Base.promote_rule(::Type{Double32}, ::Type{BigFloat}) = BigFloat
Base.promote_rule(::Type{Double32}, ::Type{<:Integer}) = Double32
```

Four rules govern writing these correctly in Julia:

1. **Give one ordering only.** `promote_rule` is consulted symmetrically by
   `promote_type`; defining both `(Double32, Float64)` and `(Float64, Double32)`
   is redundant and risks the two disagreeing.
2. **Never define `promote_rule(::Type{Double32}, ::Type{<:Real})`.** The
   abstract bound would capture types the package knows nothing about, including
   future ones, and would produce ambiguities against other packages' rules.
   `<:Integer` is the widest bound that is safe, and it is safe because the
   answer really is uniform across integer types.
3. **`Complex` needs nothing.** Base derives
   `promote_rule(Complex{T}, S)` from the real rules automatically.
4. **`BFloat16` and `DoubleFloats.DoubleFloat` go in extensions.** A
   `promote_rule` mentioning `BFloat16` requires `BFloat16s` to be loaded;
   putting it in the root module would make an optional dependency mandatory.
   `Double32sBFloat16sExt` and `Double32sDoubleFloatsExt` are the right homes,
   and neither is piracy — both signatures mention `Double32`.

`Test.detect_ambiguities(Double32s)` and `Aqua.test_ambiguities` should run in
CI; promotion and mixed-argument arithmetic are where ambiguities appear.

**Verified.** The lattice was checked over 18 types — the five float types, six
integer types including `BigInt`, `Rational{Int}`, `Rational{BigInt}`, and four
`Complex` instantiations — for the three properties that make a promotion
lattice usable:

| property | result |
|---|---|
| symmetry, `promote_type(A,B) === promote_type(B,A)` | 324 pairs, 0 failures |
| associativity, `promote_type(promote_type(A,B),C) === promote_type(A,promote_type(B,C))` | **5,832 triples, 0 failures** |
| no `Any` fallback | 324 pairs, 0 occurrences |
| `Test.detect_ambiguities(Double32s)` | 0 |

Two results are worth naming because neither is written down as a rule and both
are the right answer arriving for free. `Complex` derives entirely from the real
rules — `promote_type(Double32, Complex{Float32})` is `Complex{Double32}`, and
`Complex{Float64}` correctly stays `ComplexF64`. And `BigInt` and
`Rational{BigInt}` reach **`BigFloat`**, not `Double32`, through Base's own
rules overriding the `<:Integer` case: the least-loss answer, since neither fits
the `Float32` exponent range. The package deliberately does not override it, and
rule 2 above is why that is safe — a narrower bound than `<:Real` leaves Base's
better-informed rules in place.

### 11.4 `Complex`

Support `Complex{Double32}` through generic Julia methods after scalar arithmetic is stable. Do not add a separate `ComplexD32` type alias unless it materially improves usability.

Generic `Complex` arithmetic will work as soon as the real methods and
`promote_rule` exist, but it will be slower than a hand-written complex kernel
because it rounds each intermediate. Measure before optimizing, and if the
package does add complex kernels later, add them as methods on
`Complex{Double32}`, never as a new type — `ComplexDF32`-style aliases in
`DoubleFloats.jl` exist for historical reasons and do not need reproducing.

### 11.5 `show` and parsing

Initial `show` displays the exact pair, and the two-argument method must produce
output that parses back to the same value — that is Base's contract for
two-argument `show`, and the pair form satisfies it:

```julia
Base.show(io::IO, x::Double32) = print(io, "Double32(", x.hi, ", ", x.lo, ")")
```

```text
julia> Double32(1.0f0) / 3
Double32(0.33333334f0, -9.934107f-9)
```

Do not define a three-argument `show(io, ::MIME"text/plain", x)` yet. Base's
fallback routes it to the two-argument method, which is the desired behavior
until a decimal rendering exists.

A decimal rendering that promises round-trip reconstruction should be deferred; when it arrives it belongs in the three-argument method, with the pair form retained for two-argument `show` and for `repr`. Parsing should initially route through `BigFloat` on the host, never in device code.

---

## 12. Julia implementation practice

This section states the language-level rules the rest of the design assumes.
They are ordered from "applies to every line" to "applies to the project."

### 12.1 Inference and type stability

Every method in the scalar core must return a concrete type inferable from the
argument types alone. For this package that is nearly automatic — all fields are
concrete, all EFT returns are homogeneous tuples — and the two ways to break it
are both worth naming because both are tempting:

**A guard path that returns a different type.** `isfinite(hi) || return hi`
returns a `Float32` from a method whose fast path returns a `Double32`. The
inferred return becomes `Union{Float32,Double32}`, every caller becomes a
union-splitting branch, and the GPU compiler may reject it outright. Always
return the wrapped type: `isfinite(hi) || return canonical_nonfinite(hi)`.

**A `nothing` sentinel.** `Union{Nothing,Double32}` is idiomatic Julia in host
code and unusable in kernels. Where a "no result" case is needed on the device,
return a tuple `(ok::Bool, value::Double32)` — both fields concrete, no union.

Testing is mechanical and belongs in the test suite from the first commit:

```julia
@test @inferred(Double32(1.0f0) + Double32(2.0f0)) isa Double32
@test @inferred(two_sum(1.0f0, 2.0f0)) isa Tuple{Float32,Float32}
```

`Test.@inferred` fails loudly when inference degrades, which is what makes it
useful as a regression test rather than as a one-time check. Add
`JET.@test_opt` for the same call sites to catch runtime dispatch that inference
alone tolerates.

### 12.2 Allocation freedom

`Double32` is `isbits`, so scalar arithmetic allocates nothing by construction.
What breaks that property is always at the edges:

- a closure capturing a non-`const` global (boxed, allocates on call);
- a `Ref`, `Array`, or `String` in an error path that runs unconditionally;
- an abstractly typed field in a config struct;
- string interpolation in an exception message that is constructed before the
  branch is taken — hence `lazy"..."` in section 7.2.

Make every module-level binding `const`. Where a closure must capture a variable
that is reassigned, wrap the capture in `let`:

```julia
policy = user_policy(kwargs)
kernel = let p = policy          # `p` is a fresh, never-reassigned binding
    (out, x, y) -> add(p, x, y)
end
```

Check with `@allocated` for scalar methods and `AllocCheck.@check_allocs` for
whole call trees; the latter proves allocation-freedom statically rather than
observing it once.

```julia
@test @allocated(Double32(1.0f0) * Double32(3.0f0)) == 0
```

Note the pitfall: `@allocated` measures the first call including compilation
unless the method is already specialized. Warm up, or use `@ballocated` from
`BenchmarkTools`.

### 12.3 Inlining and code size

The inlining policy for this package is not "let the compiler decide," because
the compiler's cost model does not know that a six-instruction function is the
whole point of the abstraction.

| Construct | Annotation | Why |
|---|---|---|
| EFT primitives, `raw`, `words`, accessors | `@inline` | zero-cost only if inlined; bodies are tiny |
| layer-1 tuple kernels | `@inline` | callers must see through them to schedule FMAs |
| layer-2 policy methods | `@inline` | one branch plus a call |
| layer-3 `Base.:+` etc. | none needed | already trivial; the callee is `@inline` |
| non-finite and scaled fallbacks | `@noinline` | keeps hot-path register pressure and inliner cost down |
| host-only conversion, parsing, `show` | none | not hot, and inlining them bloats callers |

`@noinline` on cold paths is the annotation most often omitted and most often
worth having. `DoubleFloats.jl` already applies it
(`@noinline zero_error_result`), and the reasoning generalizes: an out-of-line
cold path removes its constants and its live ranges from the hot block, which on
a GPU directly translates into occupancy.

Verify rather than assume. On the host:

```julia
@code_llvm debuginfo=:none optimize=true Double32(1f0) + Double32(2f0)
```

should contain no `call` instructions to package functions and no `alloca`. On
the device, use `CUDA.@device_code_ptx` / `AMDGPU.@device_code_gcn` and read the
register report.

### 12.4 Branch-free idioms

GPU divergence and CPU misprediction both reward the same style. The
substitutions worth knowing:

| Instead of | Write | Note |
|---|---|---|
| `a && b` on cheap booleans | `a & b` | no branch; both operands already computed |
| `a \|\| b` on cheap booleans | `a \| b` | same |
| `cond ? x : y` on cheap values | `ifelse(cond, x, y)` | evaluates both arms — that is the point |
| `x < 0 ? -x : x` | `abs(x)`, `flipsign`, `copysign` | already sign-bit operations in Base |
| `if s == 0; ...` guard | compute, then check the result once | section 8.0 |

The rule for `ifelse` is that both arms must be cheap *and safe* to evaluate. It
is wrong for a branch that guards against overflow or a division by zero, and
right for selecting between two already-computed values. When the arms are
`Double32` structs, `ifelse` still applies — Julia will emit a pair of selects,
not a memory copy, for an isbits struct of two `Float32`.

### 12.5 The floating-point contract in Julia

This is the section to reread whenever an EFT produces a surprising residual.

**Julia does not contract `a * b + c` into an FMA.** The compiler emits
`fmul` followed by `fadd` with no fast-math flags, and LLVM's `fp-contract` is
off for ordinary Julia arithmetic. This is why `two_prod_`'s `p = a * b` on one
line and `fma(a, b, -p)` on the next is safe: nothing will fuse the first line.
Packages in other languages need `#pragma STDC FP_CONTRACT OFF` here; Julia does
not.

**`muladd` may contract; `fma` always does.** `muladd(a, b, c)` lowers to
`llvm.fmuladd`, which is *permitted* to fuse and is *permitted* not to — the
choice varies by backend and optimization level. Never use `muladd` where the
rounding behavior matters. Use `fma` when fusion is required for correctness or
accuracy, and plain `*`/`+` when separate roundings are required.

**`@fastmath` is prohibited package-wide.** It rewrites operations to
`Base.FastMath` equivalents that permit reassociation, which destroys every
error-free transform in the package. Add a grep-based test:

```julia
@test isempty(filter(f -> occursin("@fastmath", read(f, String)),
                     all_source_files()))
```

Command-line `--math-mode=fast` no longer affects generated code in supported
Julia versions, but `@cuda fastmath=true` and equivalent backend options still
do; keep them off and assert it in the launch wrappers.

**`@simd` implies reassociation.** `@simd for` grants the compiler permission to
reorder floating-point reductions. That is exactly what a compensated summation
loop must not permit. Never annotate a `Double32` accumulation loop with
`@simd`; if vectorization is wanted, vectorize across independent accumulators
explicitly.

**Check the generated code for unwanted fusion too.** The PTX inspection of
section 21.4 should look for `fma.rn.f32` where the algorithm requires it *and*
confirm `mul.rn.f32`/`add.rn.f32` remain separate where it requires that.
`ptxas` can contract at its own level; if it does, disable it explicitly rather
than working around the symptom.

### 12.6 Keeping generic code generic and concrete code concrete

The package uses parametric methods in exactly one place — the EFT and tuple
kernel layers — and concrete methods everywhere else. Two rules keep that from
drifting:

**Constrain to `Base.IEEEFloat`, not `AbstractFloat`.** `IEEEFloat` is
`Union{Float16,Float32,Float64}`, which is precisely the set on which the
transforms are valid. `AbstractFloat` would admit `BigFloat`, where `two_sum`
is meaningless (arbitrary precision has no fixed residual) and where the code
would silently allocate.

**Do not parameterize what has one instantiation.** A `DoubleWord{T}` type used
only at `T === Float32` costs compile time, method-table size, and specialization
churn while buying nothing. If a second width is ever needed, adding it then is
a mechanical refactor; guessing now is not.

### 12.7 Flags, keywords, and dispatch

A user-facing keyword argument is fine; a keyword argument reaching a kernel is
not. Keyword arguments do not participate in dispatch, so their values reach the
specializer only through constant propagation, which is best-effort.

Resolve keywords to types exactly once, at the host API boundary:

```julia
function compensated_sum(A::AbstractArray{Double32};
                         policy::ArithmeticPolicy = DEFAULT_POLICY,
                         deterministic::Bool = false)
    return _sum(policy, Val(deterministic), A)   # dispatch-visible from here on
end

_sum(policy::ArithmeticPolicy, ::Val{true},  A) = ...   # fixed tree
_sum(policy::ArithmeticPolicy, ::Val{false}, A) = ...   # tuned tree
```

`Val(deterministic)` introduces one dynamic dispatch at the boundary — a
function-call-sized cost, once per launch, in exchange for a fully specialized
kernel. That trade is correct here and wrong inside a loop; the general rule is
that `Val` belongs at boundaries, never in hot paths.

Where a keyword must stay a keyword through several layers, annotate the entry
point with `Base.@constprop :aggressive` and verify with `@code_typed` that the
constant actually propagates. Prefer the `Val` boundary.

### 12.8 Type piracy

Defining a method where neither the function nor any argument type belongs to
this package is type piracy: it changes behavior for code that never asked for
it, and it makes load order observable. The cases this package will be tempted
by:

| Tempting definition | Verdict |
|---|---|
| `Base.sum(::AbstractArray{Float32})::Double32` | piracy — use `sum(Double32, A)` |
| `LinearAlgebra.dot(::Vector{Float32}, ::Vector{Float32})::Double32` | piracy — use `compensated_dot` |
| `Base.sum(::AbstractArray{Double32})` | fine — `Double32` is ours |
| `Base.promote_rule(::Type{Double32}, ::Type{BFloat16})` | fine, but belongs in an extension |
| `KernelAbstractions.get_backend(::SplitDouble32Array)` | fine — the type is ours |
| `Base.:+(::CuArray{Float32}, ::Float32)` in our CUDA ext | piracy — nothing there is ours |

`Aqua.test_piracies` catches these mechanically. Run it in CI.

### 12.9 Package extensions

Extensions have exact mechanical requirements that are easy to get subtly wrong:

- the extension module name must match the `[extensions]` key in `Project.toml`,
  and the file must be `ext/<Name>.jl` (or `ext/<Name>/<Name>.jl`);
- every weak dependency needs a `[compat]` entry, same as a strong one;
- the extension does `using Double32s` and `using CUDA`, and defines methods on
  functions that already exist in the parent — it must not define *new* exported
  API, because the name would be unreachable when the extension is not loaded;
- the parent module must never `using` the weak dependency, not even inside a
  conditional;
- to call into an extension from the parent (rarely needed), use
  `Base.get_extension(@__MODULE__, :Double32sCUDAExt)` and handle `nothing`.

The pattern that keeps the parent clean is *declare the hook in the parent,
implement it in the extension*:

```julia
# src/kernels/reduce_ka.jl  (parent)
"""
    subgroup_reduce_available(backend) -> Bool

Extension hook. False unless a backend extension says otherwise.
"""
subgroup_reduce_available(::Any) = false

# ext/Double32sCUDAExt.jl
Double32s.subgroup_reduce_available(::CUDABackend) = true
```

The parent compiles and tests standalone; the extension adds capability without
the parent naming CUDA anywhere. The CI matrix of Appendix B must include the
no-extension, CUDA-only, AMDGPU-only, and both-loaded configurations, because
each exercises a different method table.

### 12.10 Documentation

Every `public` or exported name gets a docstring with the signature on the first
line, indented four spaces, then a one-sentence summary:

````julia
"""
    canonicalize(hi::Float32, lo::Float32) -> Double32

Return the canonical `Double32` whose value is the exact real sum `hi + lo`.

The result satisfies `|lo| ≤ ½ulp(hi)`; see section 5.2 of the design document
for the full invariant.

# Examples
```jldoctest
julia> canonicalize(1.0f0, 1.0f-8)
Double32(1.0f0, 1.0f-8)
```
"""
````

Doctests are the cheapest regression tests available for a numeric package —
they pin exact printed output, which catches both arithmetic and `show`
regressions. Wire them up with `DocMeta.setdocmeta!` and run
`doctest(Double32s)` in CI.

Two documentation obligations are specific to this type and should not be left
to prose in this design document alone:

1. the `Double32` docstring must state the NaN canonicalization policy and the
   nonuniform-lattice caveat, because `?Double32` is where users will look;
2. every method that is host-only (`parse`, `BigFloat` conversion, `show`) must
   say so, because "does this run in a kernel?" is the question users of this
   package ask most.

### 12.11 Test architecture

Mirror `src/` in `test/`, and give each file a `@testset` named for its subject
so failures identify themselves. Beyond the numerical suites of sections 22 and 23, run
the following, all of which are cheap and catch entire categories:

| Tool | Catches |
|---|---|
| `Test.@inferred` | inference regressions at named call sites |
| `@allocated` / `AllocCheck.jl` | accidental allocation in scalar and kernel code |
| `JET.@test_opt` / `@test_call` | runtime dispatch, unhandled unions |
| `Aqua.test_all` | piracy, ambiguities, stale deps, missing compat, unbound type params |
| `ExplicitImports.jl` | implicit imports, stale `using` |
| `Documenter.doctest` | printed-output and example regressions |
| `Test.detect_ambiguities` | promotion and mixed-argument ambiguities |

Seed every randomized test explicitly with `Random.Xoshiro(seed)` and log the
seed on failure. A property test that cannot be replayed is a flaky test.

`@test_broken` is the right tool for a known-failing edge case with an open
issue: it fails the suite when the case starts passing, which is what turns a
deferred decision into a tracked one.

### 12.12 Benchmarking that measures what you think

```julia
using BenchmarkTools
x = Double32(1.0f0) / 3
y = Double32(2.0f0) / 7
@benchmark $x * $y          # note the $ interpolation
```

Without `$`, the benchmark measures a global variable lookup — typically an
order of magnitude more than the operation. Additional rules:

- return or accumulate results so dead-code elimination cannot delete the work;
- vary inputs across iterations for operations with data-dependent cost;
- report the median and the distribution, not just the minimum;
- on GPUs, synchronize inside the timed region
  (`KernelAbstractions.synchronize(backend)`, `CUDA.@sync`) or measure nothing;
- compare equal numerical tasks — a `Float32` kernel and a `Double32` kernel do
  different amounts of useful work, and the interesting ratio is accuracy per
  unit time, not time.

`Chairmarks.jl` is a lighter-weight alternative with the same interpolation
rules and much lower overhead for microbenchmarks of this size.

### 12.13 Precompilation and latency

A concrete scalar type keeps the specialization count low, which is the main
lever on load time. Add a small `PrecompileTools` workload covering the scalar
operations and the KA CPU kernels:

```julia
using PrecompileTools
@setup_workload begin
    x, y = Double32(1.0f0), Double32(3.0f0)
    @compile_workload begin
        x + y; x - y; x * y; x / y; sqrt(x); fma(x, y, x)
        x < y; x == y; Float64(x); Double32(0.1)
        show(devnull, x)
    end
end
```

Do not attempt to precompile GPU kernels: device code is compiled per backend at
runtime and the workload would only add package load time. Precompiling the KA
*CPU* backend path is worthwhile and catches KA API drift at build time.

### 12.14 Style

Follow the Julia style guide; the points that recur in numeric packages:

- `CamelCase` types, `lowercase` or `lower_snake_case` functions, `SCREAMING`
  reserved for a handful of constants;
- append `!` to mutating functions, and put the mutated argument first;
- no type-encoding prefixes in names (`d32sum`); dispatch carries the type;
- prefer a named intermediate over a long expression — it costs nothing at
  runtime and the double-word kernels are unreadable without it;
- keep lines within about 92 characters, which is what makes side-by-side
  comparison with reference algorithms practical;
- write `1.0f0`, `0x8000_0000`, and `2^-24` in the form that matches the
  reference literature being implemented, and cite the reference in a comment.

The comment convention that matters most here: **every unchecked precondition
gets a comment justifying it at the call site.** `two_hilo_sum_`, `raw`, and
`@inbounds` are the three places where the compiler stops helping, and a
one-line justification beside each is the difference between an auditable
kernel and a trusted one.

---

## 13. Array representation

### 13.1 Array of structures: default

```julia
CuArray{Double32}
ROCArray{Double32}
Array{Double32}
```

Advantages:

- natural Julia element type;
- one logical load/store per value;
- simple broadcast and KA indexing;
- both words are adjacent;
- minimal wrapper code;
- best default for operations that always consume both words.

Risks:

- backend may emit two scalar loads rather than an aligned 64-bit load;
- array libraries may not have optimized kernels for custom structs;
- vendor BLAS cannot consume the representation.

### 13.2 Structure of arrays: optional

The wrapper needs no element-type parameter — the element type is always
`Double32` — and it needs an inner constructor to enforce the shape invariant
that section 13.4 requires:

```julia
"""
    SplitDouble32Array(hi, lo)

Structure-of-arrays view of `Double32` data: `hi` and `lo` are separate
`Float32` arrays with identical axes, and element `i` is the pair
`(hi[i], lo[i])`.
"""
struct SplitDouble32Array{N,AH<:AbstractArray{Float32,N},AL<:AbstractArray{Float32,N}} <:
       AbstractArray{Double32,N}
    hi::AH
    lo::AL

    function SplitDouble32Array(hi::AH, lo::AL) where {N,
            AH<:AbstractArray{Float32,N}, AL<:AbstractArray{Float32,N}}
        axes(hi) == axes(lo) || throw(DimensionMismatch(
            lazy"hi has axes $(axes(hi)) but lo has axes $(axes(lo))"))
        return new{N,AH,AL}(hi, lo)
    end
end
```

The `AbstractArray` interface is four required methods plus two that matter for
performance. All of them are one-liners, and each of the annotations is
load-bearing:

```julia
Base.size(A::SplitDouble32Array)      = size(A.hi)
Base.axes(A::SplitDouble32Array)      = axes(A.hi)
Base.IndexStyle(::Type{<:SplitDouble32Array}) = IndexLinear()

Base.@propagate_inbounds function Base.getindex(A::SplitDouble32Array, i::Int)
    return raw(A.hi[i], A.lo[i])
end

Base.@propagate_inbounds function Base.setindex!(A::SplitDouble32Array, v, i::Int)
    x = convert(Double32, v)
    A.hi[i] = x.hi
    A.lo[i] = x.lo
    return A
end

Base.similar(A::SplitDouble32Array, ::Type{Double32}, dims::Dims) =
    SplitDouble32Array(similar(A.hi, Float32, dims), similar(A.lo, Float32, dims))
```

`Base.@propagate_inbounds` rather than `@inline` on the indexing methods is the
detail that is easy to miss and expensive to get wrong: it forwards the caller's
`@inbounds` context to the two inner array accesses. Without it, an `@inbounds`
loop over a `SplitDouble32Array` still pays two bounds checks per element, and
inside a GPU kernel those bounds checks drag in the exception machinery.

`IndexLinear()` is correct because both backing arrays are indexed identically;
declaring it lets Base's generic `eachindex`, `copyto!`, and reduction fallbacks
take their fast paths.

`raw` in `getindex` is justified: the elements were canonical when stored, and
`setindex!` is the only way to store them.

Two more methods make the type work with the rest of the ecosystem:

```julia
Adapt.@adapt_structure SplitDouble32Array
KernelAbstractions.get_backend(A::SplitDouble32Array) = get_backend(A.hi)
```

`@adapt_structure` generates the `Adapt.adapt_structure` method that rebuilds
the wrapper around device-converted fields, which is what allows a
`SplitDouble32Array` of `Array`s to be passed to a kernel and arrive as a
`SplitDouble32Array` of device arrays. Note that it reconstructs by calling the
constructor, so the axes check runs on the device-side rebuild too — harmless,
since it is a compile-time-constant comparison there.

Advantages:

- independent coalesced streams;
- low words can be omitted for approximate passes;
- easier interoperation with `Float32` vendor routines;
- useful for progressive refinement and correction kernels;
- direct fieldwise access for reductions.

Disadvantages:

- two pointers and two memory operations;
- more complex adaptation and alias analysis;
- generic `setindex!` must preserve pair consistency;
- vendor routines operating only on `hi` do not by themselves provide `Double32` accuracy.

### 13.3 Recommendation

Provide both, with AoS as the primary user representation.

Use SoA when:

- a kernel reads only high words during a seed phase;
- vendor `Float32` code produces a high approximation followed by a correction;
- benchmarks show better memory behavior;
- a reduction stores high and low partials separately.

Use `Adapt.@adapt_structure` for the wrapper. The scalar itself requires no adaptation.

### 13.4 Aliasing and mutation

`SplitDouble32Array` must require equal axes and element counts. Mutating only one component through the public API should not be permitted. Internal field arrays may remain accessible for expert use with documented invariant responsibility.

---

## 14. KernelAbstractions architecture

`KernelAbstractions.jl` is the required primary execution layer.

### 14.1 Elementwise kernels

Scalar methods should already permit ordinary GPU broadcast. Explicit KA kernels are still useful for:

- fused application pipelines;
- conversion between AoS and SoA;
- capability tests;
- benchmark control;
- operations requiring explicit policy selection.

Example design sketch:

```julia
@kernel function add_kernel!(out, @Const(x), @Const(y), policy::ArithmeticPolicy)
    i = @index(Global, Linear)
    @inbounds out[i] = add(policy, x[i], y[i])
end
```

Pass the policy as an ordinary singleton *value*, not as a type parameter
reconstructed with `P()`. Julia specializes the kernel on the concrete singleton
type either way, so the two forms generate identical code — but the value form
is shorter, matches the scalar API of section 5.8, and does not require the
reader to notice that `P()` is free.

Use `@Const` for read-only arrays to communicate nonaliasing and nonmutation intent. Note that `@Const` is a promise, not a check: passing an aliased or later-mutated array is undefined behavior, so the launch wrapper — not the kernel — is where aliasing should be validated on the host.

`@inbounds` here is likewise a promise. It is correct only because the launch
wrapper sets `ndrange` from `length(out)` and the arrays have equal length;
that check belongs in the wrapper, with a comment at the `@inbounds` pointing to
it (section 12.14).

### 14.2 Static workgroup sizes

KA local memory is static. Reduction kernels should be instantiated with a fixed workgroup size:

```julia
kernel = reduce_stage!(backend, WG)
kernel(partials, input, ndrange=...)
```

Candidate sizes should be benchmarked per backend, commonly among 64, 128, 256, and 512 where supported.

### 14.3 Convergent synchronization

Every work item in a workgroup must reach each `@synchronize`. Bounds handling must not place the barrier in divergent control flow.

Recommended pattern:

1. derive the global index from group and local indices;
2. every work item writes either an input value or zero to local memory;
3. synchronize;
4. perform a convergent tree reduction;
5. lane 1 writes the workgroup result.

**What KA's default actually does — read from the source, because the obvious
assumption is wrong.** With `unsafe_indices=false` (the default),
`KernelAbstractions.transform_gpu!` emits
`__active_lane__ = __validindex(__ctx__)` and then calls `split(emit_gpu, body)`.
`split` cuts the body **at each `@synchronize`**, and `emit_gpu` wraps only the
statements *between* barriers in `if __active_lane__ … end`. The barrier itself
is emitted between segments, outside the conditional.

**And a second correction, found while writing the Phase 5 matrix kernel.**
`transform_gpu!` takes the `unsafe_indices` parameter and skips the splitting
entirely when it is true. **`transform_cpu!` takes no such parameter and always
splits**, because it has to turn work items into loops. So an
`unsafe_indices=true` kernel is *not* split on the GPU path but *is* on the CPU
path — which means the scoping restriction below is imposed by the **CPU**
backend for exactly those kernels. The restriction is identical either way and
any kernel that must run on both has to obey it, but a reader who assumes the
GPU path is the strict one will conclude that `unsafe_indices=true` relaxes it.
It does not.

**The scoping restriction, stated once.** A local assigned before a
`@synchronize` is not in scope after it, and neither is one assigned before a
barrier-containing loop visible inside that loop. Three consequences, all hit in
practice:

- loop bounds must be inlined or `@uniform`, never a local computed above the
  loop;
- values derived from `@index` must be recomputed in each barrier-separated
  segment — `@index` calls *are* re-emitted per segment, so calling it again is
  free of this problem while caching its result is not;
- an accumulator that must survive the loop belongs in `@localmem`. The scalar
  `@private x = …` form does not survive, despite its docstring saying it may:
  `split` resets `private_allocations` per segment, so the initializer is
  re-emitted rather than carried. Worse, if the initializer references an
  `@index` result it is hoisted out of workitem scope and fails to compile at
  all.

So the default is **already barrier-convergent by construction**. It is not the
case that KA's bounds handling puts a barrier in divergent control flow and that
`unsafe_indices=true` is needed to rescue it; that reading is natural, and it is
false. `unsafe_indices=true` skips both the `__validindex` computation and the
splitting, emitting the body raw — so *every* lane, in range or not, executes
every statement and reaches every barrier.

**The decision, therefore, is a throughput decision, not a correctness one:**

- **Use `unsafe_indices=true` only when both hold:** the kernel contains
  `@synchronize`, **and** the launcher guarantees `ndrange` is an exact multiple
  of the workgroup size, so `__active_lane__` would be statically true and the
  per-segment branch is pure overhead. Such a kernel must then derive indices
  from `@index(Group, …)` and `@index(Local, …)` only — never `@index(Global)` —
  and must guard every *data* access against the true length, which it needs
  anyway because the data length is not the ndrange.
- **Barrier-free kernels always take the default.** With no `@synchronize` there
  is no splitting to skip, so `unsafe_indices=true` buys only the removal of one
  masked test and gives up KA's entire index-validity guarantee. That is a bad
  trade at any kernel size.

The package's only kernel today, the capability probe of section 5.9, is
barrier-free and correctly takes the default.

### 14.4 KA reduction baseline

A portable stage kernel:

```julia
@kernel unsafe_indices=true function reduce_stage!(
    partials, @Const(input), n::Int, policy::ArithmeticPolicy
)
    gid = @index(Group, Linear)
    lid = @index(Local, Linear)
    W   = @uniform prod(@groupsize())

    # Experimental choice: one local Double32 array or two Float32 arrays.
    scratch = @localmem Double32 (W,)

    i   = (gid - 1) * (2W) + lid
    acc = zero(Double32)

    # Branch-free would be wrong here: input[i] must not be read out of bounds.
    if i <= n
        @inbounds acc = input[i]
    end
    if i + W <= n
        @inbounds acc = add(policy, acc, input[i + W])
    end

    @inbounds scratch[lid] = acc
    @synchronize()

    # The strides come from a compile-time tuple, NOT a loop-carried variable.
    # See the note below: a `while` loop here does not compile.
    for s in halves(Val(W))
        if lid <= s
            @inbounds scratch[lid] = add(policy, scratch[lid], scratch[lid + s])
        end
        @synchronize()
    end

    if lid == 1
        @inbounds partials[gid] = scratch[1]
    end
end

@inline halves(::Val{W}) where {W} = ntuple(k -> W >>> k, Val(trailing_zeros(W)))
```

**A `while` loop across a barrier does not work, and this was written the wrong
way first.** The earlier form of this kernel was

```julia
stride = W >>> 1
while stride > 0
    if lid <= stride
        @inbounds scratch[lid] = add(policy, scratch[lid], scratch[lid + stride])
    end
    @synchronize()
    stride >>>= 1
end
```

with the comment "uniform trip count, so every work item reaches every barrier".
The trip-count reasoning is correct and the kernel still does not run. KA's
`split` cuts the body at every `@synchronize` and recurses into scope constructs
with **hard scoping** — it `deepcopy`s the environment, and the source carries a
`TODO` saying soft scoping is what is actually wanted — so a local assigned
before a barrier is not in scope after it.

The failure is worth reproducing exactly, because it does not look like a
scoping error:

```text
MethodError: no method matching isless(::Int64, ::typeof(stride))
```

`stride` was not merely lost; the name resolved to **`Base.stride`**, the
function, and `stride > 0` became a comparison against a function object.
Renaming the variable turns it into the honest `UndefVarError`, which is how the
real cause becomes visible. Iterating a compile-time tuple removes the problem
at the root: every stride is a literal, so nothing crosses a barrier.

Both stage kernels in `ext/Double32sKernelAbstractionsExt/reduce.jl` are written
this way, and `halves` there carries the same warning.

Four Julia-level points about this kernel generalize to every reduction the
package writes:

**`@uniform` is what makes `W` a compile-time constant.** Without it, `W` is a
per-work-item value and `@localmem Double32 (W,)` cannot be sized statically.
With it, the whole `while` loop unrolls against a known bound.

**`unsafe_indices=true` disables KA's automatic bounds handling**, so the kernel
author owns index validity entirely. Per section 14.3 it is justified here — and
only here — because this kernel has barriers *and* is launched with `ndrange` an
exact multiple of the workgroup size, which makes KA's `__active_lane__` test
statically true and its per-segment branch pure overhead. It is **not** required
for barrier convergence; the default is already convergent. Every kernel that
carries the annotation should carry a comment naming the launch invariant it
relies on, and should be reviewed as a unit.

**`@inbounds` inside a kernel is not optional for performance.** A retained
bounds check compiles to a branch plus a `throw` path, and the exception
machinery is expensive to instantiate on a device — sometimes fatal to
compilation. Combine it with `unsafe_indices` deliberately, not accidentally.

**The accumulator's type must be pinned.** `acc = zero(Double32)` followed by
`acc = input[i]` is type-stable only because `eltype(input) === Double32`. If
the kernel is ever reused for `Float32` input with a `Double32` accumulator
(section 19.2), write `acc = zero(Double32)` and `acc = add(policy, acc,
Double32(input[i]))` so the accumulator type never depends on the branch taken.
This is the most common way a reduction kernel silently becomes type-unstable.

**Experiment 1 of section 28 — `@localmem Double32 (W,)` versus two
`@localmem Float32 (W,)` arrays — is closed: no measurable difference.** Both
were implemented and timed on an RTX 5090 over 50 M `Float32` elements:

| W | AoS local memory | SoA local memory |
|---|---|---|
| 128 | 0.448 ms | 0.430 ms |
| 256 | 0.425 ms | 0.420 ms |
| 512 | 0.431 ms | 0.416 ms |

The SoA column is a first pass only while the AoS column is the full multipass
reduce, so the ≤4% gap is smaller than it looks and sits inside run-to-run
noise. The reason the layout does not matter is visible in the absolute
numbers: a `Float32` `sum` of the same array takes 0.161 ms, so at 0.43 ms this
kernel is **arithmetic bound on `Double32` addition**, not bound by local-memory
banking. Bank conflicts are not the constraint, so removing them buys nothing.

**Decision: `@localmem Double32`.** With performance tied, the tiebreakers are
the ones stated above — it keeps the scalar API, the reduction body is
`op(scratch[lid], scratch[lid + s])` instead of four indexed word loads and two
stores, and there is one array to size instead of two.

### 14.5 Multipass reduction

Never rely on a `Double32` atomic for the final result. Launch repeated stages until one partial remains.

Benefits:

- no custom atomic;
- deterministic tree for fixed launch parameters;
- portable across CPU, CUDA, and AMDGPU;
- easy validation.

### 14.6 Grid-stride accumulation

For large inputs, each work item should accumulate multiple values in registers before local reduction. The optimal number depends on:

- memory bandwidth;
- operation latency;
- register pressure;
- occupancy;
- workgroup size.

Provide a launch policy object selected on the host:

```julia
struct ReductionConfig
    groupsize::Int
    items_per_thread::Int
    use_native_subgroup::Bool
end
```

The values are compile-time parameters in the actual kernel specialization where possible.

### 14.7 Deterministic mode

`compensated_sum(A; deterministic=true)` shall choose:

- a documented workgroup size;
- a fixed pairwise tree;
- no atomics;
- no backend-specific reassociation.

Results are deterministic for the same input order, package version, policy, backend, and launch configuration. Cross-backend bitwise identity is a test target for the strict KA path, not an unconditional guarantee.

---

## 15. CUDA.jl extension

The CUDA extension is optional and loads only when `CUDA.jl` is present.

### 15.1 Responsibilities

`Double32sCUDAExt` may provide:

1. detection of `CUDABackend`;
2. CUDA-specific default workgroup sizes;
3. occupancy-informed launch tuning;
4. PTX and register-count test hooks;
5. warp-shuffle reductions;
6. direct CUDA kernels where they outperform the KA baseline;
7. CUDA-specific capability tests;
8. optional use of CUDA math functions as seeds for later refinement algorithms.

It shall not redefine scalar arithmetic semantics.

### 15.2 Warp-level reduction

CUDA exposes warp shuffle operations. A native reduction may:

1. accumulate a `Double32` in each lane;
2. shuffle `hi` and `lo` separately;
3. add the received pair;
4. reduce one warp without shared-memory barriers;
5. store one pair per warp;
6. reduce warp partials in shared memory or a second kernel.

This can reduce barriers and shared-memory traffic.

The native path must:

- use the active mask correctly;
- support partially filled final warps;
- preserve the same operand tree when deterministic mode requires it;
- benchmark against the KA local-memory path;
- fall back to KA if no benefit is measured.

**Measured, and the last clause is the one that fired: no warp-shuffle
reduction ships.** Both forms were implemented and timed on an RTX 5090,
`ak_sum` over 50 M `Float32`, against the KA local-memory tree:

| candidate | speedup vs KA |
|---|---|
| full warp-shuffle reduction, as sketched above | **0.48–0.56×** |
| hybrid: shared-memory tree down to 32 lanes, then shuffle | 0.99–1.10× |

Both produce **bit-identical results** to the KA path, so semantic parity was
never the obstacle — throughput was.

**Why this inverts the usual advice, which is the part worth keeping.** A
shuffle round makes *every* lane apply the operator; a tree round halves the
live lanes. Removing five barriers is a good trade against five extra *1-flop*
`Float32` operations, and a bad one against five extra *20-flop* `Double32`
additions. The classic "shuffle beats shared memory" result is a statement about
cheap operators, and this format's operator is not cheap.

The obvious rescue — apply shuffles only where the operator *is* cheap — was
also tested. On a `Float32` `max` reduction the hybrid reaches 1.07× at the
default workgroup size and 1.26× at 1024. Still not material, and it would buy
a second reduction path to maintain and test for a case the package uses once
(the scale pass of `ak_norm2`). Rejected.

### 15.3 Launch configuration

Compile representative kernels and query CUDA launch/occupancy information. Tune for register-heavy arithmetic rather than assuming the usual `Float32` block size.

Candidate policy:

```text
elementwise: 128–256 threads
reduction:   128 or 256 threads
dot:         128 or 256 threads, multiple items per lane
```

These are hypotheses, not fixed constants.

**Tested. The hypothesis holds; the vendor heuristic does not.** On an RTX 5090,
`ak_sum` over 50 M `Float32`:

| workgroup | 128 | **256** | 512 | 1024 |
|---|---|---|---|---|
| ms | 0.413 | **0.395** | 0.407 | 0.406 |

256 wins, inside the range this section guessed. CUDA's `launch_configuration`
suggests **768** — outside it, and not even a power of two, so it cannot drive
the tree at all. The heuristic maximizes resident warps to hide memory latency;
this kernel is arithmetic bound on `Double32` addition, so extra resident warps
buy nothing while a larger block deepens the intra-block tree by `log2(W)`
barriers. `Double32s.launch_config` reports **both** numbers rather than
silently preferring one, so the disagreement stays visible.

The whole spread is 4.5%, which is the more useful conclusion: launch shape is
not where this kernel's time goes.

**Register and occupancy figures must be read from a normal build.**
`Pkg.test()` runs under `--check-bounds=yes`, which disables every `@inbounds`
in the package, so kernels compiled during testing are not the kernels a user
gets — measured on the same machine, the reduction stage reports 0 bytes of
local memory normally and 32 bytes under `--check-bounds=yes`. The test suite
asserts generous ceilings only; `docs/Reports/gpu_report.jl` produces the
figures worth quoting.

### 15.4 CUDA math intrinsics

CUDA device math may provide fast `Float32` approximations. For a `Double32` elementary function, a CUDA intrinsic may be used only as an initial seed followed by a backend-neutral correction or polynomial evaluation.

Do not return an intrinsic’s `Float32` result directly as though it had double-word accuracy.

### 15.5 Direct CUDA kernels

Use direct `CUDA.@cuda` kernels only when all of the following hold:

- a KA implementation exists and serves as the reference;
- the direct kernel has the same documented semantics;
- the direct kernel is materially faster on representative devices;
- CI validates both paths;
- the implementation is isolated in the CUDA extension;
- users can force the KA path for reproducibility or debugging.

Likely candidates are subgroup reductions and tuned dot products, not ordinary scalar elementwise arithmetic.

**Outcome: no direct CUDA kernel ships, and the reason is instructive.** A
direct `CUDA.@cuda` reduction using the *same* tree as the KA path measured
about 8% faster, which looked like it cleared the "materially faster" bar. It
did not — the gap was an artifact of the KA *driver*, not of KA kernels. The
driver was calling `KernelAbstractions.synchronize` after every reduction pass.
Successive launches on one backend are queue-ordered and the CPU backend is
synchronous outright (`synchronize(::CPU) = nothing`), so those syncs bought
nothing. Removing them took `ak_sum` from 0.444 ms to 0.395 ms and closed the
gap to 1.5% — inside noise.

Two things follow, and both generalize past this instance:

1. **Before accepting that a vendor kernel beats the portable one, check
   whether the portable *driver* is the thing being measured.** Here the
   candidate replacement and the incumbent ran identical device code; the entire
   difference was host-side synchronization.
2. **A portable fix outranks a vendor-specific duplicate of equal speed.** The
   removed syncs speed up every backend, including ones this project cannot
   test, and leave one reduction implementation to maintain instead of two.

Post-fix, `ak_sum` is about 5% *faster* than GPUArrays' tuned `sum` on the same
data, so the KA baseline is not a compromise position.

---

## 16. AMDGPU.jl extension

The AMD extension mirrors the CUDA extension and loads only when `AMDGPU.jl` is present.

### 16.1 Responsibilities

`Double32sAMDGPUExt` may provide:

1. recognition of the AMD KA backend;
2. occupancy-informed workgroup selection;
3. wavefront-level shuffle reductions;
4. GCN/LLVM generated-code inspection hooks;
5. direct `AMDGPU.@roc` kernels where benchmarks justify them;
6. AMD-specific subnormal and FMA capability tests;
7. device-specific wavefront-width handling.

### 16.2 Wavefront width

Do not hardcode 64 lanes. AMD hardware and execution modes may use different wavefront widths. Query the backend/device primitive and specialize the reduction.

Shuffle `hi` and `lo` separately, then reconstruct a raw pair and apply the scalar add kernel.

### 16.3 Launch configuration

AMDGPU provides occupancy and launch-configuration facilities. Use them to evaluate workgroup sizes for register-heavy `Double32` kernels. The optimum may differ from CUDA because of:

- wavefront width;
- vector/general-purpose register allocation;
- local data-share usage;
- occupancy limits;
- instruction scheduling.

### 16.4 WMMA and matrix hardware

AMDGPU matrix instructions operate on supported low-precision inputs with `Float32` accumulation; they do not directly accumulate `Double32`.

Possible future use:

- produce a `Float32` high approximation;
- compute a correction with custom `Double32` kernels;
- use iterative refinement.

This is experimental and must not be presented as a direct `Double32` GEMM implementation.

### 16.5 Direct AMD kernels

Apply the same gate as CUDA: KA reference first, measured benefit, isolated extension, equivalent semantics, and user-selectable fallback.

**Outcome: none ship, and neither does a wavefront-shuffle reduction.** The gate
requires *measured* benefit, and this project has had no ROCm hardware. Shipping
unmeasured vendor code would invert the gate rather than satisfy it.

There is a structural argument that they would not help, and it is worth stating
precisely because it is an argument and not a measurement. Phase 3 measured the
exactly analogous CUDA warp-shuffle reduction at **0.48–0.56×** the tree's speed.
The cause is not NVIDIA-specific: a shuffle round makes *every* lane apply the
operator, while a tree round halves the live lanes, so the classic
"shuffle beats shared memory" result is really a claim about *cheap* operators.
A `Double32` addition is about 20 flops. On a 64-lane wavefront the arithmetic
is worse still — a shuffle round costs 64 lanes' worth of work where the CUDA
case cost 32.

That argument justifies not *prioritizing* AMD shuffle work. It does not
substitute for the measurement, and this section should be revisited by someone
with a CDNA or RDNA part in hand.

### 16.6 What Phase 4 could and could not deliver

Recorded because the gap is a property of the tooling and the hardware, not an
oversight.

**Delivered and verified without hardware:**

- the AMD capability matrix — `device`, `gcn_arch`, `wavefrontsize`,
  `max_threads_per_block`, `shared_memory_per_block`, `multiprocessors`;
- `Double32s.wavefront_size(::ROCBackend)`, satisfying section 16.2's "do not
  hardcode 64" by querying `HIP.wavefrontsize` and answering 64 conservatively
  when no device is present;
- `Double32s.launch_config(::ROCBackend)`, reporting the workgroup size, the
  wavefront width, and whether the two divide evenly;
- every no-device path, asserted in `test/extensions.jl`: absent fields are
  `nothing` rather than guessed, the report degrades instead of throwing, and
  `probe_results` still raises a clear `ArgumentError`.

Every AMDGPU API used was checked to exist and to accept the argument types
passed. The **with-device** paths remain unverified and are labelled as such in
the extension's own header.

**Not delivered — `kernel_report(::ROCBackend)` does not exist.** AMDGPU.jl
exposes no per-kernel register or VGPR count; there is no counterpart to
`CUDA.registers`/`CUDA.memory`, and the only route is manual inspection through
`AMDGPU.@device_code_gcn`. The Phase 4 gate's "VGPR/LDS analysis" is therefore
**open**, not closed. A method returning empty rows would have looked like
delivery; a `MethodError` naming the function and the backend does not.

Likewise `launch_config(::ROCBackend).suggested` is `nothing`:
`AMDGPU.launch_configuration` needs a compiled kernel, which needs a device.
Returning a field that is honestly absent beats compiling untested kernel code
inside a diagnostic call that must not throw.

---

## 17. Other GPU packages

### 17.1 `GPUArrays.jl`

Evaluate but do not require directly. CUDA and AMDGPU arrays already participate in the GPU array ecosystem and provide KA backends. Add a direct dependency only if the package implements generic `AbstractGPUArray` operations that cannot be expressed through KA and standard array interfaces.

### 17.2 `Adapt.jl`

Use directly for `SplitDouble32Array` and any launch/configuration wrapper containing arrays. The scalar is already an `isbits` value and needs no custom adaptation.

### 17.3 `StaticArrays.jl`

Do not require it for the scalar or EFTs. Tuples and immutable structs are sufficient. Consider a weak dependency only if small host-side matrices or coefficient tables gain a demonstrated benefit.

Inside KA kernels, follow KA guidance for local/private storage rather than importing StaticArrays solely for scratch memory.

### 17.4 `Atomix.jl`

No initial dependency. A pair-valued atomic add would require a compare-and-swap protocol, careful NaN handling, and a decision about the atomic linearization order. Hierarchical reductions are preferable.

### 17.5 `SpecialFunctions.jl`

Use as a weak host extension only. Device support must be verified per backend and function. It is not part of the initial scalar or GPU kernel contract.

### 17.6 `oneAPI.jl` and `Metal.jl`

The backend-neutral core and KA kernels should avoid assumptions that block later support. Add extensions only after:

- FMA semantics are verified;
- subnormal behavior is characterized;
- generated code is inspected;
- CI hardware is available.

---

## 18. Broadcasting and map operations

Because `Double32` is an `isbits` scalar with device-compatible methods, ordinary broadcast should work on supported GPU arrays:

```julia
z .= x .* y .+ w
```

Be precise about what this does and does not cost. Julia's broadcast machinery
**already fuses** the whole expression into one kernel with one pass over
memory: `z .= x .* y .+ w` reads `x`, `y`, `w` once and writes `z` once, on CPU
and GPU alike. The earlier claim that a custom kernel would "reduce memory
traffic" is wrong; there is no extra traffic to remove.

What the fused broadcast *does* cost is **arithmetic and accuracy**: it evaluates
`*` and then `+` as separate double-word operations, canonicalizing twice, where
the compensated `muladd` of section 8.7 canonicalizes once and retains cross
terms the two-step form discards. That is the real motivation for a fused
kernel, and it is worth stating correctly because it changes the remedy — the
cheapest fix needs no kernel at all:

```julia
z .= muladd.(x, y, w)          # one compensated operation per element
```

Since `Base.muladd(::Double32, ::Double32, ::Double32)` exists, this fuses
through the ordinary broadcast machinery, works on every backend, requires no
package code, and is more accurate than `x .* y .+ w`. Document it prominently;
it is the single highest-value line in the user guide.

Explicit helpers remain useful for policy selection and for launch control.
Name them for what they do, not for the element type (section 10.3), and put the
mutated argument first:

```julia
map!(f, out::AbstractArray{Double32}, args...)         # Base method, ours by eltype
axpy!(out, a::Double32, x, y; policy = DEFAULT_POLICY)
muladd!(out, x, y, z; policy = DEFAULT_POLICY)
```

These should launch KA kernels and permit backend extensions to select native alternatives. The `policy` keyword is resolved to a singleton at this boundary and passed positionally into the kernel (section 12.7).

Avoid scalar iteration on GPU arrays. Tests must enable each backend’s scalar-indexing prohibition — `CUDA.allowscalar(false)` and the AMDGPU equivalent — at the top of the test suite, not per test, so that an accidental `A[i]` in library code fails loudly.

---

## 19. Reductions and linear algebra

### 19.1 Summation

Offer:

```julia
compensated_sum(A; policy = DEFAULT_POLICY, deterministic = false)
```

Modes:

- `A::Array{Double32}`: CPU KA or optimized CPU pairwise path;
- GPU arrays: KA multipass reduction;
- CUDA/AMDGPU native subgroup path when selected and faster.

Do not overload `Base.sum` until behavior, allocations, return type, and backend dispatch are stable. A later release should add `Base.sum(::AbstractArray{Double32})` — which is not piracy, since the element type is ours — delegating to `compensated_sum` with default options. Plan the transition now: `compensated_sum` keeps the keyword surface, `Base.sum` gets the zero-configuration default, and neither becomes a permanent misnamed alias.

### 19.2 Float32 input with Double32 accumulation

A principal use case is accurate reduction of `Float32` data, and Base already
spells it — no package function required:

```julia
sum(Double32, A::AbstractArray{Float32})   # Double32 accumulator, Base machinery
```

Each input converts exactly to `(x, 0)`, and the tree adds in double-word precision. Because the accumulator type comes from `Base.mapreduce`'s promotion of the mapped element type, this is correct with no package code at all beyond `Double32(::Float32)` and `+`.

Optimized backend paths attach by adding methods whose signature mentions
`Double32`:

```julia
# in Double32sCUDAExt — dispatch, not piracy: Type{Double32} is ours
Base.sum(::Type{Double32}, A::CuArray{Float32}) = ...
```

A package-level `compensated_sum(A::AbstractArray{Float32})::Double32` would be
type piracy in all but name — it would not *be* piracy, since the function is
ours, but it duplicates a Base spelling that already works and splits the API
for no benefit. Prefer `sum(Double32, A)` and document it.

### 19.3 Dot product

For `Float32` inputs, form each product exactly when possible:

```julia
p, e = two_prod_(a[i], b[i])
acc  = add(policy, acc, raw(p, e))     # p, e nonoverlapping by construction
```

This is more accurate than rounding the product to `Float32` and then accumulating: the pair `(p, e)` is the *exact* product, so no information is lost before accumulation. The `raw` call is justified because `two_prod_` returns a magnitude-ordered nonoverlapping pair by construction.

Unlike summation, the dot product has no Base spelling that produces this — `dot` is in `LinearAlgebra` and defining it for two `Float32` arrays would be piracy — so it needs a package name:

```julia
compensated_dot(x::AbstractArray{Float32}, y::AbstractArray{Float32})
compensated_dot(x::AbstractArray{Double32}, y::AbstractArray{Double32})
```

For `Double32` inputs, use the selected multiplication kernel and a pairwise reduction. When `LinearAlgebra` is loaded, a `Double32sLinearAlgebraExt` may add `LinearAlgebra.dot(::AbstractArray{Double32}, ::AbstractArray{Double32})` — legitimate, since the element type is ours — delegating to `compensated_dot`.

### 19.4 Norms

Implement after dot product:

```julia
norm2(x) = sqrt(compensated_dot(x, x))
```

Use scaled sum-of-squares for extreme magnitudes to avoid premature overflow and underflow. The scaling factor must be an exact power of two computed from `exponent(maximum(abs, x))`, so that scaling and unscaling introduce no error of their own.

### 19.5 Matrix multiplication

#### Initial KA implementation

A tiled KA kernel may:

- load tiles to local memory;
- use `Double32` register accumulators;
- support `Float32×Float32→Double32` first;
- support `Double32×Double32→Double32` later.

This will not initially match vendor SGEMM throughput.

#### Mixed-precision use

Recommended first matrix use case:

```text
Float32 inputs, Double32 accumulation, Double32 output
```

The exact product pair from `TwoProd` can improve small or cancellation-prone matrices.

#### Vendor library interoperability

Do not assume cuBLAS or rocBLAS supports `Double32`. A hybrid algorithm that uses SGEMM for `hi` and a custom correction is experimental because the SGEMM accumulation error is not an exact high component.

### 19.6 Reproducibility

Reduction order affects finite-precision results.

Provide:

- `deterministic=false`: backend-tuned tree;
- `deterministic=true`: fixed documented KA tree;
- optional `reproducible=:cross_backend` experimental mode with fixed workgroup size and no subgroup specialization.

**Outcome, and one guarantee was traded away.** The first implementation was a
hand-written multipass tree that delivered all three modes, including
cross-backend bit-identity, asserted in CI. The reductions were then moved onto
`AcceleratedKernels` (now a hard dependency) to delete that hand-written driver
and its barrier-scoping hazards.

AK does **not** preserve cross-backend bit-identity: its CPU path is
task-parallel and its GPU path block-parallel, so the trees differ and
`Double32` addition is not associative. Measured on `randn` data and on data
with exponents spread over `2^-40..2^40`, a CPU and a CUDA `ak_sum` differ.

What survives:

| mode | spelling | status |
|---|---|---|
| tuned | `sum(A)` | unchanged |
| deterministic, per backend | `ak_sum(A; block_size = 256)` | holds; asserted by repetition |
| cross-backend | reductions | **withdrawn** |
| cross-backend | `ak_matmul` | holds — still a hand-written kernel with a fixed tile loop |

**A verification trap worth recording.** The first evidence that AK preserved
cross-backend identity was a test on `1/k` data, which passed — and was
vacuous. When every element is a `Float32` with narrowly-spread exponents the
*exact* sum fits in 48 bits, so **every** tree returns the same answer and the
test proves nothing about tree shape. Determinism tests must use data whose
exact sum does not fit, and the regression now does.

---

## 20. Elementary-function strategy

### 20.1 Initial release

Implement only functions with compact residual refinement:

- `sqrt`;
- `inv`;
- integer powers;
- `abs`, `abs2`;
- possibly `hypot` with scaling;
- possibly `cbrt` after validation.

### 20.2 Seed-and-refine functions

Functions that may use a `Float32` backend result as a seed:

- reciprocal;
- square root;
- cube root;
- reciprocal square root;
- selected roots.

A backend intrinsic is a seed, not the final `Double32` result.

### 20.3 Polynomial/rational functions

`exp`, `exp2`, `expm1`, `log`, `log2`, and `log1p` require:

- range reduction in double-word arithmetic;
- coefficients stored as `Double32` pairs;
- carefully bounded polynomial or rational evaluation;
- explicit handling near cancellation points;
- backend-independent reference tests.

Implement `log1p` and `expm1` before using `log(1+x)` or `exp(x)-1` near zero.

**Implemented in Phase 6.** Three findings from doing it are worth keeping,
because each was a wrong first implementation that measured as wrong rather than
looking wrong:

1. **`exp10` must reduce in `x`, not hand `x·ln(10)` to `exp`.** `exp` converts
   an *absolute* argument error into a *relative* result error, and `x·ln(10)`
   has absolute error growing with `x`. Measured **56.1 ulp48** at `x ≈ 29.4`
   before reduction was moved into `x` itself; **0.868** after. Any `exp`-like
   function reduced through a multiplication has this bug latent in it.
2. **`log1p` must not use `2·atanh(x/(2+x))` for small `x`.** That form needs a
   division, and this package's `/` carries 4.00 ulp48 which passes undiminished
   into the answer — so `log1p` was division-limited in exactly the region it
   exists to serve. Measured **4.06 ulp48**; a direct series in `x` gives
   **1.35**.
3. **`log2` and `log10` must keep the exponent as its own term.** `log(x)/LN2`
   folds the exact integer `k·ln(2)` into the logarithm and then divides the
   sum, spending division's error on a term that was exact. Measured **1.70**
   that way against **0.80** with `k` kept separate.

`expm1` also gained the doubling identity `expm1(2u) = expm1(u)·(expm1(u)+2)`
for arguments just past the series band, where the reduced path would compute
`2·(1 + expm1(r)) - 1` and lose to cancellation. *(Replaced in the 20.6
rebuild: the ladder itself measured 4.63 ulp48 worst, and a series band
widened to `ln 2` plus direct subtraction past it measures 1.66.)*

### 20.4 Trigonometric functions

Defer until the package has:

- a high-quality double-word π split;
- robust argument reduction;
- bounded reduced intervals;
- tested polynomial kernels;
- special handling near roots;
- large-argument policy.

Using CUDA or AMDGPU `Float32` trig as the final result would not meet the package precision goal.

**Implemented in Phase 6, with the large-argument policy as a stated limit.**
Each prerequisite above is met: `π/2` is split three ways (`PIO2_C1`, `PIO2_C2`
each 8 significand bits so `q·PIO2_Ci` is exact in `Float32`, remainder a full
`Double32`), reduction is Cody–Waite to `|r| <= π/4`, and the kernels are
Horner series in `r²` with coefficients generated from 400-bit `BigFloat`.

The large-argument policy is `Double32s.TRIG_ARG_LIMIT = 2^20`, **beyond which
`sin`, `cos`, `tan`, and `sincos` return `NaN`.** This was not the first
choice. The first version simply reduced whatever it was given and documented
that accuracy degraded — and then `sin(2^30)` returned **9.88e8**, a number
outside `[-1, 1]` with nothing to indicate anything had gone wrong. A silently
wrong answer is the worst outcome available to a numerical routine, so the
policy is now to fail loudly: a caller can test for `NaN` and cannot test for
"this plausible number is fictional". Correct reduction past the limit needs
Payne–Hanek with a multi-thousand-bit `2/π`, a table and a loop that do not
belong in a GPU kernel; it is recorded as open rather than approximated.

Measured worst-case errors are in `docs/Reports/accuracy_report.md`: after the
20.6 rebuild, `sin` 1.302, `cos` 1.114, `atan` 1.093, `asin` 1.951, `acos`
2.376, `tan` 2.010 nominal-48-bit ulps on the regression pools. `tan`, `asin`,
and `acos` sit highest because each ends in a division, and this package's `/`
is measured at 4.00 ulp48 — that is their floor, not a defect in the
trigonometry.

### 20.5 Backend-specific intrinsics

CUDA/AMDGPU intrinsics may be used for:

- seeds;
- exponent extraction or scaling;
- subgroup communication;
- measured faster primitives with equivalent semantics.

They must not create backend-dependent public accuracy silently.

### 20.6 The 2026-07-29 rebuild: tables, mixed precision, and division placement

Every function in 20.3 and 20.4 (and the hyperbolic family) was rebuilt for
throughput once the whole surface existed and could be measured together. The
first implementations were correct-by-construction and deliberately
conservative: every Horner step a full policy multiply-plus-add, every
reduction the textbook one. Profiling showed the cost was almost entirely the
*serial dependency chain* of 10–26 pair Horner steps, plus a 4.00-ulp48
division wherever a reduction used one. Four changes, each measured before it
was kept (`src/math/horner.jl` carries the primitives and their contracts):

1. **Dominant-add Horner step** (`poly_step`): exact leading product, cross
   terms in two fmas, FastTwoSum against the coefficient — valid because every
   kernel's coefficient dominates its product term, and each call site states
   the ratio. ~36 cycles against ~60 for the policy pair.
2. **Mixed-precision tails**: once the remaining Horner tail is below `2^-26`
   of the result, it is evaluated in plain `Float32` muladds (error under
   `2^-49` relative). Every kernel splits at exactly that boundary.
3. **Table reductions.** `exp`: `2^(j/32)` (32 pair entries) leaves
   `|r| <= ln(2)/64`, a three-step kernel; the inverse table gives `e^{-x}`
   from the same reduction with **no division**, which is what rebuilt `sinh`
   and `cosh` at 7×. `log`: 23 bins of `(c = fl32(32/i), -log c)` replace the
   `(m-1)/(m+1)` division entirely; `r = m·c - 1` is computed *exactly* (the
   sloppy one-fma fold is a derived ~60 ulp48 defect — see `mulm1`).
4. **Division placement.** `atan` re-binned to `{0, π/6, π/3, π/2}` rotations:
   at most one division anywhere (the old reduction paid two on
   `(1, 3.73)`). `tanh` past 1 uses `1 - 2/(E+1)` so the division produces a
   small correction instead of the result. `atanh` mid-band became
   `(log1p(b) - log1p(-b))/2`: no division at all, 3.06 → 1.81 ulp48.

Measured, same 4096-point pools and harness before/after (ns/call):
exp 172.7→43.8, expm1 173.6→48.5, exp2 167.8→36.9, exp10 183.4→46.7,
log 145.0→41.0, log1p 158.7→59.3, log2 146.4→40.3, log10 147.2→42.8,
sin 194.9→39.2, cos 195.0→40.1, tan 211.0→69.4, atan 172.8→60.0,
asin 227.1→103.4, acos 238.1→117.8, sinh 345.4→49.5, cosh 350.4→49.0,
tanh 121.5→45.6, asinh 218.6→81.6, acosh 230.7→74.6, atanh 168.9→86.6.
No accuracy ceiling in `test/core/elementary.jl` regressed; the ones that
moved, moved down (expm1 4.63→1.66, tanh 3.50→1.52, atanh 2.42→1.30, and
`log` near 1 — where the old division error was amplified — 4.66→1.59 on a
dense sweep). `sin`/`cos` additionally dispatch on quadrant parity and
evaluate one kernel instead of both; `sincos` is asserted bitwise-equal to
`(sin, cos)`.

One trade is recorded rather than hidden: the tables are indexed dynamically,
which on a GPU compiles (tuple `getfield`) but may spill to local memory the
way `MATMUL_BLOCK`'s scratchpads did. Device throughput of the elementary
functions has not been re-measured since the rebuild; the host↔device
bit-identity assertions in `test/extensions.jl` cover arithmetic, not these
functions, and no cross-backend claim is made for them.

---

## 21. Performance strategy

### 21.1 Performance hypotheses

Compared with `Float32`:

- storage and bandwidth are approximately doubled;
- each live `Double32` consumes at least two scalar registers;
- addition uses roughly an order of magnitude more scalar operations;
- multiplication uses several multiplies plus at least one FMA and renormalization;
- division and square root require a hardware operation plus compensated residual work;
- reductions are sensitive to register pressure and synchronization cost.

These are hypotheses to measure, not promises.

### 21.2 Benchmark categories

#### CPU scalar

Measure:

- constructor/renormalize;
- add, subtract, multiply, divide;
- reciprocal, square root, muladd;
- comparisons and conversions.

Compare:

- `Float32`;
- `Float64`;
- `DoubleFloats.Double32`;
- proposed accurate and fast kernels.

#### KA CPU backend

Use identical KA kernels to establish abstraction overhead and correctness.

#### GPU elementwise

Measure large arrays for:

- AoS and SoA;
- add/multiply/muladd;
- conversion;
- memory-bound and compute-bound variants.

#### Reductions

Measure:

- KA local-memory tree;
- grid-stride variants;
- CUDA warp shuffle;
- AMD wavefront shuffle;
- deterministic and tuned modes;
- Float32 input/Double32 accumulator;
- Double32 input/Double32 accumulator.

#### Dot and matrix kernels

Separate product formation, accumulation, and memory costs.

### 21.3 Timing rules

- warm up compilation;
- synchronize before and after timed regions;
- exclude host-device transfer unless explicitly measuring end-to-end behavior;
- prevent constant folding and dead-code elimination;
- use large enough arrays;
- report median and distribution, not only minimum;
- record device model, driver, Julia version, package versions, compiler flags, and arithmetic policy;
- measure generated register use and occupancy;
- compare equal numerical tasks, not merely equal source expressions.

### 21.4 Generated-code inspection

CUDA CI/benchmark jobs should inspect PTX/SASS-oriented summaries for:

- use of `fma.rn.f32`;
- absence of unintended `f64`;
- absence of local-memory spills;
- register count;
- branch structure;
- vectorized or paired loads.

AMDGPU jobs should inspect LLVM/GCN output for analogous properties, especially FMA, VGPR use, LDS use, and spills.

### 21.5 Acceptance gates

No native backend path should become default until it:

1. passes the same validation suite as the KA path;
2. is faster on at least two representative devices or a clearly defined device family;
3. has no unacceptable occupancy or spill regression;
4. preserves deterministic mode where promised;
5. can be disabled.

---

## 22. Validation methodology

### 22.1 Host reference arithmetic

Use:

- `BigFloat` with at least 256-bit precision;
- `Arblib.jl` or `ArbNumerics.jl` for interval-backed checks;
- exact `Rational{BigInt}` or integer arithmetic for selected EFT identities;
- `Float64` **only** as a fast reference for the subset of canonical pairs on
  which its conversion is provably exact.

That last bullet replaces an earlier claim that `Float64` exactly represents any
finite `Double32` pair sum. It does not (section 4.2). Outside the exact subset
a `Float64` reference silently agrees with a wrong answer, which is the worst
behavior a reference can have.

**The first attempt at encoding the condition was itself wrong**, in both
directions, and is recorded here because a wrong predicate here is more
dangerous than no predicate at all:

```julia
# WRONG — accepts 31,030 inexact pairs out of a 1.7M structured sweep,
#         and rejects exact ones at hi = 2f0^-96 with a negative subnormal lo.
float64_reference_is_exact(x) =
    iszero(x.lo) || !issubnormal(x.lo) || exponent(x.hi) <= -97
```

A *normal* low word is not sufficient for exactness — canonicality bounds
`exponent(lo)` from above only, so `(2f0^-62, floatmin(Float32))` is a canonical
pair with a normal low word spanning 65 bits. Use the span condition of section
4.2:

```julia
"""
    float64_reference_is_exact(x::Double32) -> Bool

Whether `Float64(x)` reproduces the exact pair sum, and therefore whether a
`Float64` reference may be used for `x`. True iff the pair spans at most 53
bits. See section 4.2 for the proof.
"""
function float64_reference_is_exact(x::Double32)
    iszero(x.lo) && return true
    isfinite(x.hi) || return true          # canonical non-finite pairs have lo == 0
    eh = exponent(x.hi)
    # exponent(hi + lo) is eh, one less exactly when hi is a power of two and
    # lo opposes its sign; |lo| <= half-ulp bounds cancellation to one binade.
    borrow = (abs(x.hi) == ldexp(1.0f0, eh)) & (signbit(x.lo) != signbit(x.hi))
    return lsb_exponent(x.lo) >= (eh - borrow) - 52
end

reference(x::Double32) = big(x.hi) + big(x.lo)   # always exact; use by default
```

Validated against a 400-bit `BigFloat` reference over **1,701,602 canonical
pairs — 0 mismatches**. Dropping the `borrow` term gives a conservative variant
that can under-report exactness but never over-report it, which is the safe
direction for a reference gate.

Default to the `BigFloat` reference and use the `Float64` fast path only where
the predicate holds, asserting the predicate rather than assuming it. The
generator in section 22.5 must deliberately produce pairs on both sides of the
boundary, including wide-gap pairs with *normal* low words — the class the wrong
predicate above waved through.

### 22.2 Reference rounding model

For operation accuracy, define a software reference that rounds the exact result to a documented nominal 48-bit significand with `Float32` exponent limits. This reference is separate from the broader set of all canonical pair sums.

Also validate pair reconstruction directly:

```text
V(result) versus exact real operation.
```

### 22.3 Property tests

Required properties include:

```text
V(canonicalize(hi,lo)) = hi + lo, when finite and representable
abs(lo(result)) ≤ ½ ulp(hi(result))
Float64(Double32(x::Float32)) = Float64(x)
x + 0 = x
x - x = signed zero, except NaN/Inf cases
x * 1 = x
x / 1 = x
sqrt(x)^2 ≈ x for x ≥ 0
inv(x) * x ≈ 1 for finite nonzero x
comparison is consistent with the exact BigFloat pair value
hash(x) = hash(Float64(x)) for finite x
```

Order-theoretic properties belong here as well. The first group catches the
*first* `isless` defect described in section 8.10:

```text
for all non-NaN x, y:  exactly one of isless(x,y), isless(y,x), isequal(x,y)
for all x, y:          isequal(x,y) implies hash(x) == hash(y)
for all x, y:          isequal(x,y) implies !isless(x,y) && !isless(y,x)
sort! agrees with sorting the exact BigFloat values
```

**That group is necessary and demonstrably not sufficient.** The second defect
of section 8.10 — `isequal(-one(Double32), Double32(-1.0f0)) == false` — passes
every line above: trichotomy still holds, the hash implication still holds, and
`sort!` still agrees with `BigFloat` up to ties it is free to break either way.
The properties that do pin it down compare against the *represented value* and
against Base's behavior on `Float32`, not against internal consistency:

```text
for all x, y:  isequal(x,y) == (reference(x) === reference(y) as signed reals)
for all x, y:  isequal(x,y) == (x == y) whenever neither is NaN and neither is zero
for all x, y:  x == y implies !isless(x,y) && !isless(y,x)
for all x:     isequal(x, -(-x))                      # sign round-trip
for all x, y:  isequal(x,y) implies Dict(x=>1)[y] == 1
NaN pairs:     isequal(D32_NAN, -D32_NAN)             # matches isequal(NaN32, -NaN32)
```

Generate the operands for these from the canonical-pair generator including its
signed-zero and subnormal-low-word cases, not from `rand(Double32)` alone —
uniform sampling never produces `(0.0f0, -0.0f0)`, and it never produces a
value and its double negation as two distinct bit patterns.

Do not require algebraic identities that finite-precision arithmetic does not guarantee, such as associativity.

### 22.4 Metamorphic tests

- scale operands by powers of two and compare scaled results;
- negate both operands and compare sign transformations;
- swap multiplication operands;
- compare division with multiplication by reciprocal;
- compare KA CPU and scalar CPU;
- compare KA GPU and scalar reference;
- compare native CUDA/AMD path with KA on identical reduction trees;
- permute reduction inputs to characterize, not forbid, order sensitivity.

### 22.5 Random testing

Use distributions emphasizing:

- uniform bit patterns;
- logarithmically distributed exponents;
- clustered values near powers of two;
- low-word signs both equal and opposite to high;
- cancellation;
- subnormal components;
- values near overflow;
- exact halfway residuals;
- random canonical and intentionally noncanonical input pairs for constructor tests.

### 22.6 Structured exhaustive subsets

Full `Float32` pair enumeration is impossible. Exhaustively test:

- all `Float16` values widened to `Float32`;
- all pairs from selected exponent/fraction windows;
- every low word around zero, minimum subnormal, maximum subnormal, and minimum normal for selected high words;
- all classification combinations;
- all one-bit-neighbor divisors around powers of two;
- reduced mantissa models that preserve algorithm structure.

### 22.7 Device validation workflow

1. generate test operands on the host;
2. compute high-precision references in batches;
3. transfer operands once;
4. execute a KA or native test kernel;
5. transfer result arrays once;
6. classify errors on the host;
7. retain the worst cases and exact bit patterns;
8. replay worst cases in dedicated unit tests.

---

## 23. Difficult and adversarial cases

Known `Float32` bit patterns:

| Value | Bits |
|---|---:|
| `+0.0f0` | `0x00000000` |
| `-0.0f0` | `0x80000000` |
| minimum subnormal | `0x00000001` |
| maximum subnormal | `0x007fffff` |
| minimum normal | `0x00800000` |
| `prevfloat(1.0f0)` | `0x3f7fffff` |
| `1.0f0` | `0x3f800000` |
| `nextfloat(1.0f0)` | `0x3f800001` |
| maximum finite | `0x7f7fffff` |
| `+Inf32` | `0x7f800000` |
| canonical quiet NaN candidate | `0x7fc00000` |

### 23.1 Addition with cancellation

- `(1 + 2⁻²³) + (-1)`;
- two `Double32` values whose high words cancel and whose low words determine the result;
- high words equal and opposite, low words opposite in sign but unequal by one subnormal;
- cancellation yielding a subnormal result;
- cancellation yielding signed zero.

### 23.2 Widely separated magnitudes

- `floatmax(Float32) + minsubnormal(Float32)`;
- `1.0f0 + 2⁻¹⁴⁹`;
- a large high word plus a low pair that is below its `Float32` ulp but representable as `lo`;
- repeated accumulation of tiny terms into a large partial.

### 23.3 Halfway cases

- `1.0f0 + 2⁻²⁴`;
- high-word sums exactly halfway between adjacent `Float32` values;
- low-word renormalization at a tie where the high significand is even or odd.

### 23.4 Multiplication near overflow

- operands immediately below and above `sqrt(floatmax(Float32))`;
- high product rounding to maximum finite with positive cross terms;
- high product overflowing while negative cross terms may reduce the exact pair product;
- maximum finite multiplied by values around `1.0f0`.

### 23.5 Multiplication near underflow

- `floatmin(Float32) * 0.5f0`;
- minimum normal times `prevfloat(1.0f0)`;
- minimum subnormal times `2.0f0`;
- products whose high word is subnormal and residual is below minimum subnormal;
- FTZ-sensitive cases.

### 23.6 Division near powers of two

Use divisors:

```text
reinterpret(Float32, 0x3f7fffff)
reinterpret(Float32, 0x3f800000)
reinterpret(Float32, 0x3f800001)
```

and analogous values around `0.5`, `2`, minimum normal, and maximum finite.

### 23.7 Reciprocal refinement

- `3.0f0` with zero low word;
- divisors with a low word that almost cancels the high-word reciprocal residual;
- extreme exponents;
- divisors adjacent to powers of two;
- negative zero and infinities.

### 23.8 Square-root refinement

- perfect squares represented exactly;
- values one pair step above and below a perfect square;
- minimum normal and minimum subnormal;
- maximum finite;
- values whose seed square overflows without scaling;
- negative finite values and negative zero.

### 23.9 Compensated FMA cancellation

- `x*y + z` where `z` is the negative rounded high product;
- cancellation leaving only cross terms;
- cancellation leaving only `x.lo*y.lo`;
- product overflow followed by cancellation;
- result near zero with large intermediate terms.

### 23.10 Normalization

- `hi` and `lo` same sign at the half-ulp boundary;
- opposite signs causing the high word to step down;
- noncanonical pairs with `|lo| > |hi|`;
- zero high and nonzero low;
- infinite high and finite/nonfinite low;
- NaN payloads.

### 23.11 Wide-gap pairs: low words far below their permitted bound

This class is the one both the original `Float64`-exactness claim and its first
correction overlooked (section 4.2), and it is invisible to uniform random
sampling because it requires a low word many orders of magnitude below its
permitted bound. It has **two** subclasses, and a generator that produces only
the first will reproduce the second mistake:

1. **subnormal low words** under a not-tiny high word — bits pinned at `2^-149`;
2. **normal low words far below `½ulp(hi)`** — an arbitrarily wide gap of
   implicit zeros between the two words, with no subnormal in sight.

Construct pairs directly, covering both:

```julia
lo_min = reinterpret(Float32, 0x00000001)          # 2^-149
his    = (1.0f0, prevfloat(1.0f0), 2f0^-62, 2f0^-96, 2f0^-97, 2f0^-100,
          floatmin(Float32), floatmax(Float32))

# subclass 1: subnormal low words
sub_pairs  = [two_sum(hi, s*lo) for hi in his, s in (1.0f0, -1.0f0),
                                    lo in (lo_min, 3lo_min,
                                           prevfloat(floatmin(Float32)))]

# subclass 2: NORMAL low words at every gap width the invariant allows
gap_pairs  = [two_sum(hi, s*ldexp(1.0f0, exponent(hi) - 24 - k))
              for hi in his, s in (1.0f0, -1.0f0), k in 0:120]
```

and test, for each:

- `Float64(x)` versus `big(x.hi) + big(x.lo)` — must disagree exactly on the
  subset predicted by `float64_reference_is_exact` (section 22.1) and agree
  everywhere else, confirming the span condition in both directions. The three
  pairs `(2f0^-96, ±2f0^-149)` and `(2f0^-97, 2f0^-149)` must be present, since
  they are where the cancellation `borrow` term is load-bearing;
- `x == Float64(x)` — must be `false` where the conversion is inexact, which is
  what the exact mixed comparison of section 8.10 delivers and promotion does
  not;
- arithmetic against these operands, referenced against `BigFloat`;
- round-trip through `SplitDouble32Array` and through a KA kernel, since a
  backend that flushes subnormals will destroy the low word of subclass 1 and
  must be detected, not tolerated. Note that subclass 2 survives flush-to-zero
  and therefore tests a different thing.

### 23.12 Signed zeros in the low word

Reachable only through the sign operations, and never through the constructors,
so a generator based on `rand` or on `canonicalize` will not produce them.

An earlier draft listed five expressions here and asserted they were "the entire
failure surface". **They are not, on two counts.** The five expressions collapse
to three distinct pairs, and — the substantive error — the surface is not
confined to zero high words at all, because `-` flips the low word's sign bit
whatever the high word is.

Enumerate the surface by construction rather than by example. It has two parts:

```julia
# Part 1 — all four zero pairs. hi == ±0 forces iszero(lo), so this is complete.
zero_pairs = [raw(h, l) for h in (0.0f0, -0.0f0), l in (0.0f0, -0.0f0)]

# reachability, for the record:
#   ( 0.0f0,  0.0f0)  zero(Double32)
#   (-0.0f0,  0.0f0)  Double32(-0.0f0), -abs(Double32(-0.0f0))
#   ( 0.0f0, -0.0f0)  abs(Double32(-0.0f0))
#   (-0.0f0, -0.0f0)  -zero(Double32), flipsign(zero, -1), copysign(zero, -1)

# Part 2 — BOTH low-word zero signs under every nonzero high word of interest.
# This is the part the earlier list missed entirely.
lo_zero_pairs = [raw(h, l) for h in (1.0f0, -1.0f0, 1.5f0, -1.5f0,
                                     Inf32, -Inf32, NaN32, -NaN32),
                               l in (0.0f0, -0.0f0)]
```

Part 2 is where the second `isless` defect lived: `-one(Double32)` is
`(-1.0f0, -0.0f0)` and `Double32(-1.0f0)` is `(-1.0f0, +0.0f0)`, and the
guarded-on-`hi` predicates of the previous draft reported them as not `isequal`
while `==` and `hash` reported them as identical. Over the 576-pair product of
this enumeration the previous predicates showed 0 trichotomy violations and 20
semantic mismatches — which is precisely why the trichotomy property alone is
not the test to rely on here.

For every pair drawn from the full set, assert both groups of order-theoretic
properties in section 22.3 and the predicate rule of section 8.10. Enumerating
`zero_pairs × lo_zero_pairs` exhaustively costs nothing and closes the surface.

---

## 24. Implementation roadmap

### Phase 0: Naming and numerical contract

#### Entry criteria

- agreement on package/module/type naming;
- agreement on strict versus fast semantics;
- agreement that the type is double-word arithmetic, not a new IEEE format.

#### Deliverables

- short specification of invariants;
- exact special-value policy;
- promotion table;
- nominal precision and lattice caveat;
- backend capability-test plan.

#### Principal risks

- API incompatibility caused by the package/type name conflict;
- misleading `AbstractFloat` expectations.

### Phase 1: Core scalar arithmetic

#### Deliverables

- type and constructors;
- canonical special values;
- EFT primitives;
- canonicalization;
- `+`, `-`, `*`, `/`, `inv`, `sqrt`;
- comparisons and predicates;
- conversions and promotion;
- host tests;
- generated LLVM checks;
- the quality test suite of section 12.11, wired into CI.

#### Validation gate

Numerical:

- invariant tests pass;
- high-precision random and adversarial tests pass;
- no accidental `Float64` in the `Float32` arithmetic core.

Julia engineering (section 1.2), all mechanically checked:

- `Test.@inferred` passes for every public scalar operation;
- `@allocated == 0` for every scalar operation, and `AllocCheck` proves it for
  the `+`, `-`, `*`, `/`, `sqrt`, `fma` call trees;
- `JET.@test_opt` reports no runtime dispatch in the scalar core;
- `Aqua.test_all` passes, including piracy, ambiguity, and compat checks;
- no `@fastmath` anywhere in `src/`, asserted by test;
- `@code_llvm` for `+` and `*` shows no `alloca`, no `double`, and **no call
  except the single `@noinline` cold guard** — see the amendment below;
- `@code_llvm` for every operation documented as device-safe (`+`, `-`, `*`,
  `/`, `inv`, `fma`, `muladd`, `sqrt_or_nan`, `<`, `isless`, `probe_lane`)
  contains **no `throw` reference at all**;
- doctests pass.

These gates cost about a day to set up, and they are what make the later phases'
"semantic parity" claims checkable rather than asserted. They are implemented in
`test/gates.jl`.

**Amendment — "no calls to package functions" was too strong.** As first
written this gate would have failed the design's own architecture: sections 7.5
and 8 deliberately route each guard through a `@noinline` call precisely so the
hot path stays branch-free and register-light. `+` therefore calls
`add_nonfinite` and `*` calls `mul_guarded`, and satisfying the literal gate
would mean inlining the cold path into the hot one — the opposite of what is
wanted. The gate now asserts what was actually intended: every call is either a
hardware intrinsic (`@llvm.*`) or *the* one documented cold guard.

**The `throw` clause is new, and it caught real defects.** `Base.exponent`
raises `DomainError` for `0`, `Inf`, and `NaN`, and `Base.sqrt` for negatives.
Every call site in the package excludes those first, but the compiler cannot
prove it, so the exception path survived into the generated code — including in
`sqrt_or_nan`, the operation whose entire reason for existing is to keep
`throw` out of kernels. `Double32s.raw_exponent` and `Double32s.sqrt_nonneg`
replace them. Host cost: none (A/B measured at 0.98× and 0.93×, inside the
noise). Device benefit, on an RTX 5090: `/` drops from 35 to 33 registers and
96 to 88 bytes of local memory; `sqrt_or_nan` from 30 to 26 registers and 56 to
32 bytes. Local memory is spill space, so the 43% cut on `sqrt_or_nan` is the
number that matters — and some backends decline to compile a kernel containing
exception machinery at all.

**Still open from this gate:** `AllocCheck`, `JET.@test_opt`, and
`Aqua.test_all` are not wired in. `test/gates.jl` covers their intent with
cheaper stand-ins — `@allocated == 0` and `@inferred`/`Base.return_types` for
the scalar core, `Test.detect_ambiguities` in `test/core/invariants.jl` — but
these are stand-ins, not the tools, and adding the three packages remains
outstanding work rather than a closed item.

### Phase 2: KA portability

#### Deliverables

- KA elementwise kernels;
- AoS and SoA conversion kernels;
- KA CPU tests;
- GPU broadcast tests;
- KA reduction and dot-product baseline;
- device capability tests.

#### Validation gate

- identical or acceptably bounded results between scalar and KA CPU;
- CUDA and AMDGPU KA paths compile when hardware CI is available;
- no scalar GPU indexing;
- convergent barrier review completed.

### Phase 3: CUDA optimization

#### Deliverables

- CUDA extension;
- launch policy;
- warp-shuffle reduction;
- occupancy and register reports;
- optional direct CUDA reduction/dot kernels;
- CUDA capability matrix.

#### Validation gate

- semantic parity with KA;
- measured speedup;
- no spills or occupancy collapse on target devices;
- deterministic fallback available.

### Phase 4: AMDGPU optimization

#### Deliverables

- AMDGPU extension;
- wavefront-aware reduction;
- occupancy-based workgroup policy;
- optional direct ROCm kernels;
- AMD capability matrix.

#### Validation gate

- same as CUDA, with wavefront width and VGPR/LDS analysis.

### Phase 5: Reductions and linear algebra

#### Deliverables

- production `compensated_sum`, `compensated_dot`, and norm;
- deterministic and tuned modes;
- Float32-input exact-product dot;
- first tiled KA matrix kernel;
- optimize matrix multiplication
- provide additional matrix functions that have GPU support as performantly as possible
- comprehensive performance report.

### Phase 6: Elementary functions

#### Deliverables

- build functions from single precision versions wherever that is more performant
- `cbrt`, `hypot`, and selected roots;
- `exp`/`log` family;
- trigonometric range-reduction design;
- trigonometric function family and inverse trigonometric function family
- hyperbolic trigonometric function family and inverse hyperbolic trigonometric function family
- error function family
- gamma function family
- coefficient-generation scripts;
- backend seed experiments;
- function-specific error reports.

---

## 25. Design decisions and tradeoffs

### 25.1 Package name versus scalar name

**Alternatives**

1. module `Double32`, scalar `Float32x2`;
2. module `Double32s`, scalar `Double32`.

**Recommendation:** option 2.

**Rationale:** the intended scalar spelling is central to usability, and Julia cannot bind a module and type to the same root name.

### 25.2 Concrete versus parametric type

**Alternatives**

- `DoubleFloat{T}`;
- concrete `Double32`.

**Recommendation:** concrete.

**Rationale:** smaller method surface, clearer GPU contract, simpler generated-code tests, and no unsupported instantiations.

### 25.3 Strict versus relaxed normalization

**Alternatives**

- normalize every intermediate;
- carry weak pairs internally and normalize at operation boundaries.

**Recommendation:** strong public invariant, weak internal intermediates only within audited kernels.

### 25.4 Mandatory FMA versus splitting fallback

**Alternatives**

- require FMA;
- implement Dekker splitting when FMA is unavailable.

**Recommendation:** require correct `Float32` FMA on supported GPU paths; provide a host-only splitting fallback only if needed for unusual CPUs.

**Rationale:** modern target GPUs provide efficient FMA, and splitting adds operations, constants, overflow concerns, and register pressure.

### 25.5 Branch-free versus guarded kernels

**Recommendation:** one exceptional check in the public operation, branch-free finite EFTs, and a guarded fallback on non-finite output.

This matches the useful pattern observed in `DoubleFloats.jl` while reducing per-EFT divergence.

### 25.6 KA versus native backend code

**Recommendation:** KA is normative and portable. CUDA/AMDGPU native code is an optional optimization layer.

No native implementation exists without a validated KA counterpart.

### 25.7 AoS versus SoA

**Recommendation:** AoS is the default public array representation; SoA is an optional performance and interoperability representation.

### 25.8 Full `AbstractFloat` integration versus small API

**Recommendation:** subtype `AbstractFloat` for Julia interoperability, but implement a deliberately staged API and document the nonstandard lattice. Do not fake `nextfloat` or complete parsing.

An alternative is subtype `Real` initially, but that would block useful generic floating-point code and diverge from the established double-word ecosystem.

The staging rule that follows from section 11.2 is worth stating as a principle,
since it will come up for every deferred method: **implement the methods generic
code calls speculatively; defer the methods callers seek out deliberately.**
`eps`, `floatmin`, `floatmax`, `precision`, `typemin`, `typemax`, `zero`, `one`,
and `isless` are in the first group — omitting them turns working generic
algorithms into `MethodError`s and pushes users away from the type. `nextfloat`,
`prevfloat`, `parse`, `round`-to-digits, and the `rem`/`mod` family are in the
second — their absence produces a clear error at a call site whose author knows
exactly what they wanted, which is informative rather than harmful.

### 25.9 Exported surface versus discoverability

**Alternatives**

- export the accessors and reduction functions, as most numeric packages do;
- export only `Double32` and mark the rest `public`.

**Recommendation:** the second.

**Rationale:** `hi`, `lo`, and `normalize` are names users and other packages
already occupy, and Julia 1.11's `public` keyword provides documentation and
tooling visibility without namespace cost. This was not available when most
existing double-word packages were designed; the package should use it.

### 25.9 Reproducibility versus throughput

**Recommendation:** expose both deterministic KA trees and backend-tuned trees. Do not claim that the tuned result is bitwise stable across hardware.

### 25.10 Strict semantics versus native fast math

**Recommendation:** strict semantics are the default and define `+`, `*`, `/`, and `sqrt`. Fast behavior requires an explicit policy or named function.

---

## 26. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| package/type naming collision | impossible intended root API | use module `Double32s` |
| compiler reassociation | invalid EFT residuals | no `@fastmath`; generated-code tests |
| unexpected FMA contraction or decomposition | wrong residual | explicit `fma`; inspect PTX/GCN |
| subnormal flushing | degraded precision and failed exactness | capability tests; qualified guarantees |
| excess register pressure | low occupancy and spills | separate accurate/fast kernels; tune workgroups |
| shared-memory pair layout inefficiency | slow reductions | benchmark `Double32` scratch versus split `Float32` scratch |
| divergent barriers | invalid KA kernels | convergent reduction structure; review `unsafe_indices` |
| promotion ambiguity | precision loss or method errors | explicit promotion table and cross-type tests |
| accidental `Float64` device operations | severe slowdown | generated-code checks; isolate host conversions |
| noncanonical raw pairs | incorrect comparisons/predicates | private raw constructor; invariant tests |
| misleading `eps`/`nextfloat` | incorrect generic algorithm assumptions | documented nominal `eps`; defer adjacency API |
| vendor path drift | backend-specific bugs | KA reference path and shared tests |
| atomics temptation | nondeterminism and complexity | multipass reductions |
| compile latency | poor interactive use | concrete type, small core, extensions, precompile representative kernels |
| native subgroup width assumptions | incorrect AMD/CUDA reductions | query and specialize width |
| elementary-function overreach | inaccurate results | staged implementation and function-specific validation |
| matrix-kernel underperformance | poor user expectations | label initial matrix support experimental; publish comparisons |
| `muladd` used where `fma` is required | silently wrong residual on backends that decline to fuse | grep test for `muladd` in `src/core/`; PTX/GCN inspection |
| type instability from a guard path | union return, dispatch in kernels, device compile failure | `@inferred` on every public operation |
| exported name collision (`hi`, `lo`, `normalize`) | breaks user code on `using` | export only `Double32`; `public` for the rest |
| type piracy in reduction API | load-order-dependent behavior for unrelated code | `sum(Double32, A)` spelling; `Aqua.test_piracies` |
| keyword flags reaching kernels | lost specialization, silent slowdown | resolve to singletons/`Val` at the launch boundary |
| unchecked `two_hilo_sum_` precondition | silently wrong low word | true condition is residual representability, not `abs(a) ≥ abs(b)` (section 7.2); differential test against `two_sum_`; `two_hilo_sum_checked` in tests |
| assuming `Float64` represents every canonical pair exactly | wrong promotion rationale, wrong mixed comparison, reference that agrees with a wrong answer | span-based exactness predicate of section 22.1; exact mixed comparison; `BigFloat` default reference |
| **a correction that is itself wrong** | the defect is marked closed and the wrong claim now carries an argument behind it | every numerical claim in this document must be reproduced by an independent computation and the operand count reported; sections 4.2, 8.10, 8.11, 22.1 record the measurements |
| assuming a *sufficient* condition is the *necessary* one | over-strict predicate rejects valid cases; under-strict justification hides real ones | state which direction is proved; `two_hilo_sum_` and `float64_reference_is_exact` are the two live examples |
| invariant relaxed in one section, consumed unguarded in another | broken total order, silently wrong `sort!` | state relaxations as rules (section 8.10), not as local patches; order-theoretic property tests |
| relaxation scoped narrower than the operations that cause it | `isequal` false for equal values; `Dict` misses; `unique` keeps duplicates | invariant 8 of section 5.2 covers *every* high word, because `-` flips `lo`'s sign bit unconditionally |
| `min`/`max` written from `<` alone | NaN silently swallowed in reductions; signed zero unordered | mirror `Base.min`/`Base.max` on `Float32` exhaustively over `±0`, `±Inf`, `NaN` (section 8.11) |
| overflow rescued by cross terms at the multiplication boundary | finite pair result reported as `Inf` | guarded path must recompute scaled, not merely canonicalize (sections 7.5, 8.0) |
| `two_prod_` residual underflow | silent accuracy loss that `isfinite` cannot detect | subnormal capability tests; qualified accuracy claims |
| retained bounds checks in kernels | exception machinery on device, compile failure | `@inbounds` plus `@propagate_inbounds` on wrapper indexing |
| missing `[compat]` for a weak dependency | unresolvable or wrongly resolved extension | `Aqua.test_deps_compat` |

---

## 27. Recommended initial public API

```julia
module Double32s

export Double32                                    # the only exported name

public hi, lo, words, canonicalize
public SplitDouble32Array
public StrictPolicy, NativePolicy, FastPolicy
public compensated_sum, compensated_dot, compensated_mapreduce
public nominal_eps, unit_roundoff

struct Raw end                                     # internal construction tag

struct Double32 <: AbstractFloat
    hi::Float32
    lo::Float32

    Double32(::Raw, hi::Float32, lo::Float32) = new(hi, lo)
end

@inline raw(hi::Float32, lo::Float32) = Double32(Raw(), hi, lo)
@inline Double32(hi::Float32, lo::Float32) = canonicalize(hi, lo)

@inline hi(x::Double32)    = x.hi
@inline lo(x::Double32)    = x.lo
@inline words(x::Double32) = (x.hi, x.lo)

# Base integration:
# zero, one, oneunit, typemin, typemax, floatmin, floatmax, precision, eps
# promote_rule, widen, float, AbstractFloat           (convert comes free)
# predicates, comparisons, isequal, isless, hash
# +, -, *, /, inv, sqrt, fma, muladd, abs, abs2, copysign, flipsign, ldexp
# show (two-argument only)                            (broadcastable comes free)

# Array and kernel operations:
compensated_sum(A; policy = DEFAULT_POLICY, deterministic = false)
compensated_dot(x, y; policy = DEFAULT_POLICY, deterministic = false)
compensated_mapreduce(f, op, A; policy = DEFAULT_POLICY, deterministic = false)

# Provided by Base once the scalar methods exist — do not reimplement:
#   sum(Double32, A::AbstractArray{Float32})   Double32-accumulated sum
#   z .= muladd.(x, y, w)                      fused compensated map
#   sort!(::Vector{Double32})                  via isless

end
```

Do not export internal operation-mode functions such as `add_fast` or raw constructors in version 1. Everything not exported but intended for users is marked `public`, which documents it without occupying the user's namespace.

---

## 28. Experiments required before implementation decisions are frozen

1. Compare `@localmem Double32` with two `@localmem Float32` arrays on CUDA and AMDGPU.
2. Compare AoS and SoA loads for add, multiply, and dot.
3. Measure accurate versus fast addition and multiplication error/performance.
4. Verify exact emitted `Float32` FMA instructions.
5. Measure register counts and occupancy for every scalar operation embedded in a kernel.
6. Characterize subnormal behavior on each CI GPU.
7. Compare KA shared-memory reduction with CUDA warp shuffle.
8. Compare KA shared-memory reduction with AMD wavefront shuffle.
9. Test fixed-tree cross-backend reproducibility.
10. Determine whether a second division or square-root refinement is needed.
11. ~~Validate the maximum finite canonical pair.~~ **Done** — section 11.2; the
    derivation and the constant both hold. Keep the assertion as a regression
    test.
12. Evaluate whether `AbstractFloat` generic methods assume conventional `nextfloat`.
13. Measure compile latency with and without direct `GPUArrays` dependency.
14. Compare direct vendor launch tuning with KA defaults.
15. Test hash and equality consistency across `Double32`, `Float32`, `Float64`, and integers.

Julia-level experiments, each cheap and each capable of changing a decision in
sections 6 through 12:

16. Confirm that `Double32` array elements are loaded as one 64-bit transaction
    (`ld.global.v2.f32`, `global_load_dwordx2`) despite the type's 4-byte
    alignment; if not, measure the `load_pair` helper of section 6.3.
17. Compare `@inline`/`@noinline` placements on the cold guarded paths by
    register count and occupancy, not only by wall time.
18. Confirm that the tuple-based layer-1 kernels are fully SROA'd — the emitted
    code for `add_dd_` should be identical to a hand-written scalar version.
19. Measure the cost of the single output-finiteness branch versus operand
    pre-classification, on both CPU and GPU, for `+` and `*`.
20. Verify that `sum(Double32, ::CuArray{Float32})` reaches a fast path through
    Base's `mapreduce` rather than falling back to scalar indexing.
21. Measure package load time and time-to-first-operation with and without the
    `PrecompileTools` workload of section 12.13.

---

## 29. Final architectural recommendation

Build a small package around a concrete two-`Float32` scalar with a strong canonical invariant and explicit FMA-based error-free transforms. Treat the type as compensated double-word arithmetic with approximately 48-bit normal-range accuracy, not as a new IEEE floating-point format.

Make `KernelAbstractions.jl` the normative execution layer:

- all scalar arithmetic remains ordinary backend-neutral Julia;
- elementwise operations, reductions, dot products, and initial matrix kernels are written in KA;
- static workgroup sizes and convergent barriers are used for reductions;
- deterministic multipass trees are available;
- no pair-valued atomics are required.

Use `CUDA.jl` and `AMDGPU.jl` through package extensions:

- query occupancy and select launch parameters;
- use warp/wavefront shuffle reductions when they are faster;
- use backend math only as a seed for a corrected double-word result;
- permit direct native kernels only after a KA reference implementation exists and benchmarks prove a meaningful benefit.

Preserve the useful numerical architecture of `DoubleFloats.jl`, but do not inherit its broad CPU dependency graph or complete mathematical surface. The first release should be narrow, measurable, and rigorously validated.

Write it as Julia rather than as transliterated numerical C:

- one exported name, `public` for the rest;
- three layers — generic tuple kernels, concrete policy methods, one-line Base
  methods — with the layer visible in the file it lives in;
- singleton policy types instead of flags, resolved to dispatch exactly once at
  the host boundary;
- `@inline` on everything small, `@noinline` on every cold path, and a
  justification comment beside every unchecked precondition;
- `fma` where fusion is required, plain arithmetic where separate roundings are,
  and `muladd` nowhere in the numeric core;
- inference, allocation, dispatch, piracy, and ambiguity checks in CI from the
  first commit, because each of them is a one-line test and a multi-week
  retrofit.

The numerical design and the Julia design reinforce each other here. The same
concreteness that makes the type fast on a GPU makes its method table small and
its errors legible; the same layering that makes the kernels auditable makes
them reusable from the SoA path; and the same refusal to hide decisions behind
runtime flags is what lets the accuracy claims be tested rather than asserted.

Finally, resolve the Julia naming conflict before implementation. The strongest public API is:

```text
package/module: Double32s
scalar type:    Double32
```

That naming supports the intended user-facing type while leaving the architecture described here unchanged.

---

## 30. References

#### DoubleFloats.jl

- **[DF-Double]** `src/Double.jl`: pair representation, normalizing and raw constructors, aliases, conversions, and nominal precision.  
  <https://github.com/JuliaMath/DoubleFloats.jl/blob/9b37877d0f54ec8792d8954cf9e0e2dec59460d6/src/Double.jl>

- **[DF-EFT]** `src/math/errorfree.jl`: `TwoSum`, `TwoDiff`, `FastTwoSum`, FMA-based `TwoProd`, and guarded versus branch-free variants.  
  <https://github.com/JuliaMath/DoubleFloats.jl/blob/9b37877d0f54ec8792d8954cf9e0e2dec59460d6/src/math/errorfree.jl>

- **[DF-Arithmetic]** `src/math/ops/op_dddd_dd.jl` and `op_dbdb_db.jl`: pair arithmetic kernels and finite/non-finite wrapper split.  
  <https://github.com/JuliaMath/DoubleFloats.jl/blob/9b37877d0f54ec8792d8954cf9e0e2dec59460d6/src/math/ops/op_dddd_dd.jl>  
  <https://github.com/JuliaMath/DoubleFloats.jl/blob/9b37877d0f54ec8792d8954cf9e0e2dec59460d6/src/math/ops/op_dbdb_db.jl>

- **[DF-FMA]** `src/math/arithmetic/fma.jl`: compensated pair FMA patterns and cancellation refinement.  
  <https://github.com/JuliaMath/DoubleFloats.jl/blob/9b37877d0f54ec8792d8954cf9e0e2dec59460d6/src/math/arithmetic/fma.jl>

- **[DF-Refine]** `src/math/leasterror.jl`: reciprocal, division, and square-root corrections.  
  <https://github.com/JuliaMath/DoubleFloats.jl/blob/9b37877d0f54ec8792d8954cf9e0e2dec59460d6/src/math/leasterror.jl>

- **[DF-Technical]** `docs/src/technicalguide.md`: unevaluated-sum model, exponent-range limitation, precision taper, and algorithm summaries.  
  <https://github.com/JuliaMath/DoubleFloats.jl/blob/9b37877d0f54ec8792d8954cf9e0e2dec59460d6/docs/src/technicalguide.md>

#### KernelAbstractions.jl and GPU ecosystem

- **[KA-Kernels]** Kernel language, indexing, `@Const`, static `@localmem`, `@synchronize`, and launch forms.  
  <https://github.com/JuliaGPU/KernelAbstractions.jl/blob/65690fd958223a3a1902651b24dd280cd8243ff3/docs/src/kernels.md>

- **[KA-Overview]** performance-portable backend model and CUDA backend relationship.  
  <https://github.com/JuliaGPU/KernelAbstractions.jl/blob/65690fd958223a3a1902651b24dd280cd8243ff3/docs/src/index.md>

- **[CUDA-Kernel]** CUDA device arrays, shared memory, synchronization, warp shuffle operations, atomics, and device math.  
  <https://github.com/JuliaGPU/CUDA.jl/blob/a2d9472130071aae1a14ab1c4b18502c243c4299/docs/src/api/kernel.md>

- **[AMDGPU-Kernel]** AMDGPU launch configuration, occupancy, atomics, matrix instructions, and wavefront primitives.  
  <https://github.com/JuliaGPU/AMDGPU.jl/blob/f39609f9201491d8e7776d5959aed6dd2a02e427/docs/src/api/kernel_programming.md>

- **[GPUArrays-Project]** current relationship among GPUArrays, Adapt, and KernelAbstractions.  
  <https://github.com/JuliaGPU/GPUArrays.jl/blob/master/Project.toml>

---

## Appendix A. Compact operation inventory

| Operation | Required primitives | Normalization | Backend specialization |
|---|---|---|---|
| construct pair | `TwoSum` | strong | none |
| add/sub | `TwoSum`, `TwoDiff`, `FastTwoSum` | strong | generally none |
| multiply | `TwoProd`, FMA, cross terms | strong | instruction inspection |
| divide | hardware divide, product residual, correction | strong | launch only |
| reciprocal | divide/FMA correction | strong | possible seed |
| sqrt | hardware sqrt, FMA residual, correction | strong | possible seed |
| compensated FMA | `TwoProd`, FMA chain, expansion sum | strong | possible tuned kernel |
| sum | pair add, KA local memory | each tree node | CUDA/AMD subgroup |
| Float32 dot | exact `TwoProd`, pair add | each accumulation step/tree node | CUDA/AMD subgroup |
| matmul | tiled KA, pair accumulators | accumulator maintenance | future native/hybrid |
| transcendental | reduction + polynomial/refinement | function-specific | backend seed only |

## Appendix B. Minimum CI matrix

| Job | Required checks |
|---|---|
| Julia 1.12 Linux CPU | full core, BigFloat/Arb reference, KA CPU |
| Julia 1.12 Windows CPU | core and package API |
| Julia 1.12 macOS CPU | core and package API |
| Julia nightly Linux CPU | core only, allowed to fail; early warning on Base changes |
| quality (Linux CPU) | Aqua, JET, AllocCheck, `@inferred`, ExplicitImports, doctests, `@fastmath` grep |
| NVIDIA GPU | KA CUDA, native CUDA, capability tests, PTX/register inspection |
| AMD GPU | KA AMDGPU, native AMDGPU, capability tests, LLVM/GCN inspection |
| no weak dependencies | package loads and CPU scalar core works |
| CUDA only | CUDA extension loads without AMDGPU |
| AMDGPU only | AMDGPU extension loads without CUDA |
| both vendor packages | no ambiguity; backend dispatch selects by array/backend |
| docs build | Documenter build succeeds and doctests pass |

The nightly job matters more than usual for this package: it depends on
`Base.IEEEFloat`, `Random`'s sampler internals, `precision`'s keyword form, and
LLVM's non-contraction of `a*b + c`. Each is stable in practice, and each is
worth learning about from CI rather than from a user.
