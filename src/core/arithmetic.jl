# core/arithmetic.jl — the three layers of design section 8.0.
#
#   Layer 1  tuple kernels     generic, branch free, no type, no guard
#   Layer 2  policy dispatch   concrete, one branch (the single output check)
#   Layer 3  Base integration  one line each, no logic at all
#
# Layer 1 takes and returns `Tuple{T,T}` rather than `Double32` so that an
# intermediate weak pair is expressible without lying to the constructor, so
# that SoA code can call the same kernels without materializing a `Double32`,
# and so that the algorithms can be validated at `Float64` where the literature
# bounds are stated.
#
# Each public operation: run a branch-free finite kernel, test the final high
# word for non-finiteness, take a guarded or scaled fallback if required. Note
# what is *not* in that list — a pre-classification of the operands. Testing the
# output subsumes it, which halves the branch count.

# ============================================================== LAYER 1 =====

"""
    Double32s.add_dd_(x::Tuple{T,T}, y::Tuple{T,T}) -> Tuple{T,T}

Accurate double-word addition. Branch free; the caller must check the result
for non-finiteness.

This is, operation for operation, **AccurateDWPlusDW** — Algorithm 6 of
Joldeş, Muller & Popescu, *Tight and rigorous error bounds for basic building
blocks of double-word arithmetic* (ACM TOMS 44(2), 2017) — whose relative error
bound is

    |result − (x + y)| ≤ 3u²/(1 − 4u) · |x + y|,    u = 2⁻²⁴,

proved in that paper and subsequently **machine-checked** (Coq/Flocq) by
Muller & Rideau, *Formalization of double-word arithmetic, and comments on
"Tight and rigorous error bounds…"* (ACM TOMS 48(1), 2022). The formalization
is generic in the precision, so the bound applies at p = 24, and it covers the
two Fast2Sum steps — the algorithm's correctness does **not** rest on the
`|a| ≥ |b|` precondition.

!!! note "Why the local `|a| ≥ |b|` justification was wrong, and why it doesn't matter"
    An earlier draft justified the two `two_hilo_sum_` calls with "`|hi| ≥ |c|`
    by construction of `two_sum_`", which silently assumes the `two_sum_` high
    word is nonzero. Under high-word cancellation it is exactly zero while `c`
    is not: measured, 834 violations in 40,000 structured pair additions, plus
    violations with `hi ≠ 0` such as `hi = -2^-24, c = -3·2^-25`. The condition
    that actually matters is the weaker one in [`EFT.two_hilo_sum_`](@ref) —
    representability of the Fast2Sum residual — and the machine-checked proof
    above establishes the end-to-end bound without the naive precondition.
    Empirically: substituting `two_sum_` at both sites changes no bit across
    40,000 structured and 1,500,000 random finite pair additions
    (`test/core/eft.jl` keeps that differential check as a regression);
    measured worst error 2.0u², under the 3u² ceiling.

Validity: the bound assumes no overflow and canonical (nonoverlapping) inputs.
It survives into the subnormal range in absolute terms — floating-point
*addition* errors are always exactly representable (Hauser), so every EFT here
stays exact — but the *relative* form of the bound presumes a normal result.
"""
@inline function add_dd_(x::Tuple{T,T}, y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    xhi, xlo = x
    yhi, ylo = y
    h, l     = two_sum_(xhi, yhi)
    thi, tlo = two_sum_(xlo, ylo)
    c        = l + thi
    h, l     = two_hilo_sum_(h, c)
    c        = tlo + l
    h, l     = two_hilo_sum_(h, c)
    return h, l
end

"""
    Double32s.add_dd_fast_(x::Tuple{T,T}, y::Tuple{T,T}) -> Tuple{T,T}

Fewer operations and registers than [`add_dd_`](@ref), weaker last-bit error
bound. For reductions and explicitly requested fast mode.
"""
@inline function add_dd_fast_(x::Tuple{T,T}, y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    xhi, xlo = x
    yhi, ylo = y
    h, e     = two_sum_(xhi, yhi)
    t        = (xlo + ylo) + e
    return two_hilo_sum_(h, t)
end

"""
    Double32s.sub_dd_(x::Tuple{T,T}, y::Tuple{T,T}) -> Tuple{T,T}

Accurate double-word subtraction. Its own kernel rather than `add_dd_(x, -y)`,
for the signed-zero reason given in design section 7.3.

Error bound: identical to [`add_dd_`](@ref) — negation of both words is exact,
and `two_diff_(a, b)` computes the same values as `two_sum_(a, -b)` apart from
the sign of a zero, so the AccurateDWPlusDW bound `3u²/(1-4u)` transfers.
"""
@inline function sub_dd_(x::Tuple{T,T}, y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    xhi, xlo = x
    yhi, ylo = y
    h, l     = two_diff_(xhi, yhi)
    thi, tlo = two_diff_(xlo, ylo)
    c        = l + thi
    h, l     = two_hilo_sum_(h, c)
    c        = tlo + l
    h, l     = two_hilo_sum_(h, c)
    return h, l
end

@inline function sub_dd_fast_(x::Tuple{T,T}, y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    xhi, xlo = x
    yhi, ylo = y
    h, e     = two_diff_(xhi, yhi)
    t        = (xlo - ylo) + e
    return two_hilo_sum_(h, t)
end

"""
    Double32s.mul_dd_(x::Tuple{T,T}, y::Tuple{T,T}) -> Tuple{T,T}

Accurate double-word multiplication: exact leading product plus both cross
terms plus the low-low term.

`fma` here is chosen for accuracy, not exactness — each `fma` retains one
rounding a separate multiply-and-add would lose. That is not in tension with
the "never `muladd` in an EFT" rule, which is about `muladd`'s *optionality*;
these are explicit `fma` calls whose fusion is guaranteed.

Not a proof of correctly rounded `Double32` multiplication: residuals of the
cross products are not all retained.
"""
@inline function mul_dd_(x::Tuple{T,T}, y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    xhi, xlo = x
    yhi, ylo = y
    p, e = two_prod_(xhi, yhi)
    c    = fma(xhi, ylo, xlo * yhi)     # both cross terms, one rounding saved
    c    = fma(xlo, ylo, c)             # low-low term
    t    = e + c
    return two_hilo_sum_(p, t)
end

"""
    Double32s.mul_dd_fast_(x::Tuple{T,T}, y::Tuple{T,T}) -> Tuple{T,T}

Omits the `x.lo * y.lo` term, whose magnitude is normally of order `u²`
relative to the leading product. Efficient, but it can affect the final
double-word bits.
"""
@inline function mul_dd_fast_(x::Tuple{T,T}, y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    xhi, xlo = x
    yhi, ylo = y
    p, e = two_prod_(xhi, yhi)
    c    = fma(xhi, ylo, xlo * yhi)
    t    = e + c
    return two_hilo_sum_(p, t)
end

"""
    Double32s.div_dd_(x::Tuple{T,T}, y::Tuple{T,T}) -> Tuple{T,T}

One leading quotient and one compensated correction — the division kernel for
**every** policy; see [`div_dd_refined_`](@ref) for why the second correction
is measured, retained, and not used.

Accuracy: worst observed error 4.00 nominal-48-bit ulps (1.56% of 101,761
lattice-range quotients exceed 1 ulp), same family as Joldeş–Muller–Popescu's
DWDivDW2, whose proven bound for their variant is `15u²`. No formal bound has
been derived for this exact kernel; the claim is the measured one, and
faithful rounding is **not** claimed (design section 9's target revised by
measurement — see experiment 10's resolution).

The correction is *not* skipped when `y[2] == 0`: a one-word divisor such as
`3.0f0` still needs it to reach double-word precision. That skip is a tempting
"optimization" precisely because it looks like an algebraic identity.
"""
@inline function div_dd_(x::Tuple{T,T}, y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    q0 = x[1] / y[1]
    r  = sub_dd_(x, mul_dd_((q0, zero(T)), y))
    q1 = (r[1] + r[2]) / y[1]
    return two_hilo_sum_(q0, q1)
end

"""
    Double32s.div_dd_refined_(x::Tuple{T,T}, y::Tuple{T,T}) -> Tuple{T,T}

[`div_dd_`](@ref) plus a second Newton correction.

**Measured, and deliberately not selected by any policy.** Over 101,761
lattice-range quotients against a 400-bit `BigFloat` reference
(`test/core/arithmetic.jl` carries the regression):

| kernel | worst error (nominal 48-bit ulps) | not-faithful cases |
|---|---|---|
| `div_dd_` (one correction) | 4.00 | 1.56% |
| `div_dd_refined_` (two) | 2.30 | 0.66% |

The second correction roughly doubles the kernel's operation count and
register pressure — the costs that set GPU occupancy — and **still does not
reach faithful rounding**. Design section 8.4's contingency ("StrictPolicy may
select the refinement if one correction is not faithfully rounded") was
written on the assumption that the refinement would close the gap; it does
not, so paying 2× in every device kernel buys only a smaller constant. The
resolution of design experiment 10 is therefore: one correction for every
policy, with the measured bound documented, and this kernel retained under
test for any future host-only accuracy policy.

A faithfully-rounded double-word division needs a different algorithm, not a
repeated correction; that remains out of scope (design section 25.4 territory).
"""
@inline function div_dd_refined_(x::Tuple{T,T}, y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    h, l = div_dd_(x, y)
    r2   = sub_dd_(x, mul_dd_((h, l), y))
    q2   = r2[1] / y[1]
    return two_hilo_sum_(h, l + q2)
end

"""
    Double32s.inv_dd_(y::Tuple{T,T}) -> Tuple{T,T}

Reciprocal as a specialized division, saving the high-word divide's operand
setup.
"""
@inline function inv_dd_(y::Tuple{T,T}) where {T<:Base.IEEEFloat}
    q0 = inv(y[1])
    r  = sub_dd_((one(T), zero(T)), mul_dd_((q0, zero(T)), y))
    q1 = (r[1] + r[2]) / y[1]
    return two_hilo_sum_(q0, q1)
end

"""
    Double32s.sqrt_dd_(x::Tuple{T,T}) -> Tuple{T,T}

One Newton correction on the `Float32` square root. Valid for positive finite
`x` whose high word is normal; the layer-2 method handles scaling and the
exceptional cases.
"""
@inline function sqrt_dd_(x::Tuple{T,T}) where {T<:Base.IEEEFloat}
    s0 = sqrt_nonneg(x[1])
    r  = sub_dd_(x, mul_dd_((s0, zero(T)), (s0, zero(T))))
    s1 = (r[1] + r[2]) / (s0 + s0)
    return two_hilo_sum_(s0, s1)
end

"""
    Double32s.fma_dd_(x, y, z) -> Tuple{T,T}

Compensated `x*y + z` computing all major pair terms with one final
renormalization. Not correctly rounded; the derived bound is absolute, in
units of the operand scale, because under near-total cancellation no relative
bound is achievable at any cost in this format.

**Derived bound.** Let `u = 2⁻²⁴` and `M = max(|xh·yh|, |zh|)`. For canonical
finite inputs with `M ∈ [2⁻⁷⁷, 2¹²⁶]` (every intermediate then avoids overflow
and keeps its `two_prod_` residual representable):

    |fma_dd_(x, y, z) − (x·y + z)| ≤ 2u²·(1 + 80u)·M

*Derivation.* Every `two_sum_`/`two_prod_` is exact, so the chain telescopes:
`s + a4 + (b1+b2+b3+b4) + e2 + e3 + p4` accounts for every term of the exact
result except the `xl·yl` rounding (`≤ u³M`). Magnitudes: `|t| ≤ 2uM(1+u)`,
`|e1| ≤ uM`, `|p2|,|p3| ≤ uM(1+u)`, `|zl| ≤ uM`, whence
`|b1..b4| ≤ (3,4,5,6)u²M·(1+O(u))` and `|e2|,|e3|,|p4| ≤ u²M(1+O(u))`, giving
`|low| ≤ 21u²M(1+O(u))` with its six accumulation roundings `≤ 126u³M`. The
2Sum `(s, a4)` is exact with `|l| ≤ u|h| ≤ 2uM(1+5u)`, so the **only**
O(u²) rounding in the whole kernel is the final `l + low`:
`≤ u(|l| + |low|) ≤ 2u²M(1+16u)`. Total `≤ 2u²M(1+80u)`. ∎

Measured worst over structured pools: `2.0000 u²·M` — the bound is tight to
well under one percent. The ordering question of design section 23.9 is
thereby closed: with the exact-telescope structure fixed, ordering moves only
the `O(u³)` terms; the `2u²` floor is the final-recombination rounding and is
irreducible without a three-word accumulator.

The final normalization uses `two_sum_` rather than `two_hilo_sum_`
deliberately — nothing establishes the FastTwoSum residual condition here, and
design section 7.2's rule is to pay the three extra operations rather than
rely on an unproven one.

Outside the validity window the result degrades gradually with the underflow
taper of design section 5.5 (products below `2⁻⁷⁷` lose cross-term residual
bits at subnormal granularity); no rescue is attempted because, unlike `*`,
the three-operand scale structure admits no single exact rescaling.
"""
@inline function fma_dd_(x::Tuple{T,T}, y::Tuple{T,T}, z::Tuple{T,T}) where {T<:Base.IEEEFloat}
    xhi, xlo = x
    yhi, ylo = y
    zhi, zlo = z
    # Every product that can be split exactly, is. Retaining the *cross-term*
    # residuals e2 and e3 is what separates this from a plain
    # `fma(xhi, ylo, xlo*yhi)`: under cancellation those residuals are the
    # answer. Only `xlo * ylo` is left unsplit, since its own residual is below
    # what the pair format can carry.
    p1, e1 = two_prod_(xhi, yhi)
    p2, e2 = two_prod_(xhi, ylo)
    p3, e3 = two_prod_(xlo, yhi)
    p4     = xlo * ylo
    s, t   = two_sum_(p1, zhi)            # exact; the leading cancellation
    # Distill the O(u) and smaller terms with TwoSum rather than plain `+`.
    # Plain addition is what an earlier draft used, and it is catastrophically
    # wrong under cancellation: for x = (-1, -2^-24), y = (3-2^-23, 2^-24),
    # z = (3, -5.33e-15) the correction is 2.4e-7 while the answer *is* z's low
    # word, which a Float32 add rounds away entirely — measured relative error
    # of that draft on this triple: 1.67.
    a1, b1 = two_sum_(t,  e1)
    a2, b2 = two_sum_(a1, p2)
    a3, b3 = two_sum_(a2, p3)
    a4, b4 = two_sum_(a3, zlo)
    low    = ((((b1 + b2) + (b3 + b4)) + (e2 + e3)) + p4)
    # Keep the double-word structure of (a4, low) through the final combine, so
    # that a cancellation in `s + a4` still exposes `low`.
    h, l = two_sum_(s, a4)
    return two_sum_(h, l + low)
end

# ============================================================== LAYER 2 =====
#
# The struct, the policy, and the single output check. `raw` is justified in the
# finite branch because every layer-1 kernel ends in a FastTwoSum or TwoSum,
# which establishes the canonical invariant by construction.

"""
    Double32s.zero_sign(h::Float32, reference::Float32) -> Float32

Restore the IEEE sign of a zero result.

A pair kernel that ends in `s = p + t` loses the sign whenever `p` and `t` are
zeros of opposite sign: `(-0.0f0) + (+0.0f0)` is `+0.0f0`. IEEE fixes the sign
of a zero result from the *operands* — `(-0) + (-0)` is `-0`, `0 * -0` is `-0` —
so the layer-2 methods pass the corresponding `Float32` high-word operation as
the reference. The reference is already computed inside the kernel, so this
costs a select rather than an arithmetic op.

`Complex{Double32}` branch cuts and every `copysign`-based algorithm depend on
this, which is why it is not treated as a rounding detail.

Both operands must be zero for the substitution to fire, and that condition is
load-bearing rather than defensive. A pair-level cancellation can be exact while
the high-word operation is not: `fma` on the high words of an operand triple
whose *pair* product-sum is zero returns a nonzero value, and substituting it
would replace a correct zero with garbage — measured at `7.9e-8` relative before
the `iszero(reference)` conjunct was added.
"""
@inline zero_sign(h::Float32, reference::Float32) =
    ifelse(iszero(h) & iszero(reference), reference, h)

@inline function add(::StrictPolicy, x::Double32, y::Double32)
    h, l = add_dd_(words(x), words(y))
    isfinite(h) || return add_nonfinite(x, y)
    return raw(zero_sign(h, x.hi + y.hi), l)
end

@inline function add(::Union{NativePolicy,FastPolicy}, x::Double32, y::Double32)
    h, l = add_dd_fast_(words(x), words(y))
    isfinite(h) || return add_nonfinite(x, y)
    return raw(zero_sign(h, x.hi + y.hi), l)
end

@inline function sub(::StrictPolicy, x::Double32, y::Double32)
    h, l = sub_dd_(words(x), words(y))
    isfinite(h) || return sub_nonfinite(x, y)
    return raw(zero_sign(h, x.hi - y.hi), l)
end

@inline function sub(::Union{NativePolicy,FastPolicy}, x::Double32, y::Double32)
    h, l = sub_dd_fast_(words(x), words(y))
    isfinite(h) || return sub_nonfinite(x, y)
    return raw(zero_sign(h, x.hi - y.hi), l)
end

"""
    Double32s.add_nonfinite(x, y) -> Double32

Cold path for `+`. Two cases that an earlier draft conflated:

- **a non-finite operand.** `Float32` addition already gives the required table
  — `Inf + finite` is the same infinity, `+Inf + -Inf` is NaN, `NaN + anything`
  is NaN — so the high words are the whole answer.
- **overflow of two finite operands.** Here the high words are *not* the whole
  answer, because the overflow can come from the low words: adding `2f0^80` to
  `floatmax(Double32)` overflows the pair while `x.hi + y.hi` is still exactly
  `floatmax(Float32)`. Recomputing from the high words would silently return a
  finite `floatmax` for a value that overflowed. Saturate to a correctly signed
  infinity instead.

No scaled rescue exists for `+`/`-`, unlike `*` and `/`: a sum that overflows
cannot be brought back by any cross term.
"""
@noinline function add_nonfinite(x::Double32, y::Double32)
    (isfinite(x.hi) & isfinite(y.hi)) || return canonical_nonfinite(x.hi + y.hi)
    return ifelse(signbit(x.hi + y.hi), NEG_INF, POS_INF)
end

@noinline function sub_nonfinite(x::Double32, y::Double32)
    (isfinite(x.hi) & isfinite(y.hi)) || return canonical_nonfinite(x.hi - y.hi)
    return ifelse(signbit(x.hi - y.hi), NEG_INF, POS_INF)
end

# The multiplication output check tests a *range*, not just finiteness: the
# kernel is silently degraded for tiny products (see MUL_TINY in constants.jl),
# and `isfinite` cannot see that. One abs, two compares, still a single branch —
# and exact zeros route through the guard too, so the hot path needs no
# `zero_sign` fixup at all (a nonzero in-range `h` always carries the right
# sign). Warp-friendly: the branch is uniform except in the rare guard bands.
@inline mul_in_range(h::Float32) = (abs(h) >= MUL_TINY) & (abs(h) < Inf32)

@inline function mul(p::StrictPolicy, x::Double32, y::Double32)
    h, l = mul_dd_(words(x), words(y))
    mul_in_range(h) || return mul_guarded(p, x, y)
    return raw(h, l)
end

@inline function mul(p::Union{NativePolicy,FastPolicy}, x::Double32, y::Double32)
    h, l = mul_dd_fast_(words(x), words(y))
    mul_in_range(h) || return mul_guarded(p, x, y)
    return raw(h, l)
end

@inline mul_kernel(::StrictPolicy, x, y) = mul_dd_(x, y)
@inline mul_kernel(::Union{NativePolicy,FastPolicy}, x, y) = mul_dd_fast_(x, y)

"""
    Double32s.mul_guarded(policy, x, y) -> Double32

Cold path for `*`, reached in exactly four situations — a non-finite operand, a
zero operand, an overflowing product, an underflowing-band product — and the
scaled retry in the last two is a correctness requirement (design section 7.5),
not tuning. `canonical_nonfinite(x.hi * y.hi)` — correct for `+`/`-`, where no
rescue exists — would report `Inf` for representable products and garbage low
words for tiny ones.

**Trigger ⟹ the shift is safe.** Let `es = exponent(x.hi) + exponent(y.hi)`.
With finite nonzero operands the kernel's high word satisfies
`|h| = |x·y|·(1 ± 3.1u)`, and `2^es ≤ |xhi·yhi| < 2^(es+2)`, so:

- `h` non-finite requires `|xhi·yhi| ≥ (2^128 − 2^103)/(1+3.1u)`, hence
  `es ≥ 126`; scaling both operands down by `2^-64` puts their exponents in
  `[-1-64, 127-64] = [-65, 63]` — both normal — and the scaled product exponent
  in `[-2, 62]`, where the kernel is exact-residual clean (a nonzero
  `two_prod_` residual is `≥ 2^(e_p - 47) ≥ 2^-50`).
- `|h| < MUL_TINY = 2^-98` requires `es ≤ -97`; each operand exponent is then
  `≤ -97 + 149 = 52`, so scaling both **up** by `2^64` cannot overflow
  (`≤ 2^116`), and the scaled product exponent lands in `[-170, 31]` — fully
  accurate wherever the true result is representable at all; a true product
  below `2^-150` correctly becomes zero on the rescale.

The two bands are disjoint (`es ≤ -97` vs `es ≥ 126`), so one signed shift
serves both. Dropped low-word bits from operand scaling cost at most `2^-63`
relative — an operand's `lo` can underflow only if `|lo/hi| < 2^(-85-es_min)`,
far below `u² = 2^-48`.

**The rescaling is exact and maps the overflow boundary exactly.** Scaling the
result by `2^128` is exact on both words whenever the hi word stays in range,
and the boundary case cannot be misclassified: the kernel's `h` is `fl` of the
scaled pair value, and at every binade the Float32 rounding boundary
(`pair ≥ (1 - 2^-25)·2^(e+1)` rounds hi up to the next power of two) maps under
exact scaling to precisely the overflow threshold `2^128 - 2^103`, including
the round-to-even tie — both `1.0` and `2^128` have even significands. So
`ldexp` of the finite result returns `Inf` iff the true product rounds to
`Inf`. On the underflow side the rescale rounds into the subnormal taper; a
one-quantum double-rounding tie there is inside the documented taper of design
section 5.5.

**What the underflow rescue can and cannot deliver.** The scaled kernel is
full-precision, but a result near `2^-126` cannot *hold* that precision: at
`exponent(hi) = -126` the canonical bound is `|lo| ≤ ½ulp(hi) = 2^-150`, below
the least subnormal, so the format admits no low word at all — precision
tapers from 48 bits at `2^-102` to 24 bits at `2^-126` (design section 5.5).
The guard's obligation is the nearest *representable* pair, i.e. absolute
error `≤ few·u²·|V| + 2^-150`-class. Measured over the band `[2^-126, 2^-98]`:
worst error 8.2e-8·|V| (2.3e7 u², 23 bits lost) before this guard; after it,
within `4u²·|V| + 2^-149` everywhere — the representation optimum. The
regression is in `test/core/arithmetic.jl`.
"""
@noinline function mul_guarded(p::ArithmeticPolicy, x::Double32, y::Double32)
    (isfinite(x.hi) & isfinite(y.hi)) || return canonical_nonfinite(x.hi * y.hi)
    (iszero(x.hi) | iszero(y.hi))     && return raw(x.hi * y.hi, 0.0f0)  # IEEE signed zero
    es = raw_exponent(x.hi) + raw_exponent(y.hi)
    k  = ifelse(es >= 0, -RESCUE_SHIFT, RESCUE_SHIFT)
    xs = ldexp(x, k)
    ys = ldexp(y, k)
    h, l = mul_kernel(p, words(xs), words(ys))
    isfinite(h) || return canonical_nonfinite(h)
    # zero_sign: a deep-underflow product can come back ±0 from the kernel with
    # an IEEE-incorrect sign; xs.hi * ys.hi underflows to the correctly signed
    # zero and is otherwise ignored.
    return ldexp(raw(zero_sign(h, xs.hi * ys.hi), l), -2k)
end

"""
    Double32s.divide(policy, x::Double32, y::Double32) -> Double32

Layer-2 division. **A tiny numerator is scaled up first**, and this is not an
optimization to be removed.

`div_dd_` forms the residual `x - q0*y`, whose magnitude is about `|x| * 2^-48`.
When `exponent(x.hi)` drops near `-101` that residual falls below `2f0^-149` and
is lost — silently, since the result is perfectly finite. The quotient itself
may be fully representable while the algorithm computing it is not: measured on
random operands, `Double32(2.0165723f-36, 8.0f-44) / Double32(4.0026174f-18, …)`
came back with relative error `6.0e-10`, roughly 35 correct bits instead of 48.

The scaling only ever goes **up**, and that direction is load-bearing. Scaling
a large numerator *down* into a unit binade — the obvious symmetric thing —
destroys a subnormal low word, and with it `x / one(Double32) === x`:
`Double32(3.0f0, 2f0^-149)` scaled down by one binade loses its low word
outright. Scaling up is always exact.

Only the numerator needs it. `q0` cannot overflow as a result of the scaling:
after it, `exponent(x.hi) >= SCALE_FLOOR`, and `exponent(y.hi) >= -149`, so
`q0`'s exponent stays below `SCALE_FLOOR + 149 < 127`.
"""
@inline function divide(p::ArithmeticPolicy, x::Double32, y::Double32)
    (isfinite(x.hi) & isfinite(y.hi) & !iszero(x.hi) & !iszero(y.hi)) ||
        return div_exceptional(x, y)
    kx = max(0, SCALE_FLOOR - raw_exponent(x.hi))
    xs = scale2(x, kx)                     # exact: scaling up never underflows
    h, l = div_kernel(p, words(xs), words(y))
    isfinite(h) || return div_guarded(p, x, y)
    iszero(kx) && return raw(h, l)         # the common case pays nothing
    return ldexp(raw(h, l), -kx)           # guarded: may underflow
end

@inline div_kernel(::StrictPolicy, x, y) = div_dd_(x, y)
@inline div_kernel(::Union{NativePolicy,FastPolicy}, x, y) = div_dd_(x, y)

"""
    Double32s.div_guarded(policy, x, y) -> Double32

Overflow rescue for `/` (design sections 7.5, 8.0): the quotient's leading term
can overflow while the true pair quotient is representable, and before this
path existed the hot kernel turned that band into **NaN**, not merely `Inf` —
`q0 = Inf` poisons the residual `x - q0*y` with `Inf - Inf`. Measured:
`Double32(2f0^127) / Double32(0.5f0, 2f0^-25)` has true quotient
`floatmax + 2^80` (representable) and returned `NaN`.

**Reachability ⟹ the shift is safe.** The hot path reaches here only when its
`h` was non-finite with finite nonzero operands, which requires
`exponent(x.hi) - exponent(y.hi) ≥ 127`:

- that forces `exponent(x.hi) ≥ 127 + exponent(y.hi) ≥ 127 - 149 = -22`, so `kx` was zero (the
  tiny-numerator scaling and this rescue are mutually exclusive), and scaling
  `x` down by `2^-64` keeps its high word normal (`exponent ≥ -86`);
- the scaled leading quotient has exponent in `[63, 212]`: if it *still*
  overflows, the true quotient exceeds `2^211` and `Inf` is correct
  saturation; the rescuable band `exponent ∈ [127, 129]` lands near `2^64`,
  where the kernel is fully accurate;
- dropped numerator low-word bits cost at most `2^-63` relative, far below
  `u²`;
- the `2^64` rescale of the result is exact on the hi word up to the overflow
  boundary, and the boundary maps exactly for the same reason as in
  [`mul_guarded`](@ref) — `h` is `fl` of the scaled pair, and the
  round-up-to-next-binade boundary at any binade scales exactly onto the
  `2^128 - 2^103` overflow threshold, ties included.
"""
@noinline function div_guarded(p::ArithmeticPolicy, x::Double32, y::Double32)
    xs = ldexp(x, -RESCUE_SHIFT)
    h, l = div_kernel(p, words(xs), words(y))
    # Still overflowing after the 2^-64 shift means the true quotient exceeds
    # 2^211: saturate with the IEEE sign. Not `canonical_nonfinite(h)` — the
    # overflowed q0 poisons the kernel with Inf - Inf, so h is NaN here, not a
    # signed infinity.
    isfinite(h) || return ifelse(signbit(x.hi) != signbit(y.hi), NEG_INF, POS_INF)
    return ldexp(raw(h, l), RESCUE_SHIFT)
end

"""
    Double32s.div_exceptional(x, y) -> Double32

Cold path for `/`: a zero or non-finite operand on either side. Every such case
reproduces IEEE `Float32` division semantics exactly from the high words, so
`x.hi / y.hi` is the whole answer — `±0/±0` and `Inf/Inf` give NaN, `x/±0`
gives a correctly signed infinity, `±0/y` gives a correctly signed zero.
"""
@noinline div_exceptional(x::Double32, y::Double32) = canonical_nonfinite(x.hi / y.hi)

"""
    Double32s.compensated_fma(policy, x, y, z) -> Double32

High-accuracy compensated `x*y + z`. See [`fma_dd_`](@ref): no single-rounding
or correct-rounding claim is made.
"""
@inline function compensated_fma(p::ArithmeticPolicy, x::Double32, y::Double32, z::Double32)
    h, l = fma_dd_(words(x), words(y), words(z))
    isfinite(h) || return fma_nonfinite(p, x, y, z)
    return raw(zero_sign(h, fma(x.hi, y.hi, z.hi)), l)
end

@noinline function fma_nonfinite(p::ArithmeticPolicy, x::Double32, y::Double32, z::Double32)
    (isfinite(x.hi) & isfinite(y.hi) & isfinite(z.hi)) ||
        return canonical_nonfinite(fma(x.hi, y.hi, z.hi))
    # Finite operands with a non-finite result: the product overflowed. Fall
    # back to the rescued product followed by an ordinary add, which is one
    # extra rounding but never reports a representable result as infinite.
    return add(p, mul_guarded(p, x, y), z)
end

# ============================================================== LAYER 3 =====
#
# One line each; no logic. A `Base.:+` method with a body longer than a single
# call is in the wrong layer.

Base.:+(x::Double32, y::Double32) = add(DEFAULT_POLICY, x, y)
Base.:-(x::Double32, y::Double32) = sub(DEFAULT_POLICY, x, y)
Base.:*(x::Double32, y::Double32) = mul(DEFAULT_POLICY, x, y)
Base.:/(x::Double32, y::Double32) = divide(DEFAULT_POLICY, x, y)

Base.fma(x::Double32, y::Double32, z::Double32)    = compensated_fma(DEFAULT_POLICY, x, y, z)
Base.muladd(x::Double32, y::Double32, z::Double32) = compensated_fma(DEFAULT_POLICY, x, y, z)

# The Float32 mixed methods are written out because `Double32(y::Float32)` is
# `raw(y, 0.0f0)` — free — and because routing this hot case through generic
# promotion is wasteful. For every *other* Real, Base's promotion fallback plus
# core/promote.jl handles it, and each hand-written method would be a new
# opportunity for an ambiguity.
Base.:+(x::Double32, y::Float32) = add(DEFAULT_POLICY, x, Double32(y))
Base.:+(x::Float32, y::Double32) = add(DEFAULT_POLICY, Double32(x), y)
Base.:-(x::Double32, y::Float32) = sub(DEFAULT_POLICY, x, Double32(y))
Base.:-(x::Float32, y::Double32) = sub(DEFAULT_POLICY, Double32(x), y)
Base.:*(x::Double32, y::Float32) = mul(DEFAULT_POLICY, x, Double32(y))
Base.:*(x::Float32, y::Double32) = mul(DEFAULT_POLICY, Double32(x), y)
Base.:/(x::Double32, y::Float32) = divide(DEFAULT_POLICY, x, Double32(y))
Base.:/(x::Float32, y::Double32) = divide(DEFAULT_POLICY, Double32(x), y)

# ------------------------------------------------------- sign operations ---
#
# All branch free, and all preserve the canonical invariant because negating
# both words of a canonical pair yields a canonical pair. `flipsign` and
# `copysign` on Float32 are sign-bit XORs in Base, so these lower to two XORs
# each with no comparison and no branch.
#
# Each of these can produce a *negative zero low word under a nonzero high
# word* — `-one(Double32)` is `(-1.0f0, -0.0f0)`. That is invariant 8 of design
# section 5.2, and it is why core/compare.jl compares low words with `==`.

@inline Base.:-(x::Double32) = raw(-x.hi, -x.lo)
@inline Base.abs(x::Double32) = raw(abs(x.hi), flipsign(x.lo, x.hi))
@inline Base.flipsign(x::Double32, y::Real) = raw(flipsign(x.hi, y), flipsign(x.lo, y))
@inline Base.copysign(x::Double32, y::Real) = ifelse(signbit(x.hi) == signbit(y), x, -x)
@inline Base.abs2(x::Double32) = x * x

Base.:+(x::Double32) = x

# ------------------------------------------------ scaling by powers of two ---

"""
    Double32s.scale2(x::Double32, n::Int) -> Double32

Unguarded exact scaling by `2^n`, for call sites that have already established
that neither word leaves range. Use `ldexp` anywhere else.
"""
@inline scale2(x::Double32, n::Int) = raw(ldexp(x.hi, n), ldexp(x.lo, n))

Base.ldexp(x::Double32, n::Integer) = ldexp(x, Int(n))

"""
    ldexp(x::Double32, n::Int) -> Double32

Scale by `2^n`. Exact unless the low word underflows — a legitimate, documented
precision taper, not an error — or the high word leaves range.

Renormalizes rather than using `raw`, because scaling *down* can round the low
word into the subnormal range and land it exactly on the half-ulp boundary,
where the canonical invariant depends on the parity of the high word's
significand. Observed: `2.6295364f-31` with a low word of exactly `-2f0^-126`.
For canonical input that does not underflow, the `two_sum_` is an identity and
costs six adds.
"""
function Base.ldexp(x::Double32, n::Int)
    h = ldexp(x.hi, n)
    isfinite(h) || return canonical_nonfinite(h)
    return canonicalize(h, ldexp(x.lo, n))
end

# `Base.exponent` is the one place the `DomainError` is kept, because that *is*
# its contract: `exponent(0.0f0)` throws, and a `Double32` method that returned
# a number instead would be the odd one out. Every other method here excludes
# the domain first and then uses `raw_exponent`, so no unreachable `throw`
# survives into the generated code (see `Double32s.raw_exponent`).
Base.exponent(x::Double32) = exponent(x.hi)

"""
    significand(x::Double32) -> Double32

The significand of `x`, scaled into `[1, 2)`, matching `significand(::Float32)`.

Zero and non-finite arguments are returned unchanged, which is what
`Base.significand` does for `Float32` — `significand(0.0f0)` is `0.0f0`, not a
`DomainError`. An earlier version of this method inherited the throw from
`Base.exponent` and disagreed with Base at all five of `±0`, `±Inf`, and `NaN`.

# Examples
```jldoctest
julia> significand(Double32(12.0f0))
Double32(1.5f0, 0.0f0)

julia> significand(zero(Double32)), significand(Double32(Inf32))
(Double32(0.0f0, 0.0f0), Double32(Inf32, 0.0f0))
```
"""
function Base.significand(x::Double32)
    (isfinite(x.hi) & !iszero(x.hi)) || return x
    return ldexp(x, -raw_exponent(x.hi))
end

function Base.frexp(x::Double32)
    (isfinite(x.hi) & !iszero(x.hi)) || return (x, 0)
    e = raw_exponent(x.hi) + 1
    return (ldexp(x, -e), e)
end
