# core/normalize.jl — the one place a pair becomes canonical.
#
# Design sections 5.2 (canonical finite representation) and 5.3 (strong vs weak
# normalization).

"""
    canonicalize(hi::Float32, lo::Float32) -> Double32

Return the canonical `Double32` whose value is the exact real sum `hi + lo`.

This is the normalizing constructor: `Double32(hi, lo)` calls it. It is the
only public route that establishes the invariants of design section 5.2, so a
`raw` pair from anywhere else must be justified by the algorithm that produced
it.

Two behaviors are worth stating because they are not what a plain `two_sum`
would give:

- a non-finite high word (including genuine overflow of two finite words) is
  mapped to the canonical non-finite pair, so `lo` is never `NaN` beside an
  infinite `hi`;
- when both words are zero the sign of `hi` is preserved, rather than taking
  IEEE's `(-0.0f0) + (+0.0f0) == +0.0f0`. This is what makes
  `Double32(x.hi, x.lo) === x` hold for every canonical `x`, which design
  section 11.5 requires of two-argument `show`.

**This is also the byte-level normal form.** A zero low word always comes back
`+0.0f0`, so two pairs that are `isequal` — which, by design section 5.2
invariant 8, may differ in that sign bit — canonicalize to *identical bits*.
That is the tool for anything that must compare or key on representation rather
than value: serialization, byte-wise deduplication, a content hash of a raw
buffer. Verified over the signed-zero surface: 0 negative-zero low words
emitted, and every `isequal` pair byte-identical after canonicalization.

The sign is deliberately **not** normalized at construction. `raw` is the hot
path and negation is `raw(-hi, -lo)`; clearing the sign there would add a select
to every negation to fix something that is not a defect, since no value-based
operation can observe it (see the `show` docstring).

# Examples
```jldoctest
julia> Double32s.canonicalize(1.0f0, 2.0f0^-30)     # already canonical
1.0000000009313226

julia> Double32s.canonicalize(2.0f0^-30, 1.0f0)     # words out of order
1.0000000009313226

julia> Double32s.canonicalize(1.0f0, 0.5f0)         # |lo| > ulp(hi)/2: renormalizes
1.5
```
"""
@inline function canonicalize(a::Float32, b::Float32)
    s, e = two_sum_(a, b)
    isfinite(s) || return canonical_nonfinite(s)
    # Sign of a zero result: IEEE gives +0 for (-0) + (+0), but the pair
    # (-0.0f0, +0.0f0) must round-trip through `show`/`parse` and through
    # `Double32(x.hi, x.lo)`. Only fires when both inputs are zero, so a real
    # cancellation such as `1 - 1` still yields IEEE's `+0.0f0`.
    both_zero = iszero(a) & iszero(b)
    return raw(ifelse(both_zero, a, s), ifelse(both_zero, 0.0f0, e))
end

"""
    canonicalize(x::Double32) -> Double32

Renormalize a possibly weakly-normalized pair. A no-op for canonical input.
"""
@inline canonicalize(x::Double32) = canonicalize(x.hi, x.lo)

@inline canonicalize(t::Tuple{Float32,Float32}) = canonicalize(t[1], t[2])

"""
    Double32s.iscanonical(x::Double32) -> Bool

Whether `x` satisfies the canonical invariants of design section 5.2. Host and
test code only — this is the assertion that `raw` call sites are audited
against, not something a kernel should call.

Note that invariant 8 makes the *sign* of a zero low word unspecified, so this
compares low words with `==` rather than `===`.
"""
function iscanonical(x::Double32)
    isfinite(x.hi) || return iszero(x.lo)
    # Invariants 6 and 8: a zero high word forces a zero low word, and neither
    # sign is information. Checked separately because `two_sum_(-0.0f0, 0.0f0)`
    # returns `+0.0f0` — IEEE, and exactly what `canonicalize` overrides.
    iszero(x.hi) && return iszero(x.lo)
    s, e = two_sum_(x.hi, x.lo)
    return (s === x.hi) & (e == x.lo)
end
