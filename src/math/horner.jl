# math/horner.jl — the polynomial-evaluation primitives every elementary
# function is built on.
#
# Two ideas carry the whole file, and every kernel in exp.jl, log.jl, trig.jl,
# and hyperbolic.jl states its use of them:
#
# 1. **The dominant-add Horner step.** A Horner iteration `c + z*p` through the
#    public `+`/`*` costs a full policy multiply plus a full policy add — a
#    TwoSum (6 flops) where a FastTwoSum (3) would do, plus two output guards
#    that can never fire inside a kernel whose operands are bounded by
#    construction. `poly_step` fuses the iteration: exact leading product, the
#    two cross terms in two fmas, FastTwoSum against the coefficient. The
#    FastTwoSum is valid because every caller establishes **coefficient
#    dominance**: `|z*p| <= |c|/2`, so `|c.hi|` bounds the product's high word
#    and the residual is representable (see `EFT.two_hilo_sum_` for why
#    dominance is sufficient). Each call site records its dominance ratio.
#
# 2. **The mixed-precision tail.** In a Horner evaluation the high-degree terms
#    are scaled by high powers of a reduced argument; once the whole tail is
#    below `2^-26` of the result, evaluating it in plain `Float32` muladds
#    changes the result by less than `2^-26 * 2^-23 = 2^-49` relative — under
#    half an ulp48 — while replacing ~36-cycle pair steps with 4-cycle fmas.
#    Every kernel splits its coefficients at exactly that boundary and the
#    split is stated next to the coefficient slice.
#
# Measured effect of the two together, identical hardware and harness, 4096
# spread operands (docs/Checkpoint.md 2026-07-29 carries the full table):
# exp 172.7 -> 43.8 ns, log 145.0 -> 41.0 ns, sin 194.9 -> 39.2 ns,
# sinh 345.4 -> 49.5 ns. Accuracy ceilings in test/core/elementary.jl all hold;
# most tightened (expm1 4.63 -> 1.66 ulp48, tanh 3.50 -> 1.52).

"""
    Double32s.poly_step(c::Double32, z::Double32, p::Double32) -> Double32

One Horner iteration `c + z*p` under the **dominant-add contract**: the caller
guarantees `|z*p| <= |c|/2`. Exact leading product (`two_prod_`), both cross
terms fused into two `fma`s, FastTwoSum against the coefficient — valid under
the contract because `|c.hi|` then bounds the product's high word, which is the
`two_hilo_sum_` representability condition.

The `z.lo * p.lo` term is dropped; it is below `2^-48` of the step's value.
Unguarded: kernel-interior use only, with operands bounded by construction.
"""
@inline function poly_step(c::Double32, z::Double32, p::Double32)
    ph, pe = two_prod_(z.hi, p.hi)
    pe = fma(z.hi, p.lo, fma(z.lo, p.hi, pe))
    s, e = two_hilo_sum_(c.hi, ph)
    h, l = two_hilo_sum_(s, (c.lo + pe) + e)
    return raw(h, l)
end

"""
    Double32s.dmul(a::Double32, b::Double32) -> Double32

Unguarded value multiply for kernel interiors: `mul_dd_fast_` without the
range check, for call sites whose operands are bounded by construction (a
reduced argument times a polynomial value, a table entry times a correction).
Never for user-facing operands — that is what `*` and its guard are for.
"""
@inline function dmul(a::Double32, b::Double32)
    p, e = two_prod_(a.hi, b.hi)
    e = fma(a.hi, b.lo, fma(a.lo, b.hi, e))
    h, l = two_hilo_sum_(p, e)
    return raw(h, l)
end

"""
    Double32s.dsqr(a::Double32) -> Double32

Unguarded square. One fma cheaper than [`Double32s.dmul`](@ref) — the two cross
terms are equal, so `2a.hi * a.lo` replaces them — and it **keeps** the
`a.lo^2` term the generic fast multiply drops: every trigonometric and
hyperbolic kernel evaluates in `x^2`, so the square's last bit matters twice.
Measured on a dense `[-3.2, 3.2]` cos sweep: 1.78 ulp48 with the term dropped,
1.56 with it kept.
"""
@inline function dsqr(a::Double32)
    p, e = two_prod_(a.hi, a.hi)
    e = fma(a.hi + a.hi, a.lo, fma(a.lo, a.lo, e))
    h, l = two_hilo_sum_(p, e)
    return raw(h, l)
end
