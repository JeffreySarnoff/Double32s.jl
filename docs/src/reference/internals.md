# [Internals](@id internals)

```@meta
CurrentModule = Double32s
```

Not public API. These are documented because the docstrings carry the proofs,
measurements, and defect histories that make the kernels auditable. Read them
before modifying the corresponding code: several record failure modes that
were found the expensive way.

## The three-layer operation structure

Every operation follows design section 8.0: a branch-free tuple kernel
(layer 1), a policy method with the single output check (layer 2), a
one-line Base method (layer 3).

```@docs
Double32s.add_dd_
Double32s.add_dd_fast_
Double32s.sub_dd_
Double32s.mul_dd_
Double32s.mul_dd_fast_
Double32s.div_dd_
Double32s.div_dd_refined_
Double32s.inv_dd_
Double32s.sqrt_dd_
Double32s.fma_dd_
```

## Guarded paths

The cold paths, each with its trigger-and-safety proof:

```@docs
Double32s.mul_guarded
Double32s.div_guarded
Double32s.divide
Double32s.reciprocal
Double32s.compensated_fma
Double32s.square_root
```

## Construction and conversion internals

```@docs
Double32s.Raw
Double32s.raw
Double32s.to_odd_hi
Double32s.iscanonical
Double32s.canonical_nonfinite
Double32s.reference
Double32s.zero_sign
Double32s.scale2
```
