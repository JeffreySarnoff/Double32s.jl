# Double32sBFloat16sExt — promotion and conversion for BFloat16.
#
# Design section 11.3 rule 4: a `promote_rule` mentioning `BFloat16` requires
# `BFloat16s` to be loaded, so it lives here rather than making an optional
# dependency mandatory. Not piracy: every signature mentions `Double32`.

module Double32sBFloat16sExt

using Double32s
using Double32s: raw, to_odd_hi
using BFloat16s: BFloat16s, BFloat16

# Design §11.3 table: BFloat16 promotes to Double32. One ordering only
# (rule 1); `promote_rule` is consulted symmetrically.
Base.promote_rule(::Type{Double32}, ::Type{BFloat16}) = Double32

"""
    Double32(x::BFloat16) -> Double32

Exact: every `BFloat16` is a `Float32` (same exponent field, truncated
significand), so the widening needs no residual.
"""
Double32s.Double32(x::BFloat16) = Double32(Float32(x))

"""
    BFloat16(x::Double32) -> BFloat16

Correctly rounded to nearest. Goes through [`Double32s.to_odd_hi`](@ref) for
the same reason `Float16(::Double32)` does: `BFloat16(x.hi)` alone misrounds
every pair whose high word is a `BFloat16` tie broken by the low word.
`BFloat16s`' `Float32 → BFloat16` conversion is round-to-nearest-even (LLVM
`fptrunc` or the explicit bit adjustment, both RNE — verified in BFloat16s
0.6.1 source), so round-to-odd at 24 bits composes exactly for the 8-bit
target.
"""
BFloat16s.BFloat16(x::Double32) = BFloat16(to_odd_hi(x))

end # module
