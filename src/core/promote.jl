# core/promote.jl — the promotion lattice.
#
# Design section 11.3.
#
# Four rules govern writing these correctly:
#
# 1. Give one ordering only. `promote_rule` is consulted symmetrically.
# 2. Never define `promote_rule(::Type{Double32}, ::Type{<:Real})`. The abstract
#    bound would capture types the package knows nothing about, including future
#    ones. `<:Integer` is the widest safe bound, and it is safe because the
#    answer really is uniform across integer types.
# 3. `Complex` needs nothing; Base derives it from the real rules.
# 4. `BFloat16` and `DoubleFloats.DoubleFloat` belong in extensions.

Base.promote_rule(::Type{Double32}, ::Type{Float16})   = Double32
Base.promote_rule(::Type{Double32}, ::Type{Float32})   = Double32
Base.promote_rule(::Type{Double32}, ::Type{<:Integer}) = Double32

# Float64 wins: it has both a wider significand (53 bits against a nominal 48)
# and a wider exponent range, so it loses less of either operand than Double32
# would lose of the Float64 one. This is a least-loss argument, not a lossless
# one — the Double32 -> Float64 conversion is itself inexact for wide-gap pairs
# (section 4.2), which is why `==` and `<` against Float64 are implemented
# exactly rather than by promotion (section 8.10).
Base.promote_rule(::Type{Double32}, ::Type{Float64})  = Float64
Base.promote_rule(::Type{Double32}, ::Type{BigFloat}) = BigFloat
