# Double32sDoubleFloatsExt — conversions to and from DoubleFloats.jl.
#
# Design section 11.3 rule 4 and the §11.3 table row "DoubleFloats.Double32:
# extension-mediated conversion, no mandatory dependency". Every signature
# mentions `Double32s.Double32`, so nothing here is piracy.
#
# None of DoubleFloats' own promotion rules fire for our scalar — its widest
# bounds are `T<:IEEEFloat` (a `Union` of the three concrete formats) and the
# exact type `::Type{Real}`, neither of which `Double32` matches — so the two
# rules below are the only connection between the lattices, and each has one
# ordering only (design §11.3 rule 1). Verified: with both packages loaded,
# `promote_rule` has no method in the opposite direction for either pair.
#
# `DoubleFloat{Float32}` deliberately gets **no** promotion rule, per the
# design table's "extension-mediated conversion" row: the two types carry the
# same nominal precision, so either direction would silently privilege one
# package's semantics in mixed expressions. Convert explicitly; a mixed
# operation without a conversion is a promotion error, and should be.
#
# Note `DoubleFloats.Double32 === DoubleFloat{Float32}` is a *different type*
# with the same name as ours; everything below is written against
# `DoubleFloat{T}` to keep that unambiguous.

module Double32sDoubleFloatsExt

using Double32s
using Double32s: raw
using DoubleFloats: DoubleFloats, DoubleFloat

# Design §11.3 table: "wider double-word type → wider type", in both senses —
# `Double64` is strictly wider than `Double32`, which is strictly wider than
# `DoubleFloat{Float16}` (nominal 48 bits and `Float32` range against nominal
# 22 bits and `Float16` range).
Base.promote_rule(::Type{Double32}, ::Type{DoubleFloat{Float64}}) =
    DoubleFloat{Float64}
Base.promote_rule(::Type{Double32}, ::Type{DoubleFloat{Float16}}) = Double32

# ------------------------------------------------------------- in -----------

"""
    Double32(x::DoubleFloat{Float32}) -> Double32

Same word type on both sides; the canonicalizing two-word constructor absorbs
any normalization difference between the two packages' invariants.
"""
Double32s.Double32(x::DoubleFloat{Float32}) = Double32(x.hi, x.lo)

"""
    Double32(x::DoubleFloat{Float16}) -> Double32

Exact: each `Float16` word widens to `Float32` without rounding, and the
canonicalizing constructor renormalizes the pair at the wider precision.
"""
Double32s.Double32(x::DoubleFloat{Float16}) =
    Double32(Float32(x.hi), Float32(x.lo))

"""
    Double32(x::DoubleFloat{Float64}) -> Double32

Correctly rounded to the nearest canonical pair, through an exact `BigFloat`
intermediate. The precision is not the default 256: a `Double64` pair may span
up to about `1024 + 1074 + 53` bits between its high word's leading bit and
its low word's trailing bit, and a 256-bit intermediate would re-round the
value before the pair extraction sees it — the same double-rounding defect
class as the narrowing scalar conversions (`Double32s.to_odd_hi`). 2200 bits
holds every finite `Double64` exactly.
"""
function Double32s.Double32(x::DoubleFloat{Float64})
    # Zero first: `big(-0.0) + big(0.0)` is `+0.0`, so the BigFloat route
    # erases the sign of a zero. The high word's sign is the value's sign
    # (design §5.2), exactly as in `Float64(::Double32)`'s zero guard.
    (iszero(x.hi) && iszero(x.lo)) && return Double32(Float32(x.hi))
    return setprecision(BigFloat, 2200) do
        Double32(big(x.hi) + big(x.lo))            # sum exact at this precision
    end
end

# ------------------------------------------------------------- out ----------

"""
    DoubleFloat{Float64}(x::Double32) -> Double64

Exact: both `Float32` words are `Float64` values, and DoubleFloats'
canonicalizing constructor renormalizes with an error-free `two_sum`. The one
case that constructor would damage is a signed zero — IEEE `-0.0 + 0.0` is
`+0.0`, so `two_sum` erases the high word's zero sign — hence the zero guard,
which routes through DoubleFloats' non-normalizing inner constructor.
"""
function DoubleFloats.DoubleFloat{Float64}(x::Double32)
    iszero(x.hi) && return DoubleFloat{Float64}(Float64(x.hi), 0.0)
    return DoubleFloats.DoubleFloat(Float64(x.hi), Float64(x.lo))
end

"""
    DoubleFloat{Float32}(x::Double32) -> DoubleFloats.Double32

Exact and word-for-word: our canonical invariant (`hi == fl(hi + lo)`,
`|lo| ≤ ½ulp(hi)`) is DoubleFloats' own `two_sum`-stable form, so the pair
passes through the non-normalizing inner constructor unchanged — which also
preserves a negative zero high word, where the normalizing form would not.
"""
DoubleFloats.DoubleFloat{Float32}(x::Double32) =
    DoubleFloat{Float32}(x.hi, x.lo)

end # module
