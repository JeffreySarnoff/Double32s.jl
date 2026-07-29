# core/show.jl — display and parsing.
#
# Design section 11.5. Two-argument `show` must produce output that parses back
# to the same value; the three-argument `show(io, ::MIME"text/plain", x)`
# renders a decimal with a per-value round-trip guarantee: the printed string
# is the shortest one whose `parse(Double32, ·)` is `isequal` to `x`, checked
# before printing rather than promised by analysis. Parsing routes through
# `BigFloat` on the host and never appears in device code.

"""
    show(io::IO, x::Double32)

Print the pair form, `Double32(hi, lo)`, which parses back to a value `isequal`
to `x`.

**A zero low word always prints as `0.0f0`,** whatever sign bit it carries.
Design section 5.2 invariant 8 makes the sign of a zero `lo` non-information:
`-` flips it unconditionally, so `-one(Double32)` and `Double32(-1.0f0)` are
bit-distinct pairs holding one value, and `isequal` deliberately does not
distinguish them.

Printing the raw sign would make `repr` a function of the *bit pattern* rather
than of the *value*, giving `isequal(a, b)` while `repr(a) != repr(b)` — the
same shape of defect as the section 8.10 comparison bug. Measured before this
normalization: **18 disagreements over the 36 `isequal` pairs of the signed-zero
surface; after, 0.**

The sign of a zero *high* word is untouched, because there it is the value's own
sign: `NEG_ZERO` still prints `Double32(-0.0f0, 0.0f0)` and still parses back to
itself.

Nothing diagnostic is hidden — a `-0.0` low word is never a defect — and the
fields stay reachable for anyone inspecting representation rather than value.

# Examples
```jldoctest
julia> -one(Double32)                      # low word is -0.0f0 internally
-1.0

julia> repr(-one(Double32)) == repr(Double32(-1.0f0))
true

julia> Double32s.NEG_ZERO                  # a zero *high* word keeps its sign
-0.0
```
"""
function Base.show(io::IO, x::Double32)
    print(io, "Double32(")
    show(io, x.hi)
    print(io, ", ")
    # `iszero` and not `== 0.0f0 && !signbit`: this must fire for +0.0 too, so
    # that the printed form depends on nothing but the represented value.
    show(io, iszero(x.lo) ? 0.0f0 : x.lo)
    print(io, ")")
    return nothing
end

# Working precision for parsing and decimal rendering. A canonical pair spans
# at most `exponent(floatmax(Float32)) - exponent(2f0^-149) + 1 = 277` bits
# from the high word's leading bit to a subnormal low word's trailing bit;
# 384 holds every finite pair value exactly, with margin.
const _DECIMAL_PRECISION = 384

"""
    tryparse(Double32, s::AbstractString) -> Union{Double32, Nothing}

Decimal (or MPFR-accepted) string to the nearest canonical pair, via a
384-bit `BigFloat` intermediate — host-only, per design section 11.5. Returns
`nothing` when `s` is not a number. `"NaN"`, `"Inf"`, `"-Inf"`, and signed
zeros parse to their canonical pairs.

The intermediate is exact for every string this package prints. For arbitrary
input the composite rounding is correct unless the written value lies within
`2^-384` relative of a tie of the pair lattice without being that tie — below
any decimal a user can plausibly produce by accident.
"""
function Base.tryparse(::Type{Double32}, s::AbstractString)
    b = setprecision(BigFloat, _DECIMAL_PRECISION) do
        tryparse(BigFloat, s)
    end
    b === nothing && return nothing
    return setprecision(() -> Double32(b), BigFloat, _DECIMAL_PRECISION)
end

"""
    parse(Double32, s::AbstractString) -> Double32

Throwing form of [`tryparse`](@ref tryparse(::Type{Double32}, ::AbstractString)).
"""
function Base.parse(::Type{Double32}, s::AbstractString)
    x = tryparse(Double32, s)
    x === nothing &&
        throw(ArgumentError("cannot parse $(repr(s)) as Double32"))
    return x
end

"""
    Double32s.decimal(x::Double32) -> String

The shortest decimal string `s` (by significant-digit count, `%.Ng`
formatting) with `isequal(parse(Double32, s), x)` — verified by parsing back
before returning, not promised by digit-count analysis. Wide-gap pairs
(design section 23.11) genuinely need up to ~90 significant digits: a high
word near `floatmax(Float32)` above a subnormal low word pins ~277 bits.

The 97-digit fallback is unreachable — 97 digits over-determine 384 bits —
and asserted rather than silently trusted.
"""
function decimal(x::Double32)
    isnan(x.hi) && return "NaN"
    isinf(x.hi) && return signbit(x.hi) ? "-Inf" : "Inf"
    if iszero(x.hi) && iszero(x.lo)
        return signbit(x.hi) ? "-0.0" : "0.0"
    end
    return setprecision(BigFloat, _DECIMAL_PRECISION) do
        v = big(x.hi) + big(x.lo)
        for k in 1:96
            s = Printf.format(Printf.Format("%.$(k)g"), v)
            isequal(tryparse(Double32, s), x) && return _floatlike(s)
        end
        s = Printf.format(Printf.Format("%.97g"), v)
        @assert isequal(tryparse(Double32, s), x)
        return _floatlike(s)
    end
end

# `%g` drops the trailing ".0" from integer-valued results; a float type
# should not print as `1`. Round-trip is unaffected: parse("1.0") == parse("1").
_floatlike(s::String) = occursin(r"[.eE]", s) ? s : s * ".0"

"""
    show(io::IO, ::MIME"text/plain", x::Double32)

Decimal rendering with a round-trip promise: the printed string parses back —
`isequal(parse(Double32, string), x)` — because [`Double32s.decimal`](@ref)
verifies exactly that before returning. The pair form remains the two-argument
`show` and `repr`, following `BigFloat`'s precedent of a bare decimal at the
REPL.

# Examples
```jldoctest
julia> Double32(1.0f0) / 3
0.333333333333333

julia> parse(Double32, "0.33333333333333331483") === Double32(1.0f0) / 3
true

julia> repr(Double32(1.0f0) / 3)
"Double32(0.33333334f0, -9.934108f-9)"
```
"""
Base.show(io::IO, ::MIME"text/plain", x::Double32) = print(io, decimal(x))
