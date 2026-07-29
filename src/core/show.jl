# core/show.jl — display and parsing.
#
# Design section 11.5. There is exactly **one** rendering: the shortest decimal
# that parses back to the same value. It is produced by the two-argument
# `show`, so every display path in Base agrees — the REPL, array elements,
# tuples, `Dict` values, `string`, and `$` interpolation.
#
# The reconstruction contract is `parse`, not `eval`. This is `BigFloat`'s
# arrangement: `repr(big"0.1")` evaluates to a `Float64`, while
# `parse(BigFloat, repr(b))` returns `b` exactly. A bare decimal cannot do
# better for any type, because `2.23606797749979` is `Float64` literal syntax
# and no method on `Double32` changes what the parser makes of it.
#
# The earlier arrangement — pair form on two-argument `show`, decimal on a
# three-argument method — is recorded here because it was deliberate and was
# reversed for a reason. It made the rendering depend on the container:
# `[x]` printed a decimal (Base's array code tries the three-argument form
# first), while `(x,)`, `string(x)`, and `"$x"` printed
# `Double32(2.236068f0, -3.283041f-8)`, since `Tuple` has no three-argument
# method and the generic fallback discards the MIME. It also broke numeric
# column alignment: `Base.alignment(::IO, ::Real)` measures
# `sprint(show, x)` and splits it at the first `.`, which under the pair form
# landed inside the high word and described text that was never printed.
#
# Parsing routes through `BigFloat` on the host and never appears in device
# code.

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

The shortest decimal string `s` with `isequal(parse(Double32, s), x)` —
verified by parsing back before returning, not promised by digit-count
analysis. This is what `show`, `repr`, `string`, and interpolation all print.

A short decimal is not always available, and the length is not slack. Wide-gap
pairs (design section 23.11) need up to ~90 significant digits, because a high
word near `floatmax(Float32)` above a subnormal low word pins ~277 bits. Even
a pair whose low word is zero can be long: `Double32(1.0f-7)` needs 39 digits,
since the pair lattice is fine enough that no shorter decimal selects that
exact value over its neighbors.

The 97-digit fallback is unreachable — 97 digits over-determine 384 bits —
and is asserted rather than silently trusted.

The sign of a zero low word never appears, because the rendering is computed
from the represented value. Design section 5.2 invariant 8 makes that sign
non-information: `-` flips it unconditionally, so `-one(Double32)` and
`Double32(-1.0f0)` are bit-distinct pairs holding one value. A rendering that
exposed the sign would make `repr` a function of the bit pattern rather than
of the value, giving `isequal(a, b)` while `repr(a) != repr(b)`. Measured
against the signed-zero surface before the value-based rendering existed:
**18 disagreements over its 36 `isequal` pairs; now 0.** The sign of a zero
*high* word does appear, because there it is the value's own sign.

Use [`Double32s.HILO`](@ref) to inspect the representation.

# Examples
```jldoctest
julia> Double32s.decimal(sqrt(Double32(5)))
"2.23606797749979"

julia> Double32s.HILO(sqrt(Double32(5)))         # the representation
(2.236068f0, -3.283041f-8)
```
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
            s = _render(v, k)
            isequal(tryparse(Double32, s), x) && return s
        end
        s = _render(v, 97)
        @assert isequal(tryparse(Double32, s), x)
        return s
    end
end

# Render `v` with `k` significant digits, choosing positional or exponential
# form from the magnitude alone.
#
# `%g` cannot be used directly: it switches to exponential as soon as the
# decimal exponent reaches the digit count, so the shortest round-tripping
# rendering of ten would be `1e+01`. Deciding by magnitude instead follows
# Julia's own convention for `Float64`, and keeps a visible fractional part in
# both forms so the result reads as a float rather than an integer.
function _render(v::BigFloat, k::Int)
    normalized = Printf.format(Printf.Format("%.$(k - 1)e"), v)
    # `Base.split`: the package's own `split` (arrays) shadows the name here.
    mantissa, exponent = Base.split(normalized, 'e')
    e = parse(Int, exponent)
    if -5 < e < 16
        return Printf.format(Printf.Format("%.$(max(k - 1 - e, 1))f"), v)
    end
    occursin('.', mantissa) || (mantissa *= ".0")
    return string(mantissa, "e", e)          # `e+07` would read as padding
end

"""
    show(io::IO, x::Double32)

Print [`Double32s.decimal`](@ref): the shortest decimal that reconstructs `x`
under `parse(Double32, ·)`, verified for this value before printing.

This is the only rendering the type defines, so every Base display path agrees
— the REPL, array elements, tuples, `Dict` values, `string`, and `\$`
interpolation.

!!! note "The round trip is `parse`, not `eval`"
    `eval(Meta.parse(repr(x)))` yields a `Float64`, because a bare decimal is
    `Float64` literal syntax. `parse(Double32, repr(x))` returns `x`. This is
    exactly `BigFloat`'s arrangement, and it is not avoidable for any type
    that prints a bare decimal.

# Examples
```jldoctest
julia> x = sqrt(Double32(5))
2.23606797749979

julia> (x,)                                # containers agree with the scalar
(2.23606797749979,)

julia> parse(Double32, repr(x)) === x      # exact reconstruction
true

julia> -one(Double32)                      # low word is -0.0f0 internally
-1.0

julia> repr(-one(Double32)) == repr(Double32(-1.0f0))
true

julia> Double32s.NEG_ZERO                  # a zero *high* word keeps its sign
-0.0
```
"""
Base.show(io::IO, x::Double32) = print(io, decimal(x))
