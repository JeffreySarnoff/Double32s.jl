# core/show.jl — display.
#
# Design section 11.5. Two-argument `show` must produce output that parses back
# to the same value.
#
# No three-argument `show(io, ::MIME"text/plain", x)` yet: Base's fallback
# routes it to the two-argument method, which is the desired behavior until a
# decimal rendering with a round-trip promise exists.

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
Double32(-1.0f0, 0.0f0)

julia> repr(-one(Double32)) == repr(Double32(-1.0f0))
true

julia> Double32s.NEG_ZERO                  # a zero *high* word keeps its sign
Double32(-0.0f0, 0.0f0)
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
