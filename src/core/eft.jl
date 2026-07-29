# core/eft.jl — the error-free transformation layer.
#
# Design section 7. A submodule so that the EFT names can be `public` in their
# own namespace without any of them reaching `using Double32s`.

"""
    Double32s.EFT

Error-free transformations on IEEE floats. Reached as `Double32s.EFT.two_sum`
and friends; nothing here enters the namespace of `using Double32s`.

Trailing-underscore names are the *unguarded* kernels: branch free, no
finiteness check, and the caller owns the preconditions. The names without the
underscore add the guard.
"""
module EFT

public two_sum, two_sum_, two_hilo_sum, two_hilo_sum_, two_hilo_sum_checked
public two_diff, two_diff_, two_prod, two_prod_, nonfinite_pair

"""
    nonfinite_pair(s::T) -> Tuple{T,T}

Canonical non-finite EFT result: the non-finite word and a positive zero.
"""
@inline nonfinite_pair(s::T) where {T<:Base.IEEEFloat} = (s, zero(T))

"""
    two_sum_(a::T, b::T) -> (s::T, e::T)

Knuth's TwoSum. `s + e == a + b` exactly whenever `s` is finite, for any `a`
and `b`. Six add/sub, no branch, no precondition.
"""
@inline function two_sum_(a::T, b::T) where {T<:Base.IEEEFloat}
    s = a + b
    v = s - a
    e = (a - (s - v)) + (b - v)
    return s, e
end

"""
    two_sum(a::T, b::T) -> (s::T, e::T)

Guarded [`two_sum_`](@ref): returns `(s, zero(T))` when `s` is not finite.
"""
@inline function two_sum(a::T, b::T) where {T<:Base.IEEEFloat}
    s = a + b
    isfinite(s) || return nonfinite_pair(s)
    return two_sum_(a, b)
end

"""
    two_hilo_sum_(a::T, b::T) -> (s::T, e::T)

Dekker's FastTwoSum. Three add/sub instead of six.

**Exactness condition.** `s + e == a + b` exactly, for finite `s`, iff
`fl(a + b) - a` is itself representable. `abs(a) >= abs(b)` is the familiar
*sufficient* condition, **not** a necessary one; `a == 0`, and more generally
"`a + b` is exact", are further sufficient cases that the pair kernels of
design section 8 actually rely on.

Stating the sufficient condition as though it were the requirement is what let
the section 8.1 justification stand unchallenged for so long. The uses inside
`add_dd_`/`sub_dd_` are covered end-to-end by the machine-checked
AccurateDWPlusDW bound — see [`Double32s.add_dd_`](@ref) — and every other call
site pairs this with a magnitude argument (`|correction| ≤ u|leading|`) in a
comment. The condition is *not* checked here. Use
[`two_hilo_sum_checked`](@ref) in host tests.
"""
@inline function two_hilo_sum_(a::T, b::T) where {T<:Base.IEEEFloat}
    s = a + b
    e = b - (s - a)
    return s, e
end

"""
    two_hilo_sum(a::T, b::T) -> (s::T, e::T)

Guarded [`two_hilo_sum_`](@ref).
"""
@inline function two_hilo_sum(a::T, b::T) where {T<:Base.IEEEFloat}
    s = a + b
    isfinite(s) || return nonfinite_pair(s)
    return two_hilo_sum_(a, b)
end

"""
    two_hilo_sum_checked(a::T, b::T) -> (s::T, e::T)

Differentially-checked FastTwoSum for host code and tests. Not for kernels.

Checks the property that actually matters — agreement with [`two_sum_`](@ref) —
rather than the merely sufficient `abs(a) >= abs(b)`, which the pair kernels
violate legitimately under cancellation. `@boundscheck` makes the check elidable
under `@inbounds` at the call site.
"""
@inline function two_hilo_sum_checked(a::T, b::T) where {T<:Base.IEEEFloat}
    fast = two_hilo_sum_(a, b)
    @boundscheck if isfinite(a) && isfinite(b)
        slow = two_sum_(a, b)
        (fast[1] === slow[1] && fast[2] === slow[2]) || throw(ArgumentError(
            lazy"FastTwoSum residual is not representable for a=$a, b=$b: \
                 got $fast, TwoSum gives $slow"))
    end
    return fast
end

"""
    two_diff_(a::T, b::T) -> (s::T, e::T)

Knuth's TwoDiff. `s + e == a - b` exactly whenever `s` is finite.

Written out symmetrically rather than as `two_sum_(a, -b)`: the two are
algebraically identical and disagree on the sign of a zero result.
"""
@inline function two_diff_(a::T, b::T) where {T<:Base.IEEEFloat}
    s  = a - b
    bb = s - a
    e  = (a - (s - bb)) - (b + bb)
    return s, e
end

"""
    two_diff(a::T, b::T) -> (s::T, e::T)

Guarded [`two_diff_`](@ref).
"""
@inline function two_diff(a::T, b::T) where {T<:Base.IEEEFloat}
    s = a - b
    isfinite(s) || return nonfinite_pair(s)
    return two_diff_(a, b)
end

"""
    two_prod_(a::T, b::T) -> (p::T, e::T)

FMA-based TwoProduct. `p + e == a * b` exactly, provided `p` is finite, the
exact residual is representable (no underflow below the least subnormal), and
`fma` is a genuine single-rounding fused multiply-add.

!!! warning "Silent residual underflow"
    When the exact residual falls below `2f0^-149`, `e` comes back a **signed
    zero**, not a non-finite value, so no `isfinite` check anywhere can see the
    loss. The trigger is the residual's magnitude, not whether `p` is subnormal:
    `two_prod_(floatmin(Float32), 0.5f0)` is exact while
    `two_prod_(3*2f0^-149, 0.5f0)` loses `7.0e-46`. A nonzero residual of a
    48-bit product is `≥ 2^(exponent(p) - 47)`, so exactness is guaranteed for
    `exponent(p) ≥ -101` (given gradual underflow) and impossible to observe
    failing above it.

    Every consumer in this package neutralizes the hazard explicitly:

    | consumer | resolution |
    |---|---|
    | `*` ([`Double32s.mul_guarded`](@ref)) | output range check `MUL_TINY`; scaled retry |
    | `/`, `inv` | operands scaled to `SCALE_FLOOR` / `INV_SCALE_FLOOR` before the kernel |
    | `sqrt` | operand scaled to an even exponent near zero |
    | `fma`/`muladd` ([`Double32s.fma_dd_`](@ref)) | bound stated with validity window `M ≥ 2⁻⁷⁷`; taper documented |
    | `compensated_dot` | exact only for products `≥ 2⁻¹⁰¹`; documented on the function |

    All of this presumes **gradual underflow**. A backend that flushes
    subnormals moves every boundary above; gate any accuracy claim on
    [`Double32s.has_gradual_underflow`](@ref), which is written in pure `Float32` so the
    same probe can run inside a device kernel (design section 5.9).

**`fma`, never `muladd`.** `Base.muladd` lowers to `llvm.fmuladd`, which
*permits* but does not require fusion. An EFT built on `muladd` is silently
wrong on whichever backend declines to fuse.
"""
@inline function two_prod_(a::T, b::T) where {T<:Base.IEEEFloat}
    p = a * b
    e = fma(a, b, -p)
    return p, e
end

"""
    two_prod(a::T, b::T) -> (p::T, e::T)

Guarded [`two_prod_`](@ref).
"""
@inline function two_prod(a::T, b::T) where {T<:Base.IEEEFloat}
    p = a * b
    isfinite(p) || return nonfinite_pair(p)
    return (p, fma(a, b, -p))
end

end # module EFT

using .EFT: two_sum, two_sum_, two_hilo_sum, two_hilo_sum_, two_hilo_sum_checked,
            two_diff, two_diff_, two_prod, two_prod_, nonfinite_pair
