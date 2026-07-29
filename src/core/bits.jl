# core/bits.jl — bit-level helpers.
#
# Design section 7.6. Kept narrow on purpose: Base predicates are preferred
# unless generated-code inspection shows a benefit.

"""
    Double32s.lsb_exponent(z::Float32) -> Int

Binary exponent of the lowest set significand bit of a nonzero `z`, i.e. the
`k` with `z == m * 2^k` for an odd integer `m`.

Returns `typemin(Int)` for zero, so that "no constraint" compares correctly
against any bound. Non-finite `z` is not meaningful here and is not checked.

This is the quantity the `Float64`-exactness condition of design section 4.2 is
stated in: a canonical pair converts exactly iff `lo == 0` or
`lsb_exponent(lo) >= exponent(hi + lo) - 52`.

# Examples
```jldoctest
julia> Double32s.lsb_exponent(1.0f0), Double32s.lsb_exponent(1.5f0)
(0, -1)

julia> Double32s.lsb_exponent(reinterpret(Float32, 0x00000001))   # 2^-149
-149
```
"""
@inline function lsb_exponent(z::Float32)
    u = reinterpret(UInt32, z)
    m = u & 0x007fffff
    e = Int(u >> 23) & 0xff
    iszero(z) && return typemin(Int)
    return e == 0 ? -149 + trailing_zeros(m) :
                    (e - 150) + trailing_zeros(m | 0x00800000)
end

"""
    Double32s.is_power_of_two(z::Float32) -> Bool

Whether `abs(z)` is an exact power of two. Finite nonzero `z` only; the
significand field being all zero is the test.
"""
@inline is_power_of_two(z::Float32) = iszero(reinterpret(UInt32, z) & 0x007fffff)

"""
    Double32s.raw_exponent(z::Float32) -> Int

Binary exponent of `z`, agreeing with `Base.exponent` on every finite nonzero
argument — and **never throwing**.

`Base.exponent` raises `DomainError` for `0`, `Inf`, and `NaN`. Every call site
in this package excludes those first, so the throw is unreachable; the compiler
cannot prove it, and the exception path survives into the generated code. Design
section 14 forbids exactly that in device code, and it was measured rather than
assumed: before this function existed, `@code_llvm` for `/`, `inv`, `norm2`, and
even `sqrt_or_nan` — the operation documented as *the device-safe one* — each
contained calls to `throw1`/`throw2` from `Base.exponent`. After it, `+`, `-`,
`*`, `/`, `inv`, `fma`, and `sqrt_or_nan` contain **no throw reference, no
`alloca`, and no `double`**, which `test/gates.jl` now asserts.

**What it bought, measured on both sides.** On the host: nothing — an A/B of `/`
and `sqrt_or_nan` against the `Base.exponent` versions came out at 0.98× and
0.93×, i.e. inside the noise. The gain is on the device, where the dead
exception path still costs registers and spill space:

| kernel | registers | local memory |
|---|---|---|
| `/` with `Base.exponent` | 35 | 96 B |
| `/` with `raw_exponent` | **33** | **88 B** |
| `sqrt_or_nan` with `Base.exponent` + `Base.sqrt` | 30 | 56 B |
| `sqrt_or_nan` with `raw_exponent` + `sqrt_nonneg` | **26** | **32 B** |

(RTX 5090, `CUDA.registers` on an elementwise kernel.) Local memory is spill
space, so the 43% cut on `sqrt_or_nan` is the number that matters. The other
reason to do this is categorical rather than quantitative: some backends decline
to compile a kernel containing exception machinery at all.

Out-of-domain inputs return a value rather than raising: `0` gives
`typemin(Int) >> 1`, and `Inf`/`NaN` give `typemax(Int) >> 1`. Both are chosen
so that arithmetic on the result (the `+`, `-`, and `max` that call sites do
with it) cannot overflow, and both are far outside the range any guard band
tests. A caller that has not already excluded these gets a meaningless scale
factor rather than a fault — which is why every call site still excludes them,
and says so.

# Examples
```jldoctest
julia> Double32s.raw_exponent(1.0f0), Double32s.raw_exponent(0.5f0)
(0, -1)

julia> Double32s.raw_exponent(reinterpret(Float32, 0x00000001))   # 2^-149
-149

julia> Double32s.raw_exponent(0.0f0) == typemin(Int) >> 1
true
```
"""
@inline function raw_exponent(z::Float32)
    u = reinterpret(UInt32, z)
    e = Int((u >> 23) & 0xff)
    m = u & 0x007fffff
    e == 0xff && return typemax(Int) >> 1                    # Inf, NaN
    e != 0 && return e - 127                                 # normal
    iszero(m) && return typemin(Int) >> 1                    # ±0
    return -149 + (31 - leading_zeros(m))                    # subnormal
end

"""
    Double32s.sqrt_nonneg(z::Float32) -> Float32

`sqrt` with no domain check: `llvm.sqrt.f32` directly, returning `NaN` for a
negative argument instead of throwing.

`Base.sqrt(::Float32)` tests the sign and raises `DomainError`, and that branch
is unreachable at the one call site here — [`Double32s.sqrt_dd_`](@ref) is
reached only after the layer-2 method has excluded negatives — but survives into
the generated code, dragging `throw_complex_domainerror` into every kernel that
takes a square root. Same reasoning as [`Double32s.raw_exponent`](@ref).
"""
@inline sqrt_nonneg(z::T) where {T<:Base.IEEEFloat} = Base.sqrt_llvm(z)
