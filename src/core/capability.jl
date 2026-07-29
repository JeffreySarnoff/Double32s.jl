# core/capability.jl — backend capability probes.
#
# Design section 5.9: backends or compilation flags may flush subnormal inputs
# or results to zero, and may decline to fuse a multiply-add. Either one
# invalidates the exactness results that the strict kernels' error bounds rest
# on. The design requires a capability test whose results qualify test
# expectations and documentation.
#
# The probe is written as independent *lanes*, each a pure function of a lane
# code and two `Float32` operands. That shape is what lets the identical body
# answer for the host (a short loop, allocation free) and for a device (a
# six-element broadcast, or a kernel launched by the KernelAbstractions
# extension) without a second implementation that could drift from the first.

"""
    Double32s.probe_lane(code::Int32, a::Float32, b::Float32) -> Bool

One lane of the backend capability probe. Pure `Float32`, no allocation, no
`throw`: safe to call from inside a device kernel.

| code | check | fails under |
|---|---|---|
| 1 | subnormal addition, `2⁻¹⁴⁹ + 2⁻¹⁴⁹ == 2⁻¹⁴⁸` | FTZ |
| 2 | subnormal multiplication result, `floatmin·½ == 2⁻¹²⁷` | FTZ |
| 3 | subnormal `fma` result | FTZ |
| 4 | [`EFT.two_prod_`](@ref) whose exact residual `2⁻¹⁴⁵` is subnormal | FTZ, unfused `fma` |
| 5 | sign of a subnormal survives negation | FTZ |
| 6 | [`EFT.two_prod_`](@ref) with the *normal* residual `2⁻⁴⁶` | unfused `fma` only |

Lane 6 is what separates the two failure modes. Lanes 1–5 all die under
flush-to-zero, and lane 4 additionally dies if `fma` is not a genuine
single-rounding fused operation; lane 6 stays in the normal range, so it
isolates fusion. A backend that passes 6 and fails 1–5 has FTZ; one that fails
both has no fused multiply-add and cannot host this package's error bounds at
all.

Operands come from the caller — never from literals inside the lane. With
constant operands the compiler folds the whole probe to `true` at compile time,
and a folded probe ignores `set_zero_subnormals` and device FTZ flags entirely.
This is not hypothetical; it is how the first version of this probe failed its
own test. Expected values, by contrast, *are* literals, and must be, as
`reinterpret` bit patterns: under FTZ even `ldexp(1.0f0, -145)` flushes to zero,
so a probe that computes its own expectation with the faulty arithmetic reports
the fault as health.

See [`Double32s.has_gradual_underflow`](@ref),
[`Double32s.has_fused_multiply_add`](@ref), and
[`Double32s.capability_report`](@ref) for the host and per-backend entry points.
"""
@inline function probe_lane(code::Int32, a::Float32, b::Float32)
    if code == Int32(1)
        return (a + b) === reinterpret(Float32, 0x00000002)      # 2^-148
    elseif code == Int32(2)
        return (a * b) === reinterpret(Float32, 0x00400000)      # 2^-127
    elseif code == Int32(3)
        return fma(a, b, -0.0f0) === reinterpret(Float32, 0x00400000)
    elseif code == Int32(4)
        # (1 + 2^-23)·1.25·2^-120 rounds its 25-bit significand, leaving
        # exactly 2^-145 behind — a subnormal residual from a normal product.
        p, e = two_prod_(a, b)
        return (e === reinterpret(Float32, 0x00000010)) & isfinite(p)
    elseif code == Int32(5)
        return signbit(-a) & !signbit(a)
    else
        # nextfloat(1f0)^2 = 1 + 2^-22 + 2^-46; the residual 2^-46 is normal, so
        # only a missing fusion can lose it.
        p, e = two_prod_(a, b)
        return (e === reinterpret(Float32, 0x28800000)) & isfinite(p)
    end
end

"""
    Double32s.PROBE_LANES

Number of lanes in the capability probe. See [`Double32s.probe_lane`](@ref).
"""
const PROBE_LANES = 6

"""
    Double32s.UNDERFLOW_LANES

The lane range that tests gradual underflow; the remaining lane tests `fma`
fusion. See [`Double32s.probe_lane`](@ref).
"""
const UNDERFLOW_LANES = 1:5

const _PROBE_CODE = Int32[1, 2, 3, 4, 5, 6]

const _PROBE_A = Float32[
    reinterpret(Float32, 0x00000001),                 # 2^-149, least subnormal
    reinterpret(Float32, 0x00800000),                 # floatmin(Float32)
    reinterpret(Float32, 0x00800000),
    reinterpret(Float32, 0x21800001),                 # nextfloat(1f0)·2^-60
    reinterpret(Float32, 0x00000001),
    reinterpret(Float32, 0x3f800001),                 # nextfloat(1f0)
]

const _PROBE_B = Float32[
    reinterpret(Float32, 0x00000001),
    0.5f0,
    0.5f0,
    reinterpret(Float32, 0x21a00000),                 # 1.25f0·2^-60
    0.0f0,                                            # unused by lane 5
    reinterpret(Float32, 0x3f800001),
]

"""
    Double32s.probe_operands() -> (codes, a, b)

The three host vectors that drive [`Double32s.probe_lane`](@ref), each of length
[`Double32s.PROBE_LANES`](@ref). Copy them to a device to run the probe there;
[`Double32s.probe_results`](@ref) does exactly that.
"""
probe_operands() = (_PROBE_CODE, _PROBE_A, _PROBE_B)

"""
    Double32s.probe_results() -> NTuple{PROBE_LANES,Bool}
    Double32s.probe_results(prototype::AbstractArray) -> NTuple{PROBE_LANES,Bool}
    Double32s.probe_results(backend::KernelAbstractions.Backend) -> NTuple{PROBE_LANES,Bool}

Run every lane of [`Double32s.probe_lane`](@ref) and return the outcomes as a
host tuple.

The no-argument method answers for the host thread and allocates nothing. The
`prototype` method answers for whatever device owns that array, by copying the
probe operands there, broadcasting, and copying six `Bool`s back; it needs no
GPU package — only an array type supporting `similar`, `copyto!`, and
broadcast, which every `GPUArrays`-based backend does. The
`KernelAbstractions.Backend` method is added by the KernelAbstractions
extension and launches a real kernel instead, which is what a backend without
an array type still has to offer.
"""
function probe_results()
    # `Val` so the tuple is built unrolled and inferrably — the host method is
    # asserted allocation free, and `ntuple(f, ::Int)` would not be.
    return ntuple(Val(PROBE_LANES)) do i
        probe_lane(@inbounds(_PROBE_CODE[i]),
                   @inbounds(_PROBE_A[i]),
                   @inbounds(_PROBE_B[i]))
    end
end

function probe_results(prototype::AbstractArray)
    codes = similar(prototype, Int32, PROBE_LANES)
    a     = similar(prototype, Float32, PROBE_LANES)
    b     = similar(prototype, Float32, PROBE_LANES)
    copyto!(codes, _PROBE_CODE)
    copyto!(a, _PROBE_A)
    copyto!(b, _PROBE_B)
    device_out = probe_lane.(codes, a, b)
    host_out = Vector{Bool}(undef, PROBE_LANES)
    copyto!(host_out, device_out)
    return ntuple(i -> @inbounds(host_out[i]), Val(PROBE_LANES))
end

"""
    Double32s.has_gradual_underflow() -> Bool
    Double32s.has_gradual_underflow(prototype::AbstractArray) -> Bool
    Double32s.has_gradual_underflow(backend::KernelAbstractions.Backend) -> Bool

Whether an execution environment honors gradual underflow, checked by running
the design section 5.9 probe list: subnormal addition, a subnormal
multiplication result, a subnormal `fma` result, an [`EFT.two_prod_`](@ref) case
whose exact residual is subnormal, and subnormal sign preservation.

Returns `false` under flush-to-zero — e.g. after `set_zero_subnormals(true)`, or
on a device compiled with FTZ — in which case the strict kernels still run but
their subnormal-range exactness claims (design sections 7.4, 9) do not hold, and
validation must widen its tolerances near the underflow boundary. Reach for
[`Double32s.capability_report`](@ref) when the distinction between FTZ and a
missing fused multiply-add matters, which it does when the answer is `false`.

The no-argument method allocates nothing, so the identical call is available on
a device. The result is deliberately **not cached**: `set_zero_subnormals` is a
per-thread runtime switch and CUDA's math mode is a per-task setting, so a
load-time answer can go stale. The probe is a handful of flops; a backend
should cache it per device at launch-configuration time.

# Examples
```jldoctest
julia> Double32s.has_gradual_underflow()
true

julia> Double32s.has_gradual_underflow(Float32[])   # host array, host answer
true
```
"""
function has_gradual_underflow()
    ok = true
    for i in UNDERFLOW_LANES
        ok &= probe_lane(@inbounds(_PROBE_CODE[i]),
                         @inbounds(_PROBE_A[i]),
                         @inbounds(_PROBE_B[i]))
    end
    return ok
end

has_gradual_underflow(prototype::AbstractArray) =
    _underflow_ok(probe_results(prototype))

"""
    Double32s.has_fused_multiply_add() -> Bool
    Double32s.has_fused_multiply_add(prototype::AbstractArray) -> Bool
    Double32s.has_fused_multiply_add(backend::KernelAbstractions.Backend) -> Bool

Whether `fma` is a genuine single-rounding fused multiply-add in this execution
environment.

This is the package's hardest requirement. Every error-free transformation on a
product goes through [`EFT.two_prod_`](@ref), whose residual is identically zero
when the multiply and the add round separately — so an unfused backend does not
produce a slightly worse `Double32`, it produces a `Float32` wearing two words.
The check stays in the normal range, so unlike
[`Double32s.has_gradual_underflow`](@ref) it is unaffected by flush-to-zero.

# Examples
```jldoctest
julia> Double32s.has_fused_multiply_add()
true
```
"""
function has_fused_multiply_add()
    i = PROBE_LANES
    return probe_lane(@inbounds(_PROBE_CODE[i]),
                      @inbounds(_PROBE_A[i]),
                      @inbounds(_PROBE_B[i]))
end

has_fused_multiply_add(prototype::AbstractArray) =
    @inbounds probe_results(prototype)[PROBE_LANES]

@inline _underflow_ok(r::NTuple{PROBE_LANES,Bool}) =
    r[1] & r[2] & r[3] & r[4] & r[5]

"""
    Double32s.capability_report(target = nothing) -> NamedTuple

Everything the capability probe can say about an execution target, in one call.

`target` is `nothing` for the host thread, an array whose backend should be
questioned, or — with `KernelAbstractions` loaded — a `Backend`. Loading `CUDA`
or `AMDGPU` adds vendor fields to the report for their backends.

Fields:

- `target` — a short description of what was probed;
- `gradual_underflow::Bool` — see [`Double32s.has_gradual_underflow`](@ref);
- `fused_multiply_add::Bool` — see [`Double32s.has_fused_multiply_add`](@ref);
- `lanes::NTuple{PROBE_LANES,Bool}` — the raw per-lane outcomes;
- `functional::Bool` — whether there was anything to probe. Always `true` for
  the host and for an array prototype; a `KernelAbstractions.Backend` with no
  device present reports `false`, with every other field falsified rather than
  guessed;
- `usable::Bool` — `fused_multiply_add`, without which no error bound in this
  package holds. Gradual underflow is *not* required for usability: it narrows
  the exactness claims near the underflow boundary rather than voiding them.

Vendor extensions append their own fields. The CUDA extension adds `device`,
`capability`, and `math_mode` — the last one matters, because
`CUDA.FAST_MATH` enables flush-to-zero and permits contraction, so a report
that says `gradual_underflow = false` on NVIDIA hardware is far more often a
task-local math mode than a property of the silicon.

# Examples
```jldoctest
julia> r = Double32s.capability_report();

julia> r.usable && r.fused_multiply_add
true
```
"""
function capability_report(target = nothing)
    lanes = target === nothing ? probe_results() : probe_results(target)
    return build_report(target, lanes)
end

"""
    Double32s.build_report(target, lanes) -> NamedTuple

Assemble a [`Double32s.capability_report`](@ref) from lane outcomes that have
already been collected. Backend extensions use this to report a device that
could not be probed at all, rather than letting the probe throw out of a
diagnostic call.
"""
function build_report(target, lanes::NTuple{PROBE_LANES,Bool})
    base = (target = _describe(target),
            gradual_underflow = _underflow_ok(lanes),
            fused_multiply_add = lanes[PROBE_LANES],
            lanes = lanes,
            functional = true,
            usable = lanes[PROBE_LANES])
    return merge(base, _vendor_fields(target))
end

"""
    Double32s.NO_LANES

All-`false` lane outcomes, for reporting a target that could not be probed.
"""
const NO_LANES = ntuple(_ -> false, Val(PROBE_LANES))

# Extension hooks. `_describe` names the probed target; `_vendor_fields` lets a
# backend extension append to the report without redefining it.
_describe(::Nothing) = "host thread"
_describe(x::AbstractArray) = summary(x)
_describe(x) = string(x)

_vendor_fields(::Any) = NamedTuple()

"""
    Double32s.allocate_split(backend, dims...) -> SplitDouble32Array

Allocate a [`Double32s.SplitDouble32Array`](@ref) of the given shape on a
`KernelAbstractions` backend, with both word arrays uninitialized.

Defined by the KernelAbstractions extension; `using KernelAbstractions` (or any
package that loads it, such as `CUDA` or `AMDGPU`) brings it into existence.
"""
function allocate_split end
