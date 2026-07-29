# ext/Double32sAMDGPUExt.jl — AMD specifics.
#
# Design section 16, roadmap Phase 4. The same division of labor as the CUDA
# extension: `ROCArray{Double32}` works with no package code, `Adapt` and
# `get_backend` are handled generically, the reduction is the portable KA tree,
# and what remains here is what only AMDGPU.jl can answer.
#
# **This file was written and precompiled without ROCm hardware.** That is
# stated because it changes what may be claimed. Every AMDGPU API used here was
# checked to exist and to accept the argument types passed (`HIP.name`,
# `HIP.gcn_arch`, `HIP.wavefrontsize`, `HIP.properties`, all
# `(::HIPDevice)`, and `AMDGPU.launch_configuration`), and every no-device path
# is exercised by the test suite. The *with-device* paths are unverified. They
# are not marked as tested anywhere, and `docs/Checkpoint.md` records the gap.
#
# Two things are deliberately absent rather than approximated:
#
#   1. **No wavefront-shuffle reduction.** Section 16.5 applies the CUDA gate —
#      measured benefit — and no measurement is possible here. Phase 3 measured
#      the exactly analogous CUDA warp-shuffle reduction at 0.48–0.56× the
#      tree's speed, because a shuffle round makes *every* lane apply the
#      operator while a tree round halves the live lanes, and a `Double32`
#      addition is ~20 flops. That argument is structural, not NVIDIA-specific,
#      and if anything is stronger on a 64-lane wavefront. It remains an
#      argument, not a measurement, and shipping unmeasured vendor code on the
#      strength of it would invert the gate.
#   2. **No `kernel_report` method.** AMDGPU.jl exposes no per-kernel register
#      or VGPR count; there is no counterpart to `CUDA.registers`. A
#      `MethodError` naming the function and the backend is more honest than
#      empty rows.

module Double32sAMDGPUExt

using Double32s
using Double32s: DEFAULT_GROUPSIZE
using AMDGPU
using AMDGPU: ROCBackend, ROCArray

# Design section 16 "AMD capability matrix". `wavefrontsize` is queried, never
# assumed to be 64 — section 16.2 is explicit about that, and RDNA parts run
# wave32 while CDNA parts run wave64.
function _amdgpu_fields()
    AMDGPU.functional() || return (device = nothing, gcn_arch = nothing,
                                   wavefrontsize = nothing,
                                   max_threads_per_block = nothing,
                                   shared_memory_per_block = nothing,
                                   multiprocessors = nothing)
    dev = AMDGPU.device()
    props = AMDGPU.HIP.properties(dev)
    return (device = AMDGPU.HIP.name(dev),
            gcn_arch = AMDGPU.HIP.gcn_arch(dev),
            wavefrontsize = Int(AMDGPU.HIP.wavefrontsize(dev)),
            max_threads_per_block = Int(props.maxThreadsPerBlock),
            shared_memory_per_block = Int(props.sharedMemPerBlock),
            multiprocessors = Int(props.multiProcessorCount))
end

Double32s._vendor_fields(::ROCBackend) = _amdgpu_fields()
Double32s._vendor_fields(::ROCArray) = _amdgpu_fields()

function Double32s._describe(::ROCBackend)
    AMDGPU.functional() || return "ROCBackend (no device)"
    return "ROCBackend on " * AMDGPU.HIP.name(AMDGPU.device())
end

Double32s.capability_report(::Type{ROCArray}) =
    Double32s.capability_report(ROCBackend())

"""
    Double32s.wavefront_size(::ROCBackend) -> Int

Lanes per wavefront on the current AMD device, **queried** per design section
16.2 — 64 on CDNA, 32 on RDNA in wave32 mode, and never hardcoded. Returns 64
when no device is present, which is the value that makes
[`Double32s.DEFAULT_GROUPSIZE`](@ref) checks conservative rather than optimistic.
"""
Double32s.wavefront_size(::ROCBackend) =
    AMDGPU.functional() ? Int(AMDGPU.HIP.wavefrontsize(AMDGPU.device())) : 64

# Design section 16.3. AMDGPU's occupancy facility needs a compiled kernel,
# which needs a device; with none, only the package default is reported. The
# wavefront size travels with the answer because it is what a caller needs in
# order to pick a non-default groupsize that still fills whole waves.
function Double32s.launch_config(backend::ROCBackend; kind::Symbol = :reduction)
    ws = Double32s.wavefront_size(backend)
    base = (kind = kind, groupsize = DEFAULT_GROUPSIZE, wavefrontsize = ws,
            whole_waves = iszero(DEFAULT_GROUPSIZE % ws))
    AMDGPU.functional() || return merge(base, (suggested = nothing, device = nothing))
    return merge(base, (suggested = nothing, device = AMDGPU.HIP.name(AMDGPU.device())))
end

end # module Double32sAMDGPUExt
