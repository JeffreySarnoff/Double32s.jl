# ext/Double32sCUDAExt.jl — NVIDIA specifics.
#
# The portable work is already done elsewhere: `Double32` is an 8-byte `isbits`
# struct, so `CuArray{Double32}` needs no package code; `Adapt` recursion into
# `SplitDouble32Array` lives in `arrays/splitarray.jl`; kernel launches and
# `get_backend` live in the KernelAbstractions extension. What is left here is
# context that no abstraction layer can see: which device answered, and under
# what task-local compilation settings.
#
# On the math mode, measured rather than assumed. Setting
# `CUDA.math_mode!(CUDA.FAST_MATH)` on an RTX 5090 (sm_120, CUDA 13.3) changed
# *nothing* this package depends on: all six probe lanes still pass, `Float32`
# division and `sqrt` return the same bits as PEDANTIC_MATH and as the host, and
# `Double32` `/` and `sqrt` hold their relative error at 8.3e-16 and 1.0e-15
# across all three modes. CUDA.jl's math mode selects approximate intrinsics
# where Julia asks for them; it does not retroactively relax arithmetic written
# without `@fastmath`.
#
# So `math_mode` goes in the report as *context*, not as a verdict. The verdict
# is the probe. The setting that actually destroys an error-free transformation
# is a literal `@fastmath` or `CUDA.@fastmath` around the call site, because the
# LLVM `reassoc` flag it sets lets the compiler cancel `(a - (s - v))` to zero —
# and no report field can detect that from outside the expression.

module Double32sCUDAExt

using Double32s
using CUDA
using CUDA: CuDevice, CUDABackend, AnyCuArray

using Double32s: Double32, DEFAULT_GROUPSIZE, PolicyOp, add, DEFAULT_POLICY,
                 sqrt_or_nan

# `CUDA.math_mode()` is task local, so it is read at report time, never cached.
#
# The rest is the design section 15 "CUDA capability matrix": the device
# properties that decide whether a launch configuration is sane. `warpsize` is
# here because it is the one number that differs structurally from AMD's 64-wide
# wavefront, and `multiprocessors` because occupancy is meaningless without it.
function _cuda_fields()
    mode = CUDA.math_mode()
    CUDA.functional() || return (device = nothing, capability = nothing,
                                 math_mode = mode, warpsize = nothing,
                                 max_threads_per_block = nothing,
                                 shared_memory_per_block = nothing,
                                 multiprocessors = nothing)
    dev = CUDA.device()
    return (device = CUDA.name(dev),
            capability = CUDA.capability(dev),
            math_mode = mode,
            warpsize = CUDA.warpsize(dev),
            max_threads_per_block =
                CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
            shared_memory_per_block =
                CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK),
            multiprocessors =
                CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
end

Double32s._vendor_fields(::CUDABackend) = _cuda_fields()
Double32s._vendor_fields(::AnyCuArray) = _cuda_fields()

function Double32s._describe(::CUDABackend)
    CUDA.functional() || return "CUDABackend (no device)"
    return "CUDABackend on " * CUDA.name(CUDA.device())
end

"""
    Double32s.capability_report(CuArray)

Probe the current CUDA device, adding `device`, `capability`, and `math_mode` to
the report. Equivalent to `Double32s.capability_report(CUDA.CUDABackend())`.

On NVIDIA hardware `Float32` subnormals and a genuine fused multiply-add are
native, so a `gradual_underflow = false` or `fused_multiply_add = false` here is
a real finding worth chasing rather than a known limitation. `math_mode` is
reported for context; measurements on sm_120 show `FAST_MATH` leaving every
lane and every `Double32` operation unchanged.
"""
Double32s.capability_report(::Type{CuArray}) =
    Double32s.capability_report(CUDABackend())

"""
    Double32s.wavefront_size(::CUDABackend) -> Int

Lanes per warp on the current CUDA device. 32 on every NVIDIA part shipped to
date, but queried rather than assumed, for the same reason design section 16.2
gives on the AMD side. Returns 32 when no device is present.
"""
Double32s.wavefront_size(::CUDABackend) =
    CUDA.functional() ? Int(CUDA.warpsize(CUDA.device())) : 32

# --------------------------------------------------- launch and occupancy ---
#
# Design section 15.3. The candidate numbers there were explicitly labelled
# hypotheses; this is the test. Both the used value and the occupancy API's
# suggestion are reported, because on this hardware they disagree and the
# measurement wins — see the `Double32s.launch_config` docstring.

# Representative kernels, compiled but never launched. Elementwise shapes match
# what broadcast generates; the reduction shape matches `ka_sum`'s stage.
function _elementwise2!(dest, a, b, f)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if i <= length(dest)
        dest[i] = f(a[i], b[i])
    end
    return nothing
end

function _elementwise1!(dest, a, f)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if i <= length(dest)
        dest[i] = f(a[i])
    end
    return nothing
end

function _reduction_stage!(partials, input, n::Int, op, ::Val{W}) where {W}
    tid = threadIdx().x
    gid = blockIdx().x
    sh = CuStaticSharedArray(Double32, W)
    i = (gid - 1) * (2W) + tid
    acc = zero(Double32)
    if i <= n
        @inbounds acc = Double32(input[i])
    end
    @inbounds sh[tid] = acc
    sync_threads()
    if tid == 1
        @inbounds partials[gid] = sh[1]
    end
    return nothing
end

function _compile_probes()
    d = CuArray{Double32}(undef, 1)
    a = CuArray{Double32}(undef, 1)
    b = CuArray{Double32}(undef, 1)
    p = CuArray{Double32}(undef, 1)
    return (("+",           (@cuda launch=false _elementwise2!(d, a, b, +))),
            ("*",           (@cuda launch=false _elementwise2!(d, a, b, *))),
            ("/",           (@cuda launch=false _elementwise2!(d, a, b, /))),
            ("fma",         (@cuda launch=false _elementwise2!(d, a, b,
                                (x, y) -> fma(x, y, x)))),
            ("sqrt_or_nan", (@cuda launch=false _elementwise1!(d, a, sqrt_or_nan))),
            ("reduce_stage", (@cuda launch=false _reduction_stage!(p, a, 1,
                                PolicyOp(add, DEFAULT_POLICY), Val(DEFAULT_GROUPSIZE)))))
end

function Double32s.kernel_report(::CUDABackend)
    CUDA.functional() || return NamedTuple[]
    dev = CUDA.device()
    warps_per_sm = CUDA.attribute(dev,
        CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR) ÷ CUDA.warpsize(dev)
    rows = NamedTuple[]
    for (name, k) in _compile_probes()
        cfg = CUDA.launch_configuration(k.fun)
        mem = CUDA.memory(k)
        blocks_per_sm = CUDA.active_blocks(k.fun, DEFAULT_GROUPSIZE)
        occ = blocks_per_sm * (DEFAULT_GROUPSIZE ÷ CUDA.warpsize(dev)) / warps_per_sm
        push!(rows, (operation = name,
                     registers = CUDA.registers(k),
                     local_bytes = mem.local,
                     shared_bytes = mem.shared,
                     max_threads = cfg.threads,
                     occupancy = occ))
    end
    return rows
end

Double32s.kernel_report(::Type{CuArray}) = Double32s.kernel_report(CUDABackend())

function Double32s.launch_config(::CUDABackend; kind::Symbol = :reduction)
    CUDA.functional() || return (kind = kind, groupsize = DEFAULT_GROUPSIZE,
                                 suggested = nothing, device = nothing)
    # The reduction stage is the register-heaviest of the package's kernels, so
    # it is the one the suggestion is taken from (section 15.3: "tune for
    # register-heavy arithmetic rather than assuming the usual Float32 block
    # size").
    probes = _compile_probes()
    k = kind === :reduction ? last(probes)[2] : first(probes)[2]
    return (kind = kind,
            groupsize = DEFAULT_GROUPSIZE,
            suggested = CUDA.launch_configuration(k.fun).threads,
            device = CUDA.name(CUDA.device()))
end

end # module Double32sCUDAExt
