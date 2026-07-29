# gpu/arrays.jl — backend registration and structure-of-arrays allocation.
#
# `KernelAbstractions` is a direct dependency now (AcceleratedKernels brings it,
# and `gpu/linalg.jl` writes an `@kernel` that AK has no equivalent for), so
# these are ordinary package methods rather than an extension.

# KA requires this of custom array types. Correct because both word arrays are
# allocated on the same backend and `SplitDouble32Array` enforces equal axes.
KA.get_backend(A::SplitDouble32Array) = KA.get_backend(A.hi)

function allocate_split(backend::KA.Backend, dims::Integer...)
    h = KA.allocate(backend, Float32, dims...)
    l = KA.allocate(backend, Float32, dims...)
    # Unchecked: both were just allocated with identical dims.
    return SplitDouble32Array(Raw(), h, l)
end

allocate_split(backend::KA.Backend, dims::Dims) =
    allocate_split(backend, dims...)

"""
    wavefront_size(::KernelAbstractions.CPU) -> Int

`1`. A CPU work item is a loop iteration, not a lane of a SIMD group, so there
is no wave to fill and no divisibility constraint on the workgroup size.
"""
wavefront_size(::KA.CPU) = 1
