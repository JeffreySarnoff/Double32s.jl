# gpu/capability.jl — the section 5.9 probe, launched on a real backend.

# One work item per lane, through `AcceleratedKernels.foreachindex` rather than
# a hand-written `@kernel`. There is no barrier here, so none of design section
# 14.3's scoping hazards apply, and delegating removes the last reason this file
# needed a kernel of its own.
#
# `probe_lane` is branchy on the lane code, which would be a warp-divergence
# problem in a hot kernel and is irrelevant in a six-element diagnostic launch.
function probe_results(backend::KA.Backend)
    KA.functional(backend) || throw(ArgumentError(
        lazy"$backend has no device available to probe"))
    codes, a, b = probe_operands()
    dcodes = KA.allocate(backend, Int32, PROBE_LANES)
    da     = KA.allocate(backend, Float32, PROBE_LANES)
    db     = KA.allocate(backend, Float32, PROBE_LANES)
    dout   = KA.allocate(backend, Bool, PROBE_LANES)
    copyto!(dcodes, codes)
    copyto!(da, a)
    copyto!(db, b)
    KA.synchronize(backend)

    AK.foreachindex(dout, backend) do i
        @inbounds dout[i] = probe_lane(dcodes[i], da[i], db[i])
    end
    KA.synchronize(backend)

    host = Vector{Bool}(undef, PROBE_LANES)
    copyto!(host, dout)
    return ntuple(i -> @inbounds(host[i]), Val(PROBE_LANES))
end

has_gradual_underflow(backend::KA.Backend) = _underflow_ok(probe_results(backend))

has_fused_multiply_add(backend::KA.Backend) =
    @inbounds probe_results(backend)[PROBE_LANES]

_describe(backend::KA.Backend) = string(nameof(typeof(backend)))

# A backend whose device is absent is a normal state — AMDGPU loads fine on a
# machine with no ROCm hardware — so the report degrades instead of throwing.
# Every capability is reported `false`, never guessed from the backend's name.
function capability_report(backend::KA.Backend)
    KA.functional(backend) || return merge(
        build_report(backend, NO_LANES), (functional = false,))
    return build_report(backend, probe_results(backend))
end
