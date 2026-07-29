# test/extensions.jl — the weak-dependency execution layer.
#
# `KernelAbstractions`, `CUDA`, and `AMDGPU` are weak dependencies, so these
# tests must survive three situations: the extension not loaded at all, loaded
# with no device present, and loaded with a working device. Only the last one
# can check numerics, and it is skipped rather than faked when there is no
# hardware — a green test on a machine with no GPU must not read as evidence
# that the GPU path works.

using KernelAbstractions
using CUDA
using AMDGPU
using Adapt
using TOML

const KA = KernelAbstractions

@testset "extensions" begin

    @testset "extensions load" begin
        for name in (:Double32sCUDAExt, :Double32sAMDGPUExt)
            @test Base.get_extension(Double32s, name) !== nothing
        end
        @test Base.ispublic(Double32s, :allocate_split)
        @test hasmethod(Double32s.allocate_split, Tuple{KA.Backend,Int})

        # The scalar layer must not have grown a hard dependency on any GPU
        # package: a host-only user pays for `Adapt` and nothing else. This is
        # the assertion that catches a stray `Pkg.add` turning a weakdep back
        # into a dep.
        proj = TOML.parsefile(joinpath(pkgdir(Double32s), "Project.toml"))
        # The vendor stacks stay weak — a host-only user must never pull CUDA or
        # AMDGPU. `AcceleratedKernels` and `KernelAbstractions` are now hard
        # dependencies: AK supplies the reductions and KA the one kernel AK has
        # no equivalent for (the tiled matmul).
        for pkg in ("CUDA", "AMDGPU")
            @test !haskey(proj["deps"], pkg)
            @test haskey(proj["weakdeps"], pkg)
        end
        @test sort(collect(keys(proj["deps"]))) ==
              ["AcceleratedKernels", "Adapt", "KernelAbstractions",
               "LinearAlgebra", "Printf", "Random"]
        @test sort(collect(keys(proj["extensions"]))) ==
              ["Double32sAMDGPUExt", "Double32sBFloat16sExt",
               "Double32sCUDAExt", "Double32sDoubleFloatsExt"]
    end

    @testset "probe lanes" begin
        lanes = Double32s.probe_results()
        @test length(lanes) == Double32s.PROBE_LANES
        @test all(lanes)
        @test Double32s.has_gradual_underflow()
        @test Double32s.has_fused_multiply_add()

        # Lane 6 isolates fusion from flush-to-zero: under FTZ the underflow
        # lanes must fail while the fusion lane survives.
        try
            Base.set_zero_subnormals(true)
            ftz = Double32s.probe_results()
            @test !Double32s.has_gradual_underflow()
            @test ftz[Double32s.PROBE_LANES]      # fusion unaffected
            @test Double32s.has_fused_multiply_add()
        finally
            Base.set_zero_subnormals(false)
        end
        @test Double32s.has_gradual_underflow()

        # Allocation free and inferrable, so the same entry points are callable
        # from a device kernel and from a hot host loop.
        @test (@allocated Double32s.probe_results()) == 0
        @test (@allocated Double32s.has_fused_multiply_add()) == 0
        @test @inferred(Double32s.probe_results()) isa NTuple{Double32s.PROBE_LANES,Bool}

        # Operand vectors and lane codes stay in step.
        codes, a, b = Double32s.probe_operands()
        @test length(codes) == length(a) == length(b) == Double32s.PROBE_LANES
        @test all(i -> Double32s.probe_lane(codes[i], a[i], b[i]), 1:Double32s.PROBE_LANES)
    end

    @testset "host report" begin
        r = Double32s.capability_report()
        @test r.target == "host thread"
        @test r.gradual_underflow && r.fused_multiply_add && r.usable && r.functional
        @test r.lanes === Double32s.probe_results()

        # An array prototype routes through broadcast; a host array must agree
        # with the host thread.
        ra = Double32s.capability_report(Float32[])
        @test ra.lanes === r.lanes
        @test Double32s.has_gradual_underflow(Float32[])
        @test Double32s.has_fused_multiply_add(Float32[])
    end

    @testset "KernelAbstractions backend" begin
        r = Double32s.capability_report(CPU())
        @test r.target == "CPU"
        @test r.functional && r.usable
        @test r.lanes === Double32s.probe_results()
        @test Double32s.has_gradual_underflow(CPU())
        @test Double32s.has_fused_multiply_add(CPU())

        S = Double32s.allocate_split(CPU(), 3)
        @test S isa Double32s.SplitDouble32Array
        @test size(S) == (3,)
        @test KA.get_backend(S) === CPU()
        @test Double32s.allocate_split(CPU(), (2, 2)) isa Double32s.SplitDouble32Array
    end

    @testset "structure-of-arrays round trip" begin
        A = Double32.(Float32[1, 3, 7, 11]) ./ 3
        S = Double32s.split(A)
        @test S isa Double32s.SplitDouble32Array
        @test all(i -> S[i] === A[i], eachindex(A))
        @test Double32s.unsplit(S) == A
        @test all(Double32s.unsplit(S) .=== A)

        # Adapt recurses into both words and skips the checking constructor.
        @test Adapt.adapt(Array, S) isa Double32s.SplitDouble32Array
        @test Adapt.adapt(Array, S).hi == S.hi
    end

    # --- device tests -----------------------------------------------------
    # Guarded on `functional`, and each guard reports what it skipped.

    if CUDA.functional()
        @testset "CUDA device" begin
            r = Double32s.capability_report(CUDABackend())
            @test r.functional
            @test r.usable
            @test r.fused_multiply_add           # non-negotiable
            @test r.gradual_underflow            # native on NVIDIA Float32
            @test r.device isa AbstractString
            @test haskey(r, :math_mode)
            @test Double32s.capability_report(CuArray).lanes === r.lanes

            # The array-prototype route and the kernel route must agree.
            @test Double32s.capability_report(CUDA.zeros(Float32, 1)).lanes === r.lanes

            # Arithmetic on device must reproduce the host bit for bit: the
            # kernels are branch-free `Float32`, so there is nothing that may
            # legitimately differ.
            x = Double32.(Float32[1, 3, 7, 11, 1e-30, 1e30])
            y = Double32.(Float32[3, 7, 11, 13, 3e-30, 7f29])
            dx, dy = CuArray(x), CuArray(y)
            for op in (+, -, *, /)
                @test all(Array(op.(dx, dy)) .=== op.(x, y))
            end
            @test all(Array(Double32s.sqrt_or_nan.(dx)) .=== Double32s.sqrt_or_nan.(x))

            # Reductions are the one place device and host may legitimately
            # differ: `+` on `Double32` is not associative and the device tree
            # order is not the host's. Compare against the exact sum instead.
            w = Double32.(Float32.(1 ./ (1:4096)))
            exact = sum(Float64.(Float64.(w)))
            @test abs(Float64(sum(CuArray(w))) - exact) <= 8e-15 * abs(exact)

            # Structure-of-arrays on device: split, mutate in a kernel, unsplit.
            S = Double32s.split(dx)
            @test S.hi isa CuArray{Float32}
            @test all(Array(Double32s.unsplit(S)) .=== x)

            Sk = Double32s.allocate_split(CUDABackend(), length(x))
            copyto!(Sk.hi, Double32s.HI.(x))
            copyto!(Sk.lo, Double32s.LO.(x))
            @test all(Array(Double32s.unsplit(Sk)) .=== x)
            @test KA.get_backend(Sk) isa CUDABackend
        end
    else
        @info "CUDA not functional; device tests skipped"
    end

    if AMDGPU.functional()
        @testset "AMDGPU device" begin
            r = Double32s.capability_report(ROCBackend())
            @test r.functional
            @test r.fused_multiply_add
            @test r.usable
            x = Double32.(Float32[1, 3, 7, 11])
            y = Double32.(Float32[3, 7, 11, 13])
            dx, dy = ROCArray(x), ROCArray(y)
            for op in (+, -, *, /)
                @test all(Array(op.(dx, dy)) .=== op.(x, y))
            end
            @test all(Array(Double32s.unsplit(Double32s.split(dx))) .=== x)
        end
    else
        # Not a failure: AMDGPU loads on machines with no ROCm hardware, and the
        # report is required to say so rather than throw. This is the only AMD
        # path this project can verify, and it is verified thoroughly for that
        # reason — see the Phase 4 note in docs/Checkpoint.md.
        @testset "AMDGPU absent" begin
            r = Double32s.capability_report(ROCBackend())
            @test !r.functional
            @test !r.usable
            @test r.lanes === Double32s.NO_LANES
            @test_throws ArgumentError Double32s.probe_results(ROCBackend())

            # Every AMD capability field must be present and `nothing` — absent,
            # not guessed from the backend's name.
            for k in (:device, :gcn_arch, :wavefrontsize, :max_threads_per_block,
                      :shared_memory_per_block, :multiprocessors)
                @test haskey(r, k)
                @test getfield(r, k) === nothing
            end
            @test Double32s._describe(ROCBackend()) == "ROCBackend (no device)"

            lc = Double32s.launch_config(ROCBackend())
            @test lc.groupsize == Double32s.DEFAULT_GROUPSIZE
            @test lc.suggested === nothing
            @test lc.device === nothing
            @test lc.wavefrontsize == 64          # conservative default
            @test lc.whole_waves                   # 256 is 4 whole wave64s

            # Design 16.2: never hardcode 64. The query exists and is used; with
            # no device it answers conservatively rather than optimistically.
            @test Double32s.wavefront_size(ROCBackend()) == 64
            @test iszero(Double32s.DEFAULT_GROUPSIZE % 64)
            @test iszero(Double32s.DEFAULT_GROUPSIZE % 32)

            # `kernel_report` is deliberately not defined for AMD: AMDGPU.jl has
            # no per-kernel register count. The MethodError is the contract.
            @test_throws MethodError Double32s.kernel_report(ROCBackend())
        end
        @info "AMDGPU not functional; device tests skipped"
    end

    @testset "wavefront_size is queried, never hardcoded (design 16.2)" begin
        @test Double32s.wavefront_size(CPU()) == 1
        if CUDA.functional()
            @test Double32s.wavefront_size(CUDABackend()) == 32
            @test Double32s.wavefront_size(CUDABackend()) ==
                  Double32s.capability_report(CUDABackend()).warpsize
        end
        # The default workgroup size fills whole waves on both vendors, which is
        # why no package kernel needs to branch on the wavefront width.
        for ws in (1, 32, 64)
            @test iszero(Double32s.DEFAULT_GROUPSIZE % ws)
        end
    end
end
