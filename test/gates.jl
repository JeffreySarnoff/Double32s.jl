# test/gates.jl — the Phase 1 validation gate, mechanically checked.
#
# Design section 24, Phase 1: "these gates cost about a day to set up, and they
# are what make the later phases' semantic parity claims checkable rather than
# asserted." Phase 2 makes exactly such a claim — that a CPU and a CUDA
# reduction return the same bits — so the gate is closed here rather than left
# as a note.
#
# `Test.@inferred` and `@allocated == 0` for the scalar operations live with
# those operations in `test/core/`. What is added here is the generated-code
# inspection and the source-level prohibitions, which are global claims and
# belong in one place.

using InteractiveUtils

"LLVM IR for `f(::tt...)`, optimized, as a string."
function llvm_ir(f, tt)
    io = IOBuffer()
    code_llvm(io, f, tt; optimize = true, debuginfo = :none)
    return String(take!(io))
end

count_throws(ir) = length(collect(eachmatch(r"@[a-zA-Z0-9_]*throw[a-zA-Z0-9_]*", ir)))

@testset "Phase 1 validation gate" begin

    @testset "no @fastmath anywhere in src/ or ext/" begin
        # The arithmetic *is* rounding-error bookkeeping; `@fastmath` licenses
        # the compiler to reassociate it away. This is a source-level ban, not
        # a style preference, so it is checked as text.
        offenders = String[]
        for root in (joinpath(pkgdir(Double32s), "src"),
                     joinpath(pkgdir(Double32s), "ext"))
            for (dir, _, files) in walkdir(root), file in files
                endswith(file, ".jl") || continue
                path = joinpath(dir, file)
                for (n, line) in enumerate(eachline(path))
                    # Skip prose: every mention in this package is in a comment
                    # or docstring explaining why it is forbidden.
                    occursin("@fastmath", line) || continue
                    stripped = lstrip(line)
                    startswith(stripped, "#") && continue
                    occursin("`@fastmath`", line) && continue
                    push!(offenders, "$path:$n: $stripped")
                end
            end
        end
        @test isempty(offenders)
        isempty(offenders) || @info "@fastmath found" offenders
    end

    @testset "generated code: no throw, no alloca, no Float64" begin
        # The design names `+` and `*`; every operation documented as
        # device-safe is held to the same standard, because a `throw` in a
        # kernel is what `sqrt_or_nan` exists to avoid and what
        # `raw_exponent`/`sqrt_nonneg` were introduced to remove.
        device_safe = (("+",           +,                     (Double32, Double32)),
                       ("-",           -,                     (Double32, Double32)),
                       ("*",           *,                     (Double32, Double32)),
                       ("/",           /,                     (Double32, Double32)),
                       ("inv",         inv,                   (Double32,)),
                       ("fma",         fma,                   (Double32, Double32, Double32)),
                       ("muladd",      muladd,                (Double32, Double32, Double32)),
                       ("sqrt_or_nan", Double32s.sqrt_or_nan, (Double32,)),
                       ("<",           <,                     (Double32, Double32)),
                       ("isless",      isless,                (Double32, Double32)),
                       ("probe_lane",  Double32s.probe_lane,  (Int32, Float32, Float32)))
        for (name, f, tt) in device_safe
            ir = llvm_ir(f, tt)
            @test count_throws(ir) == 0
            @test !occursin("alloca", ir)
            # No accidental Float64 in the Float32 arithmetic core.
            @test !occursin("double", ir)
        end

        # The design's Phase 1 gate says `+` and `*` show "no calls to package
        # functions". Taken literally that is now wrong, and the design has been
        # amended: sections 7.5 and 8 deliberately put each guard behind a
        # `@noinline` call so the hot path stays branch-free and small. `+`
        # calls `add_nonfinite`, `*` calls `mul_guarded`, and removing those
        # calls would mean inlining the cold path into the hot one.
        #
        # The checkable claim is the one that was actually wanted: every call is
        # either a hardware intrinsic or *the* single cold guard.
        for (f, tt, guard) in ((+, (Double32, Double32), "add_nonfinite"),
                               (*, (Double32, Double32), "mul_guarded"))
            calls = [m.match for m in eachmatch(r"(?:tail )?call [^\n]*", llvm_ir(f, tt))]
            package_calls = filter(c -> !occursin("@llvm.", c), calls)
            @test length(package_calls) == 1
            @test occursin(guard, only(package_calls))
        end

        # Host routines that are not kernel entry points still must not carry a
        # dead `Base.exponent` throw. This list exists because the first sweep
        # patched `hypot` and missed `norm2` — same defect, different file.
        for (name, f, tt) in (("norm2",       Double32s.norm2, (Vector{Double32},)),
                              ("significand", significand,     (Double32,)),
                              ("frexp",       frexp,           (Double32,)),
                              ("eps",         eps,             (Double32,)),
                              ("float64_reference_is_exact",
                               Double32s.float64_reference_is_exact, (Double32,)))
            @test count_throws(llvm_ir(f, tt)) == 0
        end

        # Two counterexamples, so the check above is known to be able to fail.
        # Both of these *should* throw: it is their documented contract.
        @test count_throws(llvm_ir(sqrt, (Double32,))) > 0
        @test count_throws(llvm_ir(exponent, (Double32,))) > 0
    end

    @testset "Base parity for exponent/significand/frexp" begin
        # `significand(::Double32)` used to inherit `Base.exponent`'s
        # `DomainError` and disagreed with Base at all five of these.
        for z in (0.0f0, -0.0f0, Inf32, -Inf32, NaN32)
            got = significand(Double32(z))
            want = significand(z)
            @test isequal(Float32(got), want)
            @test isequal(frexp(Double32(z))[1], Double32(z))
            @test frexp(Double32(z))[2] == 0
        end
        for z in (1.0f0, 12.0f0, 0.75f0, -3.5f0, floatmin(Float32), floatmax(Float32))
            @test Float32(significand(Double32(z))) === significand(z)
            @test exponent(Double32(z)) == exponent(z)
            m, e = frexp(Double32(z))
            @test e == frexp(z)[2]
            @test ldexp(m, e) === Double32(z)
        end
        # `exponent` keeps Base's contract, throw included.
        @test_throws DomainError exponent(zero(Double32))
        @test_throws DomainError exponent(Double32(Inf32))
        @test_throws DomainError exponent(Double32(NaN32))
    end

    @testset "raw_exponent agrees with Base.exponent" begin
        # Sampled across the whole Float32 bit pattern space, plus every
        # boundary by construction. The stride is coprime with the binade size
        # so the sample is not aligned to exponent boundaries.
        mismatches = 0
        sampled = 0
        u = UInt32(1)
        while u < typemax(UInt32) - UInt32(9973)
            z = reinterpret(Float32, u)
            if isfinite(z) && !iszero(z)
                sampled += 1
                Double32s.raw_exponent(z) == exponent(z) || (mismatches += 1)
            end
            u += UInt32(9973)
        end
        @test sampled > 400_000
        @test mismatches == 0

        for z in (floatmin(Float32), nextfloat(0.0f0), prevfloat(floatmin(Float32)),
                  floatmax(Float32), 1.0f0, prevfloat(1.0f0), nextfloat(1.0f0),
                  3.0f0, reinterpret(Float32, 0x00000001), -1.0f0, -floatmin(Float32))
            @test Double32s.raw_exponent(z) == exponent(z)
        end

        # Out of domain: a value, never a throw, and far outside any guard band.
        @test Double32s.raw_exponent(0.0f0) == typemin(Int) >> 1
        @test Double32s.raw_exponent(-0.0f0) == typemin(Int) >> 1
        @test Double32s.raw_exponent(Inf32) == typemax(Int) >> 1
        @test Double32s.raw_exponent(NaN32) == typemax(Int) >> 1
        @test (@allocated Double32s.raw_exponent(3.0f0)) == 0

        # sqrt_nonneg: NaN, not DomainError.
        @test Double32s.sqrt_nonneg(4.0f0) === 2.0f0
        @test isnan(Double32s.sqrt_nonneg(-1.0f0))
        @test Double32s.sqrt_nonneg(4.0) === 2.0
        @test_throws DomainError sqrt(-1.0f0)
    end

    @testset "no runtime dispatch in the scalar core" begin
        # A cheap stand-in for JET: every scalar operation must infer to a
        # concrete type from concrete arguments.
        @test @inferred(Double32(1.0f0) + Double32(2.0f0)) isa Double32
        @test @inferred(Double32(1.0f0) * Double32(2.0f0)) isa Double32
        @test @inferred(Double32(1.0f0) / Double32(3.0f0)) isa Double32
        @test @inferred(inv(Double32(3.0f0))) isa Double32
        @test @inferred(Double32s.sqrt_or_nan(Double32(2.0f0))) isa Double32
        @test @inferred(fma(Double32(1.0f0), Double32(2.0f0), Double32(3.0f0))) isa Double32
        @test @inferred(Double32(1.0f0) < Double32(2.0f0)) isa Bool
        @test @inferred(Double32s.capability_report()) isa NamedTuple
        for (f, tt) in ((+, (Double32, Double32)), (*, (Double32, Double32)),
                        (/, (Double32, Double32)), (inv, (Double32,)))
            @test Base.return_types(f, tt) == Any[Double32]
        end
    end
end
