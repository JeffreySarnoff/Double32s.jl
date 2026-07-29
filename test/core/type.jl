@testset "type, constructors, canonicalization" begin
    @testset "layout and inference" begin
        @test sizeof(Double32) == 8
        @test isbitstype(Double32)
        @test Double32 <: AbstractFloat
        @test fieldnames(Double32) == (:hi, :lo)
        @test @inferred(Double32(1.0f0)) isa Double32
        @test @inferred(canonicalize(1.0f0, 2.0f0^-30)) isa Double32
        @test HILO(Double32(1.0f0, 2.0f0^-30)) isa Tuple{Float32,Float32}
    end

    @testset "Float32 constructor is exact and sign-preserving" begin
        for b in (0x00000000, 0x80000000, 0x00000001, 0x007fffff, 0x00800000,
                  0x3f7fffff, 0x3f800000, 0x3f800001, 0x7f7fffff, 0x7f800000,
                  0xff800000, 0x7fc00000)
            f = reinterpret(Float32, b)
            x = Double32(f)
            @test x.hi === f
            @test x.lo === 0.0f0
        end
        @test Double32(-0.0f0).hi === -0.0f0     # the one-arg path must not lose it
        @test signbit(Double32(-0.0f0))
    end

    @testset "canonicalize establishes the invariants" begin
        for x in vcat(structured_pool(), wide_gap_pairs(), random_pool(300))
            @test iscanonical(x)
            isfinite(x) || continue
            iszero(x.hi) && continue
            @test abs(BigFloat(x.lo)) <= big(2.0)^(exponent(x.hi) - 24)
            @test Float32(exact(x)) === x.hi          # invariant 3
        end
    end

    @testset "canonicalize is idempotent and order-free" begin
        for x in vcat(structured_pool(), wide_gap_pairs())
            @test canonicalize(x) === x
            @test Double32(x.hi, x.lo) === x          # round-trip, incl. signed zero
        end
        @test canonicalize(2.0f0^-30, 1.0f0) === canonicalize(1.0f0, 2.0f0^-30)
        @test canonicalize(1.0f0, 0.5f0) === Double32(1.5f0)
    end

    @testset "canonicalize handles overflow and non-finite input" begin
        @test canonicalize(floatmax(Float32), floatmax(Float32)) === POS_INF
        @test canonicalize(-floatmax(Float32), -floatmax(Float32)) === NEG_INF
        @test canonicalize(Inf32, 1.0f0) === POS_INF
        @test canonicalize(Inf32, -Inf32) === D32_NAN     # lo is never NaN beside Inf
        @test canonicalize(NaN32, 1.0f0) === D32_NAN
        for x in NONFINITE
            @test iszero(canonicalize(x.hi, x.lo).lo)
        end
    end

    @testset "signed zero round-trips through the two-word constructor" begin
        @test Double32(-0.0f0, 0.0f0) === NEG_ZERO
        @test Double32(0.0f0, 0.0f0) === POS_ZERO
        @test signbit(Double32(-0.0f0, 0.0f0))
        # a genuine cancellation still follows IEEE, which gives +0
        @test !signbit(canonicalize(1.0f0, -1.0f0))
    end

    @testset "integer and Float64 constructors" begin
        for n in (0, 1, -1, 3, 2^23, 2^24 + 1, 2^52 + 1, -(2^53) - 1,
                  typemax(Int32), typemin(Int32), 12345678901234567)
            @test exact(Double32(n)) == big(n) ||
                  abs(exact(Double32(n)) - big(n)) <= abs(big(n)) * 2.0^-47
        end
        @test Double32(2^24 + 1) |> exact == big(2^24 + 1)     # needs the low word
        @test Double32(Int8(-5)) === Double32(-5.0f0)
        @test Double32(true) === one(Double32)
        @test Double32(false) === zero(Double32)

        for v in (1.0, -1.0, 0.1, 1e300, -1e300, 1e-300, Inf, -Inf, NaN, 0.0, -0.0)
            x = Double32(v)
            if isnan(v)
                @test isnan(x)
            elseif abs(v) > big(2.0)^128
                @test isinf(x) && signbit(x) == signbit(v)
            elseif iszero(v)
                @test iszero(x) && signbit(x) == signbit(v)
            elseif abs(v) < big(2.0)^-149
                @test iszero(x)                       # underflows out of Float32 range
            else
                @test relerr(x, big(v)) <= 2.0^-47
            end
        end
        @test iscanonical(Double32(0.1))
        @test Double32(1 // 3) |> exact ≈ big(1) / 3 atol = 2.0^-46
    end

    @testset "conversions out" begin
        for x in vcat(structured_pool(), wide_gap_pairs(), random_pool(200))
            isfinite(x) || continue
            @test Float32(x) === x.hi                       # invariant 3
            @test BigFloat(x) == exact(x)                   # always exact
            @test float(x) === x
            @test AbstractFloat(x) === x
        end
        @test widen(Double32) === Float64
        @test Int(Double32(42)) == 42
    end
end
