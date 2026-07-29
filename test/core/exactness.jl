@testset "Float64 exactness condition (design section 4.2)" begin
    @testset "lsb_exponent" begin
        @test lsb_exponent(1.0f0) == 0
        @test lsb_exponent(1.5f0) == -1
        @test lsb_exponent(2.0f0) == 1
        @test lsb_exponent(3.0f0) == 0
        @test lsb_exponent(LOMIN) == -149
        @test lsb_exponent(3LOMIN) == -149
        @test lsb_exponent(reinterpret(Float32, 0x00000002)) == -148
        @test lsb_exponent(floatmin(Float32)) == -126        # a power of two
        @test lsb_exponent(floatmax(Float32)) == 104
        @test lsb_exponent(0.0f0) == typemin(Int)
        for b in rand(MersenneTwister(3), UInt32, 5000)
            z = reinterpret(Float32, b)
            (isfinite(z) && !iszero(z)) || continue
            k = lsb_exponent(z)
            m = big(z) / big(2.0)^k
            @test isinteger(m) && isodd(BigInt(m))
        end
    end

    @testset "is_power_of_two" begin
        @test is_power_of_two(1.0f0)
        @test is_power_of_two(-2.0f0)
        @test is_power_of_two(2f0^-96)
        @test !is_power_of_two(3.0f0)
        @test !is_power_of_two(floatmax(Float32))
    end

    @testset "the predicate matches BigFloat on every canonical pair tested" begin
        n = 0
        for x in vcat(wide_gap_pairs(), structured_pool(), random_pool(400))
            isfinite(x) || continue
            n += 1
            @test float64_reference_is_exact(x) == (big(Float64(x)) == exact(x))
        end
        @test n > 400
    end

    @testset "structured sweep across every gap width" begin
        bad = 0
        for eh in -149:8:127
            base = ldexp(1.0f0, eh)
            (isfinite(base) && !iszero(base)) || continue
            for h0 in (base, nextfloat(base), base * 1.5f0), sh in (1.0f0, -1.0f0)
                h = sh * h0
                (isfinite(h) && !iszero(h)) || continue
                for k in 0:6:150, s in (1.0f0, -1.0f0)
                    l = ldexp(1.0f0, exponent(h) - 24 - k)
                    (isfinite(l) && !iszero(l)) || continue
                    x = canonicalize(h, s * l)
                    (isfinite(x) && !iszero(x.lo)) || continue
                    float64_reference_is_exact(x) == (big(Float64(x)) == exact(x)) ||
                        (bad += 1)
                end
            end
        end
        @test bad == 0
    end

    @testset "the two directions the design's first correction got wrong" begin
        # (a) a NORMAL low word does not imply exactness: canonicality bounds
        #     exponent(lo) from above only, so the gap can be arbitrarily wide.
        for (h, l) in ((2f0^-62, floatmin(Float32)), (1.0f0, 2f0^-100),
                       (1.0f0, 2f0^-53))
            x = canonicalize(h, l)
            @test iscanonical(x)
            @test !issubnormal(x.lo)                       # lo is normal ...
            @test !float64_reference_is_exact(x)           # ... and yet inexact
            @test big(Float64(x)) != exact(x)
        end
        # (b) `exponent(hi) <= -97` is not the sharp boundary: cancellation at a
        #     power-of-two high word buys one more bit.
        x = canonicalize(2f0^-96, -LOMIN)
        @test iscanonical(x)
        @test issubnormal(x.lo)
        @test exponent(x.hi) == -96
        @test float64_reference_is_exact(x)
        @test big(Float64(x)) == exact(x)
        # the same magnitude with the opposite sign is *not* exact
        y = canonicalize(2f0^-96, LOMIN)
        @test !float64_reference_is_exact(y)
        @test big(Float64(y)) != exact(y)
    end

    @testset "the conservative variant never over-reports" begin
        conservative(x) = iszero(x.lo) || !isfinite(x.hi) ||
                          lsb_exponent(x.lo) >= exponent(x.hi) - 52
        for x in vcat(wide_gap_pairs(), structured_pool(), random_pool(400))
            isfinite(x) || continue
            conservative(x) && @test big(Float64(x)) == exact(x)
        end
    end

    @testset "reference() is always exact" begin
        for x in vcat(wide_gap_pairs(), structured_pool(), random_pool(200))
            isfinite(x) || continue
            @test reference(x) == big(x.hi) + big(x.lo)
            @test BigFloat(x) == reference(x)
        end
    end
end
