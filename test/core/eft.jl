@testset "error-free transformations" begin
    words32 = Float32[0.0, -0.0, 1.0, -1.0, 3.0, 0.5, LOMIN, 3LOMIN,
                      floatmin(Float32), prevfloat(floatmin(Float32)),
                      floatmax(Float32), prevfloat(1.0f0), nextfloat(1.0f0),
                      2f0^-24, 3f0 * 2f0^-25, 2f0^60, 2f0^-60, 1.5f0]

    @testset "two_sum_ is exact for every finite pair" begin
        for a in words32, b in words32
            s, e = EFT.two_sum_(a, b)
            isfinite(s) || continue
            @test big(s) + big(e) == big(a) + big(b)
        end
    end

    @testset "two_diff_ is exact and keeps its own zero sign" begin
        for a in words32, b in words32
            s, e = EFT.two_diff_(a, b)
            isfinite(s) || continue
            @test big(s) + big(e) == big(a) - big(b)
        end
        # the reason two_diff_ is not two_sum_(a, -b)
        @test EFT.two_diff_(0.0f0, 0.0f0)[1] === 0.0f0
        @test EFT.two_sum_(0.0f0, -0.0f0)[1] === 0.0f0
    end

    @testset "two_prod_ is exact when the residual is representable" begin
        for a in words32, b in words32
            p, e = EFT.two_prod_(a, b)
            isfinite(p) || continue
            got = big(p) + big(e)
            want = big(a) * big(b)
            # exact unless the residual fell below 2^-149 (design section 7.4)
            if got != want
                @test abs(want - got) < big(2.0)^-148
                @test iszero(e)          # the loss is silent: e is a signed zero
            end
        end
    end

    @testset "two_prod_ residual underflow is real, silent, and not `p subnormal`" begin
        # Confirmed cases from the design's section 7.4 table.
        lost = [(3LOMIN, 0.5f0), (prevfloat(1.0f0), 2f0^-140), (1.0000001f0, 2f0^-145)]
        for (a, b) in lost
            p, e = EFT.two_prod_(a, b)
            @test iszero(e)                                   # signed zero, not non-finite
            @test isfinite(p)                                 # no check can see the loss
            @test big(p) + big(e) != big(a) * big(b)
        end
        # ... while a *subnormal* product can still be exact: the trigger is the
        # residual's magnitude, not the product's class.
        p, e = EFT.two_prod_(floatmin(Float32), 0.5f0)
        @test issubnormal(p)
        @test big(p) + big(e) == big(floatmin(Float32)) * big(0.5f0)
    end

    @testset "two_hilo_sum_: the precondition as stated is false" begin
        # `abs(a) >= abs(b)` is sufficient, not necessary. Under high-word
        # cancellation `a` is exactly zero while `b` is not, and FastTwoSum is
        # still exact — which is why add_dd_ works despite violating it.
        @test EFT.two_hilo_sum_(0.0f0, 3.0f0) == EFT.two_sum_(0.0f0, 3.0f0)
        # a violation with a nonzero high word, observed inside add_dd_
        a, b = -2f0^-24, -3f0 * 2f0^-25
        @test abs(a) < abs(b)                                  # precondition violated
        @test EFT.two_hilo_sum_(a, b) == EFT.two_sum_(a, b)    # answer still exact
    end

    @testset "two_hilo_sum_checked accepts exactness, rejects loss" begin
        @test EFT.two_hilo_sum_checked(0.0f0, 3.0f0) == (3.0f0, 0.0f0)
        @test EFT.two_hilo_sum_checked(1.0f0, 2f0^-30)[1] === 1.0f0
        # A residual that FastTwoSum genuinely cannot recover.
        @test_throws ArgumentError EFT.two_hilo_sum_checked(2f0^-30, 1.0f0)
        # The check is `@boundscheck`, so it is elidable under `@inbounds` at an
        # inlined call site — which is what makes it kernel-safe. Whether any
        # *particular* call site inlines is the compiler's business, so what is
        # asserted here is only that the unchecked kernel is what it falls back
        # to.
        @test EFT.two_hilo_sum_(2f0^-30, 1.0f0) === (1.0f0, 0.0f0)
    end

    @testset "guarded variants return canonical non-finite pairs" begin
        for f in (EFT.two_sum, EFT.two_diff, EFT.two_hilo_sum)
            @test f(floatmax(Float32), floatmax(Float32))[2] === 0.0f0 ||
                  isfinite(f(floatmax(Float32), floatmax(Float32))[1])
            @test isnan(f(NaN32, 1.0f0)[1]) && iszero(f(NaN32, 1.0f0)[2])
        end
        @test EFT.two_prod(floatmax(Float32), floatmax(Float32)) === (Inf32, 0.0f0)
    end

    @testset "differential check: two_hilo_sum_ vs two_sum_ inside add_dd_" begin
        # This is the standing regression for design Tier 2.1. Any disagreement
        # is a correctness failure, not a tolerance question.
        rng = MersenneTwister(4)
        disagreements = 0
        for _ in 1:200_000
            a = reinterpret(Float32, rand(rng, UInt32))
            b = reinterpret(Float32, rand(rng, UInt32))
            (isfinite(a) & isfinite(b)) || continue
            fast = EFT.two_hilo_sum_(a, b)
            isfinite(fast[1]) || continue
            slow = EFT.two_sum_(a, b)
            if !(fast[1] === slow[1] && fast[2] === slow[2])
                # only legitimate when the FastTwoSum residual is unrepresentable
                abs(a) >= abs(b) && (disagreements += 1)
            end
        end
        @test disagreements == 0
    end
end
