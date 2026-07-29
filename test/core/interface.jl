@testset "Julia numeric interface" begin
    @testset "constants" begin
        @test zero(Double32) === POS_ZERO
        @test one(Double32) === raw(1.0f0, 0.0f0)
        @test oneunit(Double32) === one(Double32)
        @test typemin(Double32) === NEG_INF
        @test typemax(Double32) === POS_INF
        @test floatmin(Double32).hi === floatmin(Float32)
        @test zero(one(Double32)) === zero(Double32)
        @test one(zero(Double32)) === one(Double32)
    end

    @testset "floatmax derivation" begin
        # ulp(floatmax(Float32)) == 2^104, so the canonical bound is 2^103; the
        # significand is odd, so a low word of exactly 2^103 is a tie that
        # rounds up and overflows.
        fm = floatmax(Float32)
        @test exponent(fm) - 23 == 104
        @test isodd(reinterpret(UInt32, fm) & 0x007fffff)
        @test canonicalize(fm, ldexp(1.0f0, 103)) === POS_INF
        @test isfinite(floatmax(Double32))
        @test floatmax(Double32) === raw(fm, prevfloat(ldexp(1.0f0, 103)))
        @test iscanonical(floatmax(Double32))
        @test Float64(exact(floatmax(Double32))) ≈ 3.4028235677973306e38
        # nothing canonical is larger
        @test !isfinite(floatmax(Double32) + Double32(ldexp(1.0f0, 80)))
    end

    @testset "precision and eps" begin
        @test precision(Double32) == 48
        @test precision(Double32; base = 10) == 14
        @test_throws DomainError precision(Double32; base = 1)
        @test eps(Double32) === nominal_eps(Double32)
        @test exact(eps(Double32)) == big(2.0)^-47
        @test exact(unit_roundoff(Double32)) == big(2.0)^-48
        @test exact(eps(one(Double32))) == big(2.0)^-47
        @test exact(eps(Double32(4.0f0))) == big(2.0)^-45
        @test isnan(eps(D32_NAN))
        @test eps(zero(Double32)) === raw(LOMIN, 0.0f0)
        # eps is nominal, NOT the pair-lattice successor distance
        @test canonicalize(1.0f0, LOMIN) > one(Double32)
        @test canonicalize(1.0f0, LOMIN) < one(Double32) + eps(Double32)
    end

    @testset "nextfloat/prevfloat are deliberately absent" begin
        @test_throws MethodError nextfloat(one(Double32))
        @test_throws MethodError prevfloat(one(Double32))
    end

    @testset "predicates" begin
        @test isfinite(one(Double32)) && !isfinite(POS_INF) && !isfinite(D32_NAN)
        @test isinf(POS_INF) && isinf(NEG_INF) && !isinf(D32_NAN)
        @test isnan(D32_NAN) && !isnan(POS_INF)
        @test iszero(POS_ZERO) && iszero(NEG_ZERO) && !iszero(one(Double32))
        @test isone(one(Double32)) && !isone(canonicalize(1.0f0, LOMIN))
        @test signbit(NEG_ZERO) && !signbit(POS_ZERO)
        @test isinteger(Double32(3)) && !isinteger(canonicalize(3.0f0, 2f0^-30))
        # a NORMAL Double32 routinely has a SUBNORMAL low word; issubnormal
        # classifies the high word, and no predicate inspects lo
        x = canonicalize(1.0f0, LOMIN)
        @test issubnormal(x.lo)
        @test !issubnormal(x)
        @test isnormal(x)
        @test issubnormal(Double32(LOMIN))
        @test !isnormal(Double32(LOMIN))
        @test !isnormal(zero(Double32)) && !isnormal(POS_INF)
    end

    @testset "promotion" begin
        @test promote_type(Double32, Float16) === Double32
        @test promote_type(Double32, Float32) === Double32
        @test promote_type(Double32, Float64) === Float64
        @test promote_type(Double32, BigFloat) === BigFloat
        @test promote_type(Double32, Int) === Double32
        @test promote_type(Double32, Int8) === Double32
        @test promote_type(Double32, Bool) === Double32
        @test promote_type(Complex{Double32}, Double32) === Complex{Double32}
        @test promote_type(Double32, Complex{Float32}) === Complex{Double32}
        @test one(Double32) + 1 isa Double32
        @test one(Double32) + 1.0f0 isa Double32
        @test one(Double32) + 1.0 isa Float64          # Float64 wins
        @test one(Double32) + big(1.0) isa BigFloat
        @test isempty(detect_ambiguities(Double32s))
    end

    @testset "convert comes free from the constructors" begin
        @test convert(Double32, 1) === one(Double32)
        @test convert(Double32, 1.0f0) === one(Double32)
        @test convert(Double32, one(Double32)) === one(Double32)
        @test convert(Float64, one(Double32)) === 1.0
        @test Double32[1, 2, 3] == [one(Double32), Double32(2), Double32(3)]
    end

    @testset "show round-trips" begin
        for x in vcat(structured_pool(), wide_gap_pairs(), zero_surface(), NONFINITE)
            s = repr(x)
            isnan(x.hi) && continue                  # NaN payload is canonicalized
            y = eval(Meta.parse(s))
            # Base's contract is that the output parses back to the same
            # *value*; the sign of a zero low word is not a value (invariant 8).
            @test isequal(y, x)
            # Everything with a nonzero low word is bit-exact; a zero low word's
            # sign is not information and `canonicalize` may normalize it.
            iszero(x.lo) || @test y === x
        end
        @test repr(Double32(1.0f0) / 3) == "Double32(0.33333334f0, -9.934108f-9)" ||
              startswith(repr(Double32(1.0f0) / 3), "Double32(0.33333334f0, -9.9341")
        @test repr(NEG_ZERO) == "Double32(-0.0f0, 0.0f0)"
        @test sprint(show, POS_INF) == "Double32(Inf32, 0.0f0)"
    end

    @testset "rand draws from a true 48-bit grid" begin
        rng = MersenneTwister(9)
        v = [rand(rng, Double32) for _ in 1:5000]
        @test all(x -> zero(Double32) <= x < one(Double32), v)
        @test all(iscanonical, v)
        # the failure mode of the naive Double32(rand(Float32)) is an always-zero
        # low word
        @test count(x -> !iszero(x.lo), v) > 4000
        @test 0.4 < Float64(sum(Float64, v) / length(v)) < 0.6
        @test rand(rng, Double32, 3) isa Vector{Double32}
    end

    @testset "generic Base algorithms work without package code" begin
        A = Float32[1, 2, 3, 4]
        @test sum(Double32, A) === Double32(10.0f0)
        @test sum(Double32[1, 2, 3]) === Double32(6)
        @test mapreduce(Double32, +, A) === Double32(10.0f0)
        @test prod(Double32[2, 3, 4]) === Double32(24)
        @test maximum(Double32[1, 5, 3]) === Double32(5)
        @test minimum(Double32[1, 5, 3]) === Double32(1)
        @test extrema(Double32[1, 5, 3]) === (Double32(1), Double32(5))
        v = Double32[3, 1, 2]
        @test sort(v) == Double32[1, 2, 3]
        # broadcasting comes free: Double32 <: Number is a broadcast scalar
        @test (Double32[1, 2] .+ one(Double32)) == Double32[2, 3]
        @test (one(Double32) .+ Double32[1, 2]) == Double32[2, 3]
        @test muladd.(Double32[1, 2], Double32[3, 4], Double32[5, 6]) == Double32[8, 14]
        @test Double32(4) in Double32[1, 4]
        # Float32 data summed with a Double32 accumulator actually gains accuracy
        big_then_small = Float32[1f8; fill(1.0f0, 1000)]
        @test Float64(sum(Double32, big_then_small)) == 1.00001e8
        @test Float32(sum(big_then_small)) != 1.00001f8        # what Float32 loses
    end

    @testset "capability probe (Tier 2.3, design 5.9)" begin
        @test has_gradual_underflow()
        # The probe must detect flush-to-zero, and must not be fooled into
        # computing its expected values with the faulty arithmetic.
        try
            Base.set_zero_subnormals(true)
            @test !has_gradual_underflow()
        finally
            Base.set_zero_subnormals(false)
        end
        @test has_gradual_underflow()
        # Pure Float32: no allocation, so the same probe can run in a kernel.
        has_gradual_underflow()
        @test (@allocated has_gradual_underflow()) == 0
    end

    @testset "public API surface" begin
        # `names` returns exported *and* public names on 1.11+, so exports are
        # filtered explicitly. Exactly one name enters `using Double32s`.
        exported = filter(n -> Base.isexported(Double32s, n), names(Double32s))
        @test setdiff(exported, [:Double32s]) == [:Double32]
        @test :Double32 in names(Double32s)
        for n in (:hi, :lo, :words, :canonicalize, :nominal_eps, :unit_roundoff,
                  :SplitDouble32Array, :StrictPolicy, :compensated_dot, :EFT,
                  :sqrt_or_nan, :float64_reference_is_exact, :isnormal)
            @test Base.ispublic(Double32s, n)
        end
        @test !Base.ispublic(Double32s, :raw)
        @test isdefined(Double32s.EFT, :two_sum)
        @test isdefined(Double32s.EFT, :two_prod)
    end
end
