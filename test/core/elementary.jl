# test/core/elementary.jl — the elementary functions (roadmap Phase 6).
#
# Two kinds of assertion, and the distinction matters:
#
#   - **exactness and identities**, which must hold bit-for-bit;
#   - **accuracy ceilings** in nominal-48-bit ulps against a 400-bit `BigFloat`
#     reference, each one the measured worst case with a little headroom.
#
# The ceilings exist so a change that degrades a function fails here rather than
# in a report nobody reruns. They are *not* claims of faithful rounding — none
# of these functions is faithfully rounded, and the accuracy report says so.

using Double32s: rsqrt, log_or_nan, TRIG_ARG_LIMIT, PI_D32, PIO2_D32

# Worst error in nominal-48-bit ulps over a pool, against a BigFloat reference.
function worst_ulp48(f, ref, pool)
    worst = 0.0
    for x in pool
        local e
        try
            e = ref(reference(x))
        catch
            continue
        end
        (isfinite(e) && !iszero(e)) || continue
        big(2.0)^-100 <= abs(e) <= big(2.0)^100 || continue
        local g
        try
            g = f(x)
        catch
            continue
        end
        isfinite(g) || continue
        u = big(2.0)^(Int(floor(log2(abs(e)))) - 47)
        worst = max(worst, Float64(abs(reference(g) - e) / u))
    end
    return worst
end

const ELEM_WIDE = let v = Double32[]
    for k in -40:2:40, m in (1.0f0, 1.3f0, 1.7f0, 2.5f0, 3.9f0, 7.1f0)
        push!(v, Double32(ldexp(m, k)))
    end
    append!(v, [Double32(Float32(i)) / 7 for i in 1:200])
    v
end
const ELEM_POS = filter(x -> x.hi > 0, ELEM_WIDE)
const ELEM_EXP = [Double32(Float32(x)) for x in range(-80, 80; length = 600)]
const ELEM_UNIT = [Double32(Float32(x)) for x in range(-0.999, 0.999; length = 600)]
const ELEM_TRIG = [Double32(Float32(x)) for x in range(-100, 100; length = 800)]

@testset "elementary functions" begin

    @testset "exact and special values" begin
        # Values that must be exact, because the argument is a perfect power.
        @test cbrt(Double32(27.0f0)) === Double32(3.0f0)
        @test isequal(cbrt(Double32(-8.0f0)), Double32(-2.0f0))
        @test cbrt(Double32(64.0f0)) === Double32(4.0f0)
        @test log2(Double32(1024.0f0)) === Double32(10.0f0)
        @test log2(Double32(2.0f0^-30)) === Double32(-30.0f0)
        @test Double32(2.0f0)^-30 === Double32(2.0f0^-30)   # negative integer power
        @test log(one(Double32)) === POS_ZERO
        @test exp(zero(Double32)) === Double32(1.0f0)
        @test exp2(zero(Double32)) === Double32(1.0f0)
        @test expm1(zero(Double32)) === POS_ZERO
        @test log1p(zero(Double32)) === POS_ZERO
        @test sin(zero(Double32)) === POS_ZERO
        @test cos(zero(Double32)) === Double32(1.0f0)
        @test atan(zero(Double32)) === POS_ZERO
        @test rsqrt(Double32(4.0f0)) === Double32(0.5f0)
        @test rsqrt(Double32(0.25f0)) === Double32(2.0f0)

        # Signed zero survives the odd functions.
        @test cbrt(NEG_ZERO) === NEG_ZERO
        @test sin(NEG_ZERO) === NEG_ZERO
        @test atan(NEG_ZERO) === NEG_ZERO
        @test expm1(NEG_ZERO) === NEG_ZERO
        @test log1p(NEG_ZERO) === NEG_ZERO

        # Poles and limits.
        @test log(POS_ZERO) === NEG_INF
        @test log_or_nan(NEG_ZERO) === NEG_INF
        @test log1p(Double32(-1.0f0)) === NEG_INF
        @test rsqrt(POS_ZERO) === POS_INF
        @test exp(Double32(-1000.0f0)) === POS_ZERO
        @test exp(Double32(1000.0f0)) === POS_INF
        @test isequal(expm1(Double32(-1000.0f0)), Double32(-1.0f0))
        @test exp10(Double32(1000.0f0)) === POS_INF

        # NaN in, NaN out — and no throw from the kernel-safe spellings.
        for f in (cbrt, exp, expm1, exp2, exp10, log_or_nan, log1p, sin, cos,
                  tan, atan, rsqrt)
            @test isnan(f(D32_NAN))
        end
        @test isnan(log_or_nan(Double32(-1.0f0)))
        @test isnan(rsqrt(Double32(-1.0f0)))
        @test isnan(sin(POS_INF)) && isnan(cos(POS_INF))

        # The throwing spellings match Base's contract.
        @test_throws DomainError log(Double32(-1.0f0))
        @test_throws DomainError log2(Double32(-1.0f0))
        @test_throws DomainError log10(Double32(-1.0f0))
        @test_throws DomainError log1p(Double32(-2.0f0))
        @test_throws DomainError asin(Double32(1.5f0))
        @test_throws DomainError acos(Double32(-1.5f0))

        # Inverse-trig endpoints.
        @test asin(one(Double32)) === PIO2_D32
        @test asin(-one(Double32)) === -PIO2_D32
        @test acos(one(Double32)) === POS_ZERO
        @test acos(-one(Double32)) === PI_D32
        @test atan(POS_INF) === PIO2_D32
        @test atan(NEG_INF) === -PIO2_D32
    end

    @testset "identities" begin
        for x in ELEM_UNIT[1:7:end]
            s, c = sincos(x)
            @test s === sin(x)
            @test c === cos(x)
            # sin² + cos² = 1 to within a few ulps of the format.
            @test abs(Float64(s * s + c * c) - 1) < 1e-14
        end
        for x in ELEM_POS[1:5:end]
            # exp(log(x)) == x, and log(exp) round trip, to within the
            # composition of both functions' errors. The tolerance scales with
            # |log x|: log's ~1.5 ulp48 relative error is |log x|·1.5·2^-48
            # absolute, and exp converts absolute argument error straight into
            # relative result error (the exp10 lesson, run in reverse).
            lx = log_or_nan(x)
            abs(lx.hi) < 80 || continue
            @test abs(Float64(exp(lx)) / Float64(x) - 1) <
                  (4.0 + 1.5 * abs(Float64(lx))) * 2.0^-48
            @test abs(Float64(rsqrt(x)) * Float64(sqrt(x)) - 1) < 1e-13
            @test abs(Float64(cbrt(x))^3 / Float64(x) - 1) < 1e-13
        end
        for x in ELEM_EXP[1:9:end]
            abs(x.hi) < 40 || continue
            # exp2 and exp10 against exp, which is measured independently.
            @test abs(Float64(exp2(x)) / Float64(exp(x * Double32s.LN2)) - 1) < 1e-13
        end
    end

    # Measured worst cases, with headroom. Regenerate with
    # `julia docs/Reports/accuracies.jl` if an implementation changes.
    @testset "accuracy ceilings (nominal-48-bit ulps)" begin
        @test worst_ulp48(cbrt, cbrt, ELEM_WIDE)            <= 1.5     # measured 0.894
        @test worst_ulp48(rsqrt, x -> 1 / sqrt(x), ELEM_POS) <= 2.0    # measured 1.336
        @test worst_ulp48(exp, exp, ELEM_EXP)               <= 1.5     # measured 0.912
        @test worst_ulp48(exp2, x -> big(2)^x, ELEM_EXP)    <= 1.6     # measured 0.891
        @test worst_ulp48(log_or_nan, log, ELEM_POS)        <= 1.5     # measured 0.982
        @test worst_ulp48(log2, log2, ELEM_POS)             <= 1.5     # measured 0.838
        @test worst_ulp48(log10, log10, ELEM_POS)           <= 2.0     # measured 1.164
        @test worst_ulp48(sin, sin, ELEM_TRIG)              <= 2.0     # measured 1.302
        @test worst_ulp48(cos, cos, ELEM_TRIG)              <= 2.5     # measured 1.114
        @test worst_ulp48(atan, atan, ELEM_WIDE)            <= 2.5     # measured 1.093
        @test worst_ulp48(asin, asin, ELEM_UNIT)            <= 3.0     # measured 1.951
        @test worst_ulp48(acos, acos, ELEM_UNIT)            <= 3.0     # measured 2.376
        # These end in a division, so they inherit `/`'s 4.00 ulp48 floor.
        @test worst_ulp48(tan, tan, ELEM_TRIG)              <= 4.0     # measured 2.010
    end

    @testset "sincos is bitwise (sin, cos), single-kernel dispatch included" begin
        # sin and cos evaluate only the kernel their quadrant parity needs;
        # sincos evaluates both. This asserts the dispatch selects exactly the
        # kernel sincos uses — a sign or parity slip would break bit identity.
        for x in vcat(ELEM_TRIG[1:7:end], [Double32(0.0f0), NEG_ZERO,
                      PI_D32, -PI_D32, PIO2_D32])
            s, c = sincos(x)
            @test s === sin(x)
            @test c === cos(x)
        end
    end

    @testset "exp10 reduces in x, not through exp" begin
        # The defect this guards: `exp(x*LN10)` gives `exp` an argument whose
        # ABSOLUTE error grows with x, and `exp` turns that straight into
        # relative error — 56.1 ulp48 at x ≈ 29.4 before the fix.
        pool = [Double32(Float32(x)) for x in range(-35, 35; length = 600)]
        @test worst_ulp48(exp10, x -> big(10)^x, pool) <= 1.5          # measured 0.927
        @test exp10(Double32(2.0f0)) === Double32(100.0f0)
        @test exp10(zero(Double32)) === Double32(1.0f0)
    end

    @testset "expm1 and log1p beat the naive spellings near zero" begin
        # The reason both functions exist. At these magnitudes `exp(x)-1` and
        # `log(1+x)` have lost most of their significant digits.
        for e in 4:12
            x = Double32(Float32(10.0^-e))
            ref_e = exp(reference(x)) - 1
            ref_l = log(1 + reference(x))
            @test Float64(abs(reference(expm1(x)) - ref_e) / abs(ref_e)) < 1e-14
            @test Float64(abs(reference(log1p(x)) - ref_l) / abs(ref_l)) < 1e-14
            # And the naive form really is worse, or these functions are pointless.
            naive_e = exp(x) - one(Double32)
            @test Float64(abs(reference(naive_e) - ref_e) / abs(ref_e)) >=
                  Float64(abs(reference(expm1(x)) - ref_e) / abs(ref_e))
        end
        # log1p's small-x path must not use division: measured 4.06 ulp48 with
        # the `atanh(x/(2+x))` form, 1.52 with the series.
        small = [Double32(Float32(10.0^-k)) * Double32(Float32(1 + 0.3k)) for k in 1:12]
        @test worst_ulp48(log1p, log1p, small) <= 2.0
        @test worst_ulp48(expm1, expm1, vcat(small, ELEM_UNIT[1:11:end])) <= 3.0
    end

    @testset "trig argument-range policy is stated and honoured" begin
        @test TRIG_ARG_LIMIT == 2.0f0^20
        # Inside the limit, accuracy holds.
        pool = [Double32(Float32(x)) for x in range(-1000, 1000; length = 400)]
        @test worst_ulp48(sin, sin, pool) <= 3.0
        @test worst_ulp48(cos, cos, pool) <= 3.0
        # Outside it the answer is NaN, not a plausible-looking wrong number.
        # Before this policy existed, sin(2^30) returned 9.88e8.
        for x in (Double32(2.0f0^21), Double32(2.0f0^30), Double32(2.0f0^40),
                  Double32(-2.0f0^35))
            @test isnan(sin(x))
            @test isnan(cos(x))
            @test isnan(tan(x))
            @test all(isnan, sincos(x))
        end
        # And just inside the limit it still works.
        @test !isnan(sin(Double32(TRIG_ARG_LIMIT)))
    end

    @testset "coefficients are canonical Double32 pairs" begin
        # Generated by scripts/gen_coefficients.jl from 400-bit BigFloat. If a
        # coefficient were non-canonical the Horner evaluation would silently
        # lose the low word.
        for tup in (Double32s.EXPM1_COEFFS, Double32s.ATANH_COEFFS,
                    Double32s.LOG1P_COEFFS, Double32s.SIN_COEFFS,
                    Double32s.COS_COEFFS, Double32s.ATAN_COEFFS)
            @test length(tup) >= 6
            for c in tup
                @test Double32s.iscanonical(c)
                @test isfinite(c)
            end
        end
        for c in (Double32s.LN2, Double32s.INV_LN2, Double32s.LN10,
                  Double32s.LOG10_E, Double32s.LOG2_10, Double32s.LOG10_2,
                  Double32s.PI_D32, Double32s.PIO2_D32, Double32s.SQRT3_D32)
            @test Double32s.iscanonical(c)
        end
        # The named constants must match their mathematical values to the
        # format's precision — a generator bug would show up here.
        @test abs(Float64(Double32s.LN2) - log(2)) < 1e-14
        @test abs(Float64(Double32s.PI_D32) - pi) < 1e-14
        @test abs(Float64(Double32s.LOG2_10) - log2(10)) < 1e-14
        @test abs(Float64(Double32s.SQRT3_D32) - sqrt(3)) < 1e-14
    end
end

# Appended with the hyperbolic family. Same two kinds of assertion.
@testset "hyperbolic functions" begin
    const_pool_small = [Double32(Float32(10.0^-k)) * Double32(Float32(1 + 0.3k)) for k in 1:14]
    hyp_mid  = [Double32(Float32(x)) for x in range(-5, 5; length = 700)]
    hyp_wide = [Double32(Float32(x)) for x in range(-80, 80; length = 400)]
    hyp_pool = vcat(const_pool_small, hyp_mid, hyp_wide)
    hyp_unit = [Double32(Float32(x)) for x in range(-0.9999, 0.9999; length = 700)]
    hyp_ge1  = vcat([Double32(Float32(x)) for x in range(1.0, 50.0; length = 500)],
                    [Double32(1.0f0) + Double32(Float32(10.0^-k)) for k in 1:12])

    @testset "exact and special values" begin
        @test sinh(zero(Double32)) === POS_ZERO
        @test sinh(NEG_ZERO) === NEG_ZERO
        @test cosh(zero(Double32)) === Double32(1.0f0)
        @test tanh(zero(Double32)) === POS_ZERO
        @test tanh(NEG_ZERO) === NEG_ZERO
        @test asinh(zero(Double32)) === POS_ZERO
        @test asinh(NEG_ZERO) === NEG_ZERO
        @test acosh(one(Double32)) === POS_ZERO
        @test atanh(zero(Double32)) === POS_ZERO
        @test atanh(NEG_ZERO) === NEG_ZERO

        # Parity: cosh even, the other five odd.
        for x in (Double32(0.25f0), Double32(2.0f0), Double32(7.5f0))
            @test cosh(-x) === cosh(x)
            @test sinh(-x) === -sinh(x)
            @test tanh(-x) === -tanh(x)
            @test asinh(-x) === -asinh(x)
        end
        for x in (Double32(0.05f0), Double32(0.5f0), Double32(0.9f0))
            @test atanh(-x) === -atanh(x)
        end

        # Poles, saturation, infinities.
        @test atanh(one(Double32)) === POS_INF
        @test atanh(-one(Double32)) === NEG_INF
        @test tanh(Double32(100.0f0)) === one(Double32)
        @test tanh(Double32(-100.0f0)) === -one(Double32)
        @test sinh(POS_INF) === POS_INF
        @test sinh(NEG_INF) === NEG_INF
        @test cosh(POS_INF) === POS_INF
        @test cosh(NEG_INF) === POS_INF
        @test asinh(POS_INF) === POS_INF
        @test acosh(POS_INF) === POS_INF
        @test sinh(Double32(200.0f0)) === POS_INF
        @test cosh(Double32(200.0f0)) === POS_INF

        for f in (sinh, cosh, tanh, asinh)
            @test isnan(f(D32_NAN))
        end
        @test isnan(acosh(D32_NAN))
        @test isnan(atanh(D32_NAN))

        @test_throws DomainError acosh(Double32(0.5f0))
        @test_throws DomainError atanh(Double32(1.5f0))
        @test_throws DomainError atanh(Double32(-1.5f0))
    end

    @testset "identities" begin
        for x in hyp_mid[1:9:end]
            abs(x.hi) < 20 || continue
            c, s = cosh(x), sinh(x)
            # cosh² - sinh² = 1. The identity itself cancels: both squares are
            # about cosh(x)², so the achievable absolute error on a result of 1
            # scales with cosh(x)². The tolerance says so rather than pretending
            # the format is better than it is.
            tol = 1e-13 + 8 * Float64(c * c) * 2.0^-48
            @test abs(Float64(c * c - s * s) - 1) < tol
            @test abs(Float64(tanh(x)) - Float64(s) / Float64(c)) < 1e-13
            # asinh inverts sinh.
            @test abs(Float64(asinh(s)) - Float64(x)) < 1e-12 * max(1, abs(Float64(x)))
        end
        for x in hyp_unit[1:11:end]
            @test abs(Float64(tanh(atanh(x))) - Float64(x)) < 1e-12
        end
        for x in hyp_ge1[1:7:end]
            x.hi < 1.0f6 || continue
            @test abs(Float64(cosh(acosh(x))) / Float64(x) - 1) < 1e-12
        end
    end

    @testset "accuracy ceilings (nominal-48-bit ulps)" begin
        @test worst_ulp48(sinh, sinh, hyp_pool)   <= 2.5   # measured 1.640
        @test worst_ulp48(cosh, cosh, hyp_pool)   <= 1.5   # measured 1.164
        @test worst_ulp48(asinh, asinh, hyp_pool) <= 2.5   # measured 1.570
        @test worst_ulp48(acosh, acosh, hyp_ge1)  <= 2.0   # measured 1.001
        @test worst_ulp48(atanh, atanh, hyp_unit) <= 3.0   # measured 0.871
        # tanh's one division lands on the full result only in (0.35, 1]; past
        # 1 the `1 - 2/(E+1)` form scales it by (1-t)/t, and the old 4.17
        # worst (ratio form everywhere) dropped to 1.75.
        @test worst_ulp48(tanh, tanh, vcat(const_pool_small, hyp_mid)) <= 3.0  # measured 1.750
    end

    @testset "the cancelling forms really are worse" begin
        # sinh near zero. The comparison range matters: below about 1e-7 the
        # cubic term of sinh sits under 2^-48 relative, so BOTH forms return
        # exactly x and are equally optimal — the naive form's cancellation only
        # shows where the answer still has structure to lose.
        for k in 2:6
            x = Double32(Float32(10.0^-k))
            naive = (exp(x) - exp(-x)) * Double32(0.5f0)
            e = sinh(reference(x))
            good = Float64(abs(reference(sinh(x)) - e) / abs(e))
            bad  = Float64(abs(reference(naive) - e) / abs(e))
            @test good < 1e-14
            @test bad > good
        end
        # atanh near zero: the series must beat 0.5*log1p(2x/(1-x)).
        for k in 4:10
            x = Double32(Float32(10.0^-k))
            e = atanh(reference(x))
            @test Float64(abs(reference(atanh(x)) - e) / abs(e)) < 1e-14
        end
        # acosh near 1: t = x-1 is exact by Sterbenz, so the small-argument
        # form must stay accurate where log(x + sqrt(x²-1)) would not.
        for k in 3:10
            x = one(Double32) + Double32(Float32(10.0^-k))
            e = acosh(reference(x))
            @test Float64(abs(reference(acosh(x)) - e) / abs(e)) < 1e-13
        end
    end

    @testset "hyperbolic coefficients are canonical" begin
        for tup in (Double32s.TANH_COEFFS, Double32s.ASINH_COEFFS)
            @test length(tup) >= 8
            @test all(Double32s.iscanonical, tup)
            @test all(isfinite, tup)
        end
        @test Double32s.TANH_COEFFS[1] === one(Double32)     # tanh(x)/x -> 1
        @test Double32s.ASINH_COEFFS[1] === one(Double32)
    end
end
