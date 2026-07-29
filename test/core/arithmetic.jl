@testset "arithmetic" begin
    spool = structured_pool()
    rpool = random_pool(200)
    wpool = wide_gap_pairs()
    allp  = vcat(spool, wpool, NONFINITE)

    @testset "results are canonical and inferable" begin
        @test @inferred(one(Double32) + one(Double32)) isa Double32
        @test @inferred(one(Double32) * one(Double32)) isa Double32
        @test @inferred(one(Double32) / one(Double32)) isa Double32
        @test @inferred(sqrt(one(Double32))) isa Double32
        @test @inferred(fma(one(Double32), one(Double32), one(Double32))) isa Double32
        for a in allp, b in allp
            for op in (+, -, *, /)
                @test iscanonical(op(a, b))
            end
            @test iscanonical(fma(a, b, a))
        end
        for a in allp
            @test iscanonical(-a)
            @test iscanonical(abs(a))
            @test iscanonical(sqrt_or_nan(a))
            @test iscanonical(inv(a))
            @test iscanonical(ldexp(a, 7))
        end
    end

    @testset "accuracy against BigFloat, relative bounds" begin
        # Design section 9: relative bounds exclude the unavoidable
        # underflow/overflow zones; near zero the metric is absolute.
        for (nm, op, rop, bound) in (("+", +, +, 3.0),
                                     ("-", -, -, 3.0),
                                     ("*", *, *, 4.0),
                                     ("/", /, /, 6.0))
            w, wc = worst_relerr(op, rop, spool)
            @test w <= bound * U2
            w2, _ = worst_relerr(op, rop, rpool)
            @test w2 <= bound * U2
        end
        w, _ = worst_relerr1(x -> sqrt(abs(x)), x -> sqrt(abs(x)), vcat(spool, rpool))
        @test w <= 2.0 * U2
        w, _ = worst_relerr1(inv, inv, vcat(spool, rpool))
        @test w <= 4.0 * U2
    end

    @testset "`/`, `inv`, `sqrt` keep full precision at exponent extremes" begin
        # Before the operands were scaled to a unit binade, these lost ~13 bits
        # silently: the residual `x - q0*y` fell below 2f0^-149. Keep this test
        # if the scaling is ever "optimized" away.
        hard = [canonicalize(2.0165723f-36, 8.0f-44),
                canonicalize(4.0026174f-18, -9.910894f-26),
                canonicalize(2f0^-120, 2f0^-146),
                canonicalize(2f0^120, 2f0^96),
                canonicalize(floatmin(Float32), 0.0f0),
                canonicalize(2f0^-100, 2f0^-130)]
        for a in hard, b in hard
            e = exact(a) / exact(b)
            inrelrange(e) || continue
            @test relerr(a / b, e) <= 8.0 * U2
        end
        for a in hard
            e = sqrt(exact(a))
            inrelrange(e) && @test relerr(sqrt(a), e) <= 2.0 * U2
            e = inv(exact(a))
            inrelrange(e) && @test relerr(inv(a), e) <= 4.0 * U2
        end
    end

    @testset "fma: error relative to the operand scale" begin
        # A compensated FMA cannot deliver a small *relative* error when the
        # result is a near-total cancellation; the meaningful bound is against
        # M = max(|x*y|, |z|). The derived bound (src/core/arithmetic.jl,
        # fma_dd_) is 2u²(1+80u)·M for M in [2^-77, 2^126]; the pool below sits
        # inside that window, so the assertion is the proof's constant, not a
        # tolerance.
        p = spool[1:min(40, end)]
        worst = 0.0
        for a in p, b in p, c in p
            g = fma(a, b, c)
            isfinite(g) || continue
            ea, eb, ec = exact(a), exact(b), exact(c)
            e = ea * eb + ec
            scale = max(abs(ea * eb), abs(ec))
            (isfinite(e) && big(2.0)^-77 < scale < big(2.0)^80) || continue
            worst = max(worst, Float64(abs(reference(g) - e) / scale))
        end
        @test worst <= 2.01 * U2
    end

    @testset "fma survives the cancellation case that broke the design sketch" begin
        x = canonicalize(-1.0f0, -2f0^-24)
        y = canonicalize(3.0f0 - 2f0^-23, 2f0^-24)
        z = canonicalize(3.0f0, -5.3290705f-15)
        g = fma(x, y, z)
        e = exact(x) * exact(y) + exact(z)
        @test relerr(g, e) <= 8.0 * U2
        @test muladd(x, y, z) === g
    end

    @testset "non-finite behavior matches Float32" begin
        vals = Float32[0, -0.0, 1, -1, Inf, -Inf, NaN32, floatmax(Float32)]
        for a in vals, b in vals
            for (op, ref) in ((+, +), (-, -), (*, *), (/, /))
                g = op(Double32(a), Double32(b))
                r = ref(a, b)
                if isnan(r)
                    @test isnan(g)
                elseif !isfinite(r) || iszero(r)
                    @test g.hi === r
                end
            end
        end
        @test one(Double32) / zero(Double32) === POS_INF
        @test -one(Double32) / zero(Double32) === NEG_INF
        @test isnan(zero(Double32) / zero(Double32))
        @test isnan(POS_INF - POS_INF)
        @test isnan(POS_INF * zero(Double32))
        @test one(Double32) / POS_INF === POS_ZERO
        @test one(Double32) / NEG_INF === NEG_ZERO
    end

    @testset "multiplication guard bands (Tier 2.2)" begin
        # Overflow-rescuable: leading product rounds to Inf, cross terms bring
        # the true product back inside range. Exact expected pair derived in
        # the mul_guarded docstring proof.
        x2 = Double32(2f0^64, -2f0^40)
        g  = x2 * x2
        @test g === raw(prevfloat(floatmax(Float32)), 2f0^80)
        @test exact(g) == exact(x2)^2                        # rescue is exact here
        # One canonical step further down: still finite, hi == floatmax.
        x3 = Double32(2f0^64, -2f0^39)
        g3 = x3 * x3
        @test isfinite(g3) && g3.hi === floatmax(Float32)
        @test relerr(g3, exact(x3)^2) <= 4.0 * U2
        # Just over the threshold: Inf is the correct rounding.
        x4 = Double32(2f0^64, 2f0^40)
        @test x4 * x4 === POS_INF
        @test Double32(-2f0^64) * Double32(2f0^64) === NEG_INF

        # Underflow band [2^-126, 2^-98]: pre-guard the kernel silently lost
        # ~23 bits here (measured 2.3e7 u² relative). The guard's obligation is
        # the nearest *representable* pair: below 2^-102 the canonical low word
        # itself tapers into the subnormals and dies entirely at 2^-126 (where
        # half-ulp(hi) = 2^-150 < 2^-149), so the bound is absolute —
        # a few u²·|V| for the algorithm plus one subnormal quantum for the
        # format. A pure relative bound here would be asserting bits the format
        # cannot store.
        rng = MersenneTwister(77)
        worstband = 0.0
        n = 0
        while n < 400
            ex = -80 + rand(rng, -20:20); ey = -40 - rand(rng, 0:40)
            a = canonicalize(ldexp(1.0f0 + rand(rng, Float32), ex),
                             ldexp(rand(rng, Float32), ex - 24 - rand(rng, 0:20)))
            b = canonicalize(ldexp(1.0f0 + rand(rng, Float32), ey),
                             ldexp(rand(rng, Float32), ey - 24 - rand(rng, 0:20)))
            V = exact(a) * exact(b)
            (big(2.0)^-126 < abs(V) < big(2.0)^-98) || continue
            n += 1
            excess = Float64((abs(reference(a * b) - V) - big(2.0)^-149) / abs(V))
            worstband = max(worstband, excess)
        end
        @test worstband <= 4.0 * U2
        # Exact zeros route through the guard and keep IEEE signs.
        @test Double32(0.0f0) * Double32(-0.0f0) === NEG_ZERO
        @test Double32(-2f0^-120) * zero(Double32) === NEG_ZERO
        # Products that underflow the *high* word entirely: correct signed zero.
        t = Double32(2f0^-100) * Double32(-2f0^-100)
        @test iszero(t) && signbit(t)
    end

    @testset "division overflow rescue (Tier 2.2 — NaN regression)" begin
        # Pre-rescue this returned NaN: q0 = Inf poisoned the residual with
        # Inf - Inf. True quotient is floatmax + 2^80 — representable.
        x = Double32(2f0^127)
        y = Double32(0.5f0, 2f0^-25)
        g = x / y
        @test !isnan(g)
        @test isfinite(g) && g.hi === floatmax(Float32)
        @test relerr(g, exact(x) / exact(y)) <= 8.0 * U2
        # Just past the threshold: Inf is the correct rounding.
        y2 = Double32(0.5f0, 2f0^-26)
        @test x / y2 === POS_INF
        # Saturation far beyond range stays Inf, with sign.
        @test Double32(2f0^100) / Double32(3LOMIN) === POS_INF
        @test Double32(-2f0^100) / Double32(3LOMIN) === NEG_INF
    end

    @testset "faithfulness against the 48-bit lattice (Tier 2.4)" begin
        # Error in nominal-48-bit ulps of the true value, where that lattice
        # exists (|V| >= 2^-100 keeps the result's low word representable).
        # Recorded decisions: div <= 4.0 (not faithful, single correction kept
        # for GPU cost), inv <= 1.16, sqrt faithful (0 violations).
        ulp48(V::BigFloat) = big(2.0)^(Int(floor(log2(abs(V)))) - 47)
        errU(g, V) = Float64(abs(reference(g) - V) / ulp48(V))
        inlat(V) = isfinite(V) && !iszero(V) && big(2.0)^-100 <= abs(V) <= big(2.0)^120
        pool = [x for x in vcat(spool, rpool[1:60]) if isfinite(x) && !iszero(x) &&
                -45 < exponent(x.hi) < 45]
        dmax = 0.0; imax = 0.0; smax = 0.0; sviol = 0
        for a in pool, b in pool
            V = exact(a) / exact(b)
            inlat(V) && (dmax = max(dmax, errU(a / b, V)))
        end
        for a in pool
            Vi = inv(exact(a))
            inlat(Vi) && (imax = max(imax, errU(inv(a), Vi)))
            Vs = sqrt(abs(exact(a)))
            if inlat(Vs)
                e = errU(sqrt(abs(a)), Vs)
                smax = max(smax, e)
                e >= 1.0 && (sviol += 1)
            end
        end
        @test dmax <= 4.5          # measured 4.00 over the full 101k sweep
        @test imax <= 1.3          # measured 1.16
        @test smax < 1.0           # sqrt is faithful on everything measured
        @test sviol == 0
    end

    @testset "overflow saturates and the multiplication rescue works" begin
        fm = floatmax(Double32)
        @test isfinite(fm)
        @test isinf(fm + fm)
        @test isinf(fm * Double32(2.0f0))
        @test canonicalize(floatmax(Float32), ldexp(1.0f0, 103)) === POS_INF
        @test floatmax(Double32).hi === floatmax(Float32)
        # the guarded multiplication path must recompute scaled, not simply
        # report Inf: a product whose leading term overflows can still be finite
        big1 = Double32(2f0^70)
        big2 = Double32(2f0^70)
        @test isinf(big1 * big2)                    # genuinely 2^140
        small = Double32(2f0^-70)
        @test big1 * small === one(Double32)        # exact, no spurious overflow
    end

    @testset "sign operations" begin
        for x in allp
            @test (-(-x)) === x
            @test isequal(abs(x), abs(-x))
            @test signbit(-x) != signbit(x) || isnan(x.hi)
            @test flipsign(x, 1.0f0) === x
            @test flipsign(x, -1.0f0) === -x
            @test copysign(x, 1.0f0).hi === abs(x).hi
            isfinite(x) && @test exact(abs(x)) == abs(exact(x))
            isfinite(x) && @test exact(-x) == -exact(x)
        end
        @test abs(Double32(-0.0f0)) === raw(0.0f0, -0.0f0)   # documented, invariant 8
        @test iszero(abs(Double32(-0.0f0)))
        @test abs2(Double32(3.0f0)) === Double32(9.0f0)
    end

    @testset "exact identities" begin
        for x in vcat(spool, wpool)
            isfinite(x) || continue
            # `x + 0` is `+0` for `x == -0`, per IEEE — so these are stated on
            # the value, with the zero cases split out below.
            @test isequal(x + zero(Double32), x) || iszero(x)
            @test isequal(x - zero(Double32), x) || iszero(x)
            @test x * one(Double32) === x
            @test x / one(Double32) === x
            @test iszero(x - x)
            @test !signbit(x - x)                    # IEEE: x - x is +0
            @test x + x === Double32(2.0f0) * x || !isfinite(x + x)
        end
        # IEEE signed-zero results, which the pair kernels lose without the
        # `zero_sign` fixup in the layer-2 methods.
        @test Double32(-0.0f0) + Double32(-0.0f0) === NEG_ZERO
        @test Double32(-0.0f0) + Double32(0.0f0) === POS_ZERO
        @test Double32(-0.0f0) - Double32(0.0f0) === NEG_ZERO
        @test Double32(0.0f0) * Double32(-0.0f0) === NEG_ZERO
        @test Double32(-0.0f0) * Double32(-0.0f0) === POS_ZERO
        @test Double32(-1.0f0) * Double32(0.0f0) === NEG_ZERO
    end

    @testset "ldexp, exponent, frexp, significand" begin
        for x in vcat(spool, wpool)
            (isfinite(x) && !iszero(x)) || continue
            for n in (-30, -1, 0, 1, 30)
                y = ldexp(x, n)
                isfinite(y) || continue
                @test iscanonical(y)
                # Scaling by a power of two is exact unless a word enters the
                # subnormal range — including all the way to zero — which is
                # the documented precision taper of design section 5.5. It
                # applies to the high word too: `ldexp(floatmin(Float32), -1)`
                # already rounds.
                (isnormal(raw(y.hi, 0.0f0)) &&
                 (iszero(x.lo) || isnormal(raw(y.lo, 0.0f0)))) &&
                    @test exact(y) == exact(x) * big(2.0)^n
            end
            @test exponent(x) == exponent(x.hi)
            f, e = frexp(x)
            @test isfinite(f) && 0.5 <= abs(f.hi) < 1
            (isnormal(raw(f.hi, 0.0f0)) &&
             (iszero(x.lo) || isnormal(raw(f.lo, 0.0f0)))) &&
                @test exact(ldexp(f, e)) == exact(x)
            @test 1 <= abs(significand(x).hi) < 2
        end
        @test ldexp(one(Double32), 200) === POS_INF
        @test iszero(ldexp(one(Double32), -200))
    end

    @testset "policies dispatch, do not branch" begin
        for a in spool[1:40], b in spool[1:40]
            @test Double32s.add(StrictPolicy(), a, b) === a + b
            @test Double32s.sub(StrictPolicy(), a, b) === a - b
            @test Double32s.mul(StrictPolicy(), a, b) === a * b
            @test Double32s.divide(StrictPolicy(), a, b) === a / b
            # FastPolicy trades the last-bit bound for operation count; all it
            # owes is Float32 accuracy, which is the point of having it.
            g = Double32s.add(FastPolicy(), a, b)
            e = exact(a) + exact(b)
            inrelrange(e) && @test relerr(g, e) <= U
        end
        @test @inferred(Double32s.add(FastPolicy(), one(Double32), one(Double32))) isa Double32
    end

    @testset "sqrt exceptional values and DomainError" begin
        @test sqrt(zero(Double32)) === POS_ZERO
        @test sqrt(Double32(-0.0f0)) === NEG_ZERO
        @test sqrt(POS_INF) === POS_INF
        @test isnan(sqrt(D32_NAN))
        @test_throws DomainError sqrt(-one(Double32))
        @test_throws DomainError sqrt(NEG_INF)
        @test isnan(sqrt_or_nan(-one(Double32)))     # the kernel-safe spelling
        @test isnan(sqrt_or_nan(NEG_INF))
        @test sqrt_or_nan(Double32(-0.0f0)) === NEG_ZERO
        for x in vcat(spool, rpool)
            (isfinite(x) && !signbit(x) && !iszero(x)) || continue
            e = sqrt(exact(x))
            inrelrange(e) && @test relerr(sqrt(x), e) <= 2.0 * U2
        end
        # perfect squares come back exact
        for n in (1, 4, 9, 16, 1024, 2^20)
            @test sqrt(Double32(n * n)) === Double32(n)
        end
    end

    @testset "hypot" begin
        @test hypot(Double32(3), Double32(4)) === Double32(5)
        @test hypot(POS_INF, D32_NAN) === POS_INF
        @test isnan(hypot(D32_NAN, one(Double32)))
        @test hypot(zero(Double32), zero(Double32)) === POS_ZERO
        for a in spool[1:30], b in spool[1:30]
            (isfinite(a) & isfinite(b)) || continue
            e = sqrt(exact(a)^2 + exact(b)^2)
            inrelrange(e) && @test relerr(hypot(a, b), e) <= 8.0 * U2
        end
        # no premature overflow or underflow
        @test isfinite(hypot(Double32(1f30), Double32(1f30)))
        @test !iszero(hypot(Double32(1f-30), Double32(1f-30)))
    end

    @testset "integer powers come free from Base" begin
        x = canonicalize(1.0f0, 2f0^-30)
        @test x^0 === one(Double32)
        @test x^1 === x
        @test relerr(x^2, exact(x)^2) <= 4.0 * U2
        @test relerr(x^3, exact(x)^3) <= 8.0 * U2
        @test relerr(x^-1, inv(exact(x))) <= 4.0 * U2
    end
end
