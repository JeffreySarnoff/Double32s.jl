@testset "comparisons, ordering, hashing" begin
    surface = zero_surface()
    pool    = vcat(structured_pool(), wide_gap_pairs(), surface, NONFINITE)

    # Reference semantics, taken from the *represented value* plus the sign of a
    # zero read off the HIGH word. This is what pins the predicates down;
    # trichotomy alone does not (see below).
    function want_isequal(x, y)
        isnan(x.hi) && return isnan(y.hi)
        isnan(y.hi) && return false
        return exact(x) == exact(y) && signbit(x.hi) == signbit(y.hi)
    end
    function want_isless(x, y)
        isnan(x.hi) && return false
        isnan(y.hi) && return true
        ex, ey = exact(x), exact(y)
        return ex < ey || (ex == ey && signbit(x.hi) && !signbit(y.hi))
    end

    @testset "isequal / isless match the represented value" begin
        for x in pool, y in pool
            @test isequal(x, y) == want_isequal(x, y)
            @test isless(x, y) == want_isless(x, y)
        end
    end

    @testset "trichotomy (necessary, and NOT sufficient)" begin
        for x in pool, y in pool
            @test count((isless(x, y), isless(y, x), isequal(x, y))) == 1
        end
    end

    @testset "the zero rule: a zero low word's sign is never information" begin
        # Reachable pairs that hold one value in two bit patterns. Each of these
        # broke an earlier draft that used `isequal` on the low words behind an
        # `iszero(hi)` guard — and each passes trichotomy, which is why the
        # semantic tests above are the ones that matter.
        cases = [(-one(Double32),          Double32(-1.0f0)),
                 (-Double32(1.5f0),        Double32(-1.5f0)),
                 (-D32_NAN,                D32_NAN),
                 (abs(Double32(-0.0f0)),   zero(Double32))]
        for (x, y) in cases
            @test x.lo !== y.lo                       # genuinely different bits
            @test isequal(x, y)
            @test !isless(x, y)
            @test !isless(y, x)
            @test hash(x) == hash(y)
        end
        # ... and the Base behavior they must mirror
        @test isequal(NaN32, -NaN32)
        @test isequal(-1.0f0, -1.0f0)
    end

    @testset "isequal implies Dict and unique agree" begin
        for (x, y) in ((-one(Double32), Double32(-1.0f0)),
                       (-Double32(1.5f0), Double32(-1.5f0)))
            d = Dict(x => "v")
            @test haskey(d, y)
            @test d[y] == "v"
            @test length(unique([x, y])) == 1
        end
    end

    @testset "the four zero pairs are all reachable and all equal" begin
        z = [raw(h, l) for h in (0.0f0, -0.0f0), l in (0.0f0, -0.0f0)]
        @test abs(Double32(-0.0f0)) === raw(0.0f0, -0.0f0)
        @test -zero(Double32) === raw(-0.0f0, -0.0f0)
        @test -abs(Double32(-0.0f0)) === raw(-0.0f0, 0.0f0)
        @test flipsign(zero(Double32), -1.0f0) === raw(-0.0f0, -0.0f0)
        @test copysign(zero(Double32), -1.0f0) === raw(-0.0f0, -0.0f0)
        for a in z, b in z
            @test a == b                                    # all are numerically 0
            @test isequal(a, b) == (signbit(a.hi) == signbit(b.hi))
        end
    end

    @testset "== and < follow IEEE" begin
        for x in pool, y in pool
            ex, ey = exact(x), exact(y)
            if isnan(x.hi) || isnan(y.hi)
                @test !(x == y) && !(x < y) && !(x <= y)
            else
                @test (x == y) == (ex == ey)
                @test (x < y)  == (ex < ey)
                @test (x <= y) == (ex <= ey)
                @test (x > y)  == (ex > ey)
                @test (x >= y) == (ex >= ey)
                @test (x != y) == (ex != ey)
            end
        end
        @test Double32(-0.0f0) == Double32(0.0f0)
        @test !isequal(Double32(-0.0f0), Double32(0.0f0))
    end

    @testset "sort! agrees with sorting the exact values" begin
        v = shuffle(MersenneTwister(11), filter(!isnan, pool))
        @test map(exact, sort(v)) == sort(map(exact, v))
        # a vector mixing bit-distinct spellings of equal values stays stable
        w = [-one(Double32), Double32(-1.0f0), one(Double32), zero(Double32)]
        @test issorted(sort(w))
    end

    @testset "mixed Double32/Float64 is exact, not promotion-based" begin
        # Promotion would call these equal; the values differ in the 49th bit.
        x = canonicalize(1.0f0, LOMIN)
        @test !float64_reference_is_exact(x)
        @test Float64(x) == 1.0                  # the lossy conversion
        @test !(x == 1.0)                        # the exact comparison
        @test x > 1.0
        @test 1.0 < x

        ys = Float64[0.0, -0.0, 1.0, -1.0, nextfloat(1.0), prevfloat(1.0),
                     2.0^-149, 2.0^-1074, Inf, -Inf, NaN, floatmax(Float64),
                     -floatmax(Float64), 1.5, 3.0e38, -3.0e38, 0.1]
        for a in pool, y in ys
            ea = isnan(a.hi) ? nothing : exact(a)
            ey = isnan(y) ? nothing : big(y)
            if ea === nothing || ey === nothing
                @test !(a == y) && !(a < y) && !(y < a) && !(a <= y) && !(y <= a)
            else
                @test (a == y) == (ea == ey)
                @test (y == a) == (ea == ey)
                @test (a < y)  == (ea < ey)
                @test (y < a)  == (ey < ea)
                @test (a <= y) == (ea <= ey)
                @test (y <= a) == (ey <= ea)
            end
        end
    end

    @testset "comparison against Float32 and integers is exact by promotion" begin
        for a in vcat(structured_pool(), wide_gap_pairs()), f in Float32[0, 1, -1, 1.5, 3]
            @test (a == f) == (exact(a) == big(f))
            @test (a < f)  == (exact(a) < big(f))
        end
        @test Double32(3) == 3
        @test Double32(3) < 4
        @test 2 < Double32(3)
        @test one(Double32) == 1.0f0
    end

    @testset "min / max mirror Base on Float32" begin
        specials = Float32[1, -1, 0.0, -0.0, Inf, -Inf, NaN32]
        for a in specials, b in specials
            @test isequal(min(Double32(a), Double32(b)).hi, min(a, b))
            @test isequal(max(Double32(a), Double32(b)).hi, max(a, b))
        end
        # NaN must propagate from either side — the defect the design's first
        # formulation had, which turns a poisoned accumulator clean.
        @test isnan(min(one(Double32), D32_NAN))
        @test isnan(min(D32_NAN, one(Double32)))
        @test isnan(max(one(Double32), D32_NAN))
        @test isnan(max(D32_NAN, one(Double32)))
        # signed zero must be ordered
        @test signbit(min(Double32(0.0f0), Double32(-0.0f0)))
        @test signbit(min(Double32(-0.0f0), Double32(0.0f0)))
        @test !signbit(max(Double32(0.0f0), Double32(-0.0f0)))
        for x in pool, y in pool
            (isnan(x.hi) | isnan(y.hi)) && continue
            @test minmax(x, y) === (min(x, y), max(x, y))
            @test !isless(max(x, y), min(x, y))
        end
    end

    @testset "hash contract" begin
        for x in pool, y in pool
            isequal(x, y) && @test hash(x) == hash(y)
        end
        # cross-type consistency, on the subset where the conversion is exact
        for x in vcat(structured_pool(), wide_gap_pairs())
            isfinite(x) || continue
            float64_reference_is_exact(x) || continue
            @test hash(x) == hash(Float64(x))
        end
        @test hash(Double32(3)) == hash(3)
        @test hash(one(Double32)) == hash(1.0f0) == hash(1)
        # Base does NOT hash ±0 alike, because isequal distinguishes them; the
        # requirement is only that our hash agree with Float64's on each sign.
        @test !isequal(Double32(-0.0f0), Double32(0.0f0))
        @test hash(Double32(-0.0f0)) == hash(-0.0)
        @test hash(Double32(0.0f0)) == hash(0.0)
        @test hash(NEG_ZERO) == hash(raw(-0.0f0, -0.0f0))         # isequal, so must match
    end
end
