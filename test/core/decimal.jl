# test/core/decimal.jl — three-argument decimal `show` and parsing.
#
# Design section 11.5. The promise under test is per-value and checked, not
# analytic: `Double32s.decimal(x)` must return a string whose parse is
# `isequal` to `x`, for every operand class the package recognizes as hard —
# in particular both wide-gap subclasses of section 23.11, where the value
# needs far more than the Float64-habit 17 digits.

using Printf

@testset "decimal show and parse" begin

    pool = vcat(structured_pool(), wide_gap_pairs(), zero_surface(),
                random_pool(2000), NONFINITE)

    # The pool must actually contain the hard subclasses; a pool that decays
    # to easy operands would keep passing everything below while testing
    # nothing (Tier 0.1's failure mode). Assert coverage, not hope.
    @test any(x -> isfinite(x) && !iszero(x.lo) && issubnormal(x.lo), pool)
    @test any(x -> isfinite(x) && !iszero(x.lo) && !issubnormal(x.lo) &&
                   exponent(x.lo) < exponent(x.hi) - 84, pool)

    @testset "round-trip over the pool ($(length(pool)) operands)" begin
        for x in pool
            s = Double32s.decimal(x)
            @test isequal(tryparse(Double32, s), x)
        end
    end

    @testset "the 17-digit habit is visibly wrong on the hard subclass" begin
        # A wide-gap pair carries ~173 bits between the words; 17 significant
        # digits cannot round-trip it. This keeps the pool honest: if the
        # generators ever stop producing wide-gap pairs, the coverage asserts
        # above fail; if parsing ever starts ignoring the low word, this one
        # does.
        wg = canonicalize(1.0f0, LOMIN)
        s17 = setprecision(BigFloat, 400) do
            Printf.format(Printf.Format("%.17g"), big(wg.hi) + big(wg.lo))
        end
        @test !isequal(tryparse(Double32, s17), wg)
        @test length(Double32s.decimal(wg)) > 17
    end

    @testset "specials and failure modes" begin
        @test isequal(parse(Double32, "NaN"), D32_NAN)
        @test parse(Double32, "Inf") === POS_INF
        @test parse(Double32, "-Inf") === NEG_INF
        @test parse(Double32, "0.0") === POS_ZERO
        @test parse(Double32, "-0.0") === NEG_ZERO
        @test tryparse(Double32, "not-a-number") === nothing
        @test tryparse(Double32, "") === nothing
        @test_throws ArgumentError parse(Double32, "not-a-number")
    end

    @testset "one rendering, every display path" begin
        # The defect this replaces: the decimal lived on a three-argument
        # `show` while two-argument `show` printed the pair, so the rendering
        # depended on the container. `[x]` printed a decimal (Base's array
        # code tries three-argument first) while `(x,)`, `string(x)`, and
        # interpolation printed the pair. Assert agreement directly, on every
        # path Base routes through, so a reintroduced three-argument method
        # fails here.
        for x in (Double32(1.0f0) / 3, sqrt(Double32(5)), NEG_ZERO, POS_INF,
                  canonicalize(1.0f0, LOMIN))
            d = Double32s.decimal(x)
            @test repr(x) == d
            @test string(x) == d
            @test "$x" == d
            @test sprint(show, MIME("text/plain"), x) == d
            @test sprint(show, (x,)) == "($d,)"
            @test strip(Base.split(sprint(show, MIME("text/plain"), [x]), "\n")[2]) == d
            @test occursin("=> $d", sprint(show, Dict(1 => x)))
        end

        # No three-argument method of our own: the generic Base fallback must
        # be what handles the MIME.
        @test which(show, Tuple{IO, MIME"text/plain", Double32}).module === Base.Multimedia

        # Reconstruction is `parse`, not `eval` — BigFloat's arrangement.
        x = Double32(1.0f0) / 3
        @test parse(Double32, repr(x)) === x
        @test eval(Meta.parse(repr(x))) isa Float64      # documented, not a defect

        # Numeric column alignment depends on `sprint(show, x)` being the
        # printed text (`Base.alignment(::IO, ::Real)` splits it at the first
        # `.`). Under the pair form it split inside the high word and vectors
        # lost decimal-point alignment.
        @test Base.alignment(stdout, Double32(1.0f0) / 3) ==
              Base.alignment(stdout, 0.333333333333333)
    end
end
