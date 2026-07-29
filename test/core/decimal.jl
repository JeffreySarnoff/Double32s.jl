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

    @testset "MIME text/plain is the decimal; repr stays the pair" begin
        x = Double32(1.0f0) / 3
        @test sprint(show, MIME("text/plain"), x) == Double32s.decimal(x)
        @test startswith(repr(x), "Double32(")
        # Both promises at once: repr parses back through the constructor
        # (existing contract), decimal parses back through `parse` (new one).
        @test isequal(parse(Double32, sprint(show, MIME("text/plain"), x)), x)
    end
end
