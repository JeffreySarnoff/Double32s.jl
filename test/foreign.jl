# test/foreign.jl — the BFloat16s and DoubleFloats extensions (design §11.3
# rule 4) and the narrowing-conversion rounding rule they share with
# `Float16(::Double32)`.
#
# The reference for every tie decision below is constructed independently of
# the implementation: a tie midpoint `m` of the narrow lattice is exact in
# `Float32`, so for a pair `(m, lo)` the correctly rounded narrow result is
# decided by `sign(lo)` alone (and by significand parity when `lo == 0`).
# Agreement with `to_odd_hi`'s own reasoning is not used as evidence anywhere.

using BFloat16s: BFloat16
using TOML
import DoubleFloats
const DF = DoubleFloats.DoubleFloat

# Narrow-format tie pool for a target type `T` reached from Float32.
# Returns (pair, below, above, lo) tuples: `pair` is canonical with high word
# exactly on the tie between narrow neighbors `below` and `above`.
function _narrow_ties(::Type{T}, patterns) where {T}
    out = Tuple{Double32,T,T,Float32}[]
    los = Float32[]
    for u in patterns
        b = reinterpret(T, u)
        (isfinite(b) && b >= zero(T)) || continue
        nb = nextfloat(b)
        isfinite(nb) || continue
        # Overflow-safe midpoint: (a + b)/2 overflows Float32 at the top of
        # the BFloat16 range, where a + b > floatmax(Float32).
        m = Float32(b) + (Float32(nb) - Float32(b)) / 2
        isfinite(m) || continue
        empty!(los)
        # subclass 1 (§23.11): subnormal low words
        append!(los, (LOMIN, 3LOMIN, -LOMIN, -3LOMIN))
        # subclass 2: normal low words far below the half-ulp bound
        for k in (2, 10, 20), s in (1.0f0, -1.0f0)
            l = ldexp(1.0f0, exponent(m) - 24 - k)
            (isfinite(l) && !iszero(l) && !issubnormal(l)) && push!(los, s * l)
        end
        for l in los
            abs(l) <= eps(m) / 4 || continue     # keep (m, l) canonical
            push!(out, (raw(m, l), b, nb, l))
            push!(out, (raw(-m, -l), -nb, -b, -l))   # negative mirror
        end
    end
    return out
end

@testset "foreign types" begin

    @testset "extensions load" begin
        @test Base.get_extension(Double32s, :Double32sBFloat16sExt) !== nothing
        @test Base.get_extension(Double32s, :Double32sDoubleFloatsExt) !== nothing
        # Optionality: neither may appear as a hard dependency.
        proj = TOML.parsefile(joinpath(pkgdir(Double32s), "Project.toml"))
        for pkg in ("BFloat16s", "DoubleFloats")
            @test !haskey(proj["deps"], pkg)
            @test haskey(proj["weakdeps"], pkg)
        end
    end

    @testset "narrowing conversion ties" begin
        # Float16: stride through the finite positive patterns plus the
        # boundary set; BFloat16: same idea on its patterns.
        f16 = _narrow_ties(Float16,
                vcat(collect(UInt16(0x0001):UInt16(0x009b):UInt16(0x7bfe)),
                     UInt16[0x0001, 0x03ff, 0x0400, 0x3bff, 0x3c00, 0x7bfe]))
        bf16 = _narrow_ties(BFloat16,
                vcat(collect(UInt16(0x0001):UInt16(0x009d):UInt16(0x7f7e)),
                     UInt16[0x0001, 0x007f, 0x0080, 0x3f7f, 0x3f80, 0x7f7e]))
        @test length(f16) > 2000
        @test length(bf16) > 2000
        # Coverage as an assertion: both §23.11 subclasses must be present.
        for ties in (f16, bf16)
            @test any(t -> issubnormal(t[4]), ties)
            @test any(t -> !issubnormal(t[4]), ties)
        end

        for (ties, conv) in ((f16, Float16), (bf16, BFloat16))
            naive_wrong = 0
            for (p, b, nb, l) in ties
                want = l > 0 ? nb : b       # independent reference: sign(lo) decides the tie
                @test conv(p) === want
                naive_wrong += (conv(p.hi) !== want)
            end
            # The retired rule — narrow the high word alone — must be visibly
            # wrong on this pool: ties-to-even disagrees with the lo-decided
            # direction on half the operands. If this count ever reaches zero
            # the pool has stopped containing ties and the test above is
            # passing vacuously.
            @test naive_wrong > length(ties) ÷ 4
        end

        # Ties with lo == 0 stay ties: even significand wins, matching a
        # direct narrow of the midpoint.
        for (p, b, nb, l) in f16[1:50]
            m = p.hi > 0 ? p.hi : -p.hi
            @test Float16(Double32(m)) === Float16(m)
        end
    end

    @testset "BFloat16 exhaustive round-trip (65536 patterns)" begin
        exact_count = 0
        for u in UInt16(0x0000):UInt16(0xffff)
            b = reinterpret(BFloat16, u)
            d = Double32(b)
            if isnan(b)
                @test isnan(d)
            else
                # Widening is exact: high word is the same value, low word zero.
                @test d.hi == Float32(b) && iszero(d.lo)
                @test BFloat16(d) == b
                exact_count += 1
            end
        end
        @test exact_count > 65000
        @test promote_type(Double32, BFloat16) === Double32
        x = Double32(1.0f0) / 3
        @test x + BFloat16(0.5) === x + Double32(0.5f0)
    end

    @testset "DoubleFloats: exact directions" begin
        pool = vcat(structured_pool(), wide_gap_pairs(), zero_surface(),
                    random_pool(500), NONFINITE)
        for x in pool
            d64 = DF{Float64}(x)
            d32 = DF{Float32}(x)
            if isfinite(x)
                # ours → Double64 is exact: both words are Float64 values and
                # two_sum is error free. Reference is the exact pair sum.
                @test big(d64.hi) + big(d64.lo) == reference(x)
                # ours → their Double32 is word-for-word.
                @test d32.hi === x.hi && d32.lo === x.lo
            end
            # Round trips land on the same value (canonical bits may differ
            # only where isequal deliberately ignores them).
            @test isequal(Double32(d64), x)
            @test isequal(Double32(d32), x)
        end
        # Signed zero survives both directions (the BigFloat route alone
        # would erase it: big(-0.0) + big(0.0) is +0.0).
        nz64 = DF{Float64}(NEG_ZERO)
        @test signbit(nz64.hi)
        @test Double32(nz64) === NEG_ZERO
        @test Double32(DF{Float32}(NEG_ZERO)) === NEG_ZERO
    end

    @testset "Double64 → Double32 correct rounding at lattice ties" begin
        # Construct Double64 operands sitting exactly on, and 2^-1060 to
        # either side of, midpoints of the Double32 pair lattice. Verified by
        # neighbor minimality — no adjacent pair is closer to the exact value
        # — never by re-running the converter's own algorithm.
        n = 0; minimality_failures = 0; tie_parity_failures = 0
        setprecision(BigFloat, 2500) do
            for h in (1.0f0, prevfloat(1.0f0), 2f0^-96, 2f0^-30,
                      floatmax(Float32) / 2, Float32(1.5^10)),
                l in (0.5f0 * eps(h), -0.25f0 * eps(h), 2f0^-149, -3 * 2f0^-149)

                p = Double32(h, l)
                lm = (big(p.lo) + big(nextfloat(p.lo))) / 2
                for e64 in (big(0), big(2.0)^-1060, -big(2.0)^-1060)
                    v = big(p.hi) + lm + e64
                    h64 = Float64(v); l64 = Float64(v - big(h64))
                    d = DF(h64, l64)
                    big(d.hi) + big(d.lo) == v || continue
                    r = Double32(d)
                    dist = abs(reference(r) - v)
                    for alt in (raw(r.hi, nextfloat(r.lo)),
                                raw(r.hi, prevfloat(r.lo)),
                                Double32(nextfloat(r.hi), r.lo),
                                Double32(prevfloat(r.hi), r.lo))
                        minimality_failures += (abs(reference(alt) - v) < dist)
                    end
                    # Exact tie: nearest-even on the low word — but only where
                    # the tie is interior to the high word's binade. When the
                    # midpoint coincides with a Float32 rounding boundary of
                    # the high word itself (lo at exactly ±½ulp(hi)), "even"
                    # is decided at the high word and the low-word parity
                    # claim does not apply.
                    if iszero(e64) && r.hi === p.hi &&
                       abs(r.lo) < eps(r.hi) / 2
                        tie_parity_failures +=
                            (reinterpret(UInt32, r.lo) & 0x1 != 0)
                    end
                    n += 1
                end
            end
        end
        @test n > 40                       # the sweep must not silently shrink
        @test minimality_failures == 0
        @test tie_parity_failures == 0
    end

    @testset "promotion lattice with foreign types" begin
        @test promote_type(Double32, DF{Float64}) === DF{Float64}
        @test promote_type(DF{Float64}, Double32) === DF{Float64}
        @test promote_type(Double32, DF{Float16}) === Double32
        @test promote_type(DF{Float16}, Double32) === Double32
        # Same-width pair deliberately has no rule (§11.3 table: conversion
        # only); a mixed operation without an explicit conversion must error
        # rather than silently privilege one package's semantics.
        x = Double32(1.0f0) / 3
        @test_throws ErrorException x + DF{Float32}(0.5f0)
        # Mixed ops through the defined rules, checked against BigFloat.
        y = x + DoubleFloats.Double64(1)
        @test y isa DF{Float64}
        @test abs(big(y.hi) + big(y.lo) - (reference(x) + 1)) <=
              abs(reference(x) + 1) * big(2.0)^-100
        z = x + DF{Float16}(0.5)
        @test z === x + Double32(0.5f0)
        # No new ambiguities arrive with both extensions loaded.
        @test isempty(Test.detect_ambiguities(Double32s))
    end
end
