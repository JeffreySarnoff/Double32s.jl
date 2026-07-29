# test/core/invariants.jl — the cross-cutting invariants, as regressions.
#
# Each testset here closes an item that `docs/Design/UseFableHere.md` carried as
# open. They are grouped in one file because they share a property: each is a
# claim about *every* operation rather than about one, so the natural form is a
# sweep, and the natural failure mode is a future edit quietly narrowing the
# claim rather than breaking a specific case.

using Double32s: add_dd_, add_dd_fast_, sub_dd_, sub_dd_fast_, mul_dd_,
                 mul_dd_fast_, div_dd_, div_dd_refined_, inv_dd_, sqrt_dd_,
                 fma_dd_, iscanonical, reciprocal, reference
using Base.Order

# `isnormal` is a `Double32` predicate; the word-level question needs its own.
word_is_normal(f::Float32) = !iszero(f) && abs(f) >= floatmin(Float32)

@testset "cross-cutting invariants" begin

    # ---------------------------------------------------------- Tier 0.1 ---
    # The corrected exactness condition is validated elsewhere. What is checked
    # here is that the operand pool still *contains* the subclass the wrong
    # predicate waved through — normal low words with a wide span. A generator
    # edit that dropped it would leave every exactness test passing while
    # testing nothing.
    @testset "wide-gap generator still covers the normal-low-word subclass" begin
        W = wide_gap_pairs()
        normal_lo = filter(x -> word_is_normal(x.lo), W)
        @test length(normal_lo) >= 64

        spans = [exponent(x.hi) - lsb_exponent(x.lo) for x in normal_lo]
        @test minimum(spans) >= 24          # canonicality's floor
        @test maximum(spans) >= 120         # far past Float64's 53

        # The subclass that matters: normal low word *and* inexact in Float64.
        inexact = filter(x -> word_is_normal(x.lo) && !float64_reference_is_exact(x), W)
        @test length(inexact) >= 32

        # The retired predicate — `lo == 0 || lo normal || exponent(hi) <= -97`
        # — must still be visibly wrong on this pool, or the pool has stopped
        # being adversarial.
        retired(x) = iszero(x.lo) || word_is_normal(x.lo) || exponent(x.hi) <= -97
        waved = count(x -> retired(x) && !float64_reference_is_exact(x), W)
        @test waved >= 32
    end

    # ---------------------------------------------------- Tier 0.2 / 3.5 ---
    # Invariant 8: the sign of a zero low word is not information. Anything
    # value-based must be blind to it. `objectid` is deliberately excluded —
    # it distinguishes `0.0` from `-0.0` in Base too, and `IdDict` is
    # identity-based by contract.
    @testset "no method distinguishes low-word zero signs" begin
        S = zero_surface()
        pairs_examined = 0
        for a in S, b in S
            isequal(a, b) || continue
            pairs_examined += 1
            @test isequal(hash(a), hash(b))
            @test isequal(Float64(a), Float64(b))
            @test isequal(Float32(a), Float32(b))
            @test isequal(reference(a), reference(b))
            @test repr(a) == repr(b)
            @test string(a) == string(b)
            @test lt(Forward, a, b) == lt(Forward, b, a)
            @test lt(Reverse, a, b) == lt(Reverse, b, a)
        end
        @test pairs_examined >= 36

        # The two that used to leak, named so a regression is legible.
        @test repr(-one(Double32)) == repr(Double32(-1.0f0))
        @test isequal(reference(raw(-0.0f0, -0.0f0)), reference(raw(-0.0f0, 0.0f0)))

        # A zero *high* word keeps its sign: there it is the value's own sign.
        @test repr(NEG_ZERO) != repr(POS_ZERO)
        @test repr(NEG_ZERO) == "Double32(-0.0f0, 0.0f0)"

        # Collections agree with `isequal` rather than with the bit pattern.
        @test length(Set(S)) == length(unique(S))
        @test length(Dict(a => 1 for a in S)) == length(unique(S))
        @test haskey(Dict(-one(Double32) => 1), Double32(-1.0f0))

        # `canonicalize` is the byte-level normal form, for serialization and
        # for any code that must compare representations rather than values.
        for x in S
            c = canonicalize(x)
            @test !(iszero(c.lo) && signbit(c.lo))
            @test isequal(c, x)
        end
        for a in S, b in S
            isequal(a, b) || continue
            @test reinterpret(UInt32, canonicalize(a).hi) ==
                  reinterpret(UInt32, canonicalize(b).hi)
            @test reinterpret(UInt32, canonicalize(a).lo) ==
                  reinterpret(UInt32, canonicalize(b).lo)
        end
    end

    # ------------------------------------------------------------ Tier 3.3 ---
    # The decision recorded in design section 5.3: no weak pair crosses a
    # function boundary. Every `*_dd_` kernel returns a strongly-normalized
    # pair, which is what lets every wrapper use `raw` instead of paying a
    # closing `canonicalize`.
    @testset "every tuple kernel returns a strongly-normalized pair" begin
        pool = filter(isfinite, unique(vcat(structured_pool(), wide_gap_pairs(),
                                            random_pool(120))))
        canon(t) = iscanonical(raw(t[1], t[2]))
        checked = 0
        ok = true
        function tally(t)
            isfinite(t[1]) || return
            checked += 1
            ok &= canon(t)
        end
        for a in pool
            tally(inv_dd_(words(a)))
            a.hi >= 0 && tally(sqrt_dd_(words(a)))
        end
        for a in pool, b in pool
            wa, wb = words(a), words(b)
            tally(add_dd_(wa, wb)); tally(add_dd_fast_(wa, wb))
            tally(sub_dd_(wa, wb)); tally(sub_dd_fast_(wa, wb))
            tally(mul_dd_(wa, wb)); tally(mul_dd_fast_(wa, wb))
            tally(div_dd_(wa, wb)); tally(div_dd_refined_(wa, wb))
        end
        sp = pool[1:16:end]
        for a in sp, b in sp, c in sp
            tally(fma_dd_(words(a), words(b), words(c)))
        end
        @test checked > 500_000
        @test ok
    end

    # ------------------------------------------------------------ Tier 3.4 ---
    # When `exponent(hi) <= -126` the canonical half-ulp bound is at most
    # 2^-150, and no nonzero Float32 is that small — so `lo` must be zero.
    # Checked against results that actually land there, not against operands.
    @testset "subnormal-band results carry a zero low word" begin
        tiny = Double32[]
        for k in 100:150, s in (1.0f0, -1.0f0)
            push!(tiny, Double32(s * ldexp(1.0f0, -k)))
            push!(tiny, canonicalize(s * ldexp(1.0f0, -k), s * ldexp(1.0f0, -k - 24)))
            push!(tiny, canonicalize(s * ldexp(1.5f0, -k), -s * ldexp(1.0f0, -k - 25)))
        end
        pool = unique(vcat(tiny, structured_pool(), wide_gap_pairs()))
        in_band(h::Float32) = isfinite(h) && !iszero(h) && exponent(h) <= -126

        seen = 0
        ok = true
        function band(r)
            r isa Double32 || return
            ok &= iscanonical(r)
            if in_band(r.hi)
                seen += 1
                ok &= iszero(r.lo)
            end
        end
        for a in pool
            band(reciprocal(StrictPolicy(), a)); band(sqrt_or_nan(a))
            band(-a); band(abs(a)); band(canonicalize(a.hi, a.lo))
        end
        for a in pool, b in pool
            band(a + b); band(a - b); band(a * b); band(a / b)
        end
        @test seen > 50_000
        @test ok
    end

    # ------------------------------------------------------------ Tier 3.2 ---
    @testset "promotion lattice" begin
        @test promote_type(Double32, Float16) === Double32
        @test promote_type(Double32, Float32) === Double32
        @test promote_type(Double32, Float64) === Float64
        @test promote_type(Double32, BigFloat) === BigFloat
        @test promote_type(Double32, Int64) === Double32
        @test promote_type(Double32, Bool) === Double32
        @test promote_type(Double32, Rational{Int}) === Double32
        # BigInt and Rational{BigInt} reach BigFloat through Base's own rules,
        # which is the least-loss answer: neither fits the Float32 exponent
        # range. The package deliberately does not override it.
        @test promote_type(Double32, BigInt) === BigFloat
        @test promote_type(Double32, Rational{BigInt}) === BigFloat
        # Complex is derived by Base from the real rules; nothing is defined.
        @test promote_type(Double32, Complex{Float32}) === Complex{Double32}
        @test promote_type(Double32, Complex{Float64}) === ComplexF64
        @test promote_type(Double32, Complex{Int32}) === Complex{Double32}

        types = Any[Double32, Float16, Float32, Float64, BigFloat, Bool, Int8,
                    Int32, Int64, Int128, UInt64, BigInt, Rational{Int},
                    Rational{BigInt}, Complex{Float32}, Complex{Float64},
                    Complex{Double32}, Complex{Int32}]
        for A in types, B in types
            @test promote_type(A, B) === promote_type(B, A)
            @test promote_type(A, B) !== Any
        end
        for A in types, B in types, C in types
            @test promote_type(promote_type(A, B), C) ===
                  promote_type(A, promote_type(B, C))
        end
        @test isempty(Test.detect_ambiguities(Double32s))
    end
end
