@testset "arrays and reductions" begin
    @testset "SplitDouble32Array satisfies the AbstractArray interface" begin
        v = Double32[canonicalize(1.0f0, 2f0^-30), Double32(2), Double32(-3)]
        A = split(v)
        @test A isa SplitDouble32Array
        @test eltype(A) === Double32
        @test size(A) == size(v)
        @test axes(A) == axes(v)
        @test IndexStyle(typeof(A)) === IndexLinear()
        @test collect(A) == v
        @test A[1] === v[1]
        @test unsplit(A) == v
        A[2] = canonicalize(5.0f0, 2f0^-28)
        @test A[2] === canonicalize(5.0f0, 2f0^-28)
        @test A.hi[2] === 5.0f0
        @test all(iscanonical, A)

        B = similar(A)
        @test B isa SplitDouble32Array && size(B) == size(A)
        @test_throws DimensionMismatch SplitDouble32Array(zeros(Float32, 2), zeros(Float32, 3))

        M = split(Double32[1 2; 3 4])
        @test size(M) == (2, 2)
        @test M[2, 1] === Double32(3)
        @test sum(M) === Double32(10)
    end

    @testset "SplitDouble32Array round-trips wide-gap pairs" begin
        v = wide_gap_pairs()
        @test unsplit(split(v)) == v
        @test all(i -> split(v)[i] === v[i], eachindex(v))
    end

    @testset "compensated_dot on Float32 forms each product exactly" begin
        x = Float32[1, 2, 3, 4]
        y = Float32[5, 6, 7, 8]
        @test compensated_dot(x, y) === Double32(70)
        @test_throws DimensionMismatch compensated_dot(x, Float32[1, 2])

        # the case that motivates it: products that do not fit in Float32
        rng = MersenneTwister(21)
        a = rand(rng, Float32, 500) .* 1f4
        b = rand(rng, Float32, 500) .* 1f4
        e = sum(big(ai) * big(bi) for (ai, bi) in zip(a, b))
        @test relerr(compensated_dot(a, b), e) <= 8.0 * U2
        # strictly better than rounding each product to Float32 first
        naive = big(sum(a .* b))
        @test abs(reference(compensated_dot(a, b)) - e) <= abs(naive - e)
    end

    @testset "compensated_dot on Double32" begin
        x = Double32[canonicalize(1.0f0, 2f0^-30), Double32(2), Double32(3)]
        y = Double32[Double32(4), canonicalize(5.0f0, 2f0^-28), Double32(6)]
        e = sum(exact(a) * exact(b) for (a, b) in zip(x, y))
        @test relerr(compensated_dot(x, y), e) <= 8.0 * U2
        @test relerr(compensated_dot(Float32[1, 2, 3], y), sum(big(a) * exact(b)
                     for (a, b) in zip(Float32[1, 2, 3], y))) <= 8.0 * U2
    end

    @testset "compensated_sum and compensated_mapreduce pin the order" begin
        A = Float32[1f8; fill(1.0f0, 1000)]
        @test Float64(compensated_sum(A)) == 1.00001e8
        @test compensated_sum(Double32[1, 2, 3]) === Double32(6)
        @test compensated_mapreduce(identity, +, Double32[1, 2, 3]) === Double32(6)
        @test compensated_mapreduce(x -> x * x, +, Double32[1, 2, 3]) === Double32(14)
        @test iszero(compensated_sum(Float32[]))
    end

    @testset "norm2 scales exactly" begin
        @test norm2(Double32[3, 4]) === Double32(5)
        @test iszero(norm2(Double32[]))
        @test iszero(norm2(Double32[0, 0]))
        # no premature overflow or underflow, which is the whole point
        @test isfinite(norm2(Double32[Double32(1f30), Double32(1f30)]))
        @test !iszero(norm2(Double32[Double32(1f-30), Double32(1f-30)]))
        @test isnan(norm2(Double32[one(Double32), D32_NAN]))
        @test isinf(norm2(Double32[one(Double32), POS_INF]))
        v = Double32[1, 2, 3, 4]
        @test relerr(norm2(v), sqrt(big(30))) <= 8.0 * U2
    end
end
