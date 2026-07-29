# arrays/matmul.jl — host matrix multiply for Double32 matrices.
#
# Design section 19.5 said the generic fallback "will not initially match
# vendor SGEMM throughput", and it undersold the problem: LinearAlgebra's
# generic matmul walks the full pair arithmetic — a ~10-flop `*` and a ~20-flop
# `+` per element pair, in an access order chosen for correctness rather than
# for the cache. Measured at n = 64 it took **1.70 ms**, 176× slower than
# `Float64` BLAS on the same data.
#
# The method below replaces both the loop order and the arithmetic:
#
#   * **jki order with per-column channel buffers.** Column-major arrays get
#     unit-stride access on `A` and on the accumulators; `B[k,j]` is a scalar
#     hoisted out of the inner loop. The inner loop is `@simd`-able because the
#     accumulators are plain `Float32` vectors, not `Double32` pairs.
#
#   * **Split-channel compensated accumulation** instead of pair arithmetic.
#     Each product contributes its exact high part by `two_prod_`, its cross
#     terms by two fused multiply-adds, and the accumulation runs TwoSum on the
#     high channel with **Kahan compensation on the low channel**. One
#     renormalization per output element at the end, instead of one per
#     operation. Per element pair that is ~17 flops against ~30 — and, unlike
#     the pair chain, it vectorizes.
#
# Four variants were measured before this one shipped (jki with pair
# arithmetic; plain low channel; periodic refolding; Kahan low channel). The
# plain low channel was 10% faster but its error grew linearly in K —
# 1.4e-13 normwise at n = 128 against 2.8e-14 for the Kahan channel, which
# stays flat. Speed table, worst normwise error against a 400-bit reference,
# this machine:
#
#   | n   | generic (before) | this method | speedup | error before | after   |
#   |-----|------------------|-------------|---------|--------------|---------|
#   | 8   |      3.24 µs     |    0.30 µs  |  ~11×   | 1.6e-15      | 5.4e-15 |
#   | 15  |     22.06 µs     |    2.70 µs  |   ~8×   | 3.3e-15      | 1.1e-14 |
#   | 32  |    211.69 µs     |    7.52 µs  |  ~28×   | 4.8e-15      | 1.4e-14 |
#   | 64  |   1701.49 µs     |   59.93 µs  |  ~28×   | 9.2e-15      | 2.0e-14 |
#   | 128 |      (13.6 ms)   |  435.65 µs  |  ~31×   | —            | 2.8e-14 |
#
# The error is ~2× the pair chain's and stays ~2e-14 normwise as n grows —
# inside the format's documented accuracy model, and stated here rather than
# discovered later.

"""
    mul!(C::StridedMatrix{Double32}, A::StridedMatrix{Double32},
         B::StridedMatrix{Double32}, α::Number, β::Number) -> C

Cache-aware compensated matrix multiply for `Double32` matrices; `A * B`
reaches it through LinearAlgebra's 3-argument `mul!`.

Products enter the accumulator as an exact high part (`two_prod_`) plus fused
cross terms; accumulation runs TwoSum on the high channel and Kahan
compensation on the low channel, renormalizing once per output element.
Measured ~28× faster than the generic pair-arithmetic matmul at `n = 64`, with
worst normwise error ~2e-14 that stays flat in `n` (table and methodology in
the source and in `docs/Reports/benchmark_report.md`).

The general `α, β` scaling is applied per element after the compensated
product, in `Double32` arithmetic. `C` must not alias `A` or `B` — the same
contract as every `mul!`.

Adjoint/transpose wrappers and mixed element types fall back to the generic
path; only the plain `Matrix{Double32}` case is specialized.
"""
function LinearAlgebra.mul!(C::StridedMatrix{Double32},
                            A::StridedMatrix{Double32},
                            B::StridedMatrix{Double32},
                            α::Number, β::Number)
    M, K = size(A)
    K2, N = size(B)
    K == K2 || throw(DimensionMismatch(
        lazy"A has $K columns but B has $K2 rows"))
    size(C) == (M, N) || throw(DimensionMismatch(
        lazy"C is $(size(C)) but A*B is $((M, N))"))

    ad = convert(Double32, α)
    bd = convert(Double32, β)
    sh = zeros(Float32, M)                 # high channel
    sl = zeros(Float32, M)                 # low channel
    sc = zeros(Float32, M)                 # Kahan compensator for the low channel

    @inbounds for j in 1:N
        fill!(sh, 0.0f0); fill!(sl, 0.0f0); fill!(sc, 0.0f0)
        for k in 1:K
            b = B[k, j]
            bh = b.hi
            bl = b.lo
            iszero(bh) && continue         # skips the column, not a lane
            @simd for i in 1:M
                a = A[i, k]
                # Exact high product; all three remaining cross products fused.
                p, e  = two_prod_(a.hi, bh)
                cross = muladd(a.lo, bl, muladd(a.hi, bl, a.lo * bh))
                # TwoSum on the high channel...
                s, t  = two_sum_(sh[i], p)
                sh[i] = s
                # ...Kahan on the low channel, so its own rounding does not
                # accumulate linearly in K (measured: this is what holds the
                # error flat at 2.8e-14 where the plain channel drifts to 1.4e-13).
                y     = ((t + e) + cross) - sc[i]
                u     = sl[i] + y
                sc[i] = (u - sl[i]) - y
                sl[i] = u
            end
        end
        if iszero(β)
            if isone(α)
                for i in 1:M
                    C[i, j] = _mm_finish(sh[i], sl[i])
                end
            else
                for i in 1:M
                    C[i, j] = ad * _mm_finish(sh[i], sl[i])
                end
            end
        else
            for i in 1:M
                C[i, j] = ad * _mm_finish(sh[i], sl[i]) + bd * C[i, j]
            end
        end
    end
    return C
end

# One renormalization per output element. A non-finite high channel means the
# true product overflowed (or an operand was non-finite); the low channel is
# then poisoned with Inf - Inf = NaN and must not decide the answer — Inf in
# must give Inf out, exactly as the scalar `+`/`*` guards arrange.
@inline function _mm_finish(h::Float32, l::Float32)
    isfinite(h) || return Double32(h)
    return Double32(h, l)
end

# ---------------------------------------------------------------- mat-vec ---

"""
    mul!(y::StridedVector{Double32}, A::StridedMatrix{Double32},
         x::StridedVector{Double32}, α::Number, β::Number) -> y

Matrix–vector product on the same split channels as the matrix method: the
column axpy form (`y += A[:, k] * x[k]`), unit stride throughout, one
renormalization per output element.
"""
function LinearAlgebra.mul!(y::StridedVector{Double32},
                            A::StridedMatrix{Double32},
                            x::StridedVector{Double32},
                            α::Number, β::Number)
    M, K = size(A)
    length(x) == K || throw(DimensionMismatch(
        lazy"A has $K columns but x has length $(length(x))"))
    length(y) == M || throw(DimensionMismatch(
        lazy"y has length $(length(y)) but A has $M rows"))

    ad = convert(Double32, α)
    bd = convert(Double32, β)
    sh = zeros(Float32, M); sl = zeros(Float32, M); sc = zeros(Float32, M)
    @inbounds for k in 1:K
        b = x[k]
        bh = b.hi; bl = b.lo
        iszero(bh) && continue
        @simd for i in 1:M
            a = A[i, k]
            p, e  = two_prod_(a.hi, bh)
            cross = muladd(a.lo, bl, muladd(a.hi, bl, a.lo * bh))
            s, t  = two_sum_(sh[i], p)
            sh[i] = s
            yk    = ((t + e) + cross) - sc[i]
            u     = sl[i] + yk
            sc[i] = (u - sl[i]) - yk
            sl[i] = u
        end
    end
    @inbounds if iszero(β)
        if isone(α)
            for i in 1:M; y[i] = _mm_finish(sh[i], sl[i]); end
        else
            for i in 1:M; y[i] = ad * _mm_finish(sh[i], sl[i]); end
        end
    else
        for i in 1:M; y[i] = ad * _mm_finish(sh[i], sl[i]) + bd * y[i]; end
    end
    return y
end

# ------------------------------------------------- adjoint/transpose forms ---

# `Double32 <: Real`, so `adjoint` and `transpose` agree numerically; one
# implementation serves both wrappers.
const AdjTransD32{M} =
    Union{LinearAlgebra.Adjoint{Double32,M}, LinearAlgebra.Transpose{Double32,M}} where {M<:StridedMatrix{Double32}}

# A sequential channel dot over two unit-stride columns. No `@simd`: the TwoSum
# chain is a loop-carried dependency. It still beats the generic path by an
# order of magnitude — the win here is flops and locality, not vectorization.
@inline function _cdot_cols(P::AbstractMatrix{Double32}, ci::Int,
                            B::AbstractMatrix{Double32}, cj::Int, K::Int)
    sh = 0.0f0; sl = 0.0f0; sc = 0.0f0
    @inbounds for k in 1:K
        a = P[k, ci]
        b = B[k, cj]
        p, e  = two_prod_(a.hi, b.hi)
        cross = muladd(a.lo, b.lo, muladd(a.hi, b.lo, a.lo * b.hi))
        s, t  = two_sum_(sh, p)
        sh    = s
        yk    = ((t + e) + cross) - sc
        u     = sl + yk
        sc    = (u - sl) - yk
        sl    = u
    end
    return _mm_finish(sh, sl)
end

"""
    mul!(C, A′::Adjoint/Transpose{Double32}, B::StridedMatrix{Double32}, α, β)

`A′B` by materializing the transpose and running the SIMD axpy path.

Two implementations were measured. The dot form — `C[i,j] = P[:,i] ⋅ B[:,j]`,
both columns unit stride — looks like the natural fit but its TwoSum chain is a
loop-carried dependency, so it cannot vectorize. Materializing `A′` (an O(MK)
copy) and reusing the `@simd` channel kernel beats it at **every** size
measured: 0.39 vs 0.79 µs at n = 8, 60 vs 350 µs at n = 64, 3.4 vs 23.5 ms at
n = 256. The copy is the price of vectorization and it is the smaller number
throughout — the same packing trade BLAS makes.

The matrix–vector adjoint method keeps the dot form: there the copy would cost
as much as the whole computation.
"""
function LinearAlgebra.mul!(C::StridedMatrix{Double32}, At::AdjTransD32,
                            B::StridedMatrix{Double32}, α::Number, β::Number)
    P = parent(At)
    K, M = size(P)                       # At is M × K
    K2, N = size(B)
    K == K2 || throw(DimensionMismatch(
        lazy"A' has $K inner elements but B has $K2 rows"))
    size(C) == (M, N) || throw(DimensionMismatch(
        lazy"C is $(size(C)) but A'B is $((M, N))"))
    # O(MK) copy buys the vectorized kernel; Double32 is real, so adjoint and
    # transpose are the same element copy.
    Am = Matrix{Double32}(undef, M, K)
    @inbounds for k in 1:K
        @simd for i in 1:M
            Am[i, k] = P[k, i]
        end
    end
    return LinearAlgebra.mul!(C, Am, B, α, β)
end

"""
    mul!(y, A′::Adjoint/Transpose{Double32}, x::StridedVector{Double32}, α, β)

`A′x` as one unit-stride channel dot per output element.
"""
function LinearAlgebra.mul!(y::StridedVector{Double32}, At::AdjTransD32,
                            x::StridedVector{Double32}, α::Number, β::Number)
    P = parent(At)
    K, M = size(P)
    length(x) == K || throw(DimensionMismatch(
        lazy"A' has $K inner elements but x has length $(length(x))"))
    length(y) == M || throw(DimensionMismatch(
        lazy"y has length $(length(y)) but A' has $M rows"))
    ad = convert(Double32, α)
    bd = convert(Double32, β)
    X = reshape(x, K, 1)
    @inbounds for i in 1:M
        r = _cdot_cols(P, i, X, 1, K)
        y[i] = iszero(β) ? (isone(α) ? r : ad * r) : ad * r + bd * y[i]
    end
    return y
end

"""
    mul!(C, A::StridedMatrix{Double32}, B′::Adjoint/Transpose{Double32}, α, β)

`AB′` in the same jki axpy form as `AB`; the only change is that the hoisted
scalar is `B.parent[j, k]` instead of `B[k, j]`.
"""
function LinearAlgebra.mul!(C::StridedMatrix{Double32},
                            A::StridedMatrix{Double32}, Bt::AdjTransD32,
                            α::Number, β::Number)
    P = parent(Bt)                       # Bt is K × N, P is N × K
    M, K = size(A)
    N, K2 = size(P)
    K == K2 || throw(DimensionMismatch(
        lazy"A has $K columns but B' has $K2 rows"))
    size(C) == (M, N) || throw(DimensionMismatch(
        lazy"C is $(size(C)) but AB' is $((M, N))"))
    ad = convert(Double32, α)
    bd = convert(Double32, β)
    sh = zeros(Float32, M); sl = zeros(Float32, M); sc = zeros(Float32, M)
    @inbounds for j in 1:N
        fill!(sh, 0.0f0); fill!(sl, 0.0f0); fill!(sc, 0.0f0)
        for k in 1:K
            b = P[j, k]
            bh = b.hi; bl = b.lo
            iszero(bh) && continue
            @simd for i in 1:M
                a = A[i, k]
                p, e  = two_prod_(a.hi, bh)
                cross = muladd(a.lo, bl, muladd(a.hi, bl, a.lo * bh))
                s, t  = two_sum_(sh[i], p)
                sh[i] = s
                yk    = ((t + e) + cross) - sc[i]
                u     = sl[i] + yk
                sc[i] = (u - sl[i]) - yk
                sl[i] = u
            end
        end
        if iszero(β)
            if isone(α)
                for i in 1:M; C[i, j] = _mm_finish(sh[i], sl[i]); end
            else
                for i in 1:M; C[i, j] = ad * _mm_finish(sh[i], sl[i]); end
            end
        else
            for i in 1:M; C[i, j] = ad * _mm_finish(sh[i], sl[i]) + bd * C[i, j]; end
        end
    end
    return C
end

# --------------------------------------------------------------------- dot ---

"""
    dot(x::StridedVector{Double32}, y::StridedVector{Double32}) -> Double32

Channel-compensated dot product — the same accumulation the matrix multiply
uses, one renormalization at the end. Sequential in `k` (the TwoSum chain is a
loop-carried dependency), which still measures far ahead of the generic
pairwise reduction over pair arithmetic.

`Double32` is real, so `dot` needs no conjugation.
"""
function LinearAlgebra.dot(x::StridedVector{Double32}, y::StridedVector{Double32})
    K = length(x)
    length(y) == K || throw(DimensionMismatch(
        lazy"x has length $K but y has length $(length(y))"))
    return _cdot_cols(reshape(x, K, 1), 1, reshape(y, K, 1), 1, K)
end
