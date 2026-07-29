# gpu/linalg.jl — tiled matrix multiply (design section 19.5, roadmap Phase 5).
#
# This is the one place the package still writes a `KernelAbstractions.@kernel`
# by hand: `AcceleratedKernels` provides reductions, maps, and scans, but no
# GEMM, and a tiled matrix multiply is not expressible as a map or a reduce.
# That is why `KernelAbstractions` is a direct dependency rather than being
# reached transitively through AK.
#
# Three things make this faster than the first version, and each was forced or
# enabled by a measurement:
#
# 1. **Accumulators live in `@private`, not `@localmem`.** Design section 14.3's
#    scoping restriction says a plain local does not survive a `@synchronize`,
#    which is why the first version parked the accumulator in shared memory and
#    paid a read-modify-write per tile. The *array* form `@private T (n,)` — a
#    `Scratchpad`, not the scalar form, which genuinely does not survive — turns
#    out to persist across barriers. Verified directly before relying on it.
#    That removes the shared accumulator tile and makes the inner loop private.
#
# 2. **Blocking is available but defaults to 1.** Each work item can compute an
#    `R×R` block of `C`, reusing each staged value `R` times. Measured, `R = 2`
#    is three times *slower* than `R = 1` — `@private` is local memory, not
#    registers, so the "register blocking" only multiplies local-memory traffic.
#    See `MATMUL_BLOCK`.
#
# 3. **A bounds-free path.** When the problem is square and an exact multiple of
#    the output block, every range test in the kernel is statically true. The
#    same kernel is instantiated with `Val(false)` and the branches vanish.

"""
    Double32s.MATMUL_BLOCK

Blocking factor for [`Double32s.ak_matmul!`](@ref): **`1`**.

Each work item computes a `MATMUL_BLOCK × MATMUL_BLOCK` sub-block of `C`, so
`MATMUL_BLOCK > 1` reuses each staged value that many times — the textbook fix
for a shared-memory-bound GEMM.

**Measured twice, with opposite-looking results, and the second measurement is
the one that stands.** Under the original pair-arithmetic inner op, `block = 2`
was 3× *slower* (16.08 vs 5.62 ms at 1024³): KA's `@private` maps to per-thread
**local** memory — global-backed spill space, not registers — so blocking
multiplied local-memory traffic instead of removing shared-memory traffic.
Under the split-channel inner op that replaced it, the accumulator traffic is a
fraction of the old cost and `block = 2` measures within ±5% of `block = 1`
(0.327 vs 0.340 ms at 1024³, 2.48 vs 2.62 ms at 2048³).

`1` stays the default: simpler indexing, fastest at 512³, and never worse than
5% behind. The keyword stays tunable, and both measurements stay recorded so a
future reader knows the 3× penalty was a property of the *old inner op*, not of
blocking.
"""
const MATMUL_BLOCK = 1

# Fused product-accumulate for the inner loop, on **split channels**: the
# accumulator is carried as a *weak* pair (design section 5.3 permits weak
# pairs inside one kernel body), with the exact high product entering through
# TwoSum and everything small — the TwoSum residual, the product residual, the
# cross terms — collecting in the low channel. One `two_sum_` refold per tile
# (see the kernel) keeps the low channel from drifting over long K.
#
# This replaces `op(acc, a * b)` — a ~10-flop pair multiply plus a ~20-flop
# pair add — with ~11 flops for `Float32` tiles and ~14 for `Double32` tiles.
# Measured on the device it is what closed most of the gap between the
# `Double32 × Double32` and `Float32 → Double32` paths.
#
# `Float32` operands: the high product is **exact** (`two_prod_`) and there are
# no cross terms — the entire accuracy argument for the F32 → D32 case
# survives unchanged.
@inline function acc_step(acc::Double32, a::Float32, b::Float32)
    p, e = two_prod_(a, b)
    s, t = two_sum_(acc.hi, p)
    return raw(s, acc.lo + (t + e))
end

# `Double32` operands: exact high product of the high words, all three cross
# products fused into the low channel. Nothing exact is available for the
# cross terms — those operands already spent their bits — and the dropped
# rounding is O(u²) per product, the same order as the pair multiply's own
# bound.
@inline function acc_step(acc::Double32, a::Double32, b::Double32)
    p, e  = two_prod_(a.hi, b.hi)
    cross = muladd(a.lo, b.lo, muladd(a.hi, b.lo, a.lo * b.hi))
    s, t  = two_sum_(acc.hi, p)
    return raw(s, acc.lo + ((t + e) + cross))
end

# Refold a weak pair into a canonical one: TwoSum output satisfies the
# canonical invariant by construction. Applied once per tile and once before
# the store.
@inline refold(x::Double32) = ((s, t) = two_sum_(x.hi, x.lo); raw(s, t))

# One kernel serves both element types and both bounds regimes.
#
# `TE`     tile element type (`Float32` for exact products, `Double32` otherwise)
# `T`      workgroup edge, so the group is T×T work items
# `R`      register-blocking factor; each item computes R×R outputs
# `CHECK`  `Val(true)` keeps the range tests, `Val(false)` compiles them away
@kernel unsafe_indices = true function matmul_tiled!(
        C, @Const(A), @Const(B), M::Int, N::Int, K::Int,
        ::Val{TE}, ::Val{T}, ::Val{R}, ::Val{CHECK}) where {TE,T,R,CHECK}

    As  = @localmem TE (T * R, T)
    Bs  = @localmem TE (T, T * R)
    acc = @private Double32 (R * R,)

    li, lj = @index(Local, NTuple)
    @inbounds for q in 1:(R * R)
        acc[q] = zero(Double32)
    end
    @synchronize()

    for kt in 1:cld(K, T)
        gi, gj = @index(Group, NTuple)
        li, lj = @index(Local, NTuple)

        # Each work item stages R rows of A and R columns of B.
        @inbounds for r in 0:(R - 1)
            arow = (gi - 1) * (T * R) + li + r * T
            acol = (kt - 1) * T + lj
            if CHECK
                As[li + r * T, lj] = (arow <= M) & (acol <= K) ?
                                     convert(TE, A[arow, acol]) : zero(TE)
            else
                As[li + r * T, lj] = convert(TE, A[arow, acol])
            end

            brow = (kt - 1) * T + li
            bcol = (gj - 1) * (T * R) + lj + r * T
            if CHECK
                Bs[li, lj + r * T] = (brow <= K) & (bcol <= N) ?
                                     convert(TE, B[brow, bcol]) : zero(TE)
            else
                Bs[li, lj + r * T] = convert(TE, B[brow, bcol])
            end
        end
        @synchronize()

        # The inner product on split channels; each shared value is read R
        # times when R > 1.
        @inbounds for t in 1:T
            for ri in 0:(R - 1)
                av = As[li + ri * T, t]
                for rj in 0:(R - 1)
                    acc[ri * R + rj + 1] =
                        acc_step(acc[ri * R + rj + 1], av, Bs[t, lj + rj * T])
                end
            end
        end
        # Refold the weak pairs once per tile (every T k-steps), so the low
        # channel stays small relative to the high one. Measured on the host
        # variant of this algorithm, folding at this period holds the normwise
        # error to a few 1e-14 where an unfolded channel drifts linearly in K.
        @inbounds for q in 1:(R * R)
            acc[q] = refold(acc[q])
        end
        @synchronize()
    end

    gi, gj = @index(Group, NTuple)
    li, lj = @index(Local, NTuple)
    @inbounds for ri in 0:(R - 1), rj in 0:(R - 1)
        row = (gi - 1) * (T * R) + li + ri * T
        col = (gj - 1) * (T * R) + lj + rj * T
        # A non-finite high channel means true overflow or a non-finite
        # operand; the low channel is then poisoned (Inf - Inf) and must not
        # decide the answer. Branch-free select, no throw.
        v = acc[ri * R + rj + 1]
        out = ifelse(isfinite(v.hi), refold(v), raw(v.hi, 0.0f0))
        if CHECK
            if (row <= M) & (col <= N)
                C[row, col] = out
            end
        else
            C[row, col] = out
        end
    end
end

# `Float32` operands get `Float32` tiles: half the shared memory of `Double32`
# tiles, and it is what makes the exact `two_prod_` product available.
_tile_eltype(::AbstractMatrix{Float32}) = Float32
_tile_eltype(::AbstractMatrix{Double32}) = Double32
_tile_eltype(::AbstractMatrix) = Double32

"""
    Double32s.ak_matmul!(C, A, B; kwargs...) -> C

Tiled matrix multiply with `Double32` accumulation, on any
`KernelAbstractions` backend. Design section 19.5.

Both `Float32 × Float32 → Double32` and `Double32 × Double32 → Double32` are
supported, and they differ in more than the element type:

- **`Float32` operands**: each product is formed **exactly** by
  [`EFT.two_prod_`](@ref) before it enters the accumulator, so a dot product of
  length `K` loses nothing until the additions. Shared tiles hold `Float32`,
  half the memory of the `Double32` case.
- **`Double32` operands**: the ordinary pair multiply and its `2u²` bound.
  Nothing exact is available — the operands have already spent their bits.

Measured on an RTX 5090 (normwise error against a `Float64` reference):

| size | F32→D32 | D32×D32 | F32 BLAS | error (either) | F32 BLAS error |
|-----:|--------:|--------:|---------:|---------------:|---------------:|
| 512³ | 0.05 ms | 0.07 ms | 0.015 ms | 7.7e-14 | 4.6e-07 |
| 1024³ | 0.33 ms | 0.42 ms | 0.048 ms | 1.0e-13 | 4.4e-07 |

The split-channel inner op is what made those times: replacing the pair
multiply + pair add (~30 flops, long dependency chain) with the fused
product-accumulate (~11–14 flops, short chain) took 1024³ from 5.62 ms to
0.33 ms for `Float32` operands and from 22.1 ms to 0.42 ms for `Double32`
operands — **17× and 53×** — while the error stays at the accuracy model
(~28 nominal ulps normwise at K = 1024, against ~4.4e-07 for `Float32` BLAS).

A **square** problem whose size is an exact multiple of the output block
(`tile × block`, 16 by default) takes a bounds-free instantiation: every range
test is statically true there and compiles away. Everything else takes the
checked path, which pads out-of-range loads with zeros.

At these times the kernel is within 7–9× of cuBLAS SGEMM on the same device —
for ~7 more decimal digits. No vendor library accumulates `Double32`, so there
is nothing faster to call at this accuracy.

Results are **bit-identical across backends** for a given `tile` and `block` —
the tile loop fixes the summation order. Since the reductions moved to
`AcceleratedKernels`, this is the one remaining cross-backend-reproducible
routine in the package.

# Keyword arguments
- `backend` — defaults to `KernelAbstractions.get_backend(C)`.
- `tile` — workgroup edge, default [`MATMUL_TILE`](@ref); a power of two.
- `block` — register-blocking factor, default [`MATMUL_BLOCK`](@ref).

There is no `policy` keyword: the split-channel accumulation *is* the
algorithm, and it is both faster than the `StrictPolicy` pair chain and holds
its measured error. A knob that selected a slower, equally-accurate path would
be surface area without a decision behind it.
"""
function ak_matmul!(C::AbstractMatrix{Double32},
                    A::AbstractMatrix, B::AbstractMatrix;
                    backend = KA.get_backend(C),
                    tile::Int = MATMUL_TILE,
                    block::Int = MATMUL_BLOCK)
    M, K = size(A)
    K2, N = size(B)
    K == K2 || throw(DimensionMismatch(
        lazy"A has $K columns but B has $K2 rows"))
    size(C) == (M, N) || throw(DimensionMismatch(
        lazy"C is $(size(C)) but A*B is $((M, N))"))
    (tile > 0 && ispow2(tile)) || throw(ArgumentError(
        lazy"tile must be a positive power of two, got $tile"))
    (block > 0 && ispow2(block)) || throw(ArgumentError(
        lazy"block must be a positive power of two, got $block"))
    (M == 0 || N == 0 || K == 0) && return C

    TE = promote_type(_tile_eltype(A), _tile_eltype(B))
    bs = tile * block
    # The bounds-free path: square, and an exact multiple of the output block,
    # so no work item can address outside any of the three matrices.
    exact = (M == N == K) && iszero(M % bs)
    _matmul_impl(C, A, B, M, N, K, backend,
                 Val(TE), Val(tile), Val(block), Val(exact))
    return C
end

function _matmul_impl(C, A, B, M, N, K, backend,
                      ::Val{TE}, ::Val{T}, ::Val{R}, ::Val{EXACT}) where {TE,T,R,EXACT}
    bs = T * R
    matmul_tiled!(backend, (T, T))(C, A, B, M, N, K,
                                   Val(TE), Val(T), Val(R), Val(!EXACT);
                                   ndrange = (cld(M, bs) * T, cld(N, bs) * T))
    KA.synchronize(backend)
    return C
end

"""
    Double32s.ak_matmul(A, B; kwargs...) -> AbstractMatrix{Double32}

Allocating form of [`Double32s.ak_matmul!`](@ref).

# Examples
```jldoctest
julia> A = Float32[1 2; 3 4]; B = Float32[5 6; 7 8];

julia> Float64.(Double32s.ak_matmul(A, B))
2×2 Matrix{Float64}:
 19.0  22.0
 43.0  50.0
```
"""
function ak_matmul(A::AbstractMatrix, B::AbstractMatrix;
                   backend = KA.get_backend(A), kwargs...)
    C = KA.allocate(backend, Double32, size(A, 1), size(B, 2))
    fill!(C, zero(Double32))
    return ak_matmul!(C, A, B; backend, kwargs...)
end
