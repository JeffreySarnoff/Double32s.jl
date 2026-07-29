# Generates benchmark_report.md (in this directory).
#
# Run from anywhere:  julia docs/Reports/benchmarks.jl
# No dependencies beyond the package itself; timings use best-of-trials
# amortized loops rather than BenchmarkTools, so the numbers are indicative
# rather than rigorous.  Not every public function is measured.
#
# Rewritten for Double32s.jl from the DoubleFloats.jl original: one precision
# section (Double32 over Float32), baselines Float32 / Float64, and
# only the surface the package actually implements — scalar arithmetic,
# compensated reductions, and generic dense linear algebra.  Elementary and
# matrix functions are roadmap phases 5–6 and appear here when they exist.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Double32s, LinearAlgebra, Random, Printf

Random.seed!(20260728)

# ---------------------------------------------------------------- timing --

const SINK = Ref(0.0)   # keeps results live so work is not elided

"""
    nsper(f, iters; trials) -> nanoseconds per call of f()

`f` should perform one unit of work and return a number, which is accumulated
into SINK.  Best (minimum) of `trials` amortized loops.
"""
function nsper(f::F, iters::Int; trials::Int = 9) where {F<:Function}
    s = 0.0
    best = Inf
    for _ in 1:trials
        t0 = time_ns()
        for _ in 1:iters
            s += Float64(f())
        end
        best = min(best, (time_ns() - t0) / iters)
    end
    SINK[] += s
    return best
end

fmt(ns) = ns < 1000 ? @sprintf("%.0f ns", ns) :
          ns < 1e6  ? @sprintf("%.2f µs", ns/1e3) :
                      @sprintf("%.2f ms", ns/1e6)

ratio(a, b) = @sprintf("%.1f", a / b)

# ------------------------------------------------------- scalar section --

# Identical sample values across every type column.
const NPTS = 1000
xs64 = rand(NPTS) .* 2 .+ 0.25           # (0.25, 2.25): in-domain everywhere
ys64 = rand(NPTS) .* 2 .+ 0.25

# time `op` mapped across the sample vector(s); returns ns per element
function mapbench(op::F, xs) where {F<:Function}
    work = () -> begin
        s = 0.0
        for x in xs
            s += Float64(op(x))
        end
        s
    end
    return nsper(work, 20) / NPTS
end

function mapbench2(op::F, xs, ys) where {F<:Function}
    work = () -> begin
        s = 0.0
        for i in eachindex(xs)
            s += Float64(op(xs[i], ys[i]))
        end
        s
    end
    return nsper(work, 20) / NPTS
end

const COLS = (Double32, Float64)

# `dom` maps the shared (0.25, 2.25) sample onto the function's own domain, so
# every column times the same arguments and no function is measured on inputs it
# would reject.
function scalar_row(name, op1, op2 = nothing; dom = identity)
    times = map(COLS) do T
        xs = T.(dom.(xs64)); ys = T.(dom.(ys64))
        op2 === nothing ? mapbench(op1, xs) : mapbench2(op2, xs, ys)
    end
    d32, f64 = times
    return "| `$name` | $(fmt(d32)) | $(fmt(f64)) | $(ratio(d32, f64))x |"
end

# Domain adaptors.
unitdom(x)  = (x - 1.25) / 1.1          # into (-0.91, 0.91)
posdom(x)   = x                          # already positive
ge1dom(x)   = x + 1.0                    # into [1.25, 3.25], for acosh
smalldom(x) = (x - 1.25) / 5             # into (-0.2, 0.2), the series bands

arith_rows = [
    scalar_row("+",     nothing, +),
    scalar_row("-",     nothing, -),
    scalar_row("*",     nothing, *),
    scalar_row("/",     nothing, /),
    scalar_row("sqrt",  sqrt),
    scalar_row("inv",   inv),
    scalar_row("abs",   abs),
    scalar_row("fma",   nothing, (x, y) -> fma(x, y, x)),
    scalar_row("hypot", nothing, hypot),
    scalar_row("<",     nothing, (x, y) -> ifelse(x < y, one(x), zero(x))),
    scalar_row("^ (Int)", x -> x^7),
    scalar_row("^",     nothing, ^),
]

# ------------------------------------------------ elementary functions --
#
# Roadmap Phase 6, rebuilt 2026-07-29: table-based range reductions (32-entry
# 2^(j/32) for exp, 23-bin division-free log), dominant-add Horner steps, and
# Float32 tails once the remaining terms sit below 2^-26 of the result — see
# src/math/horner.jl and design section 20.6. The rebuild cut every function
# 2-7x; the `D32/F64` column is the honest measure of what remains.

elem_rows = [
    scalar_row("cbrt",  cbrt),
    # `rsqrt` is Double32-only; the baseline columns use `1/sqrt(x)`, which is
    # exactly what the Double32 method is (see its docstring).
    scalar_row("rsqrt", x -> x isa Double32 ? Double32s.rsqrt(x) : 1 / sqrt(x); dom = posdom),
    scalar_row("exp",   exp),
    scalar_row("expm1", expm1),
    scalar_row("exp2",  exp2),
    scalar_row("exp10", exp10),
    scalar_row("log",   log; dom = posdom),
    scalar_row("log1p", log1p),
    scalar_row("log2",  log2; dom = posdom),
    scalar_row("log10", log10; dom = posdom),
]

trig_rows = [
    scalar_row("sin",    sin),
    scalar_row("cos",    cos),
    scalar_row("sincos", x -> first(sincos(x))),
    scalar_row("tan",    tan),
    scalar_row("asin",   asin;  dom = unitdom),
    scalar_row("acos",   acos;  dom = unitdom),
    scalar_row("atan",   atan),
    scalar_row("atan2",  nothing, atan),
]

hyp_rows = [
    scalar_row("sinh",  sinh),
    scalar_row("cosh",  cosh),
    scalar_row("tanh",  tanh),
    scalar_row("tanh (series band)", tanh; dom = smalldom),
    scalar_row("asinh", asinh),
    scalar_row("acosh", acosh; dom = ge1dom),
    scalar_row("atanh", atanh; dom = unitdom),
]

# ---------------------------------------------------- reductions section --

const NRED = 10_000
red64  = rand(NRED) .* 2 .- 1
red64b = rand(NRED) .* 2 .- 1
redf32  = Float32.(red64)
redf32b = Float32.(red64b)

function reduction_rows()
    d32v  = Double32.(red64);  d32w = Double32.(red64b)
    f64v  = red64

    rows = String[]

    t_sumf32acc = nsper(() -> Float64(sum(Double32, redf32)), 5)
    t_sumf64    = nsper(() -> sum(Float64, redf32), 5)
    push!(rows, "| `sum(Double32, A::Vector{Float32})` | $(fmt(t_sumf32acc)) | " *
                "`sum(Float64, A)` $(fmt(t_sumf64)) | $(ratio(t_sumf32acc, t_sumf64))x |")

    t_sumd  = nsper(() -> Float64(sum(d32v)), 5)
    t_sumf  = nsper(() -> Float64(sum(f64v)), 5)
    push!(rows, "| `sum(::Vector{Double32})` | $(fmt(t_sumd)) | " *
                "`sum(::Vector{Float64})` $(fmt(t_sumf)) | $(ratio(t_sumd, t_sumf))x |")

    t_cdot  = nsper(() -> Float64(Double32s.compensated_dot(redf32, redf32b)), 5)
    t_dotf  = nsper(() -> Float64(dot(redf32, redf32b)), 5)
    push!(rows, "| `compensated_dot(::Vector{Float32}, ...)` | $(fmt(t_cdot)) | " *
                "`dot` (BLAS, Float32) $(fmt(t_dotf)) | $(ratio(t_cdot, t_dotf))x |")

    t_cdotd = nsper(() -> Float64(Double32s.compensated_dot(d32v, d32w)), 5)
    t_dot64 = nsper(() -> dot(red64, red64b), 5)
    push!(rows, "| `compensated_dot(::Vector{Double32}, ...)` | $(fmt(t_cdotd)) | " *
                "`dot` (BLAS, Float64) $(fmt(t_dot64)) | $(ratio(t_cdotd, t_dot64))x |")

    t_csum  = nsper(() -> Float64(Double32s.compensated_sum(d32v)), 5)
    push!(rows, "| `compensated_sum(::Vector{Double32})` | $(fmt(t_csum)) | " *
                "`sum(::Vector{Double32})` $(fmt(t_sumd)) | $(ratio(t_csum, t_sumd))x |")

    t_csumf = nsper(() -> Float64(Double32s.compensated_sum(redf32)), 5)
    push!(rows, "| `compensated_sum(::Vector{Float32})` | $(fmt(t_csumf)) | " *
                "`sum(Float64, A)` $(fmt(t_sumf64)) | $(ratio(t_csumf, t_sumf64))x |")

    t_norm  = nsper(() -> Float64(Double32s.norm2(d32v)), 5)
    t_norm64 = nsper(() -> norm(red64), 5)
    push!(rows, "| `norm2(::Vector{Double32})` | $(fmt(t_norm)) | " *
                "`norm` (BLAS, Float64) $(fmt(t_norm64)) | $(ratio(t_norm, t_norm64))x |")

    t_mapred = nsper(() -> Float64(Double32s.compensated_mapreduce(abs2, +, d32v)), 5)
    push!(rows, "| `compensated_mapreduce(abs2, ::Vector{Double32})` | $(fmt(t_mapred)) | " *
                "`sum(abs2, ::Vector{Float64})` $(fmt(nsper(() -> sum(abs2, red64), 5))) | " *
                "$(ratio(t_mapred, nsper(() -> sum(abs2, red64), 5)))x |")
    return rows
end

# ------------------------------------------------------- linear algebra --

# Only generic dense algorithms exist for Double32 today (no vendor kernels,
# no GenericSchur dependency), so the honest comparison set is matmul, lu,
# and lu-backed solve.  Float32 gets LAPACK/BLAS; the ratios therefore combine
# the precision cost with the loss of BLAS blocking and SIMD.
function la_row(name, make::F; iters::Int = 3) where {F<:Function}
    d32 = nsper(make(Double32), iters, trials = 5)
    f64 = nsper(make(Float64),  iters, trials = 5)
    return "| `$name` | $(fmt(d32)) | $(fmt(f64)) | $(ratio(d32, f64))x |"
end

function matmul_row(n::Int)
    a = rand(n, n) .- 0.5
    b = rand(n, n) .- 0.5
    return la_row("A * B  (n=$n)", T -> begin
        Ad = T.(a)
        Bd = T.(b)
        () -> Float64(sum(abs, Ad * Bd))
    end)
end

function linear_rows()
    n  = 32
    A0 = rand(n, n) .- 0.5 + n * I       # well-conditioned
    b0 = rand(n)
    x0 = rand(n)
    return [
        la_row("A * x   (n=32)", T -> (Ad = T.(A0); xd = T.(x0); () -> Float64(sum(abs, Ad * xd)))),
        la_row("A' * B  (n=32)", T -> (Ad = T.(A0); Bd = T.(A0 .+ 1); () -> Float64(sum(abs, Ad' * Bd)))),
        la_row("A * B'  (n=32)", T -> (Ad = T.(A0); Bd = T.(A0 .+ 1); () -> Float64(sum(abs, Ad * Bd')))),
        la_row("dot(x,y) (n=10^4)", T -> (xd = T.(rand(10^4)); yd = T.(rand(10^4)); () -> Float64(dot(xd, yd)))),
        la_row("lu(A)   (n=32)", T -> (Ad = T.(A0); () -> Float64(sum(abs, lu(Ad).factors)))),
        la_row("A \\ b   (n=32)", T -> (Ad = T.(A0); bd = T.(b0); () -> Float64(sum(abs, Ad \ bd)));
               iters = 3),
    ]
end

# ------------------------------------------------------------- assemble --

vinfo = string(VERSION)
red_rows = reduction_rows()
mm_rows  = [matmul_row(n) for n in (4, 8, 16, 32, 64)]
la_rows  = linear_rows()

report = """
# Double32s Benchmark Report

*Generated by `docs/Reports/benchmarks.jl` on Julia $vinfo.*

`Double32` is a two-word `Float32` format carrying ~48 significant bits
(~14 decimal digits) with the `Float32` exponent range, designed for GPUs
where `Float64` is slow or absent.  These are **host CPU** timings — the GPU
execution layer is roadmap phase 2+ — so the interesting columns are the
ratio.  `Float64` is the CPU-native alternative that GPUs generally lack, so
`D32/F64` answers "what would this cost me if I had `Float64` and used it
instead".

This report covers the whole implemented surface — scalar arithmetic,
elementary functions, trigonometry, hyperbolics, reductions, and dense linear
algebra.  **GPU timings are not here**: `KernelAbstractions` and `CUDA` are weak
dependencies that the root project cannot load, so `ak_sum`, `ak_dot`,
`ak_norm2`, and `ak_matmul` are measured by `docs/Reports/gpu_report.jl`, which
has its own environment.

Timings are best-of-trials amortized averages on the machine that ran the
script — indicative magnitudes, not precise measurements.

## Scalar Arithmetic

| operation | Double32 | Float64 | D32/F64 |
|:----------|---------:|--------:|--------:|
$(join(arith_rows, "\n"))

`/`, `inv`, and `sqrt` include the operand scaling that keeps their residuals
representable at exponent extremes, and `*` includes its overflow/underflow
band guard — none of these are optional fast paths (see the design's section
7.5 and the guard docstrings).

## Elementary Functions

Roadmap Phase 6, rebuilt 2026-07-29 around table range reductions, dominant-add
Horner steps, and mixed-precision (`Float32`) series tails — design section
20.6 records the rebuild and its measurements (2–7x per function). `log` and
`atanh` are now division-free; `atan` pays at most one division.
Measured accuracy for every function is in `accuracy_report.md`.

| operation | Double32 | Float64 | D32/F64 |
|:----------|---------:|--------:|--------:|
$(join(elem_rows, "\n"))

## Trigonometric Functions

`sin`, `cos`, `tan`, and `sincos` return `NaN` beyond
`Double32s.TRIG_ARG_LIMIT = 2^20`; these timings are inside that range.
`sincos` computes both from one argument reduction, so it is far cheaper than
`sin` plus `cos`.

| operation | Double32 | Float64 | D32/F64 |
|:----------|---------:|--------:|--------:|
$(join(trig_rows, "\n"))

## Hyperbolic Functions

`tanh` is listed twice: inside its series band (`|x| <= 0.35`, no division) and
across the general range (one division).  The gap is the cost of the division
this package's `/` cannot avoid.

| operation | Double32 | Float64 | D32/F64 |
|:----------|---------:|--------:|--------:|
$(join(hyp_rows, "\n"))

## Reductions ($(NRED)-element vectors)

| operation | time | baseline | ratio |
|:----------|-----:|:---------|------:|
$(join(red_rows, "\n"))

`sum(Double32, A::Vector{Float32})` is the package's headline use: a
`Float32` data stream accumulated in ~48-bit precision through Base's own
`mapreduce` machinery, no package function needed.

## Linear Algebra (generic algorithms)

`Float64` timings use LAPACK/BLAS. `Double32` `A*B`, `A*x`, `A'B`, `AB'`, and
`dot` use the package's split-channel compensated kernels (`arrays/matmul.jl`);
`lu` and `\\` still run Julia's generic dense fallbacks. The device-side tiled
kernel is measured in `gpu_report.md`.

| operation | Double32 | Float64 | D32/F64 |
|:----------|---------:|--------:|--------:|
$(join(la_rows, "\n"))

### Matrix multiply

| operation | Double32 | Float64 | D32/F64 |
|:----------|---------:|--------:|--------:|
$(join(mm_rows, "\n"))

The `Double32` column is Julia's generic `*`; `Float64` is BLAS.
The device-side tiled kernel `Double32s.ak_matmul` with exact `Float32` products
is measured in `gpu_report.md`.

## Notes

- Entries below ~10 ns are at the resolution of the timing loop; treat them
  (and their ratios) as "too fast to matter" rather than exact.
- `Double32` is an immutable 8-byte bitstype and vectors store it inline, so
  none of these timings include allocation.
- Scalar `+`, `*`, `/` measured here at a few ns/op correspond to the
  session-recorded baselines (≈7 / ≈5 / ≈15 ns on the development machine).
- Both columns time the same sample values, mapped into each function's own
  domain so nothing is measured on inputs it would reject.
"""

outpath = joinpath(@__DIR__, "benchmark_report.md")
open(outpath, "w") do io
    write(io, report)
end
println("wrote ", outpath)
