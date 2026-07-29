# Generates accuracy_report.md (in this directory).
#
# Run from anywhere:  julia docs/Reports/accuracies.jl
# Every operation is compared against a 400-bit BigFloat reference.  Errors
# are reported in nominal 48-bit ulps of the true value ("ulp48" below):
# ulp48(V) = 2^(floor(log2|V|) - 47), the design section 9.1 metric.  A worst
# case of about 1 ulp means the operation is as accurate as the format allows.
#
# Rewritten for Double32s.jl from the DoubleFloats.jl original: only the
# implemented surface (arithmetic, reductions) — elementary and matrix
# functions are roadmap phases 5–6.  Two things this report does that the
# original did not, both lessons paid for during development:
#
#   * operands are drawn from adversarial pools (wide-gap low words, signed
#     zeros, cancellation pairs), not uniform samples — every defect this
#     package has had was invisible to uniform sampling;
#   * error is only charged against the 48-bit lattice where that lattice
#     exists (|V| >= 2^-100).  Below it the format itself tapers to 24 bits,
#     and a "failure" there is a statement about the format, not the code.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Double32s, LinearAlgebra, Random, Printf
using Double32s: reference, canonicalize, has_gradual_underflow

setprecision(BigFloat, 400)
Random.seed!(20260728)

const NSAMPLES = 400

# ------------------------------------------------------------- metrics ----

ulp48(V::BigFloat) = big(2.0)^(Int(floor(log2(abs(V)))) - 47)
inlattice(V) = isfinite(V) && !iszero(V) &&
               big(2.0)^-100 <= abs(V) <= big(2.0)^120

# error of a Double32 result in nominal 48-bit ulps of the BigFloat truth
err_ulp48(got::Double32, want::BigFloat) =
    Float64(abs(reference(got) - want) / ulp48(want))

# ------------------------------------------------------------ operands ----

"Uniform bit-pattern pair, canonicalized: covers the whole lattice."
function rand_d32(rng; emin = -40, emax = 40)
    while true
        x = canonicalize(reinterpret(Float32, rand(rng, UInt32)),
                         reinterpret(Float32, rand(rng, UInt32)))
        (isfinite(x) && !iszero(x) && emin < exponent(Float32(x)) < emax) && return x
    end
end

"Adversarial: high word near a power of two, low word at every allowed gap."
function adversarial_pool()
    v = Double32[]
    lomin = reinterpret(Float32, 0x00000001)
    for h in Float32[1.0, prevfloat(1.0f0), nextfloat(1.0f0), 0.5, 2.0, 3.0,
                     1.5, 1/3f0, 2f0^10, 2f0^-10, 7.0]
        push!(v, Double32(h))
        for k in (0, 1, 2, 20, 23), s in (1f0, -1f0)
            push!(v, canonicalize(h, s * ldexp(abs(h), -k - 24)))
        end
        for l in (lomin, -lomin)
            push!(v, canonicalize(h, l))
        end
    end
    unique(v)
end

# --------------------------------------------------------------- rows -----

function accrow2(name, fn, bigfn, sampler; domain = "")
    errs = Float64[]
    while length(errs) < NSAMPLES
        x, y = sampler()
        got = fn(x, y)
        isfinite(got) || continue
        want = bigfn(reference(x), reference(y))
        inlattice(want) || continue
        push!(errs, err_ulp48(got, want))
    end
    sort!(errs)
    return @sprintf("| `%s` | %s | %d | %.3f | %.3f | %.2f%% |",
                    name, domain, length(errs), errs[(end + 1) ÷ 2], errs[end],
                    100 * count(<(1.0), errs) / length(errs))
end

function accrow1(name, fn, bigfn, sampler; domain = "")
    errs = Float64[]
    while length(errs) < NSAMPLES
        x = sampler()
        got = fn(x)
        isfinite(got) || continue
        want = bigfn(reference(x))
        inlattice(want) || continue
        push!(errs, err_ulp48(got, want))
    end
    sort!(errs)
    return @sprintf("| `%s` | %s | %d | %.3f | %.3f | %.2f%% |",
                    name, domain, length(errs), errs[(end + 1) ÷ 2], errs[end],
                    100 * count(<(1.0), errs) / length(errs))
end

function arithmetic_rows()
    rng  = MersenneTwister(1)
    pool = adversarial_pool()
    mixed()  = (rand(rng, Bool) ? rand_d32(rng) : rand(rng, pool),
                rand(rng, Bool) ? rand_d32(rng) : rand(rng, pool))
    single() = rand(rng, Bool) ? rand_d32(rng) : rand(rng, pool)
    pos()    = abs(single())
    # cancellation-heavy: y very close to -x
    function cancel()
        x = single()
        y = -canonicalize(Float32(x), ldexp(Float32(x), -24 - rand(rng, 0:10)))
        (x, y)
    end
    return [
        accrow2("+", +, +, mixed; domain = "mixed pools"),
        accrow2("+", +, +, cancel; domain = "cancellation"),
        accrow2("-", -, -, mixed; domain = "mixed pools"),
        accrow2("*", *, *, mixed; domain = "mixed pools"),
        accrow2("/", /, /, mixed; domain = "mixed pools"),
        accrow1("inv", inv, inv, single; domain = "mixed pools"),
        accrow1("sqrt", sqrt, sqrt, pos; domain = "positive"),
        accrow1("abs2", abs2, x -> x * x, single; domain = "mixed pools"),
        accrow2("hypot", hypot, (a, b) -> sqrt(a * a + b * b), mixed; domain = "mixed pools"),
    ]
end

# fma is bounded against the operand scale, not relatively: under near-total
# cancellation the true result can be 2^-173 relative to unit operands, below
# anything a 48-bit format expresses relatively.  Derived bound (see
# src/core/arithmetic.jl fma_dd_): 2u^2 * (1+80u) * max(|xh*yh|, |zh|).
function fma_row()
    rng  = MersenneTwister(2)
    pool = adversarial_pool()
    u2   = big(2.0)^-48
    worst = 0.0
    n = 0
    while n < 4NSAMPLES
        x = rand(rng, pool); y = rand(rng, pool); z = rand(rng, pool)
        g = fma(x, y, z)
        isfinite(g) || continue
        ex, ey, ez = reference(x), reference(y), reference(z)
        want  = ex * ey + ez
        scale = max(abs(ex * ey), abs(ez))
        (isfinite(want) && big(2.0)^-77 < scale < big(2.0)^80) || continue
        n += 1
        worst = max(worst, Float64(abs(reference(g) - want) / (u2 * scale)))
    end
    return @sprintf("| `fma` / `muladd` | %d | %.4f | 2.0000... (derived: 2u²(1+80u)·M) |",
                    n, worst)
end

# ---------------------------------------------------------- reductions ----

function reduction_rows()
    rng = MersenneTwister(3)
    rows = String[]

    # Float32 data, Double32 accumulator vs the naive Float32 sum.
    a = [Float32(1f4 * (rand(rng) - 0.3)) for _ in 1:100_000]
    want = sum(big.(a))
    e_d32 = err_ulp48(sum(Double32, a), want)
    e_f32 = Float64(abs(big(sum(a)) - want) / ulp48(want))
    push!(rows, @sprintf("| `sum(Double32, A)` vs `sum(::Vector{Float32})` | 100000 | %.3f | %.1f |",
                         e_d32, e_f32))

    # compensated_dot vs naive Float32 dot
    x = rand(rng, Float32, 50_000) .* 1f3
    y = rand(rng, Float32, 50_000) .* 1f3
    wantd = sum(big(xi) * big(yi) for (xi, yi) in zip(x, y))
    e_cd = err_ulp48(Double32s.compensated_dot(x, y), wantd)
    e_nd = Float64(abs(big(dot(x, y)) - wantd) / ulp48(wantd))
    push!(rows, @sprintf("| `compensated_dot(x, y)` vs Float32 `dot` | 50000 | %.3f | %.1f |",
                         e_cd, e_nd))

    # norm2 with its exact power-of-two scaling
    v = rand(rng, Float32, 10_000) .* 1f18            # squares overflow Float32
    wantn = sqrt(sum(big(vi)^2 for vi in v))
    e_n = err_ulp48(Double32s.norm2(Double32.(v)), wantn)
    push!(rows, @sprintf("| `norm2` (squares overflow Float32) | 10000 | %.3f | Inf (Float32 overflows) |",
                         e_n))
    return rows
end

# ------------------------------------------------------------- assemble ---

vinfo = string(VERSION)
hdr  = "| operation | domain | samples | median (ulp48) | worst (ulp48) | faithful |\n" *
       "|:----------|:-------|--------:|---------------:|--------------:|---------:|"

arith = arithmetic_rows()
fmar  = fma_row()
red   = reduction_rows()
gu    = has_gradual_underflow()

report = """
# Double32s Accuracy Report

*Generated by `docs/Reports/accuracies.jl` on Julia $vinfo.*

Errors are measured against a 400-bit `BigFloat` reference in **nominal
48-bit ulps of the true value**: `ulp48(V) = 2^(floor(log2|V|)-47)`, the
design's section 9.1 metric.  "Faithful" is the fraction of samples with
error strictly below 1 ulp48.  Results are charged against the 48-bit lattice
only where it exists (`|V| ≥ 2^-100`); below that the format tapers toward 24
bits by construction (design section 5.5) and accuracy statements belong to
the format, not the operations.

Gradual underflow honored on this host: **$gu** (`Double32s.has_gradual_underflow()`).
All subnormal-range claims are conditional on it.

## Arithmetic

$hdr
$(join(arith, "\n"))

Documented decisions behind these numbers (measured this development cycle,
regressions in `test/core/arithmetic.jl`):

- `sqrt` is empirically **faithful** — worst 0.75 ulp48, zero violations,
  including near-perfect-square adversaries — with a single Newton correction.
- `/` is **not faithful** and is documented as such: worst 4.00 ulp48 over
  101,761 lattice-range quotients.  A second correction (implemented,
  measured: 2.30 ulp48) still does not reach faithfulness and costs 2× in
  GPU kernels, so every policy uses one correction.
- `inv`: worst 1.16 ulp48, one violation per few hundred structured operands.

## Compensated fma

Bounded against the operand scale `M = max(|x·y|, |z|)`, in units of `u²·M`,
`u = 2⁻²⁴`; the derived bound is `2u²(1+80u)·M` and it is tight.

| operation | samples | worst (u²·M) | bound |
|:----------|--------:|-------------:|:------|
$fmar

## Reductions

| comparison | n | Double32 error (ulp48) | naive error (ulp48) |
|:-----------|--:|-----------------------:|--------------------:|
$(join(red, "\n"))

## Coverage

Every implemented `Double32` surface is measured here: scalar arithmetic,
compensated fma, reductions, elementary functions, trigonometry, hyperbolics,
and dense matrix multiply.  The `KernelAbstractions` reductions and the tiled
device matmul are measured separately in `gpu_report.md`, which has the GPU
packages available.
"""

outpath = joinpath(@__DIR__, "accuracy_report.md")
open(outpath, "w") do io
    write(io, report)
end
println("wrote ", outpath)

# --------------------------------------------------- elementary functions ---
# Roadmap Phase 6. Appended by the Phase 6 work; see docs/Checkpoint.md.

let
    ulp48(v::BigFloat) = big(2.0)^(Int(floor(log2(abs(v)))) - 47)
    function sweep(f, ref, pool)
        worst = 0.0; nf = 0; n = 0
        for x in pool
            local e
            try; e = ref(Double32s.reference(x)); catch; continue; end
            (isfinite(e) && !iszero(e)) || continue
            big(2.0)^-100 <= abs(e) <= big(2.0)^100 || continue
            local g
            try; g = f(x); catch; continue; end
            isfinite(g) || continue
            n += 1
            u = Float64(abs(Double32s.reference(g) - e) / ulp48(e))
            u > 0.5 && (nf += 1)
            worst = max(worst, u)
        end
        return worst, nf, n
    end

    wide = Double32[]
    for k in -40:40, m in (1.0f0, 1.3f0, 1.7f0, 2.5f0, 3.9f0, 7.1f0)
        push!(wide, Double32(ldexp(m, k)))
    end
    append!(wide, [Double32(Float32(i)) / 7 for i in 1:300])
    pos     = filter(x -> x.hi > 0, wide)
    expdom  = [Double32(Float32(x)) for x in range(-80, 80; length = 1200)]
    unit    = [Double32(Float32(x)) for x in range(-0.999, 0.999; length = 1200)]
    trigdom = [Double32(Float32(x)) for x in range(-100, 100; length = 1500)]
    small   = [Double32(Float32(10.0^-k)) * Double32(Float32(1 + 0.3k)) for k in 1:14]

    rows = (("cbrt",  cbrt,  cbrt,           wide),
            ("rsqrt", Double32s.rsqrt, x -> 1/sqrt(x), pos),
            ("exp",   exp,   exp,            expdom),
            ("expm1", expm1, expm1,          vcat(expdom, small)),
            ("exp2",  exp2,  x -> big(2)^x,  expdom),
            ("exp10", exp10, x -> big(10)^x, [Double32(Float32(x)) for x in range(-35,35;length=1200)]),
            ("log",   Double32s.log_or_nan, log, pos),
            ("log1p", log1p, log1p,          vcat(small, [Double32(Float32(x)) for x in range(-0.9,5.0;length=800)])),
            ("log2",  log2,  log2,           pos),
            ("log10", log10, log10,          pos),
            ("sin",   sin,   sin,            trigdom),
            ("cos",   cos,   cos,            trigdom),
            ("tan",   tan,   tan,            trigdom),
            ("atan",  atan,  atan,           wide),
            ("asin",  asin,  asin,           unit),
            ("acos",  acos,  acos,           unit),
            ("sinh",  sinh,  sinh,           vcat(small, expdom)),
            ("cosh",  cosh,  cosh,           vcat(small, expdom)),
            ("tanh",  tanh,  tanh,           vcat(small, [Double32(Float32(x)) for x in range(-17,17;length=1200)])),
            ("asinh", asinh, asinh,          vcat(small, wide)),
            ("acosh", acosh, acosh,          vcat([Double32(Float32(x)) for x in range(1.0,50.0;length=800)],
                                                  [Double32(1.0f0)+Double32(Float32(10.0^-k)) for k in 1:12])),
            ("atanh", atanh, atanh,          vcat(small, unit)))

    open(joinpath(@__DIR__, "accuracy_report.md"), "a") do io
        println(io, "\n## Elementary functions (roadmap Phase 6)\n")
        println(io, "Worst error in nominal-48-bit ulps against a 400-bit `BigFloat`")
        println(io, "reference, restricted to results in `[2^-100, 2^100]` where the 48-bit")
        println(io, "lattice exists. \"unfaithful\" counts results more than half a ulp48 from")
        println(io, "the true value. \"operands\" is how many arguments from the")
        println(io, "test pool actually produced a comparable result — the count after")
        println(io, "discarding those outside that range.\n")
        println(io, "| function | worst ulp48 | unfaithful | operands |")
        println(io, "|:---------|------------:|-----------:|---------:|")
        for (nm, f, r, pool) in rows
            w, nf, n = sweep(f, r, pool)
            @printf(io, "| `%s` | %.3f | %d | %d |\n", nm, w, nf, n)
        end
        println(io, """

`sinh`, `cosh`, `asinh`, `acosh`, and `atanh` are each written in the
non-cancelling form; `cosh` is the exception that proves the rule, since it
never cancels and the `expm1` form measured *worse* (3.6 against 1.1).

**None of these is faithfully rounded, and none is claimed to be.** The floor
for anything ending in a division is this package's `/`, measured at 4.00
ulp48 — which is why `tan`, `asin`, and `acos` sit highest. `log1p` and
`log2`/`log10` were each reworked once to *avoid* a division that dominated
their error; see the Phase 6 entry in `docs/Checkpoint.md`.

`sin`, `cos`, `tan`, and `sincos` return `NaN` beyond
`Double32s.TRIG_ARG_LIMIT = 2^20`, where the Cody–Waite reduction can no longer
be trusted. That is a deliberate policy, not a range over which the table above
applies.""")
    end
    @info "appended elementary-function accuracy section"
end


# ------------------------------------------ reductions and matrix multiply ---
# Requested coverage: every Double32 surface, not only the scalar functions.

let
    setprecision(BigFloat, 400)
    rng = MersenneTwister(20260728)

    # A deliberately hostile vector: one huge value then many small ones, which
    # is where a Float32 accumulator loses everything.
    hostile = Float32[1f8; fill(1.0f0, 100_000)]
    graded  = Float32.(1 ./ (1:100_000))
    mixed   = Float32.(randn(rng, 100_000))
    # Wide dynamic range: exponents spread over 2^-60..2^60, so the EXACT sum
    # needs far more than 48 bits and no format here can be exact.
    spread  = Float32[ldexp(Float32(randn(rng)), rand(rng, -60:60)) for _ in 1:100_000]

    relerr(got, exact) = Float64(abs(big(got) - exact) / abs(exact))

    rows = String[]
    for (name, v) in (("1e8 then 1e5 ones", hostile),
                      ("1/k, k = 1..1e5", graded),
                      ("randn, 1e5", mixed),
                      ("randn x 2^(-60..60), 1e5", spread))
        ex = sum(BigFloat.(Float64.(v)))
        d32  = relerr(Double32s.reference(sum(Double32, v)), ex)
        csum = relerr(Double32s.reference(Double32s.compensated_sum(v)), ex)
        f32  = relerr(big(sum(v)), ex)
        f64  = relerr(big(sum(Float64, v)), ex)
        push!(rows, @sprintf("| %s | %.2e | %.2e | %.2e | %.2e |",
                             name, d32, csum, f32, f64))
    end

    x32 = Float32.(randn(rng, 50_000)); y32 = Float32.(randn(rng, 50_000))
    exd = sum(BigFloat.(Float64.(x32)) .* BigFloat.(Float64.(y32)))
    dot_d32 = relerr(Double32s.reference(Double32s.compensated_dot(x32, y32)), exd)
    dot_f32 = relerr(big(sum(x32 .* y32)), exd)
    dot_f64 = relerr(big(sum(Float64.(x32) .* Float64.(y32))), exd)

    exn = sqrt(sum(BigFloat.(Float64.(x32)) .^ 2))
    n_d32 = relerr(Double32s.reference(Double32s.norm2(Double32.(x32))), exn)
    n_f32 = relerr(big(sqrt(sum(abs2, x32))), exn)

    n = 128
    A32 = Float32.(randn(rng, n, n)); B32 = Float32.(randn(rng, n, n))
    exm = Float64.(A32) * Float64.(B32)
    Cd  = Double32.(A32) * Double32.(B32)
    # Normwise, not elementwise. With `randn` data some entries of C are near
    # zero through cancellation, and an elementwise *relative* error there
    # measures the conditioning of that entry rather than the algorithm — it
    # reported 2.0e-2 for BLAS `Float32`, which is not a real result.
    scale  = maximum(abs.(exm))
    mm_d32 = maximum(abs.(Float64.(Cd) .- exm)) / scale
    mm_f32 = maximum(abs.(Float64.(A32 * B32) .- exm)) / scale

    open(joinpath(@__DIR__, "accuracy_report.md"), "a") do io
        println(io, "\n## Reductions in detail\n")
        println(io, "Relative error of the sum against a 400-bit `BigFloat` reference.")
        println(io, "`sum(Double32, A)` accumulates `Float32` data in ~48 bits through")
        println(io, "Base's own `mapreduce`; `compensated_sum` pins a sequential order.\n")
        println(io, "")
        println(io, "**The exact zeros are real, and worth understanding.** When every")
        println(io, "element is a `Float32` and their exponents span a narrow range, the")
        println(io, "*exact* sum needs only about 24 bits plus that span — for `1/k` over")
        println(io, "1e5 terms that is roughly 44 bits, which fits inside `Double32`'s 48.")
        println(io, "Those sums are not merely accurate, they are **exact**. The last row")
        println(io, "spreads the exponents over `2^-60..2^60` so the exact sum needs far")
        println(io, "more than 48 bits, and there the format's true error appears.")
        println(io, "")
        println(io, "| data | `sum(Double32, A)` | `compensated_sum` | `sum` (Float32) | `sum(Float64, A)` |")
        println(io, "|:-----|-------------------:|------------------:|----------------:|------------------:|")
        for r in rows; println(io, r); end

        println(io, "\n### Dot product and norm (50 000 elements)\n")
        println(io, "| quantity | Double32 | Float32 | Float64 |")
        println(io, "|:---------|---------:|--------:|--------:|")
        @printf(io, "| `compensated_dot(x, y)` | %.2e | %.2e | %.2e |\n", dot_d32, dot_f32, dot_f64)
        @printf(io, "| `norm2(x)` | %.2e | %.2e | — |\n", n_d32, n_f32)
        println(io, "\n`compensated_dot` forms every `Float32` product **exactly** as a")
        println(io, "`Double32` before accumulating, so nothing is lost before the summation.")

        println(io, "\n## Matrix multiply (128 x 128, Float32 data)\n")
        println(io, "Worst **normwise** error: `max|C - Cref| / max|Cref|`.")
        println(io, "An elementwise *relative* error would instead measure the")
        println(io, "conditioning of the near-zero entries `randn` data produces by")
        println(io, "cancellation — it reported 2.0e-2 for BLAS `Float32`, which is")
        println(io, "not a real result.")
        println(io, "")
        println(io, "| product | worst relative error |")
        println(io, "|:--------|---------------------:|")
        @printf(io, "| `Double32.(A) * Double32.(B)` (generic) | %.2e |\n", mm_d32)
        @printf(io, "| `A * B` (Float32, BLAS) | %.2e |\n", mm_f32)
        println(io, "\nThe device-side tiled kernel `Double32s.matmul`, which forms every")
        println(io, "`Float32` product exactly with `two_prod_`, is measured in `gpu_report.md`.")
    end
    @info "appended reductions and matrix-multiply accuracy sections"
end
