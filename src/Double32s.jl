"""
    Double32s

A GPU-oriented double-word float built from two `Float32` words.

The package exports exactly one name, the scalar type [`Double32`](@ref).
Everything else that is supported API is marked `public` and reached as
`Double32s.name`, following the export-hygiene rules of design section 1.3.

The module is named `Double32s` and the type `Double32` because a Julia module
cannot also bind its own name to a type: inside `module Double32`, the binding
`Double32` is the module, and `using Double32` would import the module rather
than the scalar.
"""
module Double32s

using Adapt
using LinearAlgebra: LinearAlgebra
using KernelAbstractions
const KA = KernelAbstractions
using AcceleratedKernels
const AK = AcceleratedKernels

export Double32

public hi, lo, words, canonicalize, isnormal
public ArithmeticPolicy, StrictPolicy, NativePolicy, FastPolicy, DEFAULT_POLICY
public add, sub, mul, divide, reciprocal, square_root, sqrt_or_nan, compensated_fma
public nominal_eps, unit_roundoff, float64_reference_is_exact, lsb_exponent
public raw_exponent, sqrt_nonneg
public has_gradual_underflow, has_fused_multiply_add, capability_report
public probe_lane, probe_operands, probe_results, build_report
public PROBE_LANES, UNDERFLOW_LANES, NO_LANES
public allocate_split
public POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, D32_NAN
public SplitDouble32Array, split, unsplit, split!, unsplit!, allocate_split_host
public PolicyOp, DEFAULT_GROUPSIZE
public ak_map!, ak_mapreduce, ak_reduce, ak_sum, ak_dot, ak_norm2
public launch_config, kernel_report, wavefront_size
public ak_matmul!, ak_matmul, MATMUL_TILE, MATMUL_BLOCK
public compensated_dot, compensated_mapreduce, compensated_sum
public rsqrt, log_or_nan, TRIG_ARG_LIMIT
public LN2, INV_LN2, LN10, LOG10_E, LN2_C2, PI_D32, PIO2_D32, PIO4_D32
public LOG2_10, LOG10_2, LOG10_2_C2, LOG1P_COEFFS
public TWO_OVER_PI, PIO2_C3, SQRT3_D32
public EXPM1_COEFFS, ATANH_COEFFS, SIN_COEFFS, COS_COEFFS, ATAN_COEFFS
public TANH_COEFFS, ASINH_COEFFS
public EFT

include("core/type.jl")
include("core/bits.jl")
include("core/eft.jl")
include("core/capability.jl")
include("core/policy.jl")
include("core/normalize.jl")
include("core/constants.jl")
include("core/special.jl")
include("core/convert.jl")
include("core/promote.jl")
include("core/compare.jl")
include("core/arithmetic.jl")
include("math/reciprocal.jl")
include("math/sqrt.jl")
include("math/coefficients.jl")
include("math/horner.jl")
include("math/roots.jl")
include("math/exp.jl")
include("math/log.jl")
include("math/trig.jl")
include("math/hyperbolic.jl")
include("math/powers.jl")
include("core/show.jl")
include("core/random.jl")
include("arrays/splitarray.jl")
include("arrays/matmul.jl")
include("reductions.jl")
include("kernels.jl")
include("gpu/arrays.jl")
include("gpu/capability.jl")
include("gpu/reduce.jl")
include("gpu/linalg.jl")

end # module Double32s
