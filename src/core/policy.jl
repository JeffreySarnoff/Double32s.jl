# core/policy.jl — compile-time accuracy/performance selectors.
#
# Design section 5.8. Empty singleton types: they carry no data, their identity
# *is* the information, and Julia specializes the whole call tree on them at
# zero runtime cost.

"""
    ArithmeticPolicy

Compile-time selector for the accuracy/performance contract of a `Double32`
operation. Instances are singletons and are always passed by value as the first
argument, so that dispatch — not a runtime flag — chooses the kernel.

Three rules follow from making the policy a type rather than a flag, and they
are what buys the performance:

1. **Never test a policy at runtime.** `if policy isa FastPolicy` inside a
   kernel is a code smell even though the compiler usually removes it; write a
   second method instead.
2. **Never pass a policy as a keyword argument into device code.** Keyword
   arguments do not participate in dispatch. Accept keywords at the host API
   boundary, resolve them to singletons there, pass positionally from then on.
3. **Never store a policy in a mutable or abstractly typed field.** Use a type
   parameter if a policy must be carried in a struct.
"""
abstract type ArithmeticPolicy end

"""
    StrictPolicy()

Preserve the documented operation order, use explicit FMA, guard exceptional
values, and require capability tests for claims involving subnormals. The
default.
"""
struct StrictPolicy <: ArithmeticPolicy end

"""
    NativePolicy()

Preserve the same public result contract where possible but permit
backend-specific instruction selection and launch tuning.
"""
struct NativePolicy <: ArithmeticPolicy end

"""
    FastPolicy()

Explicitly relax error and exceptional-value guarantees. Never the implicit
behavior of `+`, `*`, or `/`.
"""
struct FastPolicy <: ArithmeticPolicy end

"""
    Double32s.DEFAULT_POLICY

The policy the `Base` operators use: `StrictPolicy()`.
"""
const DEFAULT_POLICY = StrictPolicy()

"""
    Double32s.PolicyOp(f, policy) -> callable

Bind a policy to a policy-taking operation, giving a plain binary callable
`(a, b) -> f(policy, a, b)`.

This is rule 2 of [`ArithmeticPolicy`](@ref) made concrete for kernels. A
reduction needs an `op` of two arguments, but every accuracy-selecting operation
in this package takes the policy first. Wrapping in a closure would work on the
host and is the wrong thing to hand a kernel: a closure over a local is an
ordinary struct with a field, and its type is not the same across two call
sites. `PolicyOp` is **zero-size** — both fields are singletons — so it costs no
register, no argument-buffer slot, and no launch-configuration difference, and
the reduction specializes on it completely.

# Examples
```jldoctest
julia> op = Double32s.PolicyOp(Double32s.add, Double32s.FastPolicy());

julia> sizeof(op)
0

julia> op(Double32(1.0f0), Double32(2.0f0))
Double32(3.0f0, 0.0f0)
```
"""
struct PolicyOp{F,P<:ArithmeticPolicy} <: Function
    f::F
    policy::P
end

@inline (o::PolicyOp)(a, b) = o.f(o.policy, a, b)

Base.show(io::IO, o::PolicyOp) = print(io, "PolicyOp(", o.f, ", ", o.policy, ")")
