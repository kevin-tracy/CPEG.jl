__precompile__(true)
module CPEG

# greet() = print("Hello World!")

using LinearAlgebra
using StaticArrays
using ForwardDiff
using SparseArrays
using SuiteSparse
using Printf


include("qp_solver.jl")
include("reference_systems.jl")
include("atmosphere.jl")
include("scaling.jl")
include("gravity.jl")
include("aero_forces.jl")
include("vehicle.jl")
include("dynamics.jl")
include("post_process.jl")
include("estimator/srekf.jl")
include("estimator/srekf_wind.jl")
include("estimator/dynamics.jl")
include("estimator/dynamicswind.jl")
include("estimator/postprocess.jl")
# include("estimator/srekf.jl")
end # module
