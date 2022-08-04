cd("/Users/kevintracy/.julia/dev/CPEG")
Pkg.activate(".")
cd("/Users/kevintracy/.julia/dev/CPEG/src")
using LinearAlgebra
using StaticArrays
using ForwardDiff
using SparseArrays
using SuiteSparse
using Printf

include("qp_solver.jl")
include("atmosphere.jl")
include("scaling.jl")
include("gravity.jl")
include("aero_forces.jl")
include("vehicle.jl")
include("dynamics.jl")
include("post_process.jl")

using LinearAlgebra
using StaticArrays
using ForwardDiff
using SparseArrays
using SuiteSparse
using MATLAB

using JLD2

@load "/Users/kevintracy/.julia/dev/CPEG/src/example_traj.jld2"

ev = CPEGWorkspace()

Rm = ev.params.gravity.Rp_e
r0 = SA[Rm+125e3, 0.0, 0.0] #Atmospheric interface at 125 km altitude
V0 = 5.845e3 #Mars-relative velocity at interface of 5.845 km/sec
γ0 = -15.474*(pi/180.0) #Flight path angle at interface
v0 = V0*SA[sin(γ0), cos(γ0), 0.0]
σ0 = deg2rad(3)

Rf = Rm+10.0e3 #Parachute opens at 10 km altitude
rf = Rf*SA[cos(7.869e3/Rf)*cos(631.979e3/Rf); cos(7.869e3/Rf)*sin(631.979e3/Rf); sin(7.869e3/Rf)]


# vehicle parameters
ev.params.aero.Cl = 0.29740410453983374
ev.params.aero.Cd = 1.5284942035954776
ev.params.aero.A = 15.904312808798327    # m²
ev.params.aero.m = 2400.0                # kg

# qp solver settings
ev.qp_solver_opts.verbose = false
ev.qp_solver_opts.atol = 1e-10
ev.qp_solver_opts.max_iters = 50

# MPC stuff
ev.cost.σ_tr = deg2rad(10) # radians
ev.cost.rf = rf # goal position (meters, MCMF)
ev.cost.γ = 1e3

# sim stuff
ev.dt = 2.0 # seconds

# CPEG settings
ev.verbose = true
ev.ndu_tol = 1e-3
ev.max_cpeg_iter = 30
ev.miss_distance_tol = 1e3  # m

ev.scale.uscale = 1e1

N = length(X)

X2 = [(@SVector zeros(7)) for i = 1:N]
X2[1] = 1*X[1]

for i = 1:(N-1)
    X2[i+1] = rk4(ev,X2[i],U[i],ev.dt/ev.scale.tscale)
end

X2m = hcat(Vector.(X2)...)
Um = vcat(Vector.(U)...)

mat"
figure
hold on
plot($X2m')
legend('px','py','pz','vx','vy','vz','sigma')
hold off
"

mat"
figure
hold on
plot($Um')
hold off
"
