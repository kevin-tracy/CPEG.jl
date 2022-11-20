using Pkg
Pkg.activate(joinpath(dirname(dirname(@__DIR__)), ".."))
import CPEG as cp
Pkg.activate(dirname(dirname(@__DIR__)))
Pkg.instantiate()

using LinearAlgebra
using StaticArrays
using ForwardDiff
import ForwardDiff as FD
using SparseArrays
using SuiteSparse
using Printf
using MATLAB
import Random
using DelimitedFiles
using Dierckx
using JLD2

Random.seed!(1)

include("/Users/kevintracy/.julia/dev/CPEG/extras/fft_trajopt/clean_cpeg/real_dynamics.jl")
include("/Users/kevintracy/.julia/dev/CPEG/extras/fft_trajopt/clean_cpeg/approx_dynamics.jl")
include("/Users/kevintracy/.julia/dev/CPEG/extras/fft_trajopt/clean_cpeg/utils.jl")
# square root stuff
function chol(A)
    # returns upper triangular Cholesky factorization of matrix A
    return cholesky(Symmetric(A)).U
end
function qrᵣ(A)
    # QR decomposition of A where only the upper triangular R is returned
    return qr(Matrix(A)).R
end

function sqrkalman_filter(μ,F,u,y,kf_sys,params)

    ΓQ, ΓR, sim_dt = kf_sys.ΓQ, kf_sys.ΓR, kf_sys.sim_dt

    # predict one step
    μ̄ = filter_discrete_dynamics(params,SVector{8}(μ),u,sim_dt)
    A = FD.jacobian(_x -> filter_discrete_dynamics(params,SVector{8}(_x),u,sim_dt), μ)
    F̄ = qrᵣ([F*A';ΓQ])

    # innovation
    C = [diagm(ones(7)) zeros(7)]
    z = y - C*μ̄
    G = qrᵣ([F̄*C';ΓR])

    # kalman gain
    L = ((F̄'*F̄*C')/G)/(G')

    # update (Joseph form for Σ)
    μ₊= μ̄ + L*z
    F₊= qrᵣ([F̄*(I - L*C)';ΓR*L'])

    return μ₊, F₊
end

function calc_kρ(params,x)
    r_scaled = x[SA[1,2,3]]
    v_scaled = x[SA[4,5,6]]

    # unscale
    r, v = cp.unscale_rv(params.ev.scale,r_scaled,v_scaled)

    # altitude
    h,_,_ = cp.altitude(params.ev.params.gravity, r)
    ρ1 = cp.density_spline(params.ev.params.dsp, h)
    ρ2 = params.ρ_spline(h)

    return ρ2/ρ1
end
let

    # let's run some MPC
    ev = cp.CPEGWorkspace()

    # vehicle parameters
    ev.params.aero.Cl = 0.29740410453983374
    ev.params.aero.Cd = 1.5284942035954776
    ev.params.aero.A = 15.904312808798327    # m²
    ev.params.aero.m = 2400.0                # kg

    # sim stuff
    ev.dt = NaN # seconds

    # CPEG settings
    ev.scale.uscale = 1e1

    # initial condition
    x0_scaled = [3.5212,0,0, -1.559452236319901, 5.633128235948198,0,0.05235987755982989]

    # get atmosphere stuff in params
    altitudes,densities, Ewind, Nwind = load_atmo()
    xg = [3.3477567254291762, 0.626903908492849, 0.03739968529144168, -0.255884401134421, 0.33667198108223073, -0.056555916829042985, -1.182682624917629]

    ρ_spline = Spline1D(reverse(altitudes), reverse(densities))
    wE_spline = Spline1D(reverse(altitudes), reverse(Ewind))
    wN_spline = Spline1D(reverse(altitudes), reverse(Nwind))

    JLD = jldopen("/Users/kevintracy/.julia/dev/CPEG/controls_for_estimator.jld2")
    U = JLD["U"]

    U = [U[i][1] for i = 1:length(U)]
    X = JLD["X"]

    X = X[50:end]
    U = U[50:end]
    uspl = Spline1D(0:1.0:(1.0*(length(U)-1)),U)
    x0 = X[1]

    # uv =

    N = length(X)
    X = [zeros(7) for i = 1:(10*N)]
    X[1] = x0
    # sim_dt_scaled = 0.2/ev.scale.tscale
    sim_dt= 0.2

    params = (ev = ev,ρ_spline = ρ_spline,wE_spline = wE_spline,wN_spline = wN_spline)


    dscale, tscale = ev.scale.dscale, ev.scale.tscale

    Q = diagm([1e-15*ones(3); 1e-8*ones(3);1e-1;1e-2])
    R = diagm( [(.1/ev.scale.dscale)^2*ones(3); (0.05/(ev.scale.dscale/ev.scale.tscale))^2*ones(3);1e-2])


    kf_sys = (sim_dt = sim_dt, ΓR = chol(R), ΓQ = chol(Q))
    Y = [zeros(7) for i = 1:N]

    μ = [zeros(8) for i = 1:N]
    μ[1] = [X[1];1.0] # start with kρ = 1
    F = [zeros(8,8) for i = 1:N]
    F[1] = chol(diagm([1e-10*ones(7);1]))

    X = [zeros(7) for i = 1:N]
    X[1] = 1*x0

    sim_dt = 0.2

    for i = 1:(N-1)

        u = SA[uspl(sim_dt*(i-1))]
        X[i+1] = real_discrete_dynamics(ev,SVector{7}(X[i][1:7]),u,sim_dt/ev.scale.tscale,params)

        Y[i+1] = X[i+1][1:7] + kf_sys.ΓR*randn(7)

        μ[i+1],F[i+1] = sqrkalman_filter(μ[i],F[i],u,Y[i+1],kf_sys,params)
        # μ[i+1], F[i+1] = cp.sqrkalman_filter(ev, SVector{8}(μ[i]),F[i],u,Y[i+1],kf_sys)
    end

    @show μ
    kρ_true = [calc_kρ(params,X[i]) for i = 1:N]
    kρ_est = [μ[i][8] for i = 1:N]

    alts = [alt_from_x(ev, X[i]) for i = 1:N]

    mat"
    figure
    hold on
    plot($kρ_true)
    plot($kρ_est)
    hold off
    "

    mat"
    figure
    hold on
    plot($alts)
    hold off
    "

    # @show norm(X2[end][1:3] - X3[end][1:3])*1e3
    # @show X2 - X


    # TODO:
    #   - give the normal stuff a shot at new frequency
    #   - do some covariance analysis between these two models
    #   - try and diagonalize this (eigenstuff?)

end
