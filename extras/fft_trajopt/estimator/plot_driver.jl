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
include("/Users/kevintracy/.julia/dev/CPEG/extras/fft_trajopt/clean_cpeg/controller.jl")

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

function initialize_kf(ev, sim_dt, N, μ0, Σ0)
    dscale, tscale = ev.scale.dscale, ev.scale.tscale

    Q = diagm([1e-10*ones(3); 1e-6*ones(3);1e-10;1e-2])
    # Q = diagm([1e-15*ones(3); 1e-8*ones(3);1e-1;1e-4])
    R = diagm( [(100.0/ev.scale.dscale)^2*ones(3); (0.2/(ev.scale.dscale/ev.scale.tscale))^2*ones(3);1e-8])


    kf_sys = (sim_dt = sim_dt, ΓR = chol(R), ΓQ = chol(Q))
    Y = [zeros(7) for i = 1:N]

    μ = [zeros(8) for i = 1:N]
    # μ[1] = [X[1];1.0] # start with kρ = 1
    μ[1] = μ0
    F = [zeros(8,8) for i = 1:N]
    # F[1] = chol(diagm([1e-10*ones(7);1]))
    F[1] = chol(Σ0)
    return μ,F,kf_sys,Y
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

    # # get atmosphere stuff in params
    # altitudes,densities, Ewind, Nwind = load_atmo()
    # xg = [3.3477567254291762, 0.626903908492849, 0.03739968529144168, -0.255884401134421, 0.33667198108223073, -0.056555916829042985, -1.182682624917629]
    #
    # ρ_spline = Spline1D(reverse(altitudes), reverse(densities))
    # wE_spline = Spline1D(reverse(altitudes), reverse(0*Ewind))
    # wN_spline = Spline1D(reverse(altitudes), reverse(0*Nwind))


    # get atmosphere stuff in params
    altitudes,densities, Ewind, Nwind = load_atmo()
    nx = 7
    nu = 2
    Q = Diagonal([0,0,0,0,0,0.0,1e-8])
    Qf = 1e8*Diagonal([1.0,1,1,0,0,0,0])
    Qf[7,7] = 1e-4
    R = Diagonal([.1,10.0])
    x_desired = [3.3477567254291762, 0.626903908492849, 0.03739968529144168, -0.255884401134421, 0.33667198108223073, -0.056555916829042985, -1.182682624917629]


    u_min = [-100, 1e-8]
    u_max =  [100, 4.0]
    δu_min = [-100,-.3]
    δu_max = [100, 0.3]

    x_min = [-1e3*ones(6); -pi]
    x_max = [1e3*ones(6);   pi]
    δx_min = [-1e3*ones(6); -deg2rad(20)]
    δx_max = [1e3*ones(6);   deg2rad(20)]
    u_desired = [0; 2.0]

    ρ_spline = Spline1D(reverse(altitudes), reverse(densities))
    wE_spline = Spline1D(reverse(altitudes), reverse(1*Ewind))
    wN_spline = Spline1D(reverse(altitudes), reverse(1*Nwind))
    reg = 10.0
    X = [zeros(nx) for i = 1:10]
    U =[zeros(nu) for i = 1:10]

    params = Params(
                    ρ_spline,
                    wE_spline,
                    wN_spline,
                    nx,nu,
                    Q,R,Qf,
                    u_min,u_max,x_min,x_max,
                    δu_min, δu_max,δx_min,δx_max,
                    x_desired, u_desired,
                    ev, reg, X, U,
                    :nominal,
                    Dict(:nominal => true, :terminal => false, :coast => false),
                    10,1.0)



    initialize_control!(params,x0_scaled)


    # main sim
    N = 5000
    sim_dt = 0.2
    X = [zeros(7) for i = 1:N]
    X[1] = x0_scaled
    U = [zeros(2) for i = 1:N-1]
    kρ_est = ones(N)
    α = 0.3
    kρ_est[1] = 1
    filter_initialized = false

    μ,F,kf_sys,Y = initialize_kf(ev, sim_dt, N, [x0_scaled;1.0], diagm([1e-10*ones(7);.01]))

    u = SA[0.0]

    for i = 1:(N-1)

        # update kρ from estimator
        params.kρ=kρ_est[i]

        if rem((i-1)*sim_dt, 1.0) == 0 # if it's time to update cpeg control
            u = SA[update_control_2!(params, X[i], sim_dt,i)]
        end

        X[i+1] = real_discrete_dynamics(ev,SVector{7}(X[i][1:7]),u,sim_dt/ev.scale.tscale,params)
        Y[i+1] = X[i+1][1:7] + kf_sys.ΓR*randn(7)

        if alt_from_x(ev, X[i+1]) < 80e3
            if filter_initialized
                μ[i+1],F[i+1] = sqrkalman_filter(μ[i],F[i],u,Y[i+1],kf_sys,params)
                kρ_est[i+1] = (1-α)*μ[i+1][8] + α*kρ_est[i]
            else
                F[i+1] = 1*F[1]
                μ[i+1] = [Y[i+1];1]
                filter_initialized = true
            end
        end

        # check sim termination
        if alt_from_x(ev,X[i+1]) < alt_from_x(ev,params.x_desired[1:3])
            @info "SIM IS DONE"
            rr = normalize(params.x_desired[1:3])
            md = norm((I - rr*rr')*(X[i+1][1:3] - params.x_desired[1:3])*ev.scale.dscale/1e3)
            @info "miss distance is: $md km"
            X = X[1:(i+1)]
            U = U[1:i]
            μ = μ[1:(i+1)]
            kρ_est = kρ_est[1:(i+1)]
            break
        end

    end

    # TODO:
    # next steps
    # 1. add this into driver.jl, wrap a function around this where we push the
    # filter and dynamics through one of the time steps
    # 2. option 2 is just call update_control! when it is time to update the
    # control (right now I vote for option 2)
    # @show μ
    N = length(X)
    kρ_true = [calc_kρ(params,X[i]) for i = 1:N]


    alts = [alt_from_x(ev, X[i]) for i = 1:N]

    mat"
    figure
    hold on
    title('krho')
    plot($kρ_true)
    plot($kρ_est)
    legend('krho true','krho estimator')
    hold off
    "

    # mat"
    # figure
    # hold on
    # plot($alts)
    # hold off
    # "

    # @show norm(X2[end][1:3] - X3[end][1:3])*1e3
    # @show X2 - X


    # TODO:
    #   - give the normal stuff a shot at new frequency
    #   - do some covariance analysis between these two models
    #   - try and diagonalize this (eigenstuff?)

end
