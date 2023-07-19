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

# let
#     JLD = jldopen("/Users/kevintracy/.julia/dev/CPEG/controls_for_estimator.jld2")
#     U = JLD["U"]
#     @show U
# end
function dynamics_fudge_mg(ev::cp.CPEGWorkspace, x::SVector{7,T}, u::SVector{1,W}, params) where {T,W}

    # scaled variables
    r_scaled = x[SA[1,2,3]]
    v_scaled = x[SA[4,5,6]]
    σ = x[7]

    # unscale
    r, v = cp.unscale_rv(ev.scale,r_scaled,v_scaled)

    # altitude
    h,lat,lon = cp.altitude(ev.params.gravity, r)

    uD,uN,uE = cp.latlongtoNED(lat, lon)
    # density
    ρ = params.ρ_spline(h)
    wE = params.wE_spline(h)
    wN = params.wN_spline(h)

    wind_pp = wN * uN + wE * uE #- wU * uD  # wind velocity in pp frame , m / s
    v_rw = v + wind_pp  # relative wind vector , m / s # if wind == 0, the velocity = v

    # lift and drag magnitudes
    L, D = cp.LD_mags(ev.params.aero,ρ,r,v_rw)

    # basis for e frame
    e1, e2 = cp.e_frame(r,v_rw)

    # drag and lift accelerations
    D_a = -(D/norm(v_rw))*v_rw
    L_a = L*sin(σ)*e1 + L*cos(σ)*e2

    # gravity
    g = cp.gravity(ev.params.gravity,r)

    # acceleration
    ω = ev.planet.ω
    a = D_a + L_a + g - 2*cross(ω,v) - cross(ω,cross(ω,r))

    # rescale units
    v,a = cp.scale_va(ev.scale,v,a)

    return SA[v[1],v[2],v[3],a[1],a[2],a[3],u[1]*ev.scale.uscale]
end

function real_dynamics(
    ev::cp.CPEGWorkspace,
    x_n::SVector{7,T},
    u::SVector{1,W},
    dt_s::T2, params) where {T,W,T2}

    k1 = dt_s*dynamics_fudge_mg(ev,x_n,u,params)
    k2 = dt_s*dynamics_fudge_mg(ev,x_n+k1/2,u,params)
    k3 = dt_s*dynamics_fudge_mg(ev,x_n+k2/2,u,params)
    k4 = dt_s*dynamics_fudge_mg(ev,x_n+k3,u,params)

    return (x_n + (1/6)*(k1 + 2*k2 + 2*k3 + k4))
end

function dynamics_fudge(ev::cp.CPEGWorkspace, x::SVector{8,T}, u::SVector{1,W}) where {T,W,T2}

    # scaled variables
    r_scaled = x[SA[1,2,3]]
    v_scaled = x[SA[4,5,6]]
    σ = x[7]
    kρ = x[8]

    # unscale
    r, v = cp.unscale_rv(ev.scale,r_scaled,v_scaled)

    # altitude
    h,_,_ = cp.altitude(ev.params.gravity, r)
    ρ = kρ*cp.density_spline(ev.params.dsp, h)

    # lift and drag magnitudes
    L, D = cp.LD_mags(ev.params.aero,ρ,r,v)

    # basis for e frame
    e1, e2 = cp.e_frame(r,v)

    # drag and lift accelerations
    D_a = -(D/norm(v))*v
    L_a = L*sin(σ)*e1 + L*cos(σ)*e2

    # gravity
    g = cp.gravity(ev.params.gravity,r)

    # acceleration
    ω = ev.planet.ω
    a = D_a + L_a + g - 2*cross(ω,v) - cross(ω,cross(ω,r))

    # rescale units
    v,a = cp.scale_va(ev.scale,v,a)

    return SA[v[1],v[2],v[3],a[1],a[2],a[3],u[1]*ev.scale.uscale, 0.0]
end

function kf_dynamics(
    ev::cp.CPEGWorkspace,
    x_n::SVector{8,T},
    u::SVector{1,W},
    dt_s::T2) where {T,W,T2}

    k1 = dt_s*dynamics_fudge(ev,x_n,u)
    k2 = dt_s*dynamics_fudge(ev,x_n+k1/2,u)
    k3 = dt_s*dynamics_fudge(ev,x_n+k2/2,u)
    k4 = dt_s*dynamics_fudge(ev,x_n+k3,u)

    return (x_n + (1/6)*(k1 + 2*k2 + 2*k3 + k4))
end

# function discrete_dynamics(p,x1,u1,k)
#     rk4_fudge(p.ev,SVector{7}(x1),SA[u1[1]],u1[2]/p.ev.scale.tscale, 1.0);
# end

function load_atmo(;path="/Users/kevintracy/.julia/dev/CPEG/src/MarsGramDataset/all/out10.csv")
    TT = readdlm(path, ',')
    alt = Vector{Float64}(TT[2:end,2])
    density = Vector{Float64}(TT[2:end,end])
    Ewind = Vector{Float64}(TT[2:end,7])
    Nwind = Vector{Float64}(TT[2:end,9])
    return alt*1000, density, Ewind, Nwind
end


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

    ΓQ, ΓR, dt = kf_sys.ΓQ, kf_sys.ΓR, kf_sys.dt

    # predict one step
    μ̄ = kf_dynamics(params.ev, SVector{8}(μ),u,kf_sys.dt)
    A = FD.jacobian(_x -> kf_dynamics(params.ev, SVector{8}(_x),u,kf_sys.dt), μ)
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

    # ρ1 = densities
    # alt2= range(altitudes[1],altitudes[end], length = 500)
    # ρ2 = ρ_spline(alt2)
    # mat"
    # figure
    # hold on
    # plot($altitudes, log($ρ1),'ro')
    # plot($alt2, log($ρ2))
    # hold off
    # "
    # ρ1 = Ewind
    # alt2= range(altitudes[1],altitudes[end], length = 500)
    # ρ2 = wE_spline(alt2)
    # mat"
    # figure
    # hold on
    # plot($altitudes, ($ρ1),'ro')
    # plot($alt2, ($ρ2),'b--')
    # hold off
    # "
    # error()

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
    sim_dt_scaled = 0.2/ev.scale.tscale

    params = (ev = ev,ρ_spline = ρ_spline,wE_spline = wE_spline,wN_spline = wN_spline,
    dt = sim_dt_scaled)


    dscale, tscale = ev.scale.dscale, ev.scale.tscale

    # Q = diagm( [1e-4*ones(3); 1e-5*ones(3);1e-10;1e-5])
    # R = diagm( [1e-10*ones(6);1e-10])
    # Q = diagm( [(.000005)^2*ones(3)/ev.scale.dscale; .000005^2*ones(3)/(ev.scale.dscale/ev.scale.tscale); (1e-10)^2;(1e-10)^2;(1e-10)^2;(1e-10)^2])
    # R = diagm( 10*[(.1)^2*ones(3)/ev.scale.dscale; (0.0002)^2*ones(3)/(ev.scale.dscale/ev.scale.tscale);1e-10])
    #Q = diagm( [(.005)^2*ones(3)/ev.scale.dscale; .005^2*ones(3)/(ev.scale.dscale/ev.scale.tscale); (1e-3)^2;(1e-3)^2])
    #R = diagm( [(.1)^2*ones(3)/ev.scale.dscale; (0.0002)^2*ones(3)/(ev.scale.dscale/ev.scale.tscale);1e-10])

    Q = diagm([1e-15*ones(3); 1e-8*ones(3);1e-1;1e-2])

    R = diagm( [(.1/ev.scale.dscale)^2*ones(3); (0.05/(ev.scale.dscale/ev.scale.tscale))^2*ones(3);1e-2])


    kf_sys = (dt = sim_dt_scaled, ΓR = chol(R), ΓQ = chol(Q))
    Y = [zeros(7) for i = 1:N]

    μ = [zeros(8) for i = 1:N]
    μ[1] = [X[1];1.0] # start with kρ = 1
    F = [zeros(8,8) for i = 1:N]
    F[1] = chol(diagm([1e-10*ones(7);1]))

    X = [zeros(8) for i = 1:N]
    X[1] = [x0;1.0]
    # X[1] = 1*x0

    # Es = [zeros(7) for i = 1:N-1]
    # for i = 1:(N-1)
    #
    #     x = X[i]
    #     u = SA[uspl(kf_sys.dt*(i-1))]
    #     x_real = real_dynamics(params.ev,SVector{7}(x[1:7]),u,params.dt,params)
    #     x_kf = kf_dynamics(params.ev, SVector{8}([x;0.7]),u,params.dt)
    #     Es[i] = x_real - x_kf[1:7]
    #     X[i+1] = x_real
    # end
    #
    # Qest = cov(hcat(Es...)')
    # @show Qest
    for i = 1:(N-1)
        # X[i+1] = real_dynamics(params.ev,SVector{7}(X[i]),U[i],params.dt,params)
        # u = SA[sin(i/100)]
        u = SA[uspl(kf_sys.dt*(i-1))]
        X[i+1][1:7] = real_dynamics(params.ev,SVector{7}(X[i][1:7]),u,params.dt,params)
        # @show typeof(u)
        # X[i+1] = kf_dynamics(params.ev, SVector{8}(X[i]),u,params.dt) + kf_sys.ΓQ*randn(8)
        # X[i+1] = kf_dynamics(params.ev,SVector{8}([X[i];1.4]),U[i],params.dt,params,kf_sys.dt,params)

        Y[i+1] = X[i+1][1:7] + kf_sys.ΓR*randn(7)

        μ[i+1],F[i+1] = sqrkalman_filter(μ[i],F[i],u,Y[i+1],kf_sys,params)
        # μ[i+1], F[i+1] = cp.sqrkalman_filter(ev, SVector{8}(μ[i]),F[i],u,Y[i+1],kf_sys)
    end

    # @show μ
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
