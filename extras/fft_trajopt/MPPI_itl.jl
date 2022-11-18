using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), ".."))
import CPEG as cp
Pkg.activate(dirname(@__DIR__))
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
using LegendrePolynomials

import FiniteDiff

function load_atmo(;path="/Users/kevintracy/.julia/dev/CPEG/src/MarsGramDataset/all/out8.csv")
    TT = readdlm(path, ',')
    alt = Vector{Float64}(TT[2:end,2])
    density = Vector{Float64}(TT[2:end,end])
    Ewind = Vector{Float64}(TT[2:end,8])
    Nwind = Vector{Float64}(TT[2:end,10])
    return alt*1000, density, Ewind, Nwind
end
function dynamics_fudge_mg(ev::cp.CPEGWorkspace, x::SVector{7,T}, u::SVector{1,W}, params) where {T,W}

    # scaled variables
    r_scaled = x[SA[1,2,3]]
    v_scaled = x[SA[4,5,6]]
    σ = x[7]

    # unscale
    r, v = cp.unscale_rv(ev.scale,r_scaled,v_scaled)

    # altitude
    # h = cp.altitude(ev.params.gravity, r)
    h,lat,lon = cp.altitude(ev.params.gravity, r)

    uD,uN,uE = cp.latlongtoNED(lat, lon)
    # density
    # ρ = kρ*cp.density(ev.params.density, h)
    # mat"$ρ = spline($params.altitudes,$params.densities, $h);"
    # mat"$wE = spline($params.altitudes,$params.Ewind, $h);"
    # mat"$wN = spline($params.altitudes,$params.Nwind, $h);"
    # @show h
    ρ = params.ρ_spline(h)
    wE = params.wE_spline(h)
    wN = params.wN_spline(h)
    # @show "oops"
#
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

function rk4_fudge_mg(
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

function alt_from_x(ev::cp.CPEGWorkspace, x)
    r_scaled = x[SA[1,2,3]]
    v_scaled = SA[2,3,4.0]
    r, v = cp.unscale_rv(ev.scale,r_scaled,v_scaled)
    h,_,_ = cp.altitude(ev.params.gravity, r)
    h
end

function legendre_policy(params,norm_v,θ)
    x = norm_v*params.a + params.b
    # @show norm_v
    # @show x
    # @assert abs(x) < 1.000001
    if abs(x)>1
        return 0.0
    else
        x = clamp(x,-1,1)

        y = (x + 1)/2
        y = y^.5 # TODO
        # y
        x = -1 + 2*y
        P = collectPl(x, lmax = (length(θ)-1))
        θ'*P
        # @evalpoly(x,θ...)
    end
end

function dynamics(params, x::SVector{7,T}, θ; verbose = false) where {T}

    # scaled variables
    r_scaled = x[SA[1,2,3]]
    v_scaled = x[SA[4,5,6]]
    σ = x[7]
    # σ = legendre_policy(params,norm(v),θ)

    ev = params.ev

    # unscale
    r, v = cp.unscale_rv(ev.scale,r_scaled,v_scaled)

    # altitude
    h,_,_ = cp.altitude(ev.params.gravity, r)
    ρ = cp.density_spline(ev.params.dsp, h)
    if verbose
        @show r,v,h
    end

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
    v_s,a_s = cp.scale_va(ev.scale,v,a)

    # σ̇ policy
    σ̇ = legendre_policy(params,norm(v),θ)
    # σ̇ = 0
    if verbose
        @show θ
        @show σ̇
        @show SA[v_s[1],v_s[2],v_s[3],a_s[1],a_s[2],a_s[3], σ̇*ev.scale.uscale]
    end
    return SA[v_s[1],v_s[2],v_s[3],a_s[1],a_s[2],a_s[3], σ̇*ev.scale.uscale]
end

function rk4(params,x_n,θ;verbose = false)

    dt_s = params.ev.dt/params.ev.scale.tscale
    k1 = dt_s*dynamics(params,x_n,θ;verbose = verbose)
    k2 = dt_s*dynamics(params,x_n+k1/2,θ;verbose = verbose)
    k3 = dt_s*dynamics(params,x_n+k2/2,θ;verbose = verbose)
    k4 = dt_s*dynamics(params,x_n+k3,θ;verbose = verbose)

    return (x_n + (1/6)*(k1 + 2*k2 + 2*k3 + k4))
end

function v_from_xs(ev,x)
    r, v = cp.unscale_rv(ev.scale,x[SA[1,2,3]],x[SA[4,5,6]])
    norm(v)
end

function rollout_to_vt(params,x0_s,θ; verbose = false)
    x_old = copy(x0_s)
    x_new = copy(x0_s)
    for i = 1:1000
        x_old = x_new
        x_new = rk4(params,x_old,θ;verbose = verbose)

        v1 = v_from_xs(params.ev, x_old)
        v2 = v_from_xs(params.ev, x_new)
        if v2 < params.vt
            tt = (params.vt - v1) / (v2 - v1)
            xt = x_old*(1 - tt) + tt*x_new
            vt = v_from_xs(params.ev, xt)
            return (norm(params.proj_mat*(xt[SA[1,2,3]] - params.xg_s[SA[1,2,3]]))*params.ev.scale.dscale/1000)^2
        end
    end
    error("didn't hit vt something is fucked")
end

function MPPI_policy(params,x0_s, θ)
    cost(_θ) = rollout_to_vt(params,x0_s,_θ)

    if sqrt(cost(θ)) < 1.0
        grad(_θ) = FiniteDiff.finite_difference_gradient(__θ -> rollout_to_vt(params,x0_s,__θ), _θ)
        hess(_θ) = FiniteDiff.finite_difference_hessian(__θ -> rollout_to_vt(params,x0_s,__θ), _θ)

        H = hess(θ)
        g = grad(θ)
        Δθ = -(H + 1e-3*I)\g
        θ = linesearch(cost, θ, Δθ)

        σ̇ = legendre_policy(params,v_from_xs(params.ev,x0_s),θ)*params.ev.scale.uscale

        return σ̇, θ, sqrt(cost(θ))
    else

        # θ = zeros(5)
        # for ga_iter = 1:1
        batch_size = 50
        Js = zeros(batch_size)
        θs = [deepcopy(θ) for i = 1:batch_size]
        Js[1] = cost(θs[1])
        # @show sqrt(Js[1])
        for i = 2:batch_size
            θs[i] += 0.1*randn(length(θ))
            Js[i] = cost(θs[i])
        end

        θ = θs[argmin(Js)]
        # end

        σ̇ = legendre_policy(params,v_from_xs(params.ev,x0_s),θ)*params.ev.scale.uscale

        return σ̇, θ, sqrt(cost(θ))
    end
end
function linesearch(cost_function,x,Δx)
    J1 = cost_function(x)
    α = 1.0
    for i = 1:20
        x2 = x + α*Δx
        J2 = cost_function(x2)
        if J2 < J1
            return x2
        else
            α /= 2
        end
    end
    error("linesearch failed")
end
function MPPI_policy_hess(params,x0_s, θ)
    cost(_θ) = rollout_to_vt(params,x0_s,_θ)
    grad(_θ) = FiniteDiff.finite_difference_gradient(__θ -> rollout_to_vt(params,x0_s,__θ), _θ)
    hess(_θ) = FiniteDiff.finite_difference_hessian(__θ -> rollout_to_vt(params,x0_s,__θ), _θ)

    H = hess(θ)
    g = grad(θ)
    Δθ = -(H + 1e-3*I)\g
    θ = linesearch(cost, θ, Δθ)

    σ̇ = legendre_policy(params,v_from_xs(params.ev,x0_s),θ)*params.ev.scale.uscale

    return σ̇, θ, sqrt(cost(θ))
end
# let
#
#     # let's run some MPC
#     ev = cp.CPEGWorkspace()
#
#     # vehicle parameters
#     ev.params.aero.Cl = 0.29740410453983374
#     ev.params.aero.Cd = 1.5284942035954776
#     ev.params.aero.A = 15.904312808798327    # m²
#     ev.params.aero.m = 2400.0                # kg
#
#     # sim stuff
#     ev.dt = 2.0 # seconds
#
#     # CPEG settings
#     ev.scale.uscale = 1e1
#
#     # initial condition
#     x0_s = SA[3.4416834024182323, 0.27591843569122476, 2.6001953644857934e-5, -1.674771215138404, 5.601162692567315, 0.003962175342049756, 0.0]
#     xg_s = SA[3.3477567254291762, 0.626903908492849, 0.03739968529144168, -0.255884401134421, 0.33667198108223073, -0.056555916829042985, 0]
#
#     v0 = v_from_xs(ev,x0_s)
#     vf = 400
#
#     proj_vec = normalize(xg_s[SA[1,2,3]])
#     proj_mat = I - proj_vec*proj_vec'
#
#     params = (ev = ev, a = 2/(vf - v0), b = 1 - vf*(2/(vf - v0)), vt = 400,
#               dt_s = (ev.dt / ev.scale.tscale), proj_mat,xg_s = xg_s)
#
#
#     # θ = 0.1*(@SVector randn(15))
#     # θ = 0.0001*randn(5)
#     θ = zeros(5)
#
#
#
#     # @show alt_from_x(ev::cp.CPEGWorkspace, x0_s)/1e3
#     #
#     # @show rollout_to_vt(params,x0_s,θ)
#
#     cost(_θ) = rollout_to_vt(params,x0_s,_θ)
#     # grad(_θ) = FiniteDiff.finite_difference_gradient(__θ -> rollout_to_vt(params,x0_s,__θ), _θ)
#     batch_size = 50
#     θ = zeros(5)
#     for ga_iter = 1:50
#
#         Js = zeros(batch_size)
#         θs = [deepcopy(θ) for i = 1:batch_size]
#         Js[1] = cost(θs[1])
#         @show sqrt(Js[1])
#         for i = 2:batch_size
#             θs[i] += 0.3*randn(length(θ))
#             Js[i] = cost(θs[i])
#         end
#
#         θ = θs[argmin(Js)]
#     end
#
#
#
# end
function process_ev_run(ev,X,U)
    # t_vec = zeros(length(X))
    # for i = 2:length(t_vec)
    #     t_vec[i] = t_vec[i-1] + U[i-1][2]
    # end
    dt = [U[i][2] for i = 1:length(U)]
    t_vec = [0;cumsum(dt)]

    X = SVector{7}.(X)
    alt, dr, cr = cp.postprocess_scaled(ev,X,X[1])
    σ = [X[i][7] for i = 1:length(X)]

    r = [zeros(3) for i = 1:length(X)]
    v = [zeros(3) for i = 1:length(X)]
    for i = 1:length(X)
        r[i],v[i] = cp.unscale_rv(ev.scale,X[i][SA[1,2,3]],X[i][SA[4,5,6]])
    end

    return alt, dr, cr, σ, dt, t_vec, r, v
end

#----------------CELAN-------------------
mutable struct Params{Tf}
    ρ_spline::Spline1D
    wE_spline::Spline1D
    wN_spline::Spline1D
    ev::cp.CPEGWorkspace
    a::Tf
    b::Tf
    vt::Tf
    dt_s::Tf
    proj_mat::SMatrix{3,3,Tf,9}
    xg_s::SVector{7,Tf}
end
# Params(::Spline1D,
#        ::Spline1D,
#        ::Spline1D,
#        ::CPEG.CPEGWorkspace, ::Float64, ::Float64, ::Float64, ::Float64, ::SMatrix{3, 3, Float64, 9}, ::SVector{7, Float64})
let

    # let's run some MPC
    ev = cp.CPEGWorkspace()

    # vehicle parameters
    ev.params.aero.Cl = 0.29740410453983374
    ev.params.aero.Cd = 1.5284942035954776
    ev.params.aero.A = 15.904312808798327    # m²
    ev.params.aero.m = 2400.0                # kg

    # sim stuff
    ev.dt = 2.0 # seconds

    # CPEG settings
    ev.scale.uscale = 1e1

    # initial condition
    x0_s = SA[3.4416834024182323, 0.27591843569122476, 2.6001953644857934e-5, -1.674771215138404, 5.601162692567315, 0.003962175342049756, 0.0]
    xg_s = SA[3.3477567254291762, 0.626903908492849, 0.03739968529144168, -0.255884401134421, 0.33667198108223073, -0.056555916829042985, 0]

    v0 = v_from_xs(ev,x0_s)
    vt = 400.0

    proj_vec = normalize(xg_s[SA[1,2,3]])
    proj_mat = I - proj_vec*proj_vec'

    # get atmosphere stuff in params
    altitudes,densities, Ewind, Nwind = load_atmo()

    ρ_spline = Spline1D(reverse(altitudes), reverse(densities))
    wE_spline = Spline1D(reverse(altitudes), reverse(Ewind))
    wN_spline = Spline1D(reverse(altitudes), reverse(Nwind))

    params = Params(ρ_spline,
                    wE_spline,
                    wN_spline,
                    ev,
                    2/(vt - v0), # a
                    1 - vt*(2/(vt - v0)),
                    vt, # vt
                    (ev.dt / ev.scale.tscale), # dt_s
                     proj_mat,
                     xg_s)

    # main sim
    T = 3000
    sim_dt = 0.1
    Xsim = [zeros(7) for i = 1:T]
    Xsim[1] = x0_s
    Usim = [zeros(2) for i = 1:T-1]

    @info "starting sim"
    θ = zeros(5)
    for i = 1:T-1

        # Usim[i] = MPPI_policy(params,x0_s, θ)
        v0 = 1.000001*v_from_xs(params.ev, Xsim[i])
        vf = 0.999999*params.vt
        params.a = 2/(vf - v0) # a
        params.b = 1 - vf*(2/(vf - v0))

        # if i < 200
            σ̇, θ, J = MPPI_policy(params,SVector{7}(Xsim[i]), θ)
        # else
            # σ̇, θ, J = MPPI_policy_hess(params,SVector{7}(Xsim[i]), θ)
        # end
        Usim[i] = SA[σ̇;sim_dt]
        if rem(i-1,7)==0
            @printf "iter    σ (deg)   alt (km)    |v| (km/s)   miss (km)             \n"
            @printf "----------------------------------------------------------------------------\n"
        end
        alt = alt_from_x(params.ev, Xsim[i])/1000
        velo = v_from_xs(params.ev, Xsim[i])/1000
        # md = params.ev.scale.dscale*norm(params.X[end][1:3] - params.x_desired[1:3])/1e3
        d_left = params.ev.scale.dscale*norm(Xsim[i][1:3] - xg_s[1:3])/1e3
        @printf("%4d  %6.2f     %6.2f     %6.2f       %6.2f\n",
          i, rad2deg(Xsim[i][7]), alt, velo, J)


        Xsim[i+1] = rk4_fudge_mg(ev,SVector{7}(Xsim[i]),SA[Usim[i][1]],sim_dt/ev.scale.tscale, params)

        # check sim termination
        if v_from_xs(ev,Xsim[i+1]) < params.vt
            @info "SIM IS DONE"
            Xsim = Xsim[1:(i+1)]
            Usim = Usim[1:i]
            break
        end

    end

    # jldsave("controls_for_estimator.jld2"; U = Usim, X = Xsim)
    # Usim = [[Usim[i];dt] for i = 1:(length(Usim))]
    alt1, dr1, cr1, σ1, dt1, t_vec1, r1, v1 = process_ev_run(ev,Xsim,Usim)
    # # alt2, dr2, cr2, σ2, dt2, t_vec2, r2, v2 = process_ev_run(ev,X2,U2)
    # # alt3, dr3, cr3, σ3, dt3, t_vec3, r3, v3 = process_ev_run(ev,X3,U3)
    #
    # # get the goals
    # Xg = [3.34795153940262, 0.6269403895311674, 0.008024160056155994, -0.255884401134421, 0.33667198108223073, -0.056555916829042985, -1.182682624917629]
    alt_g, dr_g, cr_g = cp.postprocess_scaled(ev,[SVector{7}(xg_s)],SVector{7}(Xsim[1]))
    #
    mat"
    figure
    hold on
    plot($dr1/1000,$cr1/1000)
    plot($dr_g(1)/1000, $cr_g(1)/1000,'ro')
    xlabel('downrange (km)')
    ylabel('crossrange (km)')
    hold off
    "
    #
    mat"
    figure
    hold on
    plot($dr1/1000,$alt1/1000)
    plot($dr_g(1)/1000, $alt_g(1)/1000, 'ro')
    xlabel('downrange (km)')
    ylabel('altitude (km)')
    hold off
    "
    #
    v1s = [norm(v1[i]) for i = 1:length(v1)]
    # # v2s = [norm(v2[i]) for i = 1:N]
    # # v3s = [norm(v3[i]) for i = 1:N]
    #
    mat"
    figure
    hold on
    plot($t_vec1,$v1s)
    hold off
    "
    # @show Xsim[50]
    # @show Xsim[200]
    # #
    # # mat"
    # # figure
    # # hold on
    # # title('States')
    # # plot($t_vec,$X2m')
    # # legend('px','py','pz','vx','vy','vz','sigma')
    # # xlabel('Time (s)')
    # # hold off
    # # "
    # #
    σ̇ = [Usim[i][1] for i = 1:length(Usim)]
    mat"
    figure
    hold on
    title('Controls')
    plot($σ̇)
    legend('sigma dot','dt')
    xlabel('Time (s)')
    hold off
    "

    mat"
    figure
    hold on
    plot($t_vec1,rad2deg($σ1))
    title('Bank Angle')
    ylabel('Bank Angle (degrees)')
    xlabel('Time (s)')
    hold off
    "


end
