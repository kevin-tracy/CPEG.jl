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
        y = y^.3 # TODO
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



function process_ev_run(ev,X,U)

    X = SVector{7}.(X)
    alt, dr, cr = cp.postprocess_scaled(ev,X,X[1])
    σ = [X[i][7] for i = 1:length(X)]

    r = [zeros(3) for i = 1:length(X)]
    v = [zeros(3) for i = 1:length(X)]
    for i = 1:length(X)
        r[i],v[i] = cp.unscale_rv(ev.scale,X[i][SA[1,2,3]],X[i][SA[4,5,6]])
    end

    return alt, dr, cr, σ, r, v
end

function v_from_xs(ev,x)
    r, v = cp.unscale_rv(ev.scale,x[SA[1,2,3]],x[SA[4,5,6]])
    norm(v)
end

function rollout_to_vt(params,x0_s,θ; verbose = false)
    x_old = copy(x0_s)
    x_new = copy(x0_s)
    for i = 1:1000
        # update the old one
        x_old = x_new
        x_new = rk4(params,x_old,θ;verbose = verbose)

        v1 = v_from_xs(params.ev, x_old)
        v2 = v_from_xs(params.ev, x_new)
        if v2 < params.vt
            # @info "reached 450"
            # @show v1
            # @show v2
            tt = (params.vt - v1) / (v2 - v1)
            xt = x_old*(1 - tt) + tt*x_new
            vt = v_from_xs(params.ev, xt)
            # @show vt
            # error()
            return (norm(params.proj_mat*(xt[SA[1,2,3]] - params.xg_s[SA[1,2,3]]))*params.ev.scale.dscale/1000)^2
        end
    end
    error("didn't hit 450 wtf")
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
    ev.dt = 2.0 # seconds

    # CPEG settings
    ev.scale.uscale = 1e1

    # initial condition
    x0_s = SA[3.4416834024182323, 0.27591843569122476, 2.6001953644857934e-5, -1.674771215138404, 5.601162692567315, 0.003962175342049756, 0.0]
    xg_s = SA[3.3477567254291762, 0.626903908492849, 0.03739968529144168, -0.255884401134421, 0.33667198108223073, -0.056555916829042985, 0]

    v0 = 1.0*v_from_xs(ev,x0_s)
    # vf = 1.0*v_from_xs(ev,xg_s)
    vf = 390

    proj_vec = normalize(xg_s[SA[1,2,3]])
    proj_mat = I - proj_vec*proj_vec'

    params = (ev = ev, a = 2/(vf - v0), b = 1 - vf*(2/(vf - v0)), vt = 390,
              dt_s = (ev.dt / ev.scale.tscale), proj_mat,xg_s = xg_s)

              v0 = 1.0*v_from_xs(ev,x0_s)
              vf = 1.0*v_from_xs(ev,xg_s)
    # θ = 0.1*(@SVector randn(15))
    # θ = 0.0001*randn(5)
    θ = zeros(5)



    # @show alt_from_x(ev::cp.CPEGWorkspace, x0_s)/1e3
    #
    # @show rollout_to_vt(params,x0_s,θ)

    # momentum
    γ = 0.3
    η = 1e-4
    v = η*FiniteDiff.finite_difference_gradient(_θ -> rollout_to_vt(params,x0_s,_θ), θ)
    for gd_iter = 1:10
        cost = rollout_to_vt(params,x0_s,θ)
        @show sqrt(cost)
        if sqrt(cost)<.01
            @info "success"
            break
        end
        v = γ*v + η*FiniteDiff.finite_difference_gradient(_θ -> rollout_to_vt(params,x0_s,_θ), θ)
        θ -= v
    end

    # # grad + newton
    # η = 1e-4
    # for gd_iter = 1:200
    #     cost = rollout_to_vt(params,x0_s,θ)
    #     @show sqrt(cost)
    #     if sqrt(cost)<.01
    #         @info "success"
    #         break
    #     end
    #     grad = FiniteDiff.finite_difference_gradient(_θ -> rollout_to_vt(params,x0_s,_θ), θ)
    #     if sqrt(cost)>10
    #         θ -= η*grad
    #     else
    #         @info "newton"
    #         H= FiniteDiff.finite_difference_hessian(_θ -> rollout_to_vt(params,x0_s,_θ), θ)
    #         Δθ = (H + 1e-4*I)\grad
    #         θ = θ - Δθ
    #         if norm(Δθ) < 1e-3
    #             @info "success"
    #             break
    #         end
    #     end
    # end

    # BFGS
    # θ = 0.0001*randn(5)
    n = length(θ)
    # B = 1*diagm(ones(n))
    B = FiniteDiff.finite_difference_hessian(_θ -> rollout_to_vt(params,x0_s,_θ), θ)
    grad_old = FiniteDiff.finite_difference_gradient(_θ -> rollout_to_vt(params,x0_s,_θ), θ)
    cost_old = rollout_to_vt(params,x0_s,θ)

    # θ
    α = 1.0
    for bfgs_iter = 1:10

        grad = FiniteDiff.finite_difference_gradient(_θ -> rollout_to_vt(params,x0_s,_θ), θ)
        s = -(B+ 1e-3*I)\grad


        # line search
        α = 1.0
        J = rollout_to_vt(params,x0_s,θ)
        for i = 1:100
            θ_new = θ + α*s
            # @show θ
            # @show α
            # @show s
            # @show θ_new
            # try
                J2 = rollout_to_vt(params,x0_s,θ_new; verbose = false)
            # catch
                # J2 = 1e10
            # end
            if J2<J
                # @info "good step"
                θ = 1*θ_new
                # s = α*s
                break
            else
                α /= 2
            end
        end

        grad_new = FiniteDiff.finite_difference_gradient(_θ -> rollout_to_vt(params,x0_s,_θ), θ)
        y = grad_new - grad

        # @info "before"
        # @show B
        # @show s
        # @show y
        # @show dot(y,s)
        B = B - (B*s)*(s'*B)/(s'*B*s) + (y*y')/dot(y,s)
        # @show B
        # @info "after"

        @show bfgs_iter, norm(grad_new), α, sqrt(J)
        if sqrt(J)<1e-1
            @info "bfgs success"
            break
        end
    end


    vs = Vector(range(v0,vf,length = 100))
    σ̇ = [legendre_policy(params,vs[i],θ) for i = 1:length(vs)]

    mat"
    figure
    hold on
    plot($vs, $σ̇)
    xlabel('velocity')
    ylabel('sigma dot')
    hold off
    "



    x_old = copy(x0_s)
    x_new = copy(x0_s)
    vs = zeros(1000)
    σ = zeros(1000)
    # σ[1] = x0_s[7]
    for i = 1:1000
        # update the old one
        x_old = x_new
        x_new = rk4(params,x_old,θ)
        σ[i] = x_new[7]
        v1 = v_from_xs(params.ev, x_old)
        v2 = v_from_xs(params.ev, x_new)
        vs[i] = v2
        if v2 < params.vt
            σ = σ[1:i]
            vs = vs[1:i]
            break
        end
    end

    # mat"
    # figure
    # hold on
    # xlabel('velocities')
    # ylabel('bank angle')
    # plot($vs, rad2deg($σ))
    # hold off
    # "
    # mat"
    # figure
    # hold on
    # xlabel('time step')
    # ylabel('bank angle')
    # plot(rad2deg($σ))
    # hold off
    # "

    # y = params.σ̇_spline(vs)
    σ̇ = [legendre_policy(params,vs[i],θ) for i = 1:length(vs)]

    mat"
    figure
    hold on
    plot($σ̇(1:end-1))
    xlabel('step')
    ylabel('sigma dot')
    hold off
    "

    mat"
    figure
    hold on
    plot(rad2deg($σ))
    xlabel('step')
    ylabel('sigma (degrees)')
    hold off
    "


end
