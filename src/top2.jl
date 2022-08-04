# using Pkg
# Pkg.activate(dirname(@__FILE__))
# using LinearAlgebra
# import ForwardDiff as FD
# using Printf
# using StaticArrays
# # using FiniteDiff
# using MATLAB
cd("/Users/kevintracy/.julia/dev/CPEG")
Pkg.activate(".")
cd("/Users/kevintracy/.julia/dev/CPEG/src")
using LinearAlgebra
using StaticArrays
using ForwardDiff
import ForwardDiff as FD
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

function discrete_dynamics(p::NamedTuple,x,u,k)
    dt = u[2]
    rk4(p.ev,SVector{7}(x),SA[u[1]],dt/p.ev.scale.tscale)
end
function stage_cost(p::NamedTuple,x,u,k)
    dx = x - p.Xref[k]
    du = u - p.Uref[k]
    return 0.5*dx'*p.Q*dx + 0.5*du'*p.R*du
end
function term_cost(p::NamedTuple,x)
    dx = x - p.Xref[p.N]
    return 0.5*dx'*p.Qf*dx
end
function stage_cost_expansion(p::NamedTuple,x,u,k)
    dx = x - p.Xref[k]
    du = u - p.Uref[k]
    return p.Q, p.Q*dx, p.R, p.R*du
end
function term_cost_expansion(p::NamedTuple,x)
    dx = x - p.Xref[p.N]
    return p.Qf, p.Qf*dx
end
function backward_pass!(params,X,U,P,p,d,K,reg,μ,μx,ρ,λ)

    N = params.N
    ΔJ = 0.0

    # terminal cost expansion
    P[N], p[N] = term_cost_expansion(params,X[N])
    # @show p[N]

    # add AL for x cons
    hxv = ineq_con_x(params,X[N])
    mask = eval_mask(μx[N],hxv)
    # @show mask
    ∇hx = ineq_con_x_jac(params,X[N])

    p[N]  += ∇hx'*(μx[N] + ρ*(mask * hxv))
    # @show p[N]
    # @show ∇hx'*(μx[N] + ρ*(mask * hxv))
    # @show ∇hx
    # @show μx[N]
    # @show ρ
    # @show hxv
    # @show diag(mask)
    # @show mask * hxv
    P[N]  += ρ*∇hx'*mask*∇hx

    # add goal constraint
    hxv = X[N][1:3] - params.Xref[N][1:3]
    ∇hx = [I(3) zeros(3,4)]

    p[N]  += ∇hx'*(λ + ρ*hxv)
    P[N]  += ρ*∇hx'∇hx

    # @show p[N]
    # @show P[N]
    # error()

    for k = (N-1):(-1):1

        # dynamics jacobians
        A = FD.jacobian(_x -> discrete_dynamics(params,_x,U[k],k),X[k])
        B = FD.jacobian(_u -> discrete_dynamics(params,X[k],_u,k),U[k])

        # cost expansion
        Jxx,Jx,Juu,Ju = stage_cost_expansion(params,X[k],U[k],k)

        # control constraints
        huv = ineq_con_u(params,U[k])
        mask = eval_mask(μ[k],huv)
        ∇hu = ineq_con_u_jac(params,U[k])
        Ju  += ∇hu'*(μ[k] + ρ*(mask * huv))
        Juu += ρ*∇hu'*mask*∇hu

        # state constraints
        hxv = ineq_con_x(params,X[k])
        mask = eval_mask(μx[k],hxv)
        ∇hx = ineq_con_x_jac(params,X[k])
        Jx  += ∇hx'*(μx[k] + ρ*(mask * hxv))
        Jxx += ρ*∇hx'*mask*∇hx

        # Q expansion
        gx = Jx + A'*p[k+1]
        gu = Ju + B'*p[k+1]

        Gxx = Jxx + A'*P[k+1]*A
        Guu = Juu + B'*P[k+1]*B
        Gux = B'*P[k+1]*A

        # Calculate Gains
        F = cholesky(Symmetric(Guu + reg*I))
        d[k] = F\gu
        K[k] = F\Gux

        # Cost-to-go Recurrence
        p[k] = gx - K[k]'*gu + K[k]'*Guu*d[k] - Gux'*d[k]
        P[k] = Gxx + K[k]'*Guu*K[k] - Gux'*K[k] - K[k]'*Gux
        ΔJ += gu'*d[k]
    end

    # @show [any(isnan.(x)) for x in p]
    # error()

    return ΔJ
end
function trajectory_AL_cost(params,X,U,μ,μx,ρ,λ)
    N = params.N
    J = 0.0
    for k = 1:N-1
        J += stage_cost(params,X[k],U[k],k)

        # AL terms
        huv = ineq_con_u(params,U[k])
        mask = eval_mask(μ[k],huv)
        J += dot(μ[k],huv) + 0.5*ρ*huv'*mask*huv

        hxv = ineq_con_x(params,X[k])
        mask = eval_mask(μx[k],hxv)
        J += dot(μx[k],hxv) + 0.5*ρ*hxv'*mask*hxv
    end
    J += term_cost(params,X[N])
    hxv = ineq_con_x(params,X[params.N])
    mask = eval_mask(μx[params.N],hxv)
    J += dot(μx[params.N],hxv) + 0.5*ρ*hxv'*mask*hxv

    # goal constraint
    hxv = X[N][1:3] - params.Xref[N][1:3]
    J += dot(λ,hxv) + 0.5*ρ*hxv'*hxv
    return J
end
function forward_pass!(params,X,U,K,d,ΔJ,Xn,Un,μ,μx,ρ,λ; max_linesearch_iters = 20)

    N = params.N
    α = 1.0

    J = trajectory_AL_cost(params,X,U,μ,μx,ρ,λ)
    for i = 1:max_linesearch_iters

        # Forward Rollout
        for k = 1:(N-1)
            Un[k] = U[k] - α*d[k] - K[k]*(Xn[k]-X[k])
            Xn[k+1] = discrete_dynamics(params,Xn[k],Un[k],k)
        end
        Jn = trajectory_AL_cost(params,Xn,Un,μ,μx,ρ,λ)

        # armijo line search
        if Jn < J
            X .= Xn
            U .= Un
            return Jn, α
        else
            α *= 0.5
        end
    end

    @warn "forward pass failed, adding regularization"
    α = 0.0
    return J, α
end
function update_reg(reg,reg_min,reg_max,α)
    if α == 0.0
        if reg == reg_max
            error("reached max reg")
        end
        return min(reg_max,reg*10)
    end
    if α == 1.0
        return max(reg_min,reg/10)
    end
    return reg
end
function calc_max_d(d)
    dm = 0.0
    for i = 1:length(d)
        dm = max(dm,norm(d[i]))
    end
    return dm
end
function ineq_con_u(p,u)
    [u-p.u_max;-u + p.u_min]
end
function ineq_con_u_jac(params,u)
    nu = params.nu
    Array(float([I(nu);-I(nu)]))
end
function ineq_con_x(p,x)
    [x-p.x_max;-x + p.x_min]
end
function ineq_con_x_jac(params,x)
    nx = params.nx
    Array(float([I(nx);-I(nx)]))
end
function eval_mask(μv,huv)
    # active set mask
    mask = Diagonal(zeros(length(huv)))
    for i = 1:length(huv)
        mask[i,i] = μv[i] > 0 || huv[i] > 0
    end
    mask
end

function iLQR(params,X,U,P,p,K,d,Xn,Un;atol=1e-5,max_iters = 25,verbose = true,ρ=1,ϕ=10)

    # # inital logging stuff
    # if verbose
    #     @printf "iter     J           ΔJ        |d|         α        reg         ρ\n"
    #     @printf "---------------------------------------------------------------------\n"
    # end

    # initial rollout
    N = params.N
    for i = 1:N-1
        X[i+1] = discrete_dynamics(params,X[i],U[i],i)
    end

    # @show [any(isnan.(x)) for x in X]
    # error()

    reg = 1e-6
    reg_min = 1e-6
    reg_max = 1e-1

    μ = [zeros(params.ncu) for i = 1:N-1]

    μx = [zeros(params.ncx) for i = 1:N]

    λ = zeros(3)

    for iter = 1:max_iters
        ΔJ = backward_pass!(params,X,U,P,p,d,K,reg,μ,μx,ρ,λ)
        J, α = forward_pass!(params,X,U,K,d,ΔJ,Xn,Un,μ,μx,ρ,λ)
        reg = update_reg(reg,reg_min,reg_max,α)
        dmax = calc_max_d(d)
        if verbose
            if rem(iter-1,10)==0
                @printf "iter     J           ΔJ        |d|         α        reg         ρ\n"
                @printf "---------------------------------------------------------------------\n"
            end
            @printf("%3d   %10.3e  %9.2e  %9.2e  %6.4f   %9.2e   %9.2e\n",
              iter, J, ΔJ, dmax, α, reg,ρ)
        end
        if (α > 0) & (dmax<atol)
            # check convio
            convio = 0

            # control constraints
            for k = 1:N-1
                huv = ineq_con_u(params,U[k])
                mask = eval_mask(μ[k],huv)

                # update dual
                μ[k] = max.(0,μ[k] + ρ*mask*huv)
                convio = max(convio,norm(huv + abs.(huv),Inf))
            end

            # state constraints
            for k = 1:N
                hxv = ineq_con_x(params,X[k])
                mask = eval_mask(μx[k],hxv)

                # update dual
                μx[k] = max.(0,μx[k] + ρ*mask*hxv)
                convio = max(convio,norm(hxv + abs.(hxv),Inf))
            end

            # goal constraint
            hxv = X[N][1:3] - params.Xref[N][1:3]
            λ .+= ρ*hxv
            convio = max(convio, norm(hxv,Inf))

            @show convio
            if convio <1e-4
                @info "success!"
                return nothing
            end

            ρ *= ϕ
            # ρ = min(1e6,ρ*ϕ)
        end
    end
    error("iLQR failed")
end
let
    nx = 7
    nu = 2
    N = 125
    Q = Diagonal([0,0,0,0,0,0,1e-4])
    Qf = Diagonal([1,1,1,0,0,0,1e-4])
    R = Diagonal([1,100])

    u_min = [-100, .5]
    u_max =  [100, 4]

    # state is x y v θ
    x_min = [-1e3*ones(6); -pi]
    x_max = [1e3*ones(6);   pi]


    dt = NaN
    x0 = [3.5212,0,0, -1.559452236319901, 5.633128235948198,0,0.05235987755982989]
    xg = [3.34795153940262, 0.6269403895311674, 0.008024160056155994, -0.255884401134421, 0.33667198108223073, -0.056555916829042985, -1.182682624917629]
    Xref = [copy(xg) for i = 1:N]
    Uref = [[0,2.0] for i = 1:N-1]


    ncx = 2*nx
    ncu = 2*nu

    ev = CPEGWorkspace()

    # vehicle parameters
    ev.params.aero.Cl = 0.29740410453983374
    ev.params.aero.Cd = 1.5284942035954776
    ev.params.aero.A = 15.904312808798327    # m²
    ev.params.aero.m = 2400.0                # kg

    # sim stuff
    ev.dt = 2.0 # seconds

    # CPEG settings
    ev.scale.uscale = 1e1

    params = (
        nx = nx,
        nu = nu,
        ncx = ncx,
        ncu = ncu,
        N = N,
        Q = Q,
        R = R,
        Qf = Qf,
        u_min = u_min,
        u_max = u_max,
        x_min = x_min,
        x_max = x_max,
        Xref = Xref,
        Uref = Uref,
        dt = dt,
        ev = ev
    );


    X = [deepcopy(x0) for i = 1:N]
    U = [[.0001*randn();1.8] for i = 1:N-1]

    Xn = deepcopy(X)
    Un = deepcopy(U)


    P = [zeros(nx,nx) for i = 1:N]   # cost to go quadratic term
    p = [zeros(nx) for i = 1:N]      # cost to go linear term
    d = [zeros(nu) for i = 1:N-1]    # feedforward control
    K = [zeros(nu,nx) for i = 1:N-1] # feedback gain
    iLQR(params,X,U,P,p,K,d,Xn,Un;atol=1e-2,max_iters = 3000,verbose = true,ρ = 1e3, ϕ = 10.0 )

    # # run it again MPC style
    # X = [deepcopy(x0) for i = 1:N]
    # X[1] += [1e3*randn(3)/ev.scale.dscale; zeros(4)]
    #
    # Xn = deepcopy(X)
    # Un = deepcopy(U)
    #
    #
    # P = [zeros(nx,nx) for i = 1:N]   # cost to go quadratic term
    # p = [zeros(nx) for i = 1:N]      # cost to go linear term
    # d = [zeros(nu) for i = 1:N-1]    # feedforward control
    # K = [zeros(nu,nx) for i = 1:N-1] # feedback gain
    # iLQR(params,X,U,P,p,K,d,Xn,Un;atol=1e-2,max_iters = 3000,verbose = true,ρ = 1e7, ϕ = 10.0 )

    X2m = hcat(Vector.(X)...)
    Um = hcat(Vector.(U)...)

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

    mat"
    figure
    hold on
    plot(rad2deg($X2m(7,:)))
    ylabel('Bank Angle (degrees)')
    hold off
    "
end
