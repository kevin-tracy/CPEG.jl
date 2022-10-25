# using Pkg
# Pkg.activate(dirname(@__FILE__))
# using LinearAlgebra
# import ForwardDiff as FD
# using Printf
# using StaticArrays
# # using FiniteDiff
# using MATLAB
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

using LinearAlgebra
using StaticArrays
using ForwardDiff
using SparseArrays
using SuiteSparse
using MATLAB

function mpc_quad1(params::NamedTuple, A,B,X,U; verbose = true, atol = 1e-6, constrain = false)

    # sizes for state and control
    nx = 7
    nu = 2
    N = length(X)

    # indicees for state and control
    idx_x = [(i-1)*(nx+nu) .+ (1:nx) for i = 1:length(X)]
    idx_u = [(i-1)*(nx+nu) .+ nx .+ (1:nu) for i = 1:(length(X)-1)]

    # constraint jacobian
    nz = (N*nx) + (N-1)*nu # number of variables
    nc = (N-1)*nx + nx # dynamics + IC
    A_eq = spzeros(nc,nz)
    b_eq = zeros((N)*nx) # N-1 dynamics constraints + N IC constraint

    # dynamics constraint (equality)
    idx_c = [(i-1)*(nx) .+ (1:nx) for i = 1:(N-1)]
    for i = 1:(N-1)
        A_eq[idx_c[i],idx_x[i]]   = A[i]
        A_eq[idx_c[i],idx_u[i]]   = B[i]
        A_eq[idx_c[i],idx_x[i+1]] = -I(nx)
    end
    A_eq[(N-1)*nx .+ (1:nx), idx_x[1]] = I(nx)

    # inequality constraints
    G = spzeros(2*(2*N - 1),nz)
    h = spzeros(2*(2*N - 1),nz)

    # put state and control bounds based on trust region or limits
    # (whichever comes first)
    G=sparse([I(nz);-I(nz)])
    z_max = zeros(nz)
    z_min = zeros(nz)
    for i = 1:N
        z_max[idx_x[i]] = min.(params.δx_max, params.x_max - X[i])
        z_min[idx_x[i]] = max.(params.δx_min, params.x_min - X[i])
    end
    for i = 1:N-1
        z_max[idx_u[i]] = min.(params.δu_max, params.u_max - U[i])
        z_min[idx_u[i]] = max.(params.δu_min, params.u_min - U[i])
    end
    h = [z_max;-z_min]

    # cost function terms
    P = spzeros(nz,nz)
    q = zeros(nz)
    R = 0.01
    for i = 1:(N-1)
        P[idx_u[i],idx_u[i]] = params.R
        q[idx_u[i]] = params.R*(U[i] - params.Uref[i])
        P[idx_x[i],idx_x[i]] = params.Q
        q[idx_x[i]] = params.Q*(X[i] - params.Xref[i])
    end
    P[idx_x[N],idx_x[N]] = params.Qf
    q[idx_x[N]] = params.Qf*(X[N] - params.Xref[N])

    # TODO: maybe regularize

    # solve the equality only constrained QP
    sol = lu([P A_eq';A_eq spzeros(N*nx,N*nx)])\[-q;b_eq]
    z = sol[1:length(q)]
    qp_iters = 1

    # if this violates the inequality constraints, then we send it to quadprog
    # if sum(G*z .> h) != 0
    if !all(G*z .<= h)
        z, qp_iters = cp.quadprog(P,q,A_eq,b_eq,G,h; verbose = verbose,atol = atol,max_iters = 30)
    end

    # pull out δx and δu from z
    δx = [z[idx_x[i]] for i = 1:(N)]
    δu = [z[idx_u[i]] for i = 1:(N-1)]


    return δu, qp_iters
end

function dynamics_fudge(ev::cp.CPEGWorkspace, x::SVector{7,T}, u::SVector{1,W}, kρ::T2) where {T,W,T2}

    # scaled variables
    r_scaled = x[SA[1,2,3]]
    v_scaled = x[SA[4,5,6]]
    σ = x[7]

    # unscale
    r, v = cp.unscale_rv(ev.scale,r_scaled,v_scaled)

    # altitude
    h,_,_ = cp.altitude(ev.params.gravity, r)
    # @show typeof(r)
    # @show typeof(h)

    # density
    # ρ = kρ*cp.density(ev.params.density, h)
    # if typeof(ρ) != typeof(h)
    #     @show typeof(ρ)
    #     @show typeof(h)
    #     @info "error 1"
    #     error()
    # end
    ρ = kρ*cp.density_spline(ev.params.dsp, h)
    # ρ = densisty_solin
    # if typeof(ρ) != typeof(ρ2)
    #     @show typeof(h)
    #     @show typeof(ρ)
    #     @show typeof(ρ2)
    #     @info "error 2"
    #     error()
    # end
    # @show typeof(ρ)
    # @show eltype(r)
    # @show eltype(v)

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

    return SA[v[1],v[2],v[3],a[1],a[2],a[3],u[1]*ev.scale.uscale]
end

function rk4_fudge(
    ev::cp.CPEGWorkspace,
    x_n::SVector{7,T},
    u::SVector{1,W},
    dt_s::T2, kρ::T3) where {T,W,T2,T3}

    k1 = dt_s*dynamics_fudge(ev,x_n,u,kρ)
    k2 = dt_s*dynamics_fudge(ev,x_n+k1/2,u,kρ)
    k3 = dt_s*dynamics_fudge(ev,x_n+k2/2,u,kρ)
    k4 = dt_s*dynamics_fudge(ev,x_n+k3,u,kρ)

    return (x_n + (1/6)*(k1 + 2*k2 + 2*k3 + k4))
end

function discrete_dynamics(p::NamedTuple,x,u,k)
    # x1 = x[p.idx_x[1]]
    # u1 = u[p.idx_u[1]]
    # x2 = x[p.idx_x[2]]
    # u2 = u[p.idx_u[2]]
    # x3 = x[p.idx_x[3]]
    # u3 = u[p.idx_u[3]]

    # [
    rk4_fudge(p.ev,SVector{7}(x),SA[u[1]],u[2]/p.ev.scale.tscale, 1.0);
    # rk4_fudge(p.ev,SVector{7}(x2),SA[u2[1]],u2[2]/p.ev.scale.tscale, p.kρ_2);
    # rk4_fudge(p.ev,SVector{7}(x3),SA[u3[1]],u3[2]/p.ev.scale.tscale, p.kρ_3);
    # ]
end

# function discrete_dynamics(p::NamedTuple,x,u,k)
#     dt = u[2]
#     cp.rk4(p.ev,SVector{7}(x),SA[u[1]],dt/p.ev.scale.tscale)
# end

function jacobs(params::NamedTuple,X,U)
    N = length(X)
    # A = [ForwardDiff.jacobian(_x->rk4(p.ev,SVector{7}(_x),  SA[U[i][1]],U[i][2]/p.ev.scale.tscale), X[i]) for i = 1:N-1]
    # B = [ForwardDiff.jacobian(_u->rk4(p.ev,SVector{7}(X[i]),SA[_u[1]],    _u[2]/p.ev.scale.tscale), U[i]) for i = 1:N-1]
    # A,B
    A = [FD.jacobian(_x -> discrete_dynamics(params,_x,U[k],k),X[k]) for k = 1:N-1]
    B = [FD.jacobian(_u -> discrete_dynamics(params,X[k],_u,k),U[k]) for k = 1:N-1]
    A,B
end
let
    nx = 7
    nu = 2
    N = 125
    Q = Diagonal([0,0,0,0,0,0,1e-4])
    Qf = 1e8*Diagonal([1,1,1,0,0,0,1e-4])
    R = Diagonal([1,100])


    dt = NaN
    x0 = [3.5212,0,0, -1.559452236319901, 5.633128235948198,0,0.05235987755982989]
    xg = [3.34795153940262, 0.6269403895311674, 0.008024160056155994, -0.255884401134421, 0.33667198108223073, -0.056555916829042985, -1.182682624917629]
    Xref = [copy(xg) for i = 1:N]
    Uref = [[0,2.0] for i = 1:N-1]


    ncx = 2*nx
    ncu = 2*nu

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

    σ_max =  pi
    σ_min = -σ_max
    δσ_max =  deg2rad(20)
    δσ_min =  -δσ_max

    dt_max = 3.0
    dt_min = 1.0
    δdt_max =  0.2
    δdt_min = -δdt_max

    u_min = [-100, .5]
    u_max =  [100, 4]
    δu_min = [-100,-.2]
    δu_max = [100, 0.2]

    x_min = [-1e3*ones(6); -pi]
    x_max = [1e3*ones(6);   pi]
    δx_min = [-1e3*ones(6); -deg2rad(20)]
    δx_max = [1e3*ones(6);   deg2rad(20)]


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
        δu_min = δu_min,
        δu_max = δu_max,
        δx_min = δx_min,
        δx_max = δx_max,
        Xref = Xref,
        Uref = Uref,
        dt = dt,
        ev = ev
    );


    X = [deepcopy(x0) for i = 1:N]
    U = [[.0001*randn();1.8] for i = 1:N-1]

    constrain = false


    for cpeg_iter = 1:20

        # rollout
        for i = 1:N-1
            X[i+1] = discrete_dynamics(params,X[i],U[i],i)
        end

        # linearize
        A,B = jacobs(params,X,U)

        # solve MPC
        δu, qp_iters = mpc_quad1(params, A,B,X,U; verbose = false, atol = 1e-10, constrain = constrain)
        md = norm(X[N][1:3] - xg[1:3])*ev.scale.dscale/1000
        # if md < 5
        #     @info "constrained"
        #     constrain = true
        #     params.Qf .= 1*params.Q
        # end

        @show cpeg_iter, norm(δu), md, qp_iters

        # update
        U += δu

    end


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
