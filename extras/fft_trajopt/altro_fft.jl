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

include(joinpath(@__DIR__,"simple_altro.jl"))

function dynamics_fudge(ev::cp.CPEGWorkspace, x::SVector{7,T}, u::SVector{1,W}, kρ::T2) where {T,W,T2}

    # scaled variables
    r_scaled = x[SA[1,2,3]]
    v_scaled = x[SA[4,5,6]]
    σ = x[7]

    # unscale
    r, v = cp.unscale_rv(ev.scale,r_scaled,v_scaled)

    # altitude
    h,_,_ = cp.altitude(ev.params.gravity, r)
    # @show h
    # @show cp.density_spline(ev.params.dsp, h)
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

function discrete_dynamics(p,x1,u1,k)
    rk4_fudge(p.ev,SVector{7}(x1),SA[u1[1]],u1[2]/p.ev.scale.tscale, 1.0);
end

function ineq_con_u(p::NamedTuple,u)
    [u-p.u_max;-u + p.u_min] #≦ 0
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
function v_from_xs(ev,x)
    r, v = cp.unscale_rv(ev.scale,x[SA[1,2,3]],x[SA[4,5,6]])
    norm(v)
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
    # x0 = [3.5212,0,0, -1.559452236319901, 5.633128235948198,0,0]
    x0 = [3.4440871839763183, 0.2678691116554091, 3.2123176670031554e-5, -1.67259937570187, 5.602609333212407, 0.005001821126044161, 0.87]
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

    params = (
        nx = nx,
        nu = nu,
        ncx = ncx,
        ncu = ncu,
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
        ev = ev,
        N = N,
        term_idx = 1:3
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

    X2m = hcat(Vector.(X)...)
    Um = hcat(Vector.(U)...)

    t_vec = zeros(length(X))
    for i = 2:length(t_vec)
        t_vec[i] = t_vec[i-1] + U[i-1][2]
    end

    X = SVector{7}.(X)
    alt, dr, cr = cp.postprocess_scaled(ev,X,X[1])

    mat"
    figure
    hold on
    plot($dr/1000,$cr/1000)
    xlabel('downrange (km)')
    ylabel('crossrange (km)')
    hold off
    "

    mat"
    figure
    hold on
    plot($dr/1000,$alt/1000)
    xlabel('downrange (km)')
    ylabel('altitude (km)')
    hold off
    "

    mat"
    figure
    hold on
    title('States')
    plot($t_vec,$X2m')
    legend('px','py','pz','vx','vy','vz','sigma')
    xlabel('Time (s)')
    hold off
    "

    mat"
    figure
    hold on
    title('Controls')
    plot($t_vec(1:end-1), $Um')
    legend('sigma dot','dt')
    xlabel('Time (s)')
    hold off
    "

    mat"
    figure
    hold on
    plot($t_vec,rad2deg($X2m(7,:)))
    title('Bank Angle')
    ylabel('Bank Angle (degrees)')
    xlabel('Time (s)')
    hold off
    "

    # @show norm((X[end][1:3] - xg[1:3])*ev.scale.dscale / 1e3)
    function v_from_xs(ev,x)
        r, v = cp.unscale_rv(ev.scale,x[SA[1,2,3]],x[SA[4,5,6]])
        norm(v)
    end
    vs = [v_from_xs(ev,x) for x in X]
    σ̇ = [u[1] for u in U]

    @show vs
    @show σ̇
    @show X[25]
end
