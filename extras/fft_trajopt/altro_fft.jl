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

function discrete_dynamics(p::NamedTuple,x,u,k)
    dt = u[2]
    cp.rk4(p.ev,SVector{7}(x),SA[u[1]],dt/p.ev.scale.tscale)
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
end
