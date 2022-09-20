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

# include(joinpath(@__DIR__,"simple_altro.jl"))
include(joinpath(@__DIR__,"simple_altro_coordinated.jl"))

function control_coord_con(params,u)
    u[1:6] - u[7:12]
end
function control_coord_con_jac(params)
    # [diagm(ones(6)) diagm(-ones(6))]
    FD.jacobian(_u -> control_coord_con(params,_u), ones(params.nu))
end
function skew(ω::Vector{T}) where {T}
    return [0 -ω[3] ω[2];
            ω[3] 0 -ω[1];
            -ω[2] ω[1] 0]
end
function rigid_body_dynamics(J,m,x,u)
    r = x[1:3]
    v = x[4:6]
    p = x[7:9]
    ω = x[10:12]

    f = u[1:3]
    τ = u[4:6]

    [
        v
        f/m
        ((1+norm(p)^2)/4) *(   I + 2*(skew(p)^2 + skew(p))/(1+norm(p)^2)   )*ω
        J\(τ - cross(ω,J*ω))
    ]
end
function dynamics(p::NamedTuple,x,u,k)
    [
    rigid_body_dynamics(p.J1,p.m1,x[1:12], u[1:6]);
    rigid_body_dynamics(p.J2,p.m2,x[13:24],u[7:12])
    ]
end
function discrete_dynamics(p::NamedTuple,x,u,k)
    k1 = p.dt*dynamics(p,x,        u, k)
    k2 = p.dt*dynamics(p,x + k1/2, u, k)
    k3 = p.dt*dynamics(p,x + k2/2, u, k)
    k4 = p.dt*dynamics(p,x + k3, u, k)
    x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
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
    nx = 24
    nu = 12
    N = 60
    dt = 0.1
    x0 = [5*randn(3);zeros(3);normalize(randn(3))*tand(120/4);zeros(3)]
    x0 = [x0;x0]
    xg = zeros(nx)
    Xref = [deepcopy(xg) for i = 1:N]
    Uref = [zeros(nu) for i = 1:N-1]
    Q = Diagonal(ones(nx))
    Qf = Diagonal(ones(nx))
    R = Diagonal([ones(3);10*ones(3);ones(3);10*ones(3)])


    u_min = -200*ones(nu)
    u_max =  200*ones(nu)

    # state is x y v θ
    x_min = -200*ones(nx)
    x_max =  200*ones(nx)


    ncx = 2*nx
    ncu = 2*nu


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
        N = N,
        term_idx = 1:nx,
        m1 = 1.0 ,
        m2 = 1.2,
        J1 = Diagonal(ones(3)),
        J2 = 1.2*Diagonal(ones(3))
    );


    X = [deepcopy(x0) for i = 1:N]
    U = [.0001*randn(nu) for i = 1:N-1]

    Xn = deepcopy(X)
    Un = deepcopy(U)


    P = [zeros(nx,nx) for i = 1:N]   # cost to go quadratic term
    p = [zeros(nx) for i = 1:N]      # cost to go linear term
    d = [zeros(nu) for i = 1:N-1]    # feedforward control
    K = [zeros(nu,nx) for i = 1:N-1] # feedback gain
    iLQR(params,X,U,P,p,K,d,Xn,Un;atol=1e-3,max_iters = 100,verbose = true,ρ = 1e0, ϕ = 10.0 )

    Xm = hcat(X...)
    Um = hcat(U...)
    mat"
    figure
    hold on
    plot($Xm(1:12,:)')
    hold off
    "

    mat"
    figure
    hold on
    plot($Xm(13:24,:)')
    hold off
    "

    mat"
    figure
    hold on
    plot($Um')
    hold off
    "

    @info norm(control_coord_con(params,U[1]))
end
