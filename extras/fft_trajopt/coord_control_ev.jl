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
Random.seed!(1)

# include(joinpath(@__DIR__,"simple_altro.jl"))
include(joinpath(@__DIR__,"simple_altro_coordinated_struct.jl"))

function dynamics_fudge(ev::cp.CPEGWorkspace, x::SVector{7,T}, u::SVector{1,W}, kρ::T2) where {T,W,T2}

    # scaled variables
    r_scaled = x[SA[1,2,3]]
    v_scaled = x[SA[4,5,6]]
    σ = x[7]

    # unscale
    r, v = cp.unscale_rv(ev.scale,r_scaled,v_scaled)

    # altitude
    h = cp.altitude(ev.params.gravity, r)

    # density
    ρ = kρ*cp.density(ev.params.density, h)

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
    x1 = x[p.idx_x[1]]
    u1 = u[p.idx_u[1]]
    x2 = x[p.idx_x[2]]
    u2 = u[p.idx_u[2]]
    x3 = x[p.idx_x[3]]
    u3 = u[p.idx_u[3]]

    [
    rk4_fudge(p.ev,SVector{7}(x1),SA[u1[1]],u1[2]/p.ev.scale.tscale, p.kρ_1);
    rk4_fudge(p.ev,SVector{7}(x2),SA[u2[1]],u2[2]/p.ev.scale.tscale, p.kρ_2);
    rk4_fudge(p.ev,SVector{7}(x3),SA[u3[1]],u3[2]/p.ev.scale.tscale, p.kρ_3);
    ]
end


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

    return alt, dr, cr, σ, dt, t_vec
end
function control_coord_con(params,u)
    [u[params.idx_u[1]][1] - u[params.idx_u[2]][1];
    u[params.idx_u[1]][1] - u[params.idx_u[3]][1]]
end
function term_con(params,x)
    x[params.term_idx] - params.Xref[params.N][params.term_idx]
end
function ineq_con_u(p::NamedTuple,u)
    [u-p.u_max;-u + p.u_min] #≦ 0
end
function ineq_con_x(p,x)
    [x-p.x_max;-x + p.x_min]
end

let
    nx = 7*3
    nu = 2*3
    N = 125
    Q = kron(I(3),Diagonal([0,0,0,0,0,0,1e-4]))
    Qf = kron(I(3),Diagonal([1,1,1,0,0,0,1e-4]))
    R = kron(I(3),Diagonal([1,100]))

    u_min = kron(ones(3), [-100, .5])
    u_max = kron(ones(3),  [100, 4])

    # state is x y v θ
    x_min = kron(ones(3), [-1e3*ones(6); -pi])
    x_max = kron(ones(3), [1e3*ones(6);   pi])


    dt = NaN
    x0 = kron(ones(3), [3.5212,0,0, -1.559452236319901, 5.633128235948198,0,0.05235987755982989])
    xg = kron(ones(3), [3.34795153940262, 0.6269403895311674, 0.008024160056155994, -0.255884401134421, 0.33667198108223073, -0.056555916829042985, -1.182682624917629])
    Xref = [copy(xg) for i = 1:N]
    Uref = [kron(ones(3), [0,2.0]) for i = 1:N-1]


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

    # altro settings
    solver_settings = Solver_Settings()
    solver_settings.max_iters            = 500
    solver_settings.cost_tol             = 1e-2
    solver_settings.d_tol                = 1e-2
    solver_settings.max_linesearch_iters = 10
    solver_settings.ρ0                   = 1e3
    solver_settings.ϕ                    = 10.0
    solver_settings.reg_min              = 1e-6
    solver_settings.reg_max              = 1e3
    solver_settings.convio_tol           = 1e-4

    params = (
        nx = nx,
        nu = nu,
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
        term_idx = [1,2,3,8,9,10,15,16,17],
        kρ_1 = 1.0,
        kρ_2 = 1.1,
        kρ_3 = 0.9,
        idx_x = [(i-1)*7 .+ (1:7) for i = 1:3],
        idx_u = [(i-1)*2 .+ (1:2) for i = 1:3],
        solver_settings = solver_settings
    );


    X = [deepcopy(x0) for i = 1:N]
    U = [kron(ones(3), [.0001*randn();1.8]) for i = 1:N-1]

    Xn = deepcopy(X)
    Un = deepcopy(U)


    P = [zeros(nx,nx) for i = 1:N]   # cost to go quadratic term
    p = [zeros(nx) for i = 1:N]      # cost to go linear term
    d = [zeros(nu) for i = 1:N-1]    # feedforward control
    K = [zeros(nu,nx) for i = 1:N-1] # feedback gain
    iLQR(params,X,U,P,p,K,d,Xn,Un;verbose = true)

    # X1m = hcat(Vector.(X)...)[params.idx_x[1],:]
    # U1m = hcat(Vector.(U)...)[params.idx_u[1],:]
    # X2m = hcat(Vector.(X)...)[params.idx_x[2],:]
    # U2m = hcat(Vector.(U)...)[params.idx_u[2],:]
    # X3m = hcat(Vector.(X)...)[params.idx_x[3],:]
    # U3m = hcat(Vector.(U)...)[params.idx_u[3],:]

    X1 = [X[i][params.idx_x[1]] for i = 1:length(X)]
    X2 = [X[i][params.idx_x[2]] for i = 1:length(X)]
    X3 = [X[i][params.idx_x[3]] for i = 1:length(X)]
    U1 = [U[i][params.idx_u[1]] for i = 1:length(U)]
    U2 = [U[i][params.idx_u[2]] for i = 1:length(U)]
    U3 = [U[i][params.idx_u[3]] for i = 1:length(U)]

    alt1, dr1, cr1, σ1, dt1, t_vec1 = process_ev_run(ev,X1,U1)
    alt2, dr2, cr2, σ2, dt2, t_vec2 = process_ev_run(ev,X2,U2)
    alt3, dr3, cr3, σ3, dt3, t_vec3 = process_ev_run(ev,X3,U3)

    mat"
    figure
    hold on
    plot($dr1/1000,$cr1/1000)
    plot($dr2/1000,$cr2/1000)
    plot($dr3/1000,$cr3/1000)
    xlabel('downrange (km)')
    ylabel('crossrange (km)')
    hold off
    "
    #
    mat"
    figure
    hold on
    plot($dr1/1000,$alt1/1000)
    plot($dr2/1000,$alt2/1000)
    plot($dr3/1000,$alt3/1000)
    xlabel('downrange (km)')
    ylabel('altitude (km)')
    hold off
    "
    #
    # mat"
    # figure
    # hold on
    # title('States')
    # plot($t_vec,$X2m')
    # legend('px','py','pz','vx','vy','vz','sigma')
    # xlabel('Time (s)')
    # hold off
    # "
    #
    # mat"
    # figure
    # hold on
    # title('Controls')
    # plot($t_vec(1:end-1), $Um')
    # legend('sigma dot','dt')
    # xlabel('Time (s)')
    # hold off
    # "
    #
    mat"
    figure
    hold on
    plot($t_vec1,rad2deg($σ1))
    plot($t_vec2,rad2deg($σ2))
    plot($t_vec3,rad2deg($σ3))
    title('Bank Angle')
    ylabel('Bank Angle (degrees)')
    xlabel('Time (s)')
    hold off
    "
    mat"
    figure
    hold on
    plot($dt1)
    plot($dt2)
    plot($dt3)
    title('dts')
    ylabel('dts')
    xlabel('knot point')
    hold off
    "

end
