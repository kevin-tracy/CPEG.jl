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

Random.seed!(1)

include(joinpath(@__DIR__,"simple_altro_struct.jl"))

function load_atmo(;path="/Users/kevintracy/.julia/dev/CPEG/extras/fft_trajopt/atmo_samples/samp1.csv")
    TT = readdlm(path, ',')
    alt = Vector{Float64}(TT[2:end,2])
    density = Vector{Float64}(TT[2:end,end])
    Ewind = Vector{Float64}(TT[2:end,7])
    Nwind = Vector{Float64}(TT[2:end,9])
    return alt*1000, density, Ewind, Nwind
end

function alt_from_x(ev::cp.CPEGWorkspace, x)
    r_scaled = x[SA[1,2,3]]
    v_scaled = SA[2,3,4.0]
    r, v = cp.unscale_rv(ev.scale,r_scaled,v_scaled)
    h,_,_ = cp.altitude(ev.params.gravity, r)
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
    # @show h
    if (h/1e3) > 125
        @show h
        @show x
        @show u
    end
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
    x1 = x
    u1 = u
    # x2 = x[p.idx_x[2]]
    # u2 = u[p.idx_u[2]]
    # x3 = x[p.idx_x[3]]
    # u3 = u[p.idx_u[3]]

    # [
    rk4_fudge(p.ev,SVector{7}(x1),SA[u1[1]],u1[2]/p.ev.scale.tscale, p.kρ_1);
    # rk4_fudge(p.ev,SVector{7}(x2),SA[u2[1]],u2[2]/p.ev.scale.tscale, p.kρ_2);
    # rk4_fudge(p.ev,SVector{7}(x3),SA[u3[1]],u3[2]/p.ev.scale.tscale, p.kρ_3);
    # ]
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

    r = [zeros(3) for i = 1:length(X)]
    v = [zeros(3) for i = 1:length(X)]
    for i = 1:length(X)
        r[i],v[i] = cp.unscale_rv(ev.scale,X[i][SA[1,2,3]],X[i][SA[4,5,6]])
    end

    return alt, dr, cr, σ, dt, t_vec, r, v
end
# function control_coord_con(params,u)
#     [u[params.idx_u[1]][1] - u[params.idx_u[2]][1];
#     u[params.idx_u[1]][1] - u[params.idx_u[3]][1]]
# end
function term_con(params,x)
    r,v = cp.unscale_rv(params.ev.scale,x[SA[1,2,3]],x[SA[4,5,6]])
    return SA[norm(v) - 450.0]
    # x[params.term_idx] - params.Xref[params.N][params.term_idx]

    # [
    # x[params.term_idx] - params.Xref[params.N][params.term_idx];
    # -(x[params.term_idx] - params.Xref[params.N][params.term_idx])
    # ]
end
function ineq_con_u(p::NamedTuple,u)
    [u-p.u_max;-u + p.u_min] #≦ 0
end
function ineq_con_x(p,x)
    [x-p.x_max;-x + p.x_min]
end
function term_cost(p::NamedTuple,x)
    idx = SA[1,2,3]
    r = x[idx]
    rf = p.Xref[p.N][idx]
    e = p.Prf*(r - rf)
    sqrt(dot(e,e) + 1e-5) + 0.5*p.Qf[7,7]*(x[7] - p.Xref[p.N][7])^2
end
function term_cost_expansion(p::NamedTuple,x)
    H = ForwardDiff.hessian(_x -> term_cost(p,_x), x)
    g = ForwardDiff.gradient(_x -> term_cost(p,_x), x)
    H, g
end
function cpeg_mpc(ev::cp.CPEGWorkspace, N::Ti, x0_scaled::Vector{Tf}, ρ0,λ) where {Ti,Tf}
    nx = 7
    nu = 2
    # N = 125
    Q = Diagonal(SA[0,0,0,0,0,0,1e-4])
    rf = normalize([3.34795153940262, 0.6269403895311674, 0.008024160056155994])
    Prf = 1e2*(I - rf*rf')
    # Qf = SMatrix{7,7}([1e3*Prf'*Prf zeros(3,4);zeros(4,3) diagm([0,0,0,1e-4])])
    # γ = 1e3
    Qf = Diagonal(SA[0,0,0,0,0,0,1e-1])
    @info "Qf set to the right thing"
    R = Diagonal(SA[10,10])*1e-3

    u_min = SA[-100, .01]
    u_max = SA[100, 4]

    # state is x y v θ
    x_min = SVector{nx}([-1e3*ones(6); -pi])
    x_max = SVector{nx}([1e3*ones(6);   pi])


    dt = NaN
    # x0 = kron(ones(3), [3.5212,0,0, -1.559452236319901, 5.633128235948198,0,0.05235987755982989])
    x0 = 1*SVector{nx}(x0_scaled)
    xg = SA[3.34795153940262, 0.6269403895311674, 0.008024160056155994, -0.255884401134421, 0.33667198108223073, -0.056555916829042985, -1.182682624917629]
    Xref = [copy(xg) for i = 1:N]
    Uref = [SA[0,2.0] for i = 1:N-1]


    # altro settings
    solver_settings = Solver_Settings()
    solver_settings.max_iters            = 500
    solver_settings.cost_tol             = 1e-2
    solver_settings.d_tol                = 1e-2
    solver_settings.max_linesearch_iters = 10
    solver_settings.ρ0                   = ρ0
    solver_settings.ϕ                    = 10.0
    solver_settings.reg_min              = 1e-6
    solver_settings.reg_max              = 1e3
    solver_settings.convio_tol           = 1.5e-2

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
        term_idx = [1,2,3],
        kρ_1 = 1.0,
        kρ_2 = 1.0,
        kρ_3 = 1.0,
        idx_x = [(i-1)*7 .+ SVector{7}(1:7) for i = 1:3],
        idx_u = [(i-1)*2 .+ SVector{2}(1:2) for i = 1:3],
        solver_settings = solver_settings,
        Prf = Prf
    );


    X = [deepcopy(x0) for i = 1:N]
    U = [SA[.000001*randn(),2.0] for i = 1:N-1]
    for i = 1:length(U)
        # if i <= length(U_in)
            # U[i] = 1*U_in[i]
        # else
            U[i] = SA[.00001*randn(), 2.0]
        # end
    end
    # U = deepcopy(U_in)
    Xn = deepcopy(X)
    Un = deepcopy(U)


    P = [zeros(nx,nx) for i = 1:N]   # cost to go quadratic term
    p = [zeros(nx) for i = 1:N]      # cost to go linear term
    d = [zeros(nu) for i = 1:N-1]    # feedforward control
    K = [zeros(nu,nx) for i = 1:N-1] # feedback gain
    # λ = zeros(length(term_con(params,X[N])))
    λ = iLQR(params,X,U,P,p,K,d,Xn,Un,λ;verbose = true)
    e = norm(Prf*(X[end][1:3] - params.Xref[end][1:3])*ev.scale.dscale/1000)
    @show λ
    @show e

    return X,U, λ

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

    # N
    N = 125

    # initial condition
    x0_scaled = [3.5212,0,0, -1.559452236319901, 5.633128235948198,0,0.05235987755982989]

    # get atmosphere stuff
    altitudes,densities, Ewind, Nwind = load_atmo()
    params = (altitudes = altitudes, densities = densities, Ewind = Ewind, Nwind = Nwind)

    # initial control
    U_in = [SA[0,0.0] for i = 1:N-1]
    for i = 1:N-1
        U_in[i] = SA[.0001*randn();1.8]
    end

    # initial penalty
    ρ0 = 1e3

    # call mpc
    λ = @SVector zeros(3)
    # U, λ = cpeg_mpc(ev, N, x0_scaled, U_in, ρ0, λ)

    # main sim
    # T = 300
    N_mpc = 125
    # dt = 1.0
    # Xsim = [zeros(7) for i = 1:T]
    # Xsim[1] = x0_scaled
    # Usim = [zeros(2) for i = 1:T]
    # for i = 1:T-1
        # Usim[i] = U[1][1:2] # pull just the bank angle and dt
        # Xsim[i+1] = rk4_fudge_mg(ev,SVector{7}(Xsim[i]),SA[Usim[i][1]],Usim[i][2]/ev.scale.tscale, params)
        # Xsim[i+1] = rk4_fudge_mg(ev,SVector{7}(Xsim[i]),SA[Usim[i][1]],dt/ev.scale.tscale, params)

        # if alt_from_x(ev,Xsim[i+1]) < alt_from_x(ev,[3.34795153940262, 0.6269403895311674, 0.008024160056155994])
        #     @info "SIM IS DONE"
        #     Xsim = Xsim[1:i+1]
        #     Usim = Usim[1:i]
        #     break
        # end
        # call mpc
        # @show i
        # @show (N-i)
        # @show Xsim[i+1]

        # reduce 1 on the N_mpc
        # N_mpc -= 1
        #
        # # get min and max dt
        # min_dt = minimum(hcat(U...)[2,:])
        # max_dt = minimum(hcat(U...)[2,:])
        # dts = [U[i][2] for i = 1:length(U)]
        # tf = sum(dts)
        # N_mpc = Int(ceil(tf/2.0))
        N_mpc = 125
        # if time steps are getting small, remove one off N_mpc
        # if min_dt<1.5
        #     @info "time steps getting small"
        #     N_mpc -= 2
        # end
        #
        # # if time steps are getting big, add one to N_mpc
        # if max_dt>2.5
        #     @info "time steps getting big"
        #     N_mpc += 2
        # end

        # if N_mpc is really small, stop CPEG
        # @show N_mpc
        # @show N_mpc_2
        # if N_mpc < 3
            # @info "CPEG IS OFF"
            # U = U[2:end]
        # else
            # @show Xsim[i+1]
            λ = @SVector zeros(3)
            Xsim,Usim, λ = cpeg_mpc(ev, N_mpc, x0_scaled, 1e0,λ)
        # end
    # end

    # Usim = [[Usim[i];dt] for i = 1:(length(Usim))]
    # alt1, dr1, cr1, σ1, dt1, t_vec1, r1, v1 = process_ev_run(ev,Xsim,Usim)
    # # alt2, dr2, cr2, σ2, dt2, t_vec2, r2, v2 = process_ev_run(ev,X2,U2)
    # # alt3, dr3, cr3, σ3, dt3, t_vec3, r3, v3 = process_ev_run(ev,X3,U3)
    #
    # # get the goals
    # Xg = [SA[3.34795153940262, 0.6269403895311674, 0.008024160056155994, -0.255884401134421, 0.33667198108223073, -0.056555916829042985, -1.182682624917629]]
    # alt_g, dr_g, cr_g = cp.postprocess_scaled(ev,Xg,Xsim[1])
    #
    # mat"
    # figure
    # hold on
    # plot($dr1/1000,$cr1/1000)
    # plot($dr_g(1)/1000, $cr_g(1)/1000,'ro')
    # xlabel('downrange (km)')
    # ylabel('crossrange (km)')
    # hold off
    # "
    # #
    # mat"
    # figure
    # hold on
    # plot($dr1/1000,$alt1/1000)
    # plot($dr_g(1)/1000, $alt_g(1)/1000, 'ro')
    # xlabel('downrange (km)')
    # ylabel('altitude (km)')
    # hold off
    # "
    #
    # v1s = [norm(v1[i]) for i = 1:length(v1)]
    # # v2s = [norm(v2[i]) for i = 1:N]
    # # v3s = [norm(v3[i]) for i = 1:N]
    #
    # mat"
    # figure
    # hold on
    # plot($t_vec1,$v1s)
    # hold off
    # "
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
    # # mat"
    # # figure
    # # hold on
    # # title('Controls')
    # # plot($t_vec(1:end-1), $Um')
    # # legend('sigma dot','dt')
    # # xlabel('Time (s)')
    # # hold off
    # # "
    # #
    # mat"
    # figure
    # hold on
    # plot($t_vec1,rad2deg($σ1))
    # title('Bank Angle')
    # ylabel('Bank Angle (degrees)')
    # xlabel('Time (s)')
    # hold off
    # "
    # mat"
    # figure
    # hold on
    # plot($dt1)
    # title('dts')
    # ylabel('dts')
    # xlabel('knot point')
    # hold off
    # "

end
