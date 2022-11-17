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

Random.seed!(1)

include("/Users/kevintracy/.julia/dev/CPEG/extras/fft_trajopt/geod_stuff.jl")

# function load_atmo(;path="/Users/kevintracy/.julia/dev/CPEG/extras/fft_trajopt/atmo_samples/samp1.csv")
function load_atmo(;path="/Users/kevintracy/.julia/dev/CPEG/src/MarsGramDataset/all/out8.csv")
    TT = readdlm(path, ',')
    alt = Vector{Float64}(TT[2:end,2])
    density = Vector{Float64}(TT[2:end,end])
    Ewind = Vector{Float64}(TT[2:end,8])
    Nwind = Vector{Float64}(TT[2:end,10])
    return alt*1000, density, Ewind, Nwind
end

function alt_from_x(ev::cp.CPEGWorkspace, x)
    r_scaled = x[SA[1,2,3]]
    v_scaled = SA[2,3,4.0]
    r, v = cp.unscale_rv(ev.scale,r_scaled,v_scaled)
    h,_,_ = cp.altitude(ev.params.gravity, r)
    h
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
function jacobs(params,X,U)
    N = length(X)
    A = [FD.jacobian(_x -> discrete_dynamics(params,_x,U[k],k),X[k]) for k = 1:N-1]
    B = [FD.jacobian(_u -> discrete_dynamics(params,X[k],_u,k),U[k]) for k = 1:N-1]
    A,B
end
function mpc_quad(params,X,U; verbose = true, atol = 1e-6)

    @assert length(X) == (length(U) + 1)
    A,B = jacobs(params,X,U)
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
    for i = 1:(N-1)
        P[idx_u[i],idx_u[i]] = params.R
        q[idx_u[i]] = params.R*(U[i] - params.u_desired)
        P[idx_x[i],idx_x[i]] = params.Q
        q[idx_x[i]] = params.Q*(X[i] - params.x_desired)
    end
    P[idx_x[N],idx_x[N]] = params.Qf
    q[idx_x[N]] = params.Qf*(X[N] - params.x_desired)

    # TODO: maybe regularize
    P += params.reg*I

        # @warn "couldn't solve constrained problem"
        P[idx_x[N],idx_x[N]] = params.Qf + params.reg*I
        q[idx_x[N]] = params.Qf*(X[N] - params.x_desired)
        # solve the equality only constrained QP
        # sol = lu([P A_eq';A_eq -params.reg*I(N*nx)])\[-q;b_eq]
        sol = lu([P A_eq';A_eq spzeros(N*nx,N*nx)])\[-q;b_eq]
        z = sol[1:length(q)]
        qp_iters = 1

        # if this violates the inequality constraints, then we send it to quadprog
        # if sum(G*z .> h) != 0
        if !all(G*z .<= h)
            z, qp_iters = cp.quadprog(P,q,A_eq,b_eq,G,h; verbose = verbose,atol = atol,max_iters = 50)
        end
    # end
    # pull out δx and δu from z
    δx = [z[idx_x[i]] for i = 1:(N)]
    δu = [z[idx_u[i]] for i = 1:(N-1)]


    return (U + δu), qp_iters
end

function fix_control_size(params,U_in,N)
    U = [SA[0.0,params.u_desired[2]] for i = 1:(N-1)]
    for i = 1 : min(N - 1, length(U_in))
        U[i] = U_in[i]
    end
    return U
end

function rollout(params,x0,U)
    N = length(U) + 1
    X = [deepcopy(x0) for i = 1:N]
    for i = 1:N-1
        X[i+1] = discrete_dynamics(params,X[i],U[i],i)
    end
    return X
end
function rollout2(params,x0,U_in)
    N2 = 10000
    # N = length(U) + 1
    U = [zeros(params.nu) for i = 1:(N2-1)]
    X = [deepcopy(x0) for i = 1:N2]
    for i = 1:10000
        if i < length(U_in)
            U[i] = 1*U_in[i]
        else
            U[i] = 1*params.u_desired
        end
        X[i+1] = discrete_dynamics(params,X[i],U[i],i)
        if alt_from_x(params.ev,X[i+1][1:3]) < alt_from_x(params.ev,params.x_desired[1:3])
            return X[1:(i+1)], U[1:i], (i+1)
        end
    end
    error("rollout2 didn't hit alt target")
end
function rollout_to_altitude(params,x0,U)
    N = length(U) + 1
    X = [deepcopy(x0) for i = 1:N]
    for i = 1:N-1
        # @show i
        # @show X[i]
        X[i+1] = discrete_dynamics(params,X[i],U[i],i)
        # error()
        if alt_from_x(params.ev,X[i+1][1:3]) < alt_from_x(params.ev,params.x_desired[1:3])
            return i
        end
    end
    error("didn't reach altitude mark for some reason")
end

function downsample_controls(U, dt_terminal)
    sigma_dot = [U[i][1] for i = 1:length(U)]
    dts = [U[i][2] for i = 1:length(U)]
    tf = sum(dts)
    t_vec = [0;cumsum(dts)]
    t_vec = t_vec[1:end-1]
    dts_terminal = 0:dt_terminal:floor(tf)
    # mat"$sigma_dots = spline($t_vec,$sigma_dot, $dts_terminal);"
    spl = Spline1D(t_vec, sigma_dot)
    sigma_dots = spl(dts_terminal)

    U_terminal = [[sigma_dots[i]; dt_terminal] for i = 1:length(dts_terminal)]
end


mutable struct Params{Tf,Ti}
    ρ_spline::Spline1D
    wE_spline::Spline1D
    wN_spline::Spline1D
    nx::Ti
    nu::Ti
    Q::Diagonal{Tf, Vector{Tf}}
    R::Diagonal{Tf, Vector{Tf}}
    Qf::Diagonal{Tf, Vector{Tf}}
    u_min::Vector{Tf}
    u_max::Vector{Tf}
    x_min::Vector{Tf}
    x_max::Vector{Tf}
    δu_min::Vector{Tf}
    δu_max::Vector{Tf}
    δx_min::Vector{Tf}
    δx_max::Vector{Tf}
    x_desired::Vector{Tf}
    u_desired::Vector{Tf}
    ev::cp.CPEGWorkspace
    reg::Tf
    X::Vector{Vector{Tf}}
    U::Vector{Vector{Tf}}
    state::Symbol
    states_visited::Dict{Symbol, Bool}
    N_mpc::Ti
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
    nx = 7
    nu = 2
    Q = Diagonal([0,0,0,0,0,0.0,1e-8])
    Qf = 1e8*Diagonal([1.0,1,1,0,0,0,0])
    Qf[7,7] = 1e-4
    R = Diagonal([.1,10.0])
    xg = [3.3477567254291762, 0.626903908492849, 0.03739968529144168, -0.255884401134421, 0.33667198108223073, -0.056555916829042985, -1.182682624917629]


    u_min = [-100, 1e-8]
    u_max =  [100, 4.0]
    δu_min = [-100,-.3]
    δu_max = [100, 0.3]

    x_min = [-1e3*ones(6); -pi]
    x_max = [1e3*ones(6);   pi]
    δx_min = [-1e3*ones(6); -deg2rad(20)]
    δx_max = [1e3*ones(6);   deg2rad(20)]
    x_desired = xg
    u_desired = [0; 2.0]

    ρ_spline = Spline1D(reverse(altitudes), reverse(densities))
    wE_spline = Spline1D(reverse(altitudes), reverse(Ewind))
    wN_spline = Spline1D(reverse(altitudes), reverse(Nwind))
    reg = 10.0
    X = [zeros(nx) for i = 1:10]
    U =[zeros(nu) for i = 1:10]
    # states :nominal :terminal :coast
    params = Params(
                    ρ_spline,
                    wE_spline,
                    wN_spline,
                    nx,nu,
                    Q,R,Qf,
                    u_min,u_max,x_min,x_max,
                    δu_min, δu_max,δx_min,δx_max,
                    x_desired, u_desired,
                    ev, reg, X, U,
                    :nominal,
                    Dict(:nominal => true, :terminal => false, :coast => false),
                    10)


    # initial control
    params.N_mpc = 2000
    params.U = [[0,0.0] for i = 1:params.N_mpc-1]
    for i = 1:params.N_mpc-1
        params.U[i] = [.0001*randn();0.9*params.u_desired[2]]
    end
    params.N_mpc = rollout_to_altitude(params,x0_scaled,params.U)
    params.U = params.U[1:params.N_mpc]


    # main sim
    T = 3000
    sim_dt = 1.0
    Xsim = [zeros(7) for i = 1:T]
    Xsim[1] = x0_scaled
    Usim = [zeros(2) for i = 1:T-1]

    qp_iters = -1
    @info "starting sim"
    for i = 1:T-1

        if params.state == :nominal
        # if !terminal
            # ----------------MPC-----------------------
            # get N_mpc
            dts = [params.U[i][2] for i = 1:length(params.U)]
            tf = sum(dts)
            params.N_mpc = Int(ceil((tf)/params.u_desired[2]))
            if params.N_mpc < 50
                # @info "set terminal flag"
                # terminal = true
                params.state = :terminal
                # U = downsample_controls(U, dt_terminal)
            end
        end
        if params.state == :terminal
            # @info "reached terminal status"
            # if !made_the_switch
            if !params.states_visited[:terminal]
                # @info "made the switch and downsampled controls"
                # params = params_terminal
                # params.u_min[2] = 0.1e-8
                params.u_desired[2] = 1.0
                # params.Qf = params.Qf

                params.U = downsample_controls(params.U, params.u_desired[2])
                params.states_visited[:terminal] = :true
            end
            dts = [params.U[i][2] for i = 1:length(params.U)]
            tf = sum(dts)
            params.N_mpc = Int(ceil((tf)/params.u_desired[2]))
        end


        if params.state != :coast
            # CPEG
            # adjust control if mismatch with N_mpc
            params.U = fix_control_size(params,params.U,params.N_mpc)

            # do rollout
            params.X = rollout(params,Xsim[i],params.U)
            # params.X, params.U, params.N_mpc = rollout2(params,Xsim[i],params.U)

            params.U, qp_iters = mpc_quad(params,params.X,params.U; verbose = false, atol = 1e-6)
        end


        # md = params.ev.scale.dscale*norm(X[end][1:3] - params.x_desired[1:3])/1e3
        # @show i, qp_iters, alt, N_mpc, md
        # @show i, alt
        # ----------------MPC-----------------------

        # sim
        if (params.state == :coast) || (length(params.U) == 0)
            params.state = :coast
            # Usim[i] = [Usim[i-1][1];sim_dt]
            Usim[i] = [0;sim_dt]
        else # if we are in nominal or terminal
            Usim[i] = [params.U[1][1]; sim_dt]
        end

        if params.state != :coast
            if rem(i-1,7)==0
                @printf "iter    state     altitude    N_mpc     dt     qp_iter    miss        d_left\n"
                @printf "----------------------------------------------------------------------------\n"
            end
            alt = alt_from_x(params.ev, Xsim[i])/1000
            md = params.ev.scale.dscale*norm(params.X[end][1:3] - params.x_desired[1:3])/1e3
            d_left = params.ev.scale.dscale*norm(Xsim[i][1:3] - params.x_desired[1:3])/1e3
            @printf("%4d    %-7s   %6.2f      %3d      %5.2f  %3d       %6.2f     %6.2f\n",
              i, String(params.state), alt, params.N_mpc, params.u_desired[2], qp_iters, md, d_left)
        else
            if rem(i-1,7)==0
                @printf "iter    state     altitude    N_mpc     dt     qp_iter    miss        d_left\n"
                @printf "----------------------------------------------------------------------------\n"
            end
            alt = alt_from_x(params.ev, Xsim[i])/1000
            md = NaN
            d_left = params.ev.scale.dscale*norm(Xsim[i][1:3] - params.x_desired[1:3])/1e3
            @printf("%4d    %-7s   %6.2f      %3d      %5.2f  %3d       %6.2f     %6.2f\n",
              i, String(params.state), alt, NaN, NaN, NaN, NaN, d_left)
        end





        Xsim[i+1] = rk4_fudge_mg(ev,SVector{7}(Xsim[i]),SA[Usim[i][1]],sim_dt/ev.scale.tscale, params)

        # check sim termination
        if alt_from_x(ev,Xsim[i+1]) < alt_from_x(ev,xg[1:3])
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
    alt_g, dr_g, cr_g = cp.postprocess_scaled(ev,[SVector{7}(xg)],SVector{7}(Xsim[1]))
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
    @show Xsim[50]
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
