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

function jacobs(params,X,U)
    N = length(X)
    A = [FD.jacobian(_x -> filter_discrete_dynamics(params,_x,U[k],k),X[k]) for k = 1:N-1]
    B = [FD.jacobian(_u -> filter_discrete_dynamics(params,X[k],_u,k),U[k]) for k = 1:N-1]
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
        X[i+1] = filter_discrete_dynamics(params,X[i],U[i],i)
    end
    return X
end
# function rollout2(params,x0,U_in)
#     N2 = 10000
#     # N = length(U) + 1
#     U = [zeros(params.nu) for i = 1:(N2-1)]
#     X = [deepcopy(x0) for i = 1:N2]
#     for i = 1:10000
#         if i < length(U_in)
#             U[i] = 1*U_in[i]
#         else
#             U[i] = 1*params.u_desired
#         end
#         X[i+1] = discrete_dynamics(params,X[i],U[i],i)
#         if alt_from_x(params.ev,X[i+1][1:3]) < alt_from_x(params.ev,params.x_desired[1:3])
#             return X[1:(i+1)], U[1:i], (i+1)
#         end
#     end
#     error("rollout2 didn't hit alt target")
# end
function rollout_to_altitude(params,x0,U)
    N = length(U) + 1
    X = [deepcopy(x0) for i = 1:N]
    for i = 1:N-1
        # @show i
        # @show X[i]
        X[i+1] = filter_discrete_dynamics(params,X[i],U[i],i)
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

function initialize_control!(params,x0_scaled)
    # initial control
    params.N_mpc = 2000
    params.U = [[0,0.0] for i = 1:params.N_mpc-1]
    for i = 1:params.N_mpc-1
        params.U[i] = [.0001*randn();0.9*params.u_desired[2]]
    end
    params.N_mpc = rollout_to_altitude(params,x0_scaled,params.U)
    params.U = params.U[1:params.N_mpc]
end

function update_control!(params, Xsim, Usim, sim_dt, idx)

    if params.state == :nominal
        dts = [params.U[i][2] for i = 1:length(params.U)]
        tf = sum(dts)
        params.N_mpc = Int(ceil((tf)/params.u_desired[2]))
        if params.N_mpc < 50
            params.state = :terminal
        end
    end
    if params.state == :terminal
        if !params.states_visited[:terminal]
            params.u_desired[2] = 1.0
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
        params.X = rollout(params,Xsim[idx],params.U)
        # params.X, params.U, params.N_mpc = rollout2(params,Xsim[i],params.U)

        params.U, qp_iters = mpc_quad(params,params.X,params.U; verbose = false, atol = 1e-6)
    end

    # add the control
    if (params.state == :coast) || (length(params.U) == 0)
        params.state = :coast
        # Usim[i] = [Usim[i-1][1];sim_dt]
        Usim[idx] .= [0;sim_dt]
    else # if we are in nominal or terminal
        Usim[idx] .= [params.U[1][1]; sim_dt]
    end

    if params.state != :coast
        if rem(idx-1,7)==0
            @printf "iter    state     altitude    N_mpc     dt     qp_iter    miss        d_left\n"
            @printf "----------------------------------------------------------------------------\n"
        end
        alt = alt_from_x(params.ev, Xsim[idx])/1000
        md = params.ev.scale.dscale*norm(params.X[end][1:3] - params.x_desired[1:3])/1e3
        d_left = params.ev.scale.dscale*norm(Xsim[idx][1:3] - params.x_desired[1:3])/1e3
        @printf("%4d    %-7s   %6.2f      %3d      %5.2f  %3d       %6.2f     %6.2f\n",
          idx, String(params.state), alt, params.N_mpc, params.u_desired[2], qp_iters, md, d_left)
    else
        if rem(idx-1,7)==0
            @printf "iter    state     altitude    N_mpc     dt     qp_iter    miss        d_left\n"
            @printf "----------------------------------------------------------------------------\n"
        end
        alt = alt_from_x(params.ev, Xsim[idx])/1000
        md = NaN
        d_left = params.ev.scale.dscale*norm(Xsim[idx][1:3] - params.x_desired[1:3])/1e3
        @printf("%4d    %-7s   %6.2f      %3d      %5.2f  %3d       %6.2f     %6.2f\n",
          idx, String(params.state), alt, NaN, NaN, NaN, NaN, d_left)
    end

end
