
struct IDX
    # contains the variable indexing stuff for [x;s;z;y]
    nx::Int64
    ns::Int64
    nz::Int64
    ny::Int64
    N::Int64

    x::UnitRange{Int64}
    s::UnitRange{Int64}
    z::UnitRange{Int64}
    y::UnitRange{Int64}
end

struct DELTA
    # stores all the deltas (search directions)

    # affine deltas
    x_a::Array{Float64,1}
    s_a::Array{Float64,1}
    z_a::Array{Float64,1}
    y_a::Array{Float64,1}

    # centering + correcting deltas
    x_c::Array{Float64,1}
    s_c::Array{Float64,1}
    z_c::Array{Float64,1}
    y_c::Array{Float64,1}

    # total deltas
    x::Array{Float64,1}
    s::Array{Float64,1}
    z::Array{Float64,1}
    y::Array{Float64,1}

    # constructor
    function DELTA(nx,ns,nz,ny)
        new(zeros(nx),zeros(ns),zeros(nz),zeros(ny),
            zeros(nx),zeros(ns),zeros(nz),zeros(ny),
            zeros(nx),zeros(ns),zeros(nz),zeros(ny))
    end
end

mutable struct SolverSettings
    verbose::Bool
    max_iters::Int64
    atol::Float64
    function SolverSettings()
        return new(false, 50, 1e-8)
    end
end

struct QP

    # problem data
    Q::SparseMatrixCSC{Float64,Int64}
    q::Array{Float64,1}
    A::SparseMatrixCSC{Float64,Int64}
    b::Array{Float64,1}
    G::SparseMatrixCSC{Float64,Int64}
    h::Array{Float64,1}

    # variables
    x::Array{Float64,1} # primal
    s::Array{Float64,1} # primal
    z::Array{Float64,1} # dual
    y::Array{Float64,1} # dual

    # KKT stuff
    KKT::SparseMatrixCSC{Float64,Int64}
    rhs_a::Array{Float64,1}
    rhs_c::Array{Float64,1}
    p_a::Array{Float64,1}
    p_c::Array{Float64,1}

    # indexing
    idx::IDX

    # deltas
    Δ::DELTA

    # solver settings
    opts::SolverSettings

    function QP(Q,q,A,b,G,h)

        # length of variables
        nx = length(q)
        ns = length(h)
        nz = length(h)
        ny = length(b)
        N = nx + ns + nz + ny

        # indexing when stacked [x;s;z;y]
        idx_x = 1:nx
        idx_s = (nx + 1) : (nx + ns)
        idx_z = (nx + ns + 1) : (nx + ns + nz)
        idx_y = (nx + ns + nz + 1) : (nx + ns + nz + ny)
        idx = IDX(nx,ns,nz,ny,N, idx_x, idx_s, idx_z, idx_y)

        # kkt stuff
        KKT = spzeros(N,N)
        rhs_a = zeros(N)
        rhs_c = zeros(N)
        p_a = zeros(N)
        p_c = zeros(N)

        # initialize variables to zero
        x = zeros(nx)
        s = zeros(ns)
        z = zeros(nz)
        y = zeros(ny)

        # deltas
        Δ = DELTA(nx,ns,nz,ny)

        new(Q,q,A,b,G,h,x,s,z,y, KKT, rhs_a, rhs_c, p_a, p_c, idx, Δ, SolverSettings())
    end


end


# ---------------real functions---------------

function rhs_kkt_a!(qp::QP)
    idx = qp.idx
    qp.rhs_a[idx.x] = -(qp.A'*qp.y + qp.G'*qp.z + qp.Q*qp.x + qp.q)
    qp.rhs_a[idx.s] = -(qp.s .* qp.z)
    qp.rhs_a[idx.z] = -(qp.G*qp.x + qp.s - qp.h)
    qp.rhs_a[idx.y] = -(qp.A*qp.x - qp.b)
    return nothing
end
function index_sol_a!(qp::QP)
    qp.Δ.x_a .= qp.p_a[qp.idx.x]
    qp.Δ.s_a .= qp.p_a[qp.idx.s]
    qp.Δ.z_a .= qp.p_a[qp.idx.z]
    qp.Δ.y_a .= qp.p_a[qp.idx.y]
    return nothing
end
function index_sol_c!(qp::QP)
    qp.Δ.x_c .= qp.p_c[qp.idx.x]
    qp.Δ.s_c .= qp.p_c[qp.idx.s]
    qp.Δ.z_c .= qp.p_c[qp.idx.z]
    qp.Δ.y_c .= qp.p_c[qp.idx.y]
    return nothing
end
function linesearch(x,dx)
    α = min(1.0, minimum([dx[i]<0 ? -x[i]/dx[i] : Inf for i = 1:length(x)]))
    return α
end
function centering_params(qp::QP)

    μ = dot(qp.s,qp.z)/qp.idx.ns

    α = min(linesearch(qp.s,qp.Δ.s_a), linesearch(qp.z,qp.Δ.z_a))

    σ = (dot(qp.s + α*qp.Δ.s_a, qp.z + α*qp.Δ.z_a)/dot(qp.s,qp.z))^3
    return σ, μ
end
function rhs_kkt_c!(qp::QP, σ, μ)
    idx = qp.idx
    qp.rhs_c .= 0
    qp.rhs_c[idx.s] = (σ*μ .- (qp.Δ.s_a .* qp.Δ.z_a))
    return nothing
end
function combine_deltas!(qp::QP)
    qp.Δ.x .= qp.Δ.x_a + qp.Δ.x_c
    qp.Δ.s .= qp.Δ.s_a + qp.Δ.s_c
    qp.Δ.z .= qp.Δ.z_a + qp.Δ.z_c
    qp.Δ.y .= qp.Δ.y_a + qp.Δ.y_c
    return nothing
end
function update_vars!(qp::QP,α)
    qp.x .+= α*qp.Δ.x
    qp.s .+= α*qp.Δ.s
    qp.z .+= α*qp.Δ.z
    qp.y .+= α*qp.Δ.y
    return nothing
end

struct QPMaxItersError <: Exception
    var::Symbol
end

function solveqp!(qp::QP)

    if qp.opts.verbose
        @printf "iter     objv        gap       |Ax-b|    |Gx+s-h|    step\n"
        @printf "---------------------------------------------------------\n"
    end

    initialize!(qp)

    initialize_kkt!(qp)

    for i = 1:qp.opts.max_iters

        # update linear system for solves
        update_kkt!(qp)
        # kkt_factor = qdldl(qp.KKT +
                          # 1e-10*Diagonal([ones(qp.idx.nx + qp.idx.ns);-ones(qp.idx.nz + qp.idx.ny)]))
        kkt_factor = lu(qp.KKT)

        # affine step
        rhs_kkt_a!(qp)

        qp.p_a .= kkt_factor\qp.rhs_a
        # qp.p_a .= iterative_ref(kkt_factor,qp.KKT,qp.rhs_a)
        index_sol_a!(qp)

        # centering and correcting step
        σ, μ = centering_params(qp)
        rhs_kkt_c!(qp, σ, μ)
        qp.p_c .= kkt_factor\qp.rhs_c
        # qp.p_c .= iterative_ref(kkt_factor,qp.KKT,qp.rhs_c)
        index_sol_c!(qp)

        # combine deltas
        combine_deltas!(qp)

        # last linesearch
        α = min(1,0.99*min(linesearch(qp.s,qp.Δ.s),linesearch(qp.z,qp.Δ.z)))

        update_vars!(qp,α)

        if logging(qp::QP,i,α)
            return i
            # break
        end

        # if i == qp.opts.max_iters
            # throw(QPMaxItersError)
        # end
    end
    throw(QPMaxItersError)
    # return nothing
end

function logging(qp::QP,iter,α)

    J = 0.5*qp.x'*qp.Q*qp.x + dot(qp.q,qp.x)
    gap = dot(qp.s,qp.z)/length(qp.s)
    eq_res = norm(qp.A*qp.x - qp.b,Inf)
    ineq_res = norm(qp.G*qp.x + qp.s - qp.h,Inf)


    if qp.opts.verbose
        @printf("%3d   %10.3e  %9.2e  %9.2e  %9.2e  % 6.4f\n",
              iter, J, gap, eq_res,
              ineq_res, α)
    end

    # check for convergence
    if (eq_res < 1e-10) & (ineq_res < 1e-10) & (gap<qp.opts.atol)
        return true
    else
        return false
    end
end

function initialize!(qp::QP)
    idx = qp.idx
    Ni = idx.nx + idx.nz + idx.ny
    idx_y = idx.y .- idx.ns

    A = spzeros(Ni, Ni)
    A[idx.x,idx.x] = qp.Q
    A[idx.x,idx.s] = qp.G'
    A[idx.x,idx_y] = qp.A'
    A[idx.s,idx.x] = qp.G
    A[idx.s,idx.s] = -I(idx.ns)
    A[idx_y,idx.x] = qp.A

    init = A\[-qp.q;qp.h;qp.b]

    qp.x .= init[idx.x]
    qp.z .= init[idx.s]
    qp.y .= init[idx_y]


    α_p = -minimum(-qp.z)
    if α_p < 0
        qp.s .= -qp.z
    else
        qp.s .= -qp.z .+ (1 + α_p)
    end

    α_d = -minimum(qp.z)
    if α_d >= 0
        qp.z .= qp.z .+ (1 + α_d)
    end

    return nothing
end

function initialize_kkt!(qp::QP)
    idx = qp.idx

    qp.KKT[idx.x, idx.x] = qp.Q
    qp.KKT[idx.x, idx.z] = qp.G'
    qp.KKT[idx.x, idx.y] = qp.A'
    # qp.KKT[idx.s, idx.s] = Diagonal(qp.z)
    # qp.KKT[idx.s, idx.z] = Diagonal(qp.s)
    qp.KKT[idx.z, idx.x] = qp.G
    qp.KKT[idx.z, idx.s] = I(idx.nz)
    qp.KKT[idx.y, idx.x] = qp.A

    return nothing
end

function update_kkt!(qp::QP)
    idx = qp.idx
    qp.KKT[idx.s, idx.s] = Diagonal(qp.z)#Diagonal(qp.z ./ qp.s)
    qp.KKT[idx.s, idx.z] = Diagonal(qp.s)#I(idx.ns)
    return nothing
end

function quadprog(Q,q,A,b,G,h; verbose = false, max_iters = 50, atol = 1e-8)
    qp = QP(Q,q,A,b,G,h)
    qp.opts.verbose = verbose
    qp.opts.max_iters = max_iters
    qp.opts.atol = atol
    qp_iters = solveqp!(qp::QP)
    return qp.x, qp_iters
end
