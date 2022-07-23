
function dynamics(ev::CPEGWorkspace, x::SVector{7,T}, u::SVector{1,W}) where {T,W}

    # scaled variables
    r_scaled = x[SA[1,2,3]]
    v_scaled = x[SA[4,5,6]]
    σ = x[7]

    # unscale
    r, v = unscale_rv(ev.scale,r_scaled,v_scaled)

    # altitude
    h = altitude(ev.params.gravity, r)

    # density
    ρ = density(ev.params.density, h)

    # lift and drag magnitudes
    L, D = LD_mags(ev.params.aero,ρ,r,v)

    # basis for e frame
    e1, e2 = e_frame(r,v)

    # drag and lift accelerations
    D_a = -(D/norm(v))*v
    L_a = L*sin(σ)*e1 + L*cos(σ)*e2

    # gravity
    g = gravity(ev.params.gravity,r)

    # acceleration
    ω = ev.planet.ω
    a = D_a + L_a + g - 2*cross(ω,v) - cross(ω,cross(ω,r))

    # rescale units
    v,a = scale_va(ev.scale,v,a)

    return SA[v[1],v[2],v[3],a[1],a[2],a[3],u[1]*ev.scale.uscale]
end

function rk4(
    ev::CPEGWorkspace,
    x_n::SVector{7,T},
    u::SVector{1,W},
    dt_s::Float64) where {T,W}

    k1 = dt_s*dynamics(ev,x_n,u)
    k2 = dt_s*dynamics(ev,x_n+k1/2,u)
    k3 = dt_s*dynamics(ev,x_n+k2/2,u)
    k4 = dt_s*dynamics(ev,x_n+k3,u)
    return (x_n + (1/6)*(k1 + 2*k2 + 2*k3 + k4))
end

function rollout(ev::CPEGWorkspace,x0::SVector{7,T},U::Vector{SVector{1,T}}) where T
    """everything in and out of the function is scaled"""

    # scaled dt
    dt_s = ev.dt/ev.scale.tscale

    # input U
    U_in = copy(ev.U)

    N = 1000
    X = [@SVector zeros(length(x0)) for i = 1:N]
    U = [@SVector zeros(length(U_in[1])) for i = 1:N]
    X[1] = x0
    end_idx = NaN
    for i = 1:N-1
        U[i] = (i>length(U_in)) ? SA[0.0] : U_in[i]

        X[i+1] = rk4(ev,X[i],U[i],dt_s)

        # for debugging purposes
        # @show (norm(X[i+1][1:3])*ev.scale.dscale - ev.params.gravity.R)/1e3

        if (norm(X[i+1][1:3])*ev.scale.dscale - ev.params.gravity.Rp_e) < 10e3
            # @info "hit alt"
            # @show i
            end_idx = i+1
            break
        end
    end

    # trim relavent
    if isnan(end_idx)
        error("didn't hit the altitude during the rollout")
    end

    X = X[1:end_idx]
    # ev.U = U[1:(end_idx-1)]

    return X, U[1:(end_idx-1)]
end

function get_jacobians(
    ev::CPEGWorkspace,
    X::Vector{SVector{7,T}}) where T

    dt_s = ev.dt/ev.scale.tscale

    N = length(X)
    A = [ForwardDiff.jacobian(_x->rk4(ev,_x,ev.U[i],dt_s),X[i]) for i = 1:N-1]
    B = [ForwardDiff.jacobian(_u->rk4(ev,X[i],_u,dt_s),ev.U[i]) for i = 1:N-1]
    return A,B
end



# let
#
#     ev = CPEGWorkspace()
#
#     # # @show ev.scale.uscale
#     #
#     # x = @SVector randn(7)
#     # u = SA[1.45]
#     # #
#     # @btime dynamics($ev,$x,$u)
#     #
#     # @btime rk4($ev,$x,$u,0.1)
#     # #
#     # A = ForwardDiff.jacobian(_x -> dynamics(ev,_x,u),x)
#     # # @btime ForwardDiff.jacobian(_x -> dynamics($ev,_x,$u),$x)
#     # @btime ForwardDiff.jacobian(_x -> rk4($ev,_x,$u,0.1),$x)
#     # ForwardDiff.jacobian(_x -> rk4(ev,_x,u,0.1),x)
#     Rm = ev.params.gravity.R
#     r0 = SA[Rm+125e3, 0.0, 0.0] #Atmospheric interface at 125 km altitude
#     V0 = 5.845e3 #Mars-relative velocity at interface of 5.845 km/sec
#     γ0 = -15.474*(pi/180.0) #Flight path angle at interface
#     v0 = V0*SA[sin(γ0), cos(γ0), 0.0]
#     σ0 = deg2rad(90)
#
#     r0sc,v0sc = scale_rv(ev.scale,r0,v0)
#
#     dt = 1.0/ev.scale.tscale
#
#     x0 = SA[r0sc[1],r0sc[2],r0sc[3],v0sc[1],v0sc[2],v0sc[3],σ0]
#
#     # @show norm(r0)
#     #
#     # @show norm(r0)*ev.scale.dscale - ev.params.gravity.R
#     N = 100
#     U = [SA[0.0] for i = 1:N-1]
#     #
#     X,U = rollout(ev, x0, U, dt)
#     @show length(X)
#     # N = 100
#     # X = [@SVector zeros(7) for i = 1:N]
#
#     # @show typeof([r0;v0;σ0])
#     # X[1] = SA[r0[1],r0[2],r0[3],v0[1],v0[2],v0[3],σ0]
#     # @btime $X[1] = SA[$r0[1],$r0[2],$r0[3],$v0[1],$v0[2],$v0[3],$σ0]
#
#     # @show X[1]
#     # @show typeof(X[1])
#
#     # i = 7
#     # @show ForwardDiff.jacobian(_x->rk4(ev,_x,U[i],dt),X[i])
#     # @show ForwardDiff.jacobian(_u->rk4(ev,X[i],_u,dt),U[i])
#     # A,B= get_jacobians(ev,X,U,dt)
#     #
#     # N = length(X)
#     # Xm = zeros(7,N)
#     # for i = 1:N
#     #     Xm[:,i] = X[i]
#     # end
#     #
#     # nr = [norm(X[i][1:3]) for i = 1:N]
#     # nv = [norm(X[i][4:6]) for i = 1:N]
#     # mat"
#     # figure
#     # hold on
#     # plot($Xm(1:3,:)')
#     # hold off
#     # "
#     #
#     # mat"
#     # figure
#     # hold on
#     # plot($Xm(4:6,:)')
#     # hold off
#     # "
#     # mat"
#     # figure
#     # hold on
#     # plot($nr)
#     # plot($nv)
#     # hold off"
#
#     # X = unscale_X(ev.scale,X)
#     # # @show typeof(X[1][SA[1,2,3]])
#     #
#     # alt, dr, cr = postprocess(ev,X,[r0;v0])
#     #
#     #
#     # mat"
#     # figure
#     # hold on
#     # plot($alt/1e3)
#     # hold off "
#     #
#     # mat"
#     # figure
#     # hold on
#     # plot($dr/1e3,$cr/1e3)
#     # hold off"
#
# end
