function esdensity(model,h,H)
    ρ0 = 5.25*1e-7
    ρ = ρ0*exp(-(h)/H)
    return ρ
end

function esdynamics(ev::CPEGWorkspace, x::SVector{8,T}, u::SVector{1,W}) where {T,W}
    # scaled variables
    r_scaled = x[SA[1,2,3]]
    v_scaled = x[SA[4,5,6]]
    σ = x[7]
    kρ = x[8]

    # unscale
    r, v = unscale_rv(ev.scale,r_scaled,v_scaled)
    # altitude
    h = altitude(ev.params.gravity, r)[1]
    # density
    # ρ = density(ev.params.density, h)
    ρ = esdensity(ev, h, 7.295*1e3)*(1+kρ)

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
    # println("alt ",h)
    # rescale units
    v,a = scale_va(ev.scale,v,a)

    return SA[v[1],v[2],v[3],a[1],a[2],a[3],u[1]*ev.scale.uscale,0]
end

function rk4_est(
    ev::CPEGWorkspace,
    x_n::SVector{8,T},
    u::SVector{1,W},
    dt_s::Float64) where {T,W}

    k1 = dt_s*esdynamics(ev,x_n,u)
    k2 = dt_s*esdynamics(ev,x_n+k1/2,u)
    k3 = dt_s*esdynamics(ev,x_n+k2/2,u)
    k4 = dt_s*esdynamics(ev,x_n+k3,u)
    return (x_n + (1/6)*(k1 + 2*k2 + 2*k3 + k4))
end

function get_discdyn(model,X,U,dt)
    # dt_s = dt/ev.scale.tscale
    A= ForwardDiff.jacobian(_x -> rk4(model,_x,U,dt),X)
    B= ForwardDiff.jacobian(_u -> rk4(model,X,_u,dt),U)
    return A,B
end
