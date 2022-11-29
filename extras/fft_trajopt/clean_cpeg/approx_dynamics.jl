
function approx_dynamics(ev::cp.CPEGWorkspace, x::SVector{7,T}, u::SVector{1,W}, kρ::T2) where {T,W,T2}

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
    ρ = kρ*cp.density_spline(ev.params.dsp, h)*0.9

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

function rk4_approx_dynamics(
    ev::cp.CPEGWorkspace,
    x_n::SVector{7,T},
    u::SVector{1,W},
    dt_s::T2, kρ::T3) where {T,W,T2,T3}

    k1 = dt_s*approx_dynamics(ev,x_n,u,kρ)
    k2 = dt_s*approx_dynamics(ev,x_n+k1/2,u,kρ)
    k3 = dt_s*approx_dynamics(ev,x_n+k2/2,u,kρ)
    k4 = dt_s*approx_dynamics(ev,x_n+k3,u,kρ)

    return (x_n + (1/6)*(k1 + 2*k2 + 2*k3 + k4))
end

function controller_discrete_dynamics(p,x1,u1,k) where {T1,T2}
    @assert length(x1)==7
    @assert length(u1)==2
    rk4_approx_dynamics(p.ev,SVector{7}(x1),SA[u1[1]],u1[2]/p.ev.scale.tscale, p.kρ);
end

function filter_discrete_dynamics(p,x1,u1,dt)
    @assert length(x1)==8
    @assert length(u1)==1
    # add kρ̇ = 0 onto the output
    [rk4_approx_dynamics(p.ev,SVector{7}(x1[1:7]),SA[u1[1]],dt/p.ev.scale.tscale, x1[8]);x1[8]]
end
