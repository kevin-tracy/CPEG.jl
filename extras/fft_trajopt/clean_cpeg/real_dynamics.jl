
function load_atmo(;path="/Users/kevintracy/.julia/dev/CPEG/src/MarsGramDataset/all/out8.csv")
    TT = readdlm(path, ',')
    alt = Vector{Float64}(TT[2:end,2])
    density = Vector{Float64}(TT[2:end,end])
    Ewind = Vector{Float64}(TT[2:end,8])
    Nwind = Vector{Float64}(TT[2:end,10])
    return alt*1000, density, Ewind, Nwind
end

function real_dynamics(ev::cp.CPEGWorkspace, x::SVector{7,T}, u::SVector{1,W}, params) where {T,W}

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

function real_discrete_dynamics(
    ev::cp.CPEGWorkspace,
    x_n::SVector{7,T},
    u::SVector{1,W},
    dt_s::T2, params) where {T,W,T2}

    k1 = dt_s*real_dynamics(ev,x_n,u,params)
    k2 = dt_s*real_dynamics(ev,x_n+k1/2,u,params)
    k3 = dt_s*real_dynamics(ev,x_n+k2/2,u,params)
    k4 = dt_s*real_dynamics(ev,x_n+k3,u,params)

    return (x_n + (1/6)*(k1 + 2*k2 + 2*k3 + k4))
end
