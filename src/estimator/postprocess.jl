function postprocess_es(ev::CPEGWorkspace,X,x0)
    N = length(X)
    alt = zeros(N)
    cr = zeros(N)
    dr = zeros(N)
    for i = 1:N
        alt[i] = altitude(ev,X[i])
        dr[i],cr[i] = rangedistances(ev,X[i],x0)
    end
    return alt, dr, cr
end
function epsilon(ev::CPEGWorkspace,x)
    μ = ev.params.gravity.μ
    r = x[1:3]
    v = x[4:6]
    Ω = ev.params.gravity.Ω #Set Ω = 0.0 here if you want that behavior
    Ω̂ = hat([0, 0, Ω])
    vi = v + Ω̂*r
    return (dot(vi,vi)/2 - μ/norm(r))/1e8
end

function peri_apo(ev::CPEGWorkspace,x)
    r = x[1:3]
    v = x[4:6]
    μ = ev.params.gravity.μ
    h = cross(r,v)

    e_vec = cross(v,h)/μ - normalize(r)
    e = norm(e_vec)
    if e > 1
        return NaN, NaN
    else
        a = -μ/(2*epsilon(model,x)*1e8)
        rp = a*(1-e)
        ra = a*(1+e)
        return rp, ra
    end
end
function processU(ev::CPEGWorkspace,X,U)
    N = length(X)
    AoA = zeros(N-1)
    bank = zeros(N-1)
    for i = 1:N-1
        AoA[i] = (norm(U[i])) * deg2rad(20)
        bank[i] = atan(U[i][1],U[i][2])
    end
    return AoA, bank
end

function altitude(ev,x)
    # alt = [norm(X[i][1:3])*1e3-model.evmodel.planet.R for i = 1:length(X)]

    return norm(x[1:3])*ev.scale.dscale - ev.params.gravity.Rp_e
end
function anglebetween(r1,r2)
    dp = dot(normalize(r1),normalize(r2))
    if dp >1.000000001
        error("over 1 in angle between")
    end
    if dp>1
        dp = 1
    end
    return acos(dp)
end
function rangedistances(ev,x,x0)

    # first we get the angular momentum
    r0 = x0[1:3]
    v0 = x0[4:6]
    h = normalize(cross(r0,v0))
    R = ev.params.gravity.Rp_e
    r = x[1:3]

    # downrange stuff
    r_dr = r - dot(r,h)*h
    θ_dr = anglebetween(r0,r_dr)
    dr = θ_dr*R

    # cross range stuff
    dr_r_r = r - r_dr
    θ_cr = anglebetween(r_dr,r)
    cr = dot(dr_r_r,h) > 0 ? θ_cr*R : -θ_cr*R
    return dr, cr

end
