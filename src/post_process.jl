
function postprocess(ev::CPEGWorkspace,X,x0)
    N = length(X)
    alt = zeros(N)
    cr = zeros(N)
    dr = zeros(N)
    for i = 1:N
        alt[i] = altitude(ev.params.gravity, X[i][SA[1,2,3]])
        dr[i],cr[i] = rangedistances(ev,X[i],x0)
    end
    return alt, dr, cr
end

function postprocess_scaled(ev::CPEGWorkspace,X,x0_s)
    X = unscale_X(ev.scale,X)
    x0 = x0_s/ev.scale.dscale
    alt, dr, cr = postprocess(ev,X,x0)
    return alt,dr,cr
end
# function processU(model::CPEGWorkspace,X,U)
#     N = length(X)
#     AoA = zeros(N-1)
#     bank = zeros(N-1)
#     for i = 1:N-1
#         AoA[i] = (norm(U[i])) * deg2rad(20)
#         bank[i] = atan(U[i][1],U[i][2])
#     end
#     return AoA, bank
# end

# function altitude(model,x)
#     return norm(x[1:3]) - model.evmodel.planet.R
# end
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
function rangedistances(ev::CPEGWorkspace,x::StaticVector,x0::StaticVector)

    # first we get the angular momentum
    r0 = x0[1:3]
    v0 = x0[4:6]
    h = normalize(cross(r0,v0))
    # R = model.evmodel.planet.R
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
