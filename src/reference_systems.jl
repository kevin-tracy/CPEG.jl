# NOTE: commented out because this is in gravity.jl
# struct GravityParameters
#     μ::Float64
#     J2::Float64
#     Rp_e::Float64
#     Rp_p::Float64
#     Rp_m::Float64
#     function GravityParameters()
#         GM_MARS = 4.2828375816e13
#         J2_MARS = 1960.45e-6
#         Rpe_MARS = 3.3962e6 #m equatorial radius
#         Rpp_MARS = 3.3762e6 # polar radius, m
#         Rpm_MARS = 3.3895e6  # volumetric mean radius, m
#         new(GM_MARS,J2_MARS,Rpe_MARS,Rpp_MARS,Rpm_MARS)
#     end
# end

function altitude(g::GravityParameters, rp::SVector{3, T}) where T
    # @show rp
    f = (g.Rp_e-g.Rp_p)/g.Rp_e  # flattening
    e = (1-(1-f)^2) # ellepticity (Note: considered as square)
    r = sqrt(rp[1]^2 + rp[2]^2)
    # print(rp)
    # Calculate initial guesses for reduced latitude (latr) and planet-detic latitude (latd)
    latr = atan(rp[3]/((1-f)*r))  # reduced latitude
    latd = atan((rp[3] + (e*(1-f)*g.Rp_e*sin(latr)^3)/(1-e)) / ( r - e*g.Rp_e*cos(latr)^3 ) )
    lat = latd

    # Recalculate reduced latitude based on planet-detic latitude
    latr2 = atan((1-f)*sin(latd)/cos(latd))
    diff = latr - latr2
    # print(diff, '\n')

    iter = 0
    while abs(diff) > 1e-10 && iter <= 1000
        iter += 1
        latr = latr2
        latd = atan((rp[3]+(e*(1-f)*g.Rp_e*sin(latr)^3)/(1-e))/(r - e*g.Rp_e*cos(latr)^3))
        latr2 = atan((1-f)*sin(latd)/ cos(latd))
        diff = latr - latr2
        if iter == 1000
            error("Lat failed: reached max iters")
        end
        lat = latd
    end

    #Calculate longitude
    lon = atan(rp[2],rp[1]) # -180<lon<180
    #Calculate altitude
    N = g.Rp_e / (1-e*sin(lat)^2)^0.5 # radius of curvature in the vertical prime
    # @show r
    # @show lat
    # @show e
    # @show N
    h = r*cos(lat) + (rp[3] + e*N*sin(lat))*sin(lat) - N
    # h = norm(r) - g.Rp_e
    # print("h2 - ", h2, '\n')
    # print("h1 - ", h, '\n')
    return [h, lat, lon]
end

function latlongtoNED(lat, lon)
    # Compute first in xyz coordinates(z: north pole, x - z plane: contains r, y: completes right - handed set)
    uDxyz = [-cos(lat), 0, -sin(lat)]
    uNxyz = [-sin(lat), 0, cos(lat)]
    uExyz = [0,1,0]

    # Rotate by longitude to change to PCPF frame
    L3 = [[cos(lon), -sin(lon), 0],
         [sin(lon), cos(lon), 0],
          [0, 0, 1]]
    uN = L3'uNxyz #inner product
    uE = L3'uExyz
    uD = L3'uDxyz

    return [uD', uN', uE']
end
