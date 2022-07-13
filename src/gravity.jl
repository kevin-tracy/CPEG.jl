
struct GravityParameters
    μ::Float64
    J2::Float64
    Rp_e::Float64
    Rp_p::Float64
    Rp_m::Float64
    function GravityParameters()
        GM_MARS = 4.2828375816e13
        J2_MARS = 1960.45e-6
        Rpe_MARS = 3.3962e6 #m equatorial radius
        Rpp_MARS = 3.3762e6 # polar radius, m
        Rpm_MARS = 3.3895e6  # volumetric mean radius, m
        new(GM_MARS,J2_MARS,Rpe_MARS,Rpp_MARS,Rpm_MARS)
    end
end

@inline function gravity(g::GravityParameters,r::SVector{3, T}) where T

    nr = norm(r)

    x,y,z = r

    # precompute repeated stuff
    Re_r_sqr = 1.5*g.J2*(g.Rp_e/nr)^2
    five_z_sqr = 5*z^2/nr^2

    return  (-g.μ/nr^3)*SA[x*(1 - Re_r_sqr*(five_z_sqr - 1)),
                           y*(1 - Re_r_sqr*(five_z_sqr - 1)),
                           z*(1 - Re_r_sqr*(five_z_sqr - 3))]
end

function altitude(g::GravityParameters, rp::SVector{3, T}) where T
    f = (g.Rp_e-g.Rp_p)/g.Rp_e  # flattening
    e = (1-(1-f)^2) # ellepticity (Note: considered as square)
    r = sqrt(rp[1]^2 + rp[2]^2)

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
    h = r*cos(lat) + (rp[3] + e*N*sin(lat))*sin(lat) - N

    # h = norm(r) - g.Rp_e
    # print("h2 - ", h2, '\n')
    # print("h1 - ", h, '\n')
    return h
end


# mass = 6.4185e23  # mass, kg
# g_ref = 3.71  # m/s^2
# ρ_ref = 8.748923102971180e-07#3.8*10**-8#7.3*10**-8#0.02  # kg/m^3
# μ = 4.2828e13 # gravitational parameter, m^3/s^2
# h_ref = 90*1e3
# H =6.308278108290950e+03 #10.6 * 10 ** 3  # m
# R = 188.92  # J/KgK
# γ = 1.33  # wrong check
# T = 150  # K constant # wrong anchor # the script is not considering this number but T =150
# p = 636  # N/m^2, Surface pressure
# J2 = 1.96045e-3
# k = 1.898e-4  # Sutton - Graves heating coefficient, kg ^ 0.5 / m
# μ_fluid = 13.06*10e-6 # Pa*s  Kinematic viscosity
# Lz = -4.5/1e3 # K/m
