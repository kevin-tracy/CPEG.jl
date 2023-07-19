

const MARS_SMA = 3396200
const MARS_f = 1/169.8
const ECC2 = MARS_f * (2.0 - MARS_f)

function MCMFfromGEOD(lat, lon, alt)# degrees


    # Check validity of input
    if lat < -pi/2 || lat > pi/2
        throw(ArgumentError("Lattiude, $lat, out of range. Must be between -90 and 90 degrees."))
    end

    # Compute Earth-fixed position vector
    N = MARS_SMA / sqrt(1.0 - ECC2*sin(lat)^2)
    x =           (N+alt)*cos(lat)*cos(lon)
    y =           (N+alt)*cos(lat)*sin(lon)
    z =  ((1.0-ECC2)*N+alt)*sin(lat)

    return SA[x, y, z]
end

function GEODfromMCMF(mcmf)
    # Expand ECEF coordinates
    x, y, z = mcmf

    # Compute intermediate quantities
    epsilon  = eps(Float64) * 1.0e3 * MARS_SMA # Convergence requirement as function of machine precision
    rho2 = x^2 + y^2                      # Square of the distance from the z-axis
    dz   = ECC2 * z
    N    = 0.0

    # Iteratively compute refine coordinates
    while true
        zdz    = z + dz
        Nh     = sqrt(rho2 + zdz^2)
        sinphi = zdz / Nh
        N      = MARS_SMA / sqrt(1.0 - ECC2 * sinphi^2)
        dz_new = N * ECC2 * sinphi

        # Check convergence requirement
        if abs(dz - dz_new) < epsilon
            break
        end

        dz = dz_new
    end

    # Extract geodetic coordinates
    zdz = z + dz
    lat = atan(zdz, sqrt(rho2))
    lon = atan(y, x)
    alt = sqrt(rho2 + zdz^2) - N

    return lat, lon, alt
end

# let
#
#     mcmf = [3.34795153940262, 0.6269403895311674, 0.008024160056155994]*1e6
#
#     lat, lon, alt = GEODfromMCMF(mcmf)
#
#     @show rad2deg(lat)
#     @show rad2deg(lon)
#     @show alt
#
#     mcmf2 = MCMFfromGEOD(lat + deg2rad(.5), lon, alt)
#
#     @show norm(mcmf - mcmf2)/1e3
#
#     @show mcmf2/1e6
#
#     # @show rad2deg(lat)
#     # @show rad2deg(lon)
#     # @show alt
#
#
# end
