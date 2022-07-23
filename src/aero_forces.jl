
mutable struct AeroParameters
    Cl::Float64
    Cd::Float64
    A::Float64
    m::Float64
    function AeroParameters(α=deg2rad(12))
        # MSL https://www.sciencedirect.com/science/article/pii/S0094576514002355#f0005
        δ = deg2rad(70) # rad
        nose_radius = 1.125 # meters
        base_radius = 4.5/2 # meters

        # lift (Gallais, "Atmospheric Re-Entry Vehicle Mechanics," p.82-83)
        Cl1 = 1.42
        Cl =Cl1*α

        # drag (Gallais, "Atmospheric Re-Entry Vehicle Mechanics," p.82-83)
        Cd0 = 1.65
        Cd2 = -2.77
        Cd =Cd0 + Cd2*α^2

        α = -α #mismatch between angle of attack definition
        k = nose_radius/base_radius
        CA_body = (1-sin(δ)^4)*k^2+(2*sin(δ)^2*cos(α)^2+cos(δ)^2*sin(α)^2)*(1-k^2*cos(δ)^2) # axial aerodynamic coefficient
        CN_body = (1-k^2*cos(δ)^2)*cos(δ)^2*sin(2*α) # normal aerodynamic coefficient

        Cl = CN_body*cos(α)-CA_body*sin(α)
        Cd = CA_body*cos(α)+CN_body*sin(α)
     
        # print(Cd,'\n')
        # print(Cl,'\n')

        # print(Cd2,'\n')
        # print(Cl2,'\n')
        # cross sectional area and mass
        A = π*(2.25)^2 # m²
        m = 2400.0     # kg

        return new(Cl, Cd, A, m)
    end
end

@inline function LD_mags(aero::AeroParameters,ρ::T,r::SVector{3,T},v::SVector{3,T}) where T

    dynamic_ρ = 0.5*ρ*aero.A*dot(v,v)/aero.m

    L = aero.Cl*dynamic_ρ
    D = aero.Cd*dynamic_ρ

    return L,D
end


@inline function e_frame(r::SVector{3, T},v::SVector{3, T}) where T
    e1 = cross(r,v)
    e1 = e1/norm(e1)
    e2 = cross(v,e1)
    e2 = e2/norm(e2)
    return e1,e2
end




# let
#
#
#     r = SA[1,4,.3]
#     v = SA[.3,.4,.2]
#
#
#     @btime e1,e2 = e_frame($r,$v)
#
# end
