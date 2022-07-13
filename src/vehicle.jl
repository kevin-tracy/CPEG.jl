
struct Parameters
    density::CPEGDensityParameters
    gravity::GravityParameters
    aero::AeroParameters
    function Parameters()
        new(CPEGDensityParameters(),GravityParameters(),AeroParameters())
    end
end

struct Planet
    Ω::Float64
    ω::SVector{3,Float64}
    Rp_e::Float64
    Rp_p::Float64
    Rp_m::Float64
    mass::Float64
    g_ref::Float64
    ρ_ref::Float64
    μ::Float64
    href::Float64
    H::Float64
    R::Float64
    γ::Float64
    T::Float64
    p::Float64
    J2::Float64
    k::Float64
    μ_fluid::Float64
    Lz::Float64

    function Planet()
        # mars
        Ω = 7.088218127774194e-5 # rad/s
        ω = SA[0,0,Ω]
        Rp_e = 3.3962e6  # m
        Rp_p = 3.3762e6 # polar radius, m
        Rp_m = 3.3895e6  # volumetric mean radius, m
        mass = 6.4185e23  # mass, kg
        g_ref = 3.71  # m/s^2
        ρ_ref = 8.748923102971180e-07#3.8*10**-8#7.3*10**-8#0.02  # kg/m^3
        μ = 4.2828e13 # gravitational parameter, m^3/s^2
        h_ref = 90*1e3
        H =6.308278108290950e+03 #10.6 * 10 ** 3  # m
        R = 188.92  # J/KgK
        γ = 1.33  # wrong check
        T = 150  # K constant # wrong anchor # the script is not considering this number but T =150
        p = 636  # N/m^2, Surface pressure
        J2 = 1.96045e-3
        k = 1.898e-4  # Sutton - Graves heating coefficient, kg ^ 0.5 / m
        μ_fluid = 13.06*10e-6 # Pa*s  Kinematic viscosity
        Lz = -4.5/1e3 # K/m
        return new(Ω,ω,Rp_e,Rp_p,Rp_m,mass,g_ref,ρ_ref,μ,h_ref,H,R,γ,T,p,J2,k,μ_fluid,Lz)
    end
end

mutable struct COST
    rf::SVector{3,Float64}
    γ::Float64
    σ_tr::Float64
    function COST()
        new(SA[1.,2.,3.], 1e3 , deg2rad(20))
    end
end

mutable struct CPEGWorkspace
    params::Parameters
    scale::Scaling
    planet::Planet
    qp_solver_opts::SolverSettings
    cost::COST
    U::Vector{SVector{1,Float64}}
    miss_distance::Float64
    dt::Float64
    σ::Vector{Float64}
    max_cpeg_iter::Int64
    miss_distance_tol::Float64
    ndu_tol::Float64
    verbose::Bool
    function CPEGWorkspace()
        U = [SA[0.0] for i = 1:1000]
        new(Parameters(), Scaling(), Planet(), SolverSettings(), COST(), U,0.0,0.0,zeros(2),20,1e3,1e-2,true)
    end
end

# function initialize_CPEG(p::)



# # density methods
# @inline function altitude(ev::CPEGWorkspace,r::SVector{3,T}) where T
#     norm(r) - ev.params.gravity.R
# end
#
# @inline function density(ev::CPEGWorkspace,r::SVector{3,T}) where T
#     return density(ev.params.density,altitude(ev,r))
# end
