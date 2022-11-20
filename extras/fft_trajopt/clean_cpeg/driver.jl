using Pkg
Pkg.activate(joinpath(dirname(dirname(@__DIR__)), ".."))
import CPEG as cp
Pkg.activate(dirname(dirname(@__DIR__)))
Pkg.instantiate()

using LinearAlgebra
using StaticArrays
using ForwardDiff
import ForwardDiff as FD
using SparseArrays
using SuiteSparse
using Printf
using MATLAB
import Random
using DelimitedFiles
using Dierckx

include(joinpath(@__DIR__,"utils.jl"))
include(joinpath(@__DIR__,"approx_dynamics.jl"))
include(joinpath(@__DIR__,"real_dynamics.jl"))
include(joinpath(@__DIR__,"controller.jl"))

Random.seed!(1)


let

    # get entry vehicle stuff working
    ev = cp.CPEGWorkspace()

    # vehicle parameters
    ev.params.aero.Cl = 0.29740410453983374
    ev.params.aero.Cd = 1.5284942035954776
    ev.params.aero.A = 15.904312808798327    # m²
    ev.params.aero.m = 2400.0                # kg

    # sim stuff
    ev.dt = NaN # seconds (this is NaN to make sure it doesn't get used)

    # CPEG settings
    ev.scale.uscale = 1e1

    # initial condition
    x0_scaled = [3.5212,0,0, -1.559452236319901, 5.633128235948198,0,0.05235987755982989]

    # get atmosphere stuff in params
    altitudes,densities, Ewind, Nwind = load_atmo()
    nx = 7
    nu = 2
    Q = Diagonal([0,0,0,0,0,0.0,1e-8])
    Qf = 1e8*Diagonal([1.0,1,1,0,0,0,0])
    Qf[7,7] = 1e-4
    R = Diagonal([.1,10.0])
    xg = [3.3477567254291762, 0.626903908492849, 0.03739968529144168, -0.255884401134421, 0.33667198108223073, -0.056555916829042985, -1.182682624917629]


    u_min = [-100, 1e-8]
    u_max =  [100, 4.0]
    δu_min = [-100,-.3]
    δu_max = [100, 0.3]

    x_min = [-1e3*ones(6); -pi]
    x_max = [1e3*ones(6);   pi]
    δx_min = [-1e3*ones(6); -deg2rad(20)]
    δx_max = [1e3*ones(6);   deg2rad(20)]
    x_desired = xg
    u_desired = [0; 2.0]

    ρ_spline = Spline1D(reverse(altitudes), reverse(densities))
    wE_spline = Spline1D(reverse(altitudes), reverse(Ewind))
    wN_spline = Spline1D(reverse(altitudes), reverse(Nwind))
    reg = 10.0
    X = [zeros(nx) for i = 1:10]
    U =[zeros(nu) for i = 1:10]
    # states :nominal :terminal :coast
    params = Params(
                    ρ_spline,
                    wE_spline,
                    wN_spline,
                    nx,nu,
                    Q,R,Qf,
                    u_min,u_max,x_min,x_max,
                    δu_min, δu_max,δx_min,δx_max,
                    x_desired, u_desired,
                    ev, reg, X, U,
                    :nominal,
                    Dict(:nominal => true, :terminal => false, :coast => false),
                    10,1.0)



    initialize_control!(params,x0_scaled)


    # main sim
    T = 3000
    sim_dt = 1.0
    Xsim = [zeros(7) for i = 1:T]
    Xsim[1] = x0_scaled
    Usim = [zeros(2) for i = 1:T-1]

    @info "starting sim"
    for i = 1:T-1

        update_control!(params, Xsim, Usim, sim_dt, i)

        Xsim[i+1] = real_discrete_dynamics(ev,SVector{7}(Xsim[i]),SA[Usim[i][1]],sim_dt/ev.scale.tscale, params)

        # check sim termination
        if alt_from_x(ev,Xsim[i+1]) < alt_from_x(ev,xg[1:3])
            @info "SIM IS DONE"
            Xsim = Xsim[1:(i+1)]
            Usim = Usim[1:i]
            break
        end

    end

    alt1, dr1, cr1, σ1, dt1, t_vec1, r1, v1 = process_ev_run(ev,Xsim,Usim)
    alt_g, dr_g, cr_g = cp.postprocess_scaled(ev,[SVector{7}(xg)],SVector{7}(Xsim[1]))
    #
    mat"
    figure
    hold on
    plot($dr1/1000,$cr1/1000)
    plot($dr_g(1)/1000, $cr_g(1)/1000,'ro')
    xlabel('downrange (km)')
    ylabel('crossrange (km)')
    hold off
    "
    #
    mat"
    figure
    hold on
    plot($dr1/1000,$alt1/1000)
    plot($dr_g(1)/1000, $alt_g(1)/1000, 'ro')
    xlabel('downrange (km)')
    ylabel('altitude (km)')
    hold off
    "

    v1s = [norm(v1[i]) for i = 1:length(v1)]

    mat"
    figure
    hold on
    plot($t_vec1,$v1s)
    hold off
    "

    σ̇ = [Usim[i][1] for i = 1:length(Usim)]
    mat"
    figure
    hold on
    title('Controls')
    plot($σ̇)
    legend('sigma dot')
    xlabel('Time (s)')
    hold off
    "

    mat"
    figure
    hold on
    plot($t_vec1,rad2deg($σ1))
    title('Bank Angle')
    ylabel('Bank Angle (degrees)')
    xlabel('Time (s)')
    hold off
    "

end
