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
using JLD2

Random.seed!(1)

let

    datasets = 1:20
    N_data = length(datasets)
    alts = [zeros(2) for i = 1:N_data]
    drs =[zeros(2) for i = 1:N_data]
    crs =[zeros(2) for i = 1:N_data]
    σs =[zeros(2) for i = 1:N_data]
    t_vecs =[zeros(2) for i = 1:N_data]
    σ̇s =[zeros(2) for i = 1:N_data]
    dr_errors = zeros(N_data)
    cr_errors = zeros(N_data)
    qp_iters=[zeros(2) for i = 1:N_data]
    alt_g = 0.0
    dr_g = 0.0
    cr_g = 0.0

    for i in datasets
        path = "/Users/kevintracy/.julia/dev/CPEG/src/MarsGramDataset/all/out" * string(i) * ".csv"

        alts[i], drs[i], crs[i], σs[i], t_vecs[i], σ̇s[i], dr_errors[i], cr_errors[i], qp_iters[i], alt_g, dr_g, cr_g = run_cpeg_sim(path; verbose = false)
    end

    # mat"
    # figure
    # hold on
    # for i = 1:$N_data
    #     dr1 = $drs{i};
    #     cr1 = $crs{i};
    #     plot(dr1/1000,cr1/1000)
    # end
    # plot($dr_g(1)/1000, $cr_g(1)/1000,'ro')
    # xlabel('downrange (km)')
    # ylabel('crossrange (km)')
    # hold off
    # "
    mat"
    figure
    hold on
    plot($dr_errors/1000, $cr_errors/1000,'ro')
    xlabel('downrange (km)')
    ylabel('crossrange (km)')
    grid on
    hold off
    "
    # mat"
    # figure
    # hold on
    # plot($dr1/1000,$alt1/1000)
    # plot($dr_g(1)/1000, $alt_g(1)/1000, 'ro')
    # xlabel('downrange (km)')
    # ylabel('altitude (km)')
    # hold off
    # "
    # mat"
    # figure
    # hold on
    # title('Controls')
    # plot($t_vec(1:end-1), $U)
    # legend('sigma dot')
    # xlabel('Time (s)')
    # hold off
    # "
    # mat"
    # figure
    # hold on
    # plot($t_vec,rad2deg($σ1))
    # title('Bank Angle')
    # ylabel('Bank Angle (degrees)')
    # xlabel('Time (s)')
    # hold off
    # "
    #
    # mat"
    # figure
    # title('qp iters')
    # plot($qp_iters)
    # hold off
    # "

end
