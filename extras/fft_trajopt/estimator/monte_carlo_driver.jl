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

include("/Users/kevintracy/.julia/dev/CPEG/extras/fft_trajopt/estimator/cpeg_sim_function.jl")

# let

datasets = 1:1000
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
    @show i
    path = "/Users/kevintracy/.julia/dev/CPEG/src/MarsGramDataset/all/out" * string(i) * ".csv"

    alts[i], drs[i], crs[i], σs[i], t_vecs[i], σ̇s[i], dr_errors[i], cr_errors[i], qp_iters[i], alt_g, dr_g, cr_g = run_cpeg_sim(path; verbose = false, use_filter = true)
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

errors = [dr_errors/1000 cr_errors/1000]

μ = vec(mean(errors; dims = 1))
Σ = cov(errors)
F = eigen(Σ)
R = F.vectors
a,b=sqrt.(F.values)
# a = 1
# b = 1
P = [(3*R*[a*cos(t), b*sin(t)] + μ) for t in range(0,2*pi,length = 100)]
P = hcat(P...)
mat"
figure
hold on
plot($dr_errors/1000, $cr_errors/1000,'r*','MarkerSize',5)
plot($P(1,:),$P(2,:),'b')
xlabel('downrange (km)')
ylabel('crossrange (km)')
%axis equal
p1 = plot(0,0,'r')
p2 = plot(0,0,'b')
grid on
xline(0)
yline(0)
legend([p1,p2],'Terminal Error','3 sigma')
hold off
"

qp_iters_stacked = filter(!iszero,vcat(qp_iters...))
mat"
figure
hold on
h1 = histogram($qp_iters_stacked)
h1.Normalization = 'probability';
hold off
"

mat"
figure
hold on
for i = 1:$N_data
    t = $t_vecs{i}
    sig = $σs{i}
    plot(t,rad2deg(sig))
end
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

# end
