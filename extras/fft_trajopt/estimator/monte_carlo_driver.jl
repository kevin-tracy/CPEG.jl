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
alt_g, dr_g, cr_g = cp.postprocess_scaled(cp.CPEGWorkspace(),
[SA[3.34795153940262, 0.6269403895311674, 0.008024160056155994,
 -0.255884401134421, 0.33667198108223073, -0.056555916829042985,
  -1.182682624917629]],SA[3.5212,0,0, -1.559452236319901, 5.633128235948198,
  0,0.05235987755982989])
alt_g = alt_g[1]
dr_g = dr_g[1]
cr_g = cr_g[1]
# for i in datasets
#     @show i
#     path = "/Users/kevintracy/.julia/dev/CPEG/src/MarsGramDataset/all/out" * string(i) * ".csv"
#
#     alts[i], drs[i], crs[i], σs[i], t_vecs[i], σ̇s[i], dr_errors[i], cr_errors[i], qp_iters[i], _,_,_ = run_cpeg_sim(path; verbose = false, use_filter = true)
# end

# using JLD2
# jldsave("mc_1000.jld2"; alts, drs, crs, σs, t_vecs, σ̇s, dr_errors, cr_errors, qp_iters, alt_g, dr_g, cr_g)


# define a color order

# mat"
# newcolors = [238 204 102
#              238 153 170
#              102 153 204
#              153 119 0
#              153 68 85
#              0 68 136
#              0 0 0]/255;
#
# colororder(newcolors)
# mat"
# newcolors = [0.83 0.14 0.14
#              1.00 0.54 0.00
#              0.47 0.25 0.80
#              0.25 0.80 0.54];
#
# colororder(newcolors)

# mat"
# figure
# hold on
# for i = 1:$N_data
#     dr1 = $drs{i};
#     al1 = $alts{i};
#     plot(dr1/1000,al1/1000)
# end
# hold off
# "

# interpolate gd stuff
drs_interest = range(400.0,dr_g[1]/1000,length = 100)
new_alts = [zeros(length(drs_interest)) for i = 1:N_data]
for i = 1:N_data
  spl = Spline1D(drs[i]/1000,alts[i]/1000)
  new_alts[i] = spl(drs_interest)
end
alt_min = zeros(length(drs_interest))
alt_max = zeros(length(drs_interest))
for i = 1:length(drs_interest)
    rr = [new_alts[data_num][i] for data_num = 1:N_data]
    alt_min[i] = minimum(rr)
    alt_max[i] = maximum(rr)
end

fill_x = [drs_interest; reverse(drs_interest)]
fill_y = [alt_min; reverse(alt_max)]
mat"
figure
hold on
plot($drs_interest, $alt_min)
plot($drs_interest, $alt_max)
p = fill($fill_x, $fill_y, 'k');
p.FaceAlpha = 0.3;
p2 =plot([0,1000],[$alt_g(1),$alt_g(1)]/1000,'r:','linewidth',2);
legend([p,p2],'Trajectories','Target Altitude')
legend boxoff
xlabel('Downrange, km')
ylabel('Altitude, km')
ylim([7,33])
xlim([400,640])
hold off
addpath('/Users/kevintracy/devel/GravNav/matlab2tikz-master/src')
matlab2tikz('dr_alt.tikz')
"
mat"
figure
hold on
for i = 1:$N_data
    plot($drs_interest,$new_alts{i})
end
hold off
"
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
xlabel('Downrange Error, km')
ylabel('Crossrange Error, km')
%axis equal
p1 = plot(0,0,'r');
p2 = plot(0,0,'b');
grid on
plot([0,0],[-5,5],'k','linewidth',0.5)
plot([-5,5],[0,0],'k','linewidth',0.5)
legend([p1,p2],'Terminal Error','3 sigma')
%legend boxoff
hold off
xlim([-.5,.7])
ylim([-.4,.4])

addpath('/Users/kevintracy/devel/GravNav/matlab2tikz-master/src')
matlab2tikz('terminal_errors.tikz')
"

mat"
figure
hold on
for i = 1:$N_data
    dr1 = $drs{i};
    cr1 = $crs{i};
    plot(dr1/1000,cr1/1000)
end
plot($dr_g/1000, $cr_g/1000,'ro')
xlabel('downrange (km)')
ylabel('crossrange (km)')
hold off
"

# P = [(3*R*[a*cos(t), b*sin(t)] + μ) + [dr_g[1];cr_g[1]]/1000 for t in range(0,2*pi,length = 100)]
# P = hcat(P...)

# interpolate gd stuff
drs_interest = range(300.0,dr_g/1000+2,length = 2000)
drs_interest = drs_interest[1:1991]
new_crs = [zeros(length(drs_interest)) for i = 1:N_data]
for i = 1:N_data
  spl = Spline1D(drs[i]/1000,crs[i]/1000; bc = "zero")
  new_crs[i] = spl(drs_interest)
end
cr_min = zeros(length(drs_interest))
cr_max = zeros(length(drs_interest))
for i = 1:(length(drs_interest))
    rr = [new_crs[data_num][i] for data_num = 1:N_data]
    rr = filter(!iszero, rr)
    @show i
    cr_min[i] = minimum(rr)
    cr_max[i] = maximum(rr)
end

fill_x = [drs_interest; reverse(drs_interest)]
fill_y = [cr_min; reverse(cr_max)]
mat"
figure
hold on
%plot($drs_interest, $cr_min)
%plot($drs_interest, $cr_max)
p = fill($fill_x, $fill_y, 'k');
p.FaceAlpha = 0.3;
%legend([p,p2],'Trajectories','Target Altitude')
%legend boxoff
%plot($P(1,:),$P(2,:),'b');
p2 = plot($dr_g/1000,$cr_g/1000,'r.','MarkerSize',20)
legend([p,p2],'Trajectories','Target','location','northwest')
legend boxoff

xlabel('Downrange, km')
ylabel('Crossrange, km')
%ylim([7,33])
xlim([350,640])
hold off
addpath('/Users/kevintracy/devel/GravNav/matlab2tikz-master/src')
matlab2tikz('dr_cr.tikz')
"

#-------------qp iters----------------------
mat"
figure
hold on
for i = 1:$N_data
    plot($qp_iters{i})
end
hold off
"
qp_iters_stacked = filter(!iszero,vcat(qp_iters...))
mat"
figure
hold on
h1 = histogram($qp_iters_stacked)
h1.Normalization = 'probability';
ytix = get(gca, 'YTick');
set(gca, 'YTick',ytix, 'YTickLabel',ytix*100)
xticks([1,2,3,4,5,6,7,8,9,10,11])
hold off
"

mat"
figure
hold on
for i = 1:300%$N_data
    t = $t_vecs{i};
    sig = $σs{i};
    plot(t,rad2deg(sig))
end
xlabel('Time, s')
xlim([0,220])
p51 = plot(0,0)
legend(p51,'DELETE THIS')
ylabel('Bank Angle, deg')
hold off
%addpath('/Users/kevintracy/devel/GravNav/matlab2tikz-master/src')
%matlab2tikz('bank_angle.tikz')
"

# ts_interest = range(0.0,214.95,length = 1000)
ts_interest = [range(0.0,38.0,length = 10);range(38,180,length = 30);range(180,214.95,length = 30)]
ts_interest = round.(ts_interest, digits = 3)
# drs_interest = drs_interest[1:1991]
new_σs = [NaN*zeros(length(ts_interest)) for i = 1:N_data]
for i = 1:N_data
  spl = Spline1D(t_vecs[i],σs[i]; bc = "zero")
  new_σs[i] = round.(rad2deg.(spl(ts_interest)),digits = 3)
  replace!(new_σs[i], 0.0=>NaN)
end
mat"
figure
hold on
for i = 1:300%$N_data
    plot($ts_interest,$new_σs{i})
end
xlabel('Time, s')
xlim([0,220])
p51 = plot([0,0],[0,0]);
legend(p51,'DELETE THIS')
ylabel('Bank Angle, deg')
hold off
addpath('/Users/kevintracy/devel/GravNav/matlab2tikz-master/src')
matlab2tikz('bank_angle.tikz')
"
# σ_min = zeros(length(ts_interest))
# σ_max = zeros(length(ts_interest))
# for i = 1:(length(ts_interest))
#     rr = [new_σs[data_num][i] for data_num = 1:N_data]
#     rr = filter(!iszero, rr)
#     σ_min[i] = minimum(rr)
#     σ_max[i] = maximum(rr)
# end

# fill_x = [ts_interest; reverse(ts_interest)]
# fill_y = [rad2deg.(σ_min); reverse(rad2deg.(σ_max))]

# mat"
# figure
# hold on
# p = fill($fill_x,$fill_y,'k');
# p.FaceAlpha = 0.3;
# hold off
# "
