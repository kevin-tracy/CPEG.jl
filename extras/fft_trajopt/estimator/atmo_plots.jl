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


function load_atmo(;path="/Users/kevintracy/.julia/dev/CPEG/src/MarsGramDataset/all/out4.csv")
    TT = readdlm(path, ',')
    alt = Vector{Float64}(TT[2:end,2])
    density = Vector{Float64}(TT[2:end,end])
    Ewind = Vector{Float64}(TT[2:end,8])
    Nwind = Vector{Float64}(TT[2:end,10])
    return alt, density, Ewind, Nwind
end

let


    ev = cp.CPEGWorkspace()

    # alts = []
    # ρs = []
    # Ewinds = []
    # Nwinds = []
    ev_alts = 10:1:100

    datasets = 1:100
    N_data = length(datasets)
    ρs = [zeros(length(ev_alts)) for i = 1:N_data]
    Ewinds = [zeros(length(ev_alts)) for i = 1:N_data]
    Nwinds = [zeros(length(ev_alts)) for i = 1:N_data]

    for i = datasets
        path = "/Users/kevintracy/.julia/dev/CPEG/src/MarsGramDataset/all/out" * string(i) * ".csv"
        alt, density, Ewind, Nwind = load_atmo(;path)
        ρ_spl = Spline1D(reverse(alt),reverse(density))
        Ewind_spl = Spline1D(reverse(alt),reverse(Ewind))
        Nwind_spl = Spline1D(reverse(alt),reverse(Nwind))

        ρs[i] = ρ_spl(ev_alts)
        Ewinds[i] = Ewind_spl(ev_alts)
        Nwinds[i] = Nwind_spl(ev_alts)
        # push!(alts, alt)
        # push!(ρs, density)
        # push!(Ewinds, Ewind)
        # push!(Nwinds, Nwind)
    end


    ev_ρ = [cp.density_spline(ev.params.dsp, alt*1000) for alt in ev_alts]

    c1 = [209/255,124/255,50/255]
    c2 = [50,172,209]/255;
    mat"
    figure
    hold on
    for i = 1:100
        x1 = $Ewinds{i};
        x2 = $Nwinds{i};
        plot(x1,$ev_alts,'color',$c1)
        plot(x2,$ev_alts,'color',$c2)
    end
    plot([0,0],[10,100],'k--','linewidth',2);
    p1 = plot([],[],'color',$c1);
    p2 = plot([],[],'color',$c2);
    legend([p1,p2],'East','North','location','northwest')
    xlabel('Wind Velocity, m/s')
    ylabel('Altitude, km')
    ylim([10,100])
    legend boxoff
    hold off
    %addpath('/Users/kevintracy/devel/GravNav/matlab2tikz-master/src')
    %matlab2tikz('test.tikz')
    "

    Ewind_min = zeros(length(ev_alts))
    Ewind_max = zeros(length(ev_alts))
    Nwind_min = zeros(length(ev_alts))
    Nwind_max = zeros(length(ev_alts))
    for i = 1:length(ev_alts)

        E = [Ewinds[data_iter][i] for data_iter = 1:N_data]
        N = [Nwinds[data_iter][i] for data_iter = 1:N_data]

        Ewind_min[i] = minimum(E)
        Ewind_max[i] = maximum(E)
        Nwind_min[i] = minimum(N)
        Nwind_max[i] = maximum(N)
    end

    fill_y = [ev_alts;reverse(ev_alts)]
    fill_x1 = [Ewind_min;reverse(Ewind_max)]
    fill_x2 = [Nwind_min;reverse(Nwind_max)]
    mat"
    figure
    hold on
    plot($Ewind_min, $ev_alts, 'color',$c1,'linewidth',1.5)
    plot($Ewind_max, $ev_alts, 'color',$c1,'linewidth',1.5)
    plot($Nwind_min, $ev_alts, 'color',$c2,'linewidth',1.5)
    plot($Nwind_max, $ev_alts, 'color',$c2,'linewidth',1.5)
    P1 = fill($fill_x1,$fill_y,'g')
    P1.FaceColor = $c1
    P1.EdgeColor = $c1
    P1.FaceAlpha = 0.3
    P2 = fill($fill_x2,$fill_y,'g')
    P2.FaceColor = $c2
    P2.EdgeColor = $c2
    P2.FaceAlpha = 0.3
    plot([0,0],[10,100],'k--','linewidth',1);
    p1 = plot([0,0],[0,0],'color',$c1);
    p2 = plot([0,0],[0,0],'color',$c2);
    legend([p1,p2],'East','North','location','northwest')
    xlabel('Wind Velocity, m/s')
    ylabel('Altitude, km')
    ylim([10,100])
    legend boxoff
    hold off
    addpath('/Users/kevintracy/devel/GravNav/matlab2tikz-master/src')
    matlab2tikz('wind_shaded.tikz')
    "



end
