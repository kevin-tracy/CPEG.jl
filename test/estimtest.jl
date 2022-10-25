using Random
using MATLAB
using CSV
using StatsBase
# using DataFrames
# include(joinpath(@__DIR__,"dynamics.jl"))
# # include(joinpath(@__DIR__,"post_process.jl"))
# include(joinpath(@__DIR__,"srekf.jl"))
# include("/home/josephine/.julia/dev/CPEG/src/MarsGramDataset/all/")

Random.seed!(1234)

# df = CSV.File("/home/josephine/.julia/dev/CPEG/src/MarsGramDataset/all/out1.csv")
df = CSV.File("/Users/Josephine/.julia/dev/CPEG/src/MarsGramDataset/all/out1.csv")
# print(df)
# print(size(df)[1])
ρ_real = zeros(size(df))
alt_real = zeros(size(df))
NW_real = zeros(size(df))
EW_real = zeros(size(df))
for (index, row) in enumerate(df)
    ρ_real[index] = row.DensP
    alt_real[index] = row.HgtMOLA
    NW_real[index] = row.NWind#NWTot
    EW_real[index] = row.EWind#EWTot
    # println("values: $(row.HgtMOLA), $(row.Denkgm3), $(row.DensP)")
end
# print(ρ_real)

ev = CPEG.CPEGWorkspace()

model = ev
Rm = ev.params.gravity.Rp_e
h0 = 110e3
r0 = SA[Rm+h0, 0.0, 0.0] #Atmospheric interface at 125 km altitude
V0 = 5.845e3 #Mars-relative velocity at interface of 5.845 km/sec
γ0 = -15.474*(pi/180.0) #Flight path angle at interface
v0 = V0*SA[sin(γ0), cos(γ0), 0.0]
σ0 = deg2rad(42)
kρ = 0.0
kew = 0.5
knw = 0.5
idx_trn = 1

# ev.scale.dscale = 1.0
# ev.scale.tscale = 1.0
# ev.scale.uscale = 1.0
x0 = [r0/ev.scale.dscale;v0/(ev.scale.dscale/ev.scale.tscale); σ0;kρ;kew;knw]

# first rollout
dt = 0.1/ev.scale.tscale# 1/3600/ev.scale.tscale
N = 4000
t_vec = (0:dt:((N-1)*dt))#*3600
print(t_vec)
Y = [zeros(7) for i = 1:N]

X = [@SVector zeros(length(x0)) for i = 1:N]
U = [@SVector zeros(1) for i = 1:N]

X[1] = deepcopy(x0)
μ = deepcopy(X)
μ[1] = [μ[1][1:7]; μ[1][8] + 0.1*randn(); μ[1][9] + 0.1*randn(); μ[1][10] + 0.1*randn()]
F = [zeros(10,10) for i = 1:N]
Σ = (0.01*Matrix(float(I(10))))
Σ[8,8] = (0.05)^2
Σ[9,9] = (.3)^2
Σ[10,10] = (.3)^2
F[1] = CPEG.chol(Matrix(Σ))
σ = deepcopy(F)
σm_ρ = zeros(N)
σm_ρ[1] = Σ[8,8]^0.5
σm_ew = zeros(N)
σm_ew[1] = Σ[9,9]^0.5
σm_nw = zeros(N)
σm_nw[1] = Σ[10,10]^0.5

# initialize density, winds
ρ_spline = zeros(N)
ρ_interp = zeros(N)
NW_spline = zeros(N)
NW_interp = zeros(N)
EW_spline = zeros(N)
EW_interp = zeros(N)

breaks_wind = sort([10:5:35; 37:2.:50; 51:1.:121],rev=true)
breaks_ρ = sort([10:5:60; 62:2.:70; 71:1.:121],rev=true)

# breaks_wind = sort(10:110:120,rev=true)
# breaks_ρ = sort(10:110:120,rev=true)

index_breakwind_previous = findlast(x->x>=h0*1e-3+1, breaks_wind)
index_breakρ_previous = findlast(x->x>=h0*1e-3+1, breaks_ρ)


Q = diagm( [(1e-0)^2*ones(3)/ev.scale.dscale; .005^2*ones(3)/(ev.scale.dscale/ev.scale.tscale); (1e-10)^2;(1e-10)^2;(1e-10)^2;(1e-10)^2])
R = diagm( 10*[(.1)^2*ones(3)/ev.scale.dscale; (0.0002)^2*ones(3)/(ev.scale.dscale/ev.scale.tscale);1e-10])
#Q = diagm( [(.005)^2*ones(3)/ev.scale.dscale; .005^2*ones(3)/(ev.scale.dscale/ev.scale.tscale); (1e-3)^2;(1e-3)^2])
#R = diagm( [(.1)^2*ones(3)/ev.scale.dscale; (0.0002)^2*ones(3)/(ev.scale.dscale/ev.scale.tscale);1e-10])

kf_sys = (dt = dt, ΓR = CPEG.chol(R), ΓQ = CPEG.chol(Q))


end_idx = NaN
index_alt = NaN
for i = 1:(N-1)
    U[i] = [cos(i/10)/30]
    # U[i] = [pi/2]

    ## real density for each time step
    alt = CPEG.postprocess_es(model,[X[i]],x0)[1][1]
    ρ_spline[i] = CPEG.density_spline(model.params.density, alt[1]) # this not necessary, only to keep track of modellization
    EW_spline[i],NW_spline[i] = CPEG.eswind_spline(model.params.density, alt[1])[1:3] # this not necessary, only to keep track of modellization
    # println("EW ", EW_spline[i], " NW ", NW_spline[i])
    #find \rho_real
    global index_alt
    for (index,h) in enumerate(alt_real)
        # println("h ",h," alt",alt)
        if h <= alt*1e-3
            index_alt = index
            # println("index ",index_alt," h", h," alt",alt)
            break
        end
    end
    x_interp = (alt*1e-3 - alt_real[index_alt-1]) / (alt_real[index_alt] - alt_real[index_alt-1]) # this is the true value taken from Mars Sample
    ρ_interp[i] = x_interp* (ρ_real[index_alt]-ρ_real[index_alt-1]) + ρ_real[index_alt-1] # this is the true value taken from Mars Sample
    NW_interp[i] = x_interp* (NW_real[index_alt]-NW_real[index_alt-1]) + NW_real[index_alt-1] # this is the true value taken from Mars Sample
    EW_interp[i] = x_interp* (EW_real[index_alt]-EW_real[index_alt-1]) + EW_real[index_alt-1] # this is the true value taken from Mars Sample

    index_breakwind = findlast(x->x>=alt*1e-3, breaks_wind)
    index_breakρ = findlast(x->x>=alt*1e-3, breaks_ρ)
    global index_breakwind_previous, index_breakρ_previous, kρ, kew, knw
    if index_breakwind > index_breakwind_previous
        println(i)
        println("wind ", alt)
        knw = sin(NW_interp[i]/NW_spline[i])
        println("NW_interp ",NW_interp[i], "NW_spline ", NW_spline[i]," knw ",knw)
        kew = sin(EW_interp[i]/EW_spline[i])
        if i == 1
            X[i] = [X[i][1:8];kew;knw]
        end
        index_breakwind_previous = index_breakwind
    end

    if index_breakρ > index_breakρ_previous
        println("rho ", alt)
        kρ = (ρ_interp[i]/ρ_spline[i])-1
        if i == 1
            X[i] = [X[i][1:7];kρ;X[i][9:10]]
        end
        index_breakρ_previous = index_breakρ
    end
    # if i ==1 # just initialize
    #     kρ = (ρ_interp[i]/ρ_spline[i])-1
    #     knw = (NW_interp[i]/NW_spline[i])-1
    #     kew = (EW_interp[i]/EW_spline[i])-1
    #     global kρ, knw, kew
    #     # print()
    #     X[i] = [X[i][1:7];kρ;kew;knw]
    # end
    # # println("x_interp ",  x_interp, "alt ", alt, "alt_real1 ", alt_real[index_alt-1], "alt_real2 ", alt_real[index_alt])
    if i == 1
        μ[1] = [μ[1][1:7]; kρ + 0.1*randn(); kew + 0.1*randn(); knw + 0.1*randn()]
    end
    X[i+1] = CPEG.rk4_est(model,X[i],U[i],dt)
    Y[i+1] = CPEG.measurement(model,X[i+1]) + kf_sys.ΓR*randn(7)
    μ[i+1], F[i+1] = CPEG.sqrkalman_filter(model, μ[i],F[i],U[i],Y[i+1],kf_sys)
    σ[i+1] = F[i+1]'*F[i+1]
    σm_ρ[i+1] = sqrt(σ[i+1][8,8])
    σm_ew[i+1] = sqrt(σ[i+1][9,9])
    σm_nw[i+1] = sqrt(σ[i+1][10,10])
    #break if altitude less than 10 km
    x = deepcopy(X[i+1])
    r = CPEG.unscale_rv(ev.scale,x[SA[1,2,3]],x[SA[4,5,6]])[1]
    # alt = CPEG.altitude(ev.params.gravity, r)[1]
    alt,lat,lon = CPEG.altitude(ev.params.gravity, r)
    if i%1000== 1
        println("alt ",alt," - lat ",lat," - lon ",lon)
    end
    # println(alt, i)
    global idx_trn
    idx_trn += 1
    if alt <= 1e4
        alt = CPEG.postprocess_es(model,[X[i+1]],x0)[1]
        ρ_spline[i+1] = CPEG.density_spline(model.params.density, alt[1])
        # println("alt ",alt," - lat ",lat," - lon ",lon)
        break
    end
end
# exit()
# truncate results
println(μ[1],X[1])
t_vec = (0:dt:((idx_trn-1)*dt))*ev.scale.tscale#*3600
X  = X[1:idx_trn]
U = U[1:idx_trn]
Y = Y[1:idx_trn]
μ = μ[1:idx_trn]
F = F[1:idx_trn]
σm_ρ = σm_ρ[1:idx_trn]
σm_ew = σm_ew[1:idx_trn]
σm_nw = σm_ew[1:idx_trn]
ρ_spline = ρ_spline[1:idx_trn]
ρ_interp = ρ_interp[1:idx_trn]
NW_spline = NW_spline[1:idx_trn]
NW_interp = NW_interp[1:idx_trn]
EW_spline = EW_spline[1:idx_trn]
EW_interp = EW_interp[1:idx_trn]
N = idx_trn

# println(X)
function mat_from_vec(a)
    "Turn a vector of vectors into a matrix"
    rows = length(a[1])
    columns = length(a)
    A = zeros(rows,columns)

    for i = 1:columns
        A[:,i] = a[i]
    end

    return A
end

Xm = mat_from_vec(X)

mat"
figure
hold on
plot($Xm(7,:))
hold off
"

# println(ρ_spline)


alt, dr, cr = CPEG.postprocess_es(model,X,x0)

alt_k, dr_k, cr_k = CPEG.postprocess_es(model,μ,x0)

perr = 1e3*([ev.scale.dscale*norm(X[i][1:3] - μ[i][1:3]) for i = 1:N])
verr = 1e3*([(ev.scale.dscale/ev.scale.tscale)*norm(X[i][4:6] - μ[i][4:6]) for i = 1:N])

yperr = 1e3*([ev.scale.dscale*norm(X[i][1:3] - Y[i][1:3]) for i = 2:N])
yverr = 1e3*([(ev.scale.dscale/ev.scale.tscale)*norm(X[i][4:6] - Y[i][4:6]) for i = 2:N])

# println(ρ_interp)
mat"
figure
hold on
p1 = plot($alt/10^3,$ρ_spline);
p2 = plot($alt(1:end-1)/10^3,$ρ_interp(1:end-1));
hold off
ylabel('Density, kg/m^3')
xlabel('Altitude, km')
legend([p1;p2],'ρ spline','ρ real','location','northeast')
hold off
set(gca,'FontSize',14)
set(gca, 'YScale', 'log')
%saveas(gcf,'plots/density.eps','epsc')
%pause(15)
"

mat"
figure
hold on
p1 = plot($alt/10^3,$EW_spline);
p2 = plot($alt(1:end-1)/10^3,$EW_interp(1:end-1));
hold off
ylabel('East Wind, m/s')
xlabel('Altitude, km')
legend([p1;p2],'EW spline','EW real','location','northeast')
hold off
set(gca,'FontSize',14)
%set(gca, 'YScale', 'log')
saveas(gcf,'plots/eastwind.png')
%pause(15)
"
mat"
figure
hold on
p1 = plot($alt/10^3,$NW_spline);
p2 = plot($alt(1:end-1)/10^3,$NW_interp(1:end-1));
hold off
ylabel('North Wind, m/s')
xlabel('Altitude, km')
legend([p1;p2],'NW spline','NW real','location','northeast')
hold off
set(gca,'FontSize',14)
%set(gca, 'YScale', 'log')
saveas(gcf,'plots/northwind.png')
%pause(15)
"

# exit()
mat"
figure
hold on
title('Trajectory')
%plot($t_vec, $Xm(1,:))
%plot($t_vec, $Xm(2,:))
plot($t_vec, $alt)
legend('X','Y','Z')
xlabel('Time, s')
ylabel('Position, km')
hold off
set(gca,'FontSize',14)
%saveas(gcf,'plots/traj.eps','epsc')
saveas(gcf,'plots/traj.png')
"

mat"
figure
hold on
title('Position Error')
plot($t_vec, $perr)
plot($t_vec(1:end-1), $yperr)
legend('SREKF','Measurement')
xlabel('Time (s)')
ylabel('Position Error (km)')
hold off
set(gca,'FontSize',14)
%saveas(gcf,'plots/perr.eps','epsc')
saveas(gcf,'plots/perr.png')
"
mat"
figure
hold on
title('Velocity Error')
plot($t_vec,$verr)
plot($t_vec(1:end-1),$yverr)
legend('SREKF','Measurement')
ylabel('Velocity Error (km/s)')
xlabel('Time (s)')
hold off
set(gca,'FontSize',14)
%saveas(gcf,'plots/verr.eps','epsc')
saveas(gcf,'plots/verr.png')
%pause(30)
"
# exit()
μm = mat_from_vec(μ)

# println("rho_spline ", ρ_spline," mu", (1 .+ μm[8,:]'))

# println(ρ_est)
# exit()
# print(μm[8,:])
# σm = zeros(length(μ))
# # @infiltrate
# # error()
# for i = 1:length(μ)
#     Σ = F[i]'*F[i]
#     σm = sqrt(Σ[8,8])
# end
# print(3*σm)
mat"
figure
hold on
p1 = plot($alt_k/1e3,$μm(8,:)','b');
p2= plot($alt_k/1e3,$μm(8,:)' + 3*$σm_ρ,'r--');
plot($alt_k/1e3,$μm(8,:)' - 3*$σm_ρ,'r--')
p3 = plot($alt/1e3,$Xm(8,:)','color',[0.9290, 0.6940, 0.1250]);
title('Atmospheric Correction Factor')
legend([p1;p2;p3],'SREKF krho','3 sigma bounds','True krho','location','southeast')
xlabel('Altitude, km')
ylabel('k rho')
%ylim([0.7 0.82])
hold off
set(gca,'FontSize',14)
%saveas(gcf,'plots/krho.eps','epsc')
saveas(gcf,'plots/krho.png')
"

mat"
figure
hold on
p1 = plot($t_vec,$μm(8,:)','b');
p2= plot($t_vec,$μm(8,:)' + 3*$σm_ρ,'r--');
plot($t_vec,$μm(8,:)' - 3*$σm_ρ,'r--')
p3 = plot($t_vec,$Xm(8,:)','color',[0.9290, 0.6940, 0.1250]);
title('Atmospheric Correction Factor')
legend([p1;p2;p3],'SREKF krho','3 sigma bounds','True krho','location','southeast')
xlabel('Time, s')
ylabel('k rho')
%ylim([0.7 0.82])
hold off
set(gca,'FontSize',14)
%saveas(gcf,'plots/krhotime.eps','epsc')
saveas(gcf,'plots/krhotime.png')
"
ρ_est = zeros(length(ρ_spline))
ρ_est_max = zeros(length(ρ_spline))
ρ_est_min = zeros(length(ρ_spline))
ρ_est = [ρ_spline[i]*(1+μm[8,i]') for i in range(1,length(ρ_spline))]
ρ_est_max = [ρ_spline[i]*(1+(μm[8,i]'+3*σm_ρ[i])) for i in range(1,length(ρ_spline))]
ρ_est_min = [ρ_spline[i]*(1+(μm[8,i]'-3*σm_ρ[i])) for i in range(1,length(ρ_spline))]
L2_ρ = L2dist(ρ_interp, ρ_est)

NW_est = zeros(length(NW_spline))
NW_est_max = zeros(length(NW_spline))
NW_est_min = zeros(length(NW_spline))
NW_est = [NW_spline[i]*(μm[10,i]') for i in range(1,length(NW_spline))]
NW_est_max = [NW_spline[i]*((μm[10,i]'+3*σm_nw[i])) for i in range(1,length(NW_spline))]
NW_est_min = [NW_spline[i]*((μm[10,i]'-3*σm_nw[i])) for i in range(1,length(NW_spline))]
L2_NW = L2dist(NW_interp, NW_est)

EW_est = zeros(length(EW_spline))
EW_est_max = zeros(length(EW_spline))
EW_est_min = zeros(length(EW_spline))
EW_est = [EW_spline[i]*(μm[9,i]') for i in range(1,length(EW_spline))]
EW_est_max = [EW_spline[i]*((μm[9,i]'+3*σm_ew[i])) for i in range(1,length(EW_spline))]
EW_est_min = [EW_spline[i]*((μm[9,i]'-3*σm_ew[i])) for i in range(1,length(EW_spline))]
L2_EW = L2dist(EW_interp, EW_est)
mat"
figure
hold on
p1 = plot($alt_k(1:end-1)/1e3,$ρ_est(1:end-1),'b');
p2= plot($alt_k(1:end-1)/1e3,$ρ_est_max(1:end-1),'r--');
plot($alt_k(1:end-1)/1e3,$ρ_est_min(1:end-1),'r--')
p3 = plot($alt/1e3,$ρ_interp,'color',[0.9290, 0.6940, 0.1250]);
title(['Atmospheric Density - L2 Distance - ', num2str($L2_ρ)])
legend([p1;p2;p3],'SREKF ρ','3 sigma bounds','True ρ','location','southeast')
xlabel('Altitude, km')
ylabel('ρ, kg/m^3')
%ylim([0.7 0.82])
set(gca, 'YScale', 'log')
hold off
set(gca,'FontSize',14)
%saveas(gcf,'plots/rho.eps','epsc')
saveas(gcf,'plots/rho.png')
"

mat"
figure
hold on
p1 = plot($t_vec(1:end-1),$ρ_est(1:end-1),'b');
p2= plot($t_vec(1:end-1),$ρ_est_max(1:end-1),'r--');
plot($t_vec(1:end-1),$ρ_est_min(1:end-1),'r--')
p3 = plot($t_vec,$ρ_interp','color',[0.9290, 0.6940, 0.1250]);
title(['Atmospheric Density - L2 Distance - ', num2str($L2_ρ)])
legend([p1;p2;p3],'SREKF ρ','3 sigma bounds','True ρ','location','southeast')
xlabel('Time, s')
ylabel('ρ, kg/m^2')
%ylim([0.7 0.82])
set(gca, 'YScale', 'log')
hold off
set(gca,'FontSize',14)
%saveas(gcf,'plots/rhotime.eps','epsc')
saveas(gcf,'plots/rhotime.png')
"

mat"
figure
hold on
p1 = plot($alt_k/1e3,$μm(9,:)','b');
p2= plot($alt_k/1e3,$μm(9,:)' + 3*$σm_ew,'r--');
plot($alt_k/1e3,$μm(9,:)' - 3*$σm_ew,'r--')
p3 = plot($alt/1e3,$Xm(9,:)','color',[0.9290, 0.6940, 0.1250]);
title('East Wind Correction Factor')
legend([p1;p2;p3],'SREKF kwE','3 sigma bounds','True kwE','location','southeast')
xlabel('Altitude, km')
ylabel('k East Wind')
% ylim([-1.0 1.0])
hold off
set(gca,'FontSize',14)
%saveas(gcf,'plots/keW.eps','epsc')
saveas(gcf,'plots/keW.png')
"

mat"
figure
hold on
p1 = plot($t_vec,$μm(9,:)','b');
p2= plot($t_vec,$μm(9,:)' + 3*$σm_ew,'r--');
plot($t_vec,$μm(9,:)' - 3*$σm_ew,'r--')
p3 = plot($t_vec,$Xm(9,:)','color',[0.9290, 0.6940, 0.1250]);
title('East Wind Correction Factor')
legend([p1;p2;p3],'SREKF kwE','3 sigma bounds','True kwE','location','southeast')
xlabel('Time, s')
ylabel('k East Wind')
% ylim([-1.0 1.0])
hold off
set(gca,'FontSize',14)
%saveas(gcf,'plots/keWtime.eps','epsc')
saveas(gcf,'plots/keWtime.png')
"

mat"
figure
hold on
p1 = plot($alt_k(1:end-1)/1e3,$EW_est(1:end-1),'b');
p2= plot($alt_k(1:end-1)/1e3,$EW_est_max(1:end-1),'r--');
plot($alt_k(1:end-1)/1e3,$EW_est_min(1:end-1),'r--')
p3 = plot($alt/1e3,$EW_interp,'color',[0.9290, 0.6940, 0.1250]);
title(['East Wind - L2 Distance - ', num2str($L2_EW)])
legend([p1;p2;p3],'SREKF EW','3 sigma bounds','True EW','location','southeast')
xlabel('Altitude, km')
ylabel('East Wind, m/s')
ylim([-100 100])
%set(gca, 'YScale', 'log')
hold off
set(gca,'FontSize',14)
%saveas(gcf,'plots/ew.eps','epsc')
saveas(gcf,'plots/ew.png')
"

mat"
figure
hold on
p1 = plot($t_vec(1:end-1),$EW_est(1:end-1),'b');
p2= plot($t_vec(1:end-1),$EW_est_max(1:end-1),'r--');
plot($t_vec(1:end-1),$EW_est_min(1:end-1),'r--')
p3 = plot($t_vec,$EW_interp','color',[0.9290, 0.6940, 0.1250]);
title(['East Wind - L2 Distance - ', num2str($L2_EW)])
legend([p1;p2;p3],'SREKF EW','3 sigma bounds','True EW','location','southeast')
xlabel('Time, s')
ylabel('EW, kg/m^2')
ylim([-100 100])
%set(gca, 'YScale', 'log')
hold off
set(gca,'FontSize',14)
%saveas(gcf,'plots/ewtime.eps','epsc')
saveas(gcf,'plots/ewtime.png')
"

mat"
figure
hold on
p1 = plot($alt_k/1e3,$μm(10,:)','b');
p2= plot($alt_k/1e3,$μm(10,:)' + 3*$σm_nw,'r--');
plot($alt_k/1e3,$μm(10,:)' - 3*$σm_nw,'r--')
p3 = plot($alt/1e3,$Xm(10,:)','color',[0.9290, 0.6940, 0.1250]);
title('North Wind Correction Factor')
legend([p1;p2;p3],'SREKF kwN','3 sigma bounds','True kwN','location','southeast')
xlabel('Altitude, km')
ylabel('k North Wind')
% ylim([-1.0 1.0])
hold off
set(gca,'FontSize',14)
%saveas(gcf,'plots/knW.eps','epsc')
saveas(gcf,'plots/knW.png')
"

mat"
figure
hold on
p1 = plot($t_vec,$μm(10,:)','b');
p2= plot($t_vec,$μm(10,:)' + 3*$σm_nw,'r--');
plot($t_vec,$μm(10,:)' - 3*$σm_nw,'r--')
p3 = plot($t_vec,$Xm(10,:)','color',[0.9290, 0.6940, 0.1250]);
title('North Wind Correction Factor')
legend([p1;p2;p3],'SREKF kwN','3 sigma bounds','True kwN','location','southeast')
xlabel('Time, s')
ylabel('k North Wind')
% ylim([-1.0 1.0])
hold off
set(gca,'FontSize',14)
%saveas(gcf,'plots/knWtime.eps','epsc')
saveas(gcf,'plots/knWtime.png')
"

mat"
figure
hold on
p1 = plot($alt_k(1:end-1)/1e3,$NW_est(1:end-1),'b');
p2= plot($alt_k(1:end-1)/1e3,$NW_est_max(1:end-1),'r--');
plot($alt_k(1:end-1)/1e3,$NW_est_min(1:end-1),'r--')
p3 = plot($alt/1e3,$NW_interp,'color',[0.9290, 0.6940, 0.1250]);
title(['North Wind - L2 Distance - ', num2str($L2_NW)])
legend([p1;p2;p3],'SREKF NW','3 sigma bounds','True NW','location','southeast')
xlabel('Altitude, km')
ylabel('North Wind, m/s')
ylim([-100 100])
%set(gca, 'YScale', 'log')
hold off
set(gca,'FontSize',14)
%saveas(gcf,'plots/nw.eps','epsc')
saveas(gcf,'plots/nw.png')
"

mat"
figure
hold on
p1 = plot($t_vec(1:end-1),$NW_est(1:end-1),'b');
p2= plot($t_vec(1:end-1),$NW_est_max(1:end-1),'r--');
plot($t_vec(1:end-1),$NW_est_min(1:end-1),'r--')
p3 = plot($t_vec,$NW_interp','color',[0.9290, 0.6940, 0.1250]);
title(['North Wind - L2 Distance - ', num2str($L2_NW)])
legend([p1;p2;p3],'SREKF NW','3 sigma bounds','True NW','location','southeast')
xlabel('Time, s')
ylabel('NW, kg/m^2')
ylim([-100 100])
%set(gca, 'YScale', 'log')
hold off
set(gca,'FontSize',14)
%saveas(gcf,'plots/nwtime.eps','epsc')
saveas(gcf,'plots/nwtime.png')
"
# @test a1 ≈ (a1_central + a1_j2) rtol = 1e-12
