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

estimate_wind = 0

# df = CSV.File("/home/josephine/.julia/dev/CPEG/src/MarsGramDataset/all/out1.csv")
df = CSV.File("/Users/Josephine/.julia/dev/CPEG/src/MarsGramDataset/MonteCarlo/out1.csv")

ρ_real = zeros(size(df))
alt_real = zeros(size(df))
NW_real = zeros(size(df))
EW_real = zeros(size(df))
for (index, row) in enumerate(df)
    ρ_real[index] = row.DensP
    alt_real[index] = row.HgtMOLA
    NW_real[index] = row.NWind#NWTot
    EW_real[index] = row.EWind#EWTot
end

ev = CPEG.CPEGWorkspace()

model = ev
Rm = ev.params.gravity.Rp_e
h0 = 110e3
r0 = SA[Rm+h0, 0.0, 0.0] #Atmospheric interface at 125 km altitude
V0 = 5.845e3 #Mars-relative velocity at interface of 5.845 km/sec
γ0 = -15.474*(pi/180.0) #Flight path angle at interface
v0 = V0*SA[sin(γ0), cos(γ0), 0.0]
σ0 = deg2rad(42)
idx_trn = 1

# ev.scale.dscale = 1.0
# ev.scale.tscale = 1.0
# ev.scale.uscale = 1.0
if estimate_wind == 0
    x0 = [r0/ev.scale.dscale;v0/(ev.scale.dscale/ev.scale.tscale); σ0;0]
    state_number = 8
else
    x0 = [r0/ev.scale.dscale;v0/(ev.scale.dscale/ev.scale.tscale); σ0;0;0;0]
    state_number = 10
end

# first rollout
dt = 0.5/ev.scale.tscale# 1/3600/ev.scale.tscale
N = 4000
t_vec = (0:dt:((N-1)*dt))#*3600
Y = [zeros(7) for i = 1:N]

X = [@SVector zeros(length(x0)) for i = 1:N]
U = [@SVector zeros(1) for i = 1:N]

X[1] = deepcopy(x0)
μ = deepcopy(X)
F = [zeros(10,10) for i = 1:N]
Σ = (0.01*Matrix(float(I(state_number))))
Σ[8,8] = (0.1)^2
σm_ρ = zeros(N)
σm_ρ[1] = Σ[8,8]^0.5
if estimate_wind == 1
    Σ[9,9] = (.05)^2
    Σ[10,10] = (.05)^2
    σm_ew = zeros(N)
    σm_ew[1] = Σ[9,9]^0.5
    σm_nw = zeros(N)
    σm_nw[1] = Σ[10,10]^0.5
end
F[1] = CPEG.chol(Matrix(Σ))
σ = deepcopy(F)

# initialize density, winds
ρ_spline = zeros(N)
ρ_interp = zeros(N)
NW_spline = zeros(N)
NW_interp = zeros(N)
EW_spline = zeros(N)
EW_interp = zeros(N)

kρ = zeros(N)
kew = zeros(N)
knw = zeros(N)


if estimate_wind == 0
    Q = 50*diagm( [(5e-2)^2*ones(3)/ev.scale.dscale; .0001^2*ones(3)/(ev.scale.dscale/ev.scale.tscale); (1e-10)^2;(0.5e-4)^2])
else
    Q = diagm( [(5e-2)^2*ones(3)/ev.scale.dscale; .0001^2*ones(3)/(ev.scale.dscale/ev.scale.tscale); (1e-10)^2;(1e-10)^2;(1e-4)^2;(1e-4)^2])
    # Q = diagm( [(5e-1)^2*ones(3)/ev.scale.dscale; .001^2*ones(3)/(ev.scale.dscale/ev.scale.tscale); (1e-10)^2;(1e-10)^2;(1e-6)^2;(1e-6)^2])
end
#Q = diagm( [(5e-1)^2*ones(3)/ev.scale.dscale; .001^2*ones(3)/(ev.scale.dscale/ev.scale.tscale); (1e-10)^2;(1e-10)^2;(1e-6)^2;(1e-6)^2])
R = diagm( 10*[(.1)^2*ones(3)/ev.scale.dscale; (0.0002)^2*ones(3)/(ev.scale.dscale/ev.scale.tscale);1e-10])
#Q = diagm( [(.005)^2*ones(3)/ev.scale.dscale; .005^2*ones(3)/(ev.scale.dscale/ev.scale.tscale); (1e-3)^2;(1e-3)^2])
#R = diagm( [(.1)^2*ones(3)/ev.scale.dscale; (0.0002)^2*ones(3)/(ev.scale.dscale/ev.scale.tscale);1e-10])

kf_sys = (dt = dt, ΓR = CPEG.chol(R), ΓQ = CPEG.chol(Q))


end_idx = NaN
index_alt = NaN
for i = 1:(N-1)
    U[i] = [cos(i/10)/30]

    ## real density for each time step
    alt = CPEG.postprocess_es(model,[X[i]],x0)[1][1]
    ρ_spline[i] = CPEG.density_spline(model.params.density, alt[1]) # this not necessary, only to keep track of modellization
    EW_spline[i],NW_spline[i] = CPEG.eswind_spline(model.params.density, alt[1])[1:3] # this not necessary, only to keep track of modellization

    #find \rho_real
    global index_alt
    for (index,h) in enumerate(alt_real)
        if h <= alt*1e-3
            index_alt = index
            break
        end
    end
    x_interp = (alt*1e-3 - alt_real[index_alt-1]) / (alt_real[index_alt] - alt_real[index_alt-1]) # this is the true value taken from Mars Sample
    ρ_interp[i] = x_interp* (ρ_real[index_alt]-ρ_real[index_alt-1]) + ρ_real[index_alt-1] # this is the true value taken from Mars Sample
    NW_interp[i] = x_interp* (NW_real[index_alt]-NW_real[index_alt-1]) + NW_real[index_alt-1] # this is the true value taken from Mars Sample
    EW_interp[i] = x_interp* (EW_real[index_alt]-EW_real[index_alt-1]) + EW_real[index_alt-1] # this is the true value taken from Mars Sample

    # define correction factors
    global kρ, kew, knw
    knw[i] = sin(NW_interp[i]/NW_spline[i])
    kew[i] = sin(EW_interp[i]/EW_spline[i])
    kρ[i] = (ρ_interp[i]/ρ_spline[i])-1
    # exponential moving average
    α = 0.03 # if just initialize set this to 0 or close to 0
    if estimate_wind == 1
        if i == 1
            X[i] = [X[i][1:7];kρ[i];kew[i];knw[i]]
        else
            X[i] = [X[i][1:7];(1-α)*X[i][8]+(α)*kρ[i];(1-α)*X[i][9]+(α)*kew[i];(1-α)*X[i][10]+(α)*knw[i]]
        end
    else
        if i == 1
            X[i] = [X[i][1:7];kρ[i]]
        else
            X[i] = [X[i][1:7];(1-α)*X[i][8]+(α)*kρ[i]]
        end
    end


    if i == 1
        if estimate_wind == 0
            μ[1] = [μ[1][1:7]; kρ[i] + 0.1*randn()]
        else
            μ[1] = [μ[1][1:7]; kρ[i] + 0.1*randn(); kew[i] + 0.1*randn(); knw[i] + 0.1*randn()]
        end
    end
    if estimate_wind == 0
        X[i+1] = CPEG.rk4_est(model,X[i],U[i],dt)
        Y[i+1] = CPEG.measurement(model,X[i+1]) + kf_sys.ΓR*randn(7)
        μ[i+1], F[i+1] = CPEG.sqrkalman_filter(model, μ[i],F[i],U[i],Y[i+1],kf_sys)
    else
        X[i+1] = CPEG.rk4_est_wind(model,X[i],U[i],dt)
        Y[i+1] = CPEG.measurement_wind(model,X[i+1]) + kf_sys.ΓR*randn(7)
        μ[i+1], F[i+1] = CPEG.sqrkalman_filter_wind(model, μ[i],F[i],U[i],Y[i+1],kf_sys)
    end

    σ[i+1] = F[i+1]'*F[i+1]
    σm_ρ[i+1] = sqrt(σ[i+1][8,8])
    if estimate_wind ==1
        σm_ew[i+1] = sqrt(σ[i+1][9,9])
        σm_nw[i+1] = sqrt(σ[i+1][10,10])
    end
    #break if altitude less than 10 km
    x = deepcopy(X[i+1])
    r = CPEG.unscale_rv(ev.scale,x[SA[1,2,3]],x[SA[4,5,6]])[1]
    # alt = CPEG.altitude(ev.params.gravity, r)[1]
    alt,lat,lon = CPEG.altitude(ev.params.gravity, r)
    if i%1000== 1
        println("alt ",alt," - lat ",lat," - lon ",lon)
    end

    global idx_trn
    idx_trn += 1
    if alt <= 1e4
        alt = CPEG.postprocess_es(model,[X[i+1]],x0)[1]
        ρ_spline[i+1] = CPEG.density_spline(model.params.density, alt[1])
        break
    end
end
# exit()
# truncate results
t_vec = (0:dt:((idx_trn-1)*dt))*ev.scale.tscale#*3600
X  = X[1:idx_trn]
U = U[1:idx_trn]
Y = Y[1:idx_trn]
μ = μ[1:idx_trn]
F = F[1:idx_trn]
σm_ρ = σm_ρ[1:idx_trn]
if estimate_wind ==1
    σm_ew = σm_ew[1:idx_trn]
    σm_nw = σm_ew[1:idx_trn]
end
ρ_spline = ρ_spline[1:idx_trn]
ρ_interp = ρ_interp[1:idx_trn]
NW_spline = NW_spline[1:idx_trn]
NW_interp = NW_interp[1:idx_trn]
EW_spline = EW_spline[1:idx_trn]
EW_interp = EW_interp[1:idx_trn]
kρ = kρ[1:idx_trn]
kew = kew[1:idx_trn]
knw = knw[1:idx_trn]
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

mat"
figure
hold on
p1 = plot($alt_k/1e3,$μm(8,:)','b');
p2= plot($alt_k/1e3,$μm(8,:)' + 3*$σm_ρ,'r--');
plot($alt_k/1e3,$μm(8,:)' - 3*$σm_ρ,'r--')
p3 = plot($alt/1e3,$kρ','color',[0.9290, 0.6940, 0.1250]);
p4 = plot($alt/1e3,$Xm(8,:)','color','k');
title('Atmospheric Correction Factor')
legend([p1;p2;p3;p4],'SREKF krho','3 sigma bounds','True krho','Exp. Moving Average','location','southeast')
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
p3 = plot($t_vec,$kρ,'color',[0.9290, 0.6940, 0.1250]);
p4 = plot($t_vec,$Xm(8,:)','color','k');
title('Atmospheric Correction Factor')
legend([p1;p2;p3;p4],'SREKF krho','3 sigma bounds','True krho','Exp. Moving Average','location','southeast')
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

if estimate_wind == 1
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
end
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
if estimate_wind == 1
    mat"
    figure
    hold on
    p1 = plot($alt_k/1e3,$μm(9,:)','b');
    p2= plot($alt_k/1e3,$μm(9,:)' + 3*$σm_ew,'r--');
    plot($alt_k/1e3,$μm(9,:)' - 3*$σm_ew,'r--')
    p3 = plot($alt/1e3,$kew,'color',[0.9290, 0.6940, 0.1250]);
    p4 = plot($alt/1e3,$Xm(9,:)','color','k');
    title('East Wind Correction Factor')
    legend([p1;p2;p3;p4],'SREKF kwE','3 sigma bounds','True kwE','Exp. Moving Average','location','southeast')
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
    p3 = plot($t_vec,$kew,'color',[0.9290, 0.6940, 0.1250]);
    p4 = plot($t_vec,$Xm(9,:)','color','k');
    title('East Wind Correction Factor')
    legend([p1;p2;p3;p4],'SREKF kwE','3 sigma bounds','True kwE','Exp. Moving Average','location','southeast')
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
    p3 = plot($alt/1e3,$knw,'color',[0.9290, 0.6940, 0.1250]);
    p4 = plot($alt/1e3,$Xm(10,:)','color','k');
    title('North Wind Correction Factor')
    legend([p1;p2;p3;p4],'SREKF kwN','3 sigma bounds','True kwN','Exp. Moving Average','location','southeast')
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
    p3 = plot($t_vec,$knw,'color',[0.9290, 0.6940, 0.1250]);
    p4 = plot($t_vec,$Xm(10,:)','color','k');
    title('North Wind Correction Factor')
    legend([p1;p2;p3;p4],'SREKF kwN','3 sigma bounds','True kwN','Exp. Moving Average','location','southeast')
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
end
# @test a1 ≈ (a1_central + a1_j2) rtol = 1e-12
