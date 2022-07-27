using Random
using MATLAB
# include(joinpath(@__DIR__,"dynamics.jl"))
# # include(joinpath(@__DIR__,"post_process.jl"))
# include(joinpath(@__DIR__,"srekf.jl"))


Random.seed!(1234)

ev = CPEG.CPEGWorkspace()

model = ev
Rm = ev.params.gravity.Rp_e
r0 = SA[Rm+125e3, 0.0, 0.0] #Atmospheric interface at 125 km altitude
V0 = 5.845e3 #Mars-relative velocity at interface of 5.845 km/sec
γ0 = -15.474*(pi/180.0) #Flight path angle at interface
v0 = V0*SA[sin(γ0), cos(γ0), 0.0]
σ0 = deg2rad(42)
kρ = 1.74

ev.scale.dscale = 1.0e3
ev.scale.tscale = 1.0
ev.scale.uscale = 1.0

x0 = [r0/ev.scale.dscale;v0/(ev.scale.dscale/ev.scale.tscale); σ0;kρ]

# first rollout
dt = 0.2/3600/ev.scale.tscale
N = 2000
t_vec = (0:dt:((N-1)*dt))*3600
# X = NaN*[zeros(8) for i = 1:N]
# U_in = [zeros(1) for i = 1:N-1]
Y = [zeros(7) for i = 1:N]

X = [@SVector zeros(length(x0)) for i = 1:N]
U = [@SVector zeros(1) for i = 1:N]

# print(U)
X[1] = deepcopy(x0)
μ = deepcopy(X)
μ[1] = [μ[1][1:7]; μ[1][8] + 0.1*randn()]
F = [zeros(8,8) for i = 1:N]
Σ = (0.001*Matrix(float(I(8))))
Σ[8,8] = (1)^2
F[1] = CPEG.chol(Matrix(Σ))

Q = diagm( [(.000005)^2*ones(3)/ev.scale.dscale; .000005^2*ones(3)/(ev.scale.dscale/ev.scale.tscale); (1e-10)^2;(1e-10)^2])
R = diagm( [(.1)^2*ones(3)/ev.scale.dscale; (0.0002)^2*ones(3)/(ev.scale.dscale/ev.scale.tscale);1e-10])

kf_sys = (dt = dt, ΓR = CPEG.chol(R), ΓQ = CPEG.chol(Q))

end_idx = NaN
for i = 1:(N-1)
    U[i] = [sin(i/10)/30]
    X[i+1] = CPEG.rk4_est(model,X[i],U[i],dt)
    Y[i+1] = CPEG.measurement(model,X[i+1]) + kf_sys.ΓR*randn(7)
    μ[i+1], F[i+1] = CPEG.sqrkalman_filter(model, μ[i],F[i],U[i],Y[i+1],kf_sys)
end


# print(X)
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
alt, dr, cr = CPEG.postprocess_es(model,X,x0)

alt_k, dr_k, cr_k = CPEG.postprocess_es(model,μ,x0)

perr = 1e3*([ev.scale.dscale*norm(X[i][1:3] - μ[i][1:3]) for i = 1:N])
verr = 1e3*([(ev.scale.dscale/ev.scale.tscale)*norm(X[i][4:6] - μ[i][4:6]) for i = 1:N])

yperr = 1e3*([ev.scale.dscale*norm(X[i][1:3] - Y[i][1:3]) for i = 2:N])
yverr = 1e3*([(ev.scale.dscale/ev.scale.tscale)*norm(X[i][4:6] - Y[i][4:6]) for i = 2:N])
# @infiltrate
# error()
# mat"
# figure
# hold on
# plot($alt)
# plot($alt_k)
# hold off
# "
#
# mat"
# figure
# hold on
# plot($dr,$cr)
# plot($dr_k,$cr_k)
# xlabel('downrange')
# ylabel('crossrange')
# hold off
# "
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
saveas(gcf,'plots/perr.eps','epsc')
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
saveas(gcf,'plots/verr.eps','epsc')
"
μm = mat_from_vec(μ)
σm = zeros(length(μ))
# @infiltrate
# error()
for i = 1:length(μ)
    Σ = F[i]'*F[i]
    σm = sqrt(Σ[8,8])
end
mat"
figure
hold on
p1 = plot($alt_k/1e3,$μm(8,:)','b')
p2= plot($alt_k/1e3,$μm(8,:)' + 3*$σm,'r--')
plot($alt_k/1e3,$μm(8,:)' - 3*$σm,'r--')
p3 = plot($alt/1e3,$Xm(8,:)','color',[0.9290, 0.6940, 0.1250])
title('Atmospheric Correction Factor')
legend([p1;p2;p3],'SREKF krho','3 sigma bounds','True krho','location','southeast')
xlabel('Altitude, km')
ylabel('k rho')
ylim([1.7 1.77])
hold off
set(gca,'FontSize',14)
saveas(gcf,'plots/krho.eps','epsc')
"
# @test a1 ≈ (a1_central + a1_j2) rtol = 1e-12
