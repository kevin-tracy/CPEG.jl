include(joinpath(@__DIR__,"dynamics.jl"))
include(joinpath(@__DIR__,"post_process.jl"))
include(joinpath(@__DIR__,"srekf.jl"))


Random.seed!(1234)

ev = CPEG.CPEGWorkspace()

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

x0 = [r0/dscale;v0/(dscale/tscale); σ0;kρ]

# first rollout
dt = 0.2/3600/tscale
N = 500
t_vec = (0:dt:((N-1)*dt))*3600
X = NaN*[zeros(8) for i = 1:N]
U = [zeros(1) for i = 1:N-1]
Y = [zeros(7) for i = 1:N]

X[1] = deepcopy(x0)
μ = deepcopy(X)
μ[1][8] = μ[1][8] + 0.1*randn()
F = [zeros(8,8) for i = 1:N]
Σ = (0.001*Matrix(float(I(8))))
Σ[8,8] = (1)^2
F[1] = chol(Matrix(Σ))

Q = diagm( [(.000005)^2*ones(3)/dscale; .000005^2*ones(3)/(dscale/tscale); (1e-10)^2;(1e-10)^2])
R = diagm( [(.1)^2*ones(3)/dscale; (0.0002)^2*ones(3)/(dscale/tscale);1e-10])

kf_sys = (dt = dt, ΓR = chol(R), ΓQ = chol(Q))

end_idx = NaN
for i = 1:(N-1)
    U[i] = [sin(i/10)/30]
    X[i+1] = rk4(model,X[i],U[i],dt)
    Y[i+1] = measurement(model,X[i+1]) + kf_sys.ΓR*randn(7)

    μ[i+1], F[i+1] = sqrkalman_filter(model, μ[i],F[i],U[i],Y[i+1],kf_sys)
end
