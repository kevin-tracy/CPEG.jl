using LinearAlgebra
using MATLAB
using BenchmarkTools
using StaticArrays
using LegendrePolynomials

function legendre(x::T, θ::SVector{N,T}) where {N,T}

    P = SA[1,x]
    for n = 3:N
        Pnew = ((2*n - 1)/n)*x*P[n-1] - ((n-1)/n)*P[n-2]
        # Pnew = (1/(n + 1))*((2*n + 1)*x*P[n-1] + n*P[n-2])
        P = [P; Pnew]
        # push!(P,Pnew)
    end
    P
end

let

    θ = @SVector randn(5)

    x = 0.32
    P = legendre(x,θ)

    # @show P

    mat"$P2=legendreP(0:7,$x)'"

    @show P2

    P3 = [1,x,.5*(3*x^2-1),0.5*(5*x^3 - 3*x),(1/8)*(35*x^4 - 30*x^2 + 3),
    (1/8)*(63*x^5 - 70*x^3 + 15*x),
    (1/16)*(231*x^6 - 315*x^4 + 105*x^2 - 5),
    (1/16)*(429*x^7 - 693*x^5 + 315*x^3 - 35*x),

    ]
    @show P3

    P4 = collectPl(x, lmax = 7)

    # x = zeros(8)
    # @btime $x .= collectPl($x, lmax = 7)

    @btime Pl($x,7)
    # Pl(x,3)
    @show P4
end
