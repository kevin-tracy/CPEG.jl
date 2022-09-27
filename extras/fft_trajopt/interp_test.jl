
using DelimitedFiles
# using BSplineKit
# using Interpolations
using MATLAB
p = "/Users/kevintracy/.julia/dev/CPEG/extras/fft_trajopt/atmo_samples/samp1.csv"

TT = readdlm(p, ',')

alt = Vector{Float64}(TT[2:end,2])
density = Vector{Float64}(TT[2:end,end])

# itp = interpolate(alt, density, BSplineOrder(4))


des_alts = Vector{Float64}(5:5:125)

mat"
$spl = spline($alt,$density,$des_alts)
disp('done')
"
@show "done"

mat"
figure
hold on
plot($alt, log($density))
plot($des_alts, log($spl))
hold off
"
