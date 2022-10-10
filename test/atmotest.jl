using MATLAB



# import CPEG
#
# p = CPEG.CPEGDensityParameters()
#
#
# # this data came from a marsgramm output
# hs = [5*1e3, 30e3, 50e3, 75e3, 100e3]
# ρs = [8.277E-03, 7.555E-04, 7.081E-05, 5.867E-06, 3.612E-07]
#
# for i = 1:length(hs)
#
#     @test log(ρs[i]) ≈ log(CPEG.density(p,hs[i])) rtol = 1e-2
#
# end

p = CPEG.CPEGDensityParameters()


# this data came from a marsgramm output
hs = [10:5:60; 62:2.:70; 71:1.:120]*1e3
# ρs = [8.277E-03, 7.555E-04, 7.081E-05, 5.867E-06, 3.612E-07]
density = zeros(length(hs))#[zeros(1) for i = 1:length(hs)]
# show(density)
for i = 1:length(hs)
    # show(density[i])
    density[i] = CPEG.density_spline(p,hs[i])
    # @test log(ρs[i]) ≈ log(CPEG.density(p,hs[i])) rtol = 1e-2

end

mat"
figure
hold on
title('Density')
plot($hs, $density)
ylabel('Density, kg/m^3')
xlabel('Altitude, km')
hold off
set(gca,'FontSize',14)
set(gca, 'YScale', 'log')
saveas(gcf,'plots/density.eps','epsc')
"
