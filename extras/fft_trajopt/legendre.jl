using LinearAlgebra
using MATLAB
using BenchmarkTools
using StaticArrays
using LegendrePolynomials

let
    vs = [5846.951772540256, 5839.732431897648, 5829.943457016204, 5816.1542528040945, 5797.114404238737, 5771.954819769013, 5739.629357355113, 5698.497668636788, 5646.716569886262, 5583.13043553353, 5507.400025179468, 5419.246573878202, 5317.520841413681, 5200.402656633509, 5065.942670295066, 4912.64005192082, 4739.879056197717, 4548.705311547184, 4342.769099819083, 4128.162000960868, 3912.210426828762, 3701.3506946327757, 3499.778522550993, 3309.6929628235885, 3131.915735663983, 2966.4584131394836, 2812.9101317294585, 2670.661093230561, 2539.0224597243073, 2417.285915143196, 2304.7444649945533, 2200.7051591069203, 2104.499012558687, 2015.488298123299, 1933.0715267128835, 1856.6864598619793, 1785.8114940057676, 1719.9657443460771, 1658.708134946914, 1601.6357701799877, 1548.3818233398401, 1498.61313558042, 1452.0276765098984, 1408.3519798415914, 1367.3386350825454, 1328.763889859789, 1292.4253968854662, 1258.140124069375, 1225.742435052293, 1195.082339597538, 1166.023908062617, 1138.443840916189, 1112.2301824330605, 1087.2811668730562, 1063.5041853106225, 1040.8148615950752, 1019.1362265156262, 998.3979799989559, 978.5358319963556, 959.4909135651102, 941.2092504781725, 923.6412924842911, 906.7414920746339, 890.4679272854048, 874.7819636771876, 859.6479511820725, 845.0329520019068, 830.9064961793466, 817.2403618522053, 804.0083775455813, 791.1862441598472, 778.7513745802382, 766.6827490693399, 754.9607848110912, 743.5672181571346, 732.4849982865979, 721.6981911312305, 711.1918925416657, 700.9521497794818, 690.965890515575, 681.2208585997021, 671.7055559403763, 662.409189899818, 653.3216256664864, 644.4333431188459, 635.7353977392239, 627.2193851767508, 618.8774090940075, 610.7020519637704, 602.6863485106521, 594.8237615179225, 587.1081597427925, 579.5337977042713, 572.0952971267185, 564.7876298396591, 557.6061019505504, 550.5463391222246, 543.6042728008151, 536.7761272533106, 530.0584072865661, 523.4478865317415, 516.9415961898435, 510.53681414534196, 504.23105436579107, 498.0220545424544, 491.90769753733997, 485.8858452869654, 479.9542931085429, 474.1107619949491, 468.3528931042239, 462.6782440371176, 457.08428687432615, 451.56840794628965, 446.12790930921443, 440.76001190106797, 435.46186035039034, 430.23052940864244, 425.063031973172, 419.95632866251765, 414.90733889846507, 409.9129534398353, 404.97004830133926, 400.07549997689375, 395.2262018706241, 390.4252242955695]
    σ̇ = [4.335293949091313, 4.202983590126629, 4.022397490909791, 3.775980957315257, 3.4534847652159097, 3.0499602002442665, 2.557887138662824, 1.9676338174841084, 1.276457804991852, 0.49333325310175546, -0.3671882144823324, -1.2959042075561857, -2.289889561589978, -3.344698786164477, -4.444328787373314, -5.550397592591635, -6.590153744125464, -7.449908654151643, -7.997324036837147, -8.140798221190009, -7.8849815688646725, -7.324807736646528, -6.587640172905432, -5.7828607141666, -4.98419302616279, -4.233691795568228, -3.5517441919182997, -2.9456641940521013, -2.4154138267376393, -1.957016362639381, -1.5645209307774819, -1.2311523929030717, -0.949985990767196, -0.7143399673627385, -0.5179957262514757, -0.3553092187855251, -0.22125249641542793, -0.11141064604750084, -0.02195118888671945, 0.05042224267706954, 0.10852380350377716, 0.1547414264802026, 0.19109137415004765, 0.2192692324374052, 0.2406962233718439, 0.2565603907476843, 0.26785265177294815, 0.27539797505016494, 0.27988209216039195, 0.2818742160356951, 0.2818462530791802, 0.280188977940215, 0.2772256040286099, 0.27322313880260324, 0.26840186614462025, 0.2629432527717607, 0.25699653337449024, 0.2506841910599755, 0.2441065161636708, 0.23734539732881152, 0.23046747375991725, 0.2235267563340779, 0.21656680729180555, 0.2096225531721288, 0.20272179303535545, 0.1958864535134151, 0.18913363345376172, 0.18247647366567024, 0.1759248812141825, 0.1694861327281757, 0.16316537701152126, 0.15696605381811055, 0.15089024279336352, 0.14493895421759137, 0.13911237122702294, 0.13341005155094895, 0.12783109545536134, 0.1223742854535091, 0.11703820239908813, 0.11182132180430975, 0.10672209356103073, 0.10173900769905003, 0.09687064835991765, 0.09211573776291687, 0.0874731716353434, 0.08294204728657908, 0.07852168528108426, 0.07421164546294376, 0.07001173792512742, 0.06592202936329766, 0.06194284513896481, 0.05807476727759173, 0.05431862853258513, 0.050675502580621555, 0.047146690354755084, 0.04373370246876476, 0.04043823766003339, 0.037262157146229484, 0.03420745477258788, 0.03127622283185886, 0.02847061342889698, 0.025792795287772816, 0.023244905897399363, 0.020828999021861753, 0.018546980127823867, 0.016400519430885187, 0.01439098594470087, 0.01251941334730891, 0.010786462938918957, 0.009192380276628888, 0.007736944828259589, 0.006419411947891801, 0.005238446412617293, 0.0041920466798931145, 0.003277458967231785, 0.0024910801963858135, 0.0018283488057019295, 0.0012836224024697271, 0.0008500412245755172, 0.000519376391836158, 0.0002818619739912432, 0.0001260099758675225, 3.840745047073849e-5, 3.495097056301539e-6]

    # mat"
    # figure
    # hold on
    # plot($vs(1:end-1), $σ̇)
    # hold off"

    v0 = vs[1]*1.00001
    vt = vs[end-1]*0.9999
    @show v0
    @show vt
    a = 2/(vt - v0)
    b = 1 - vt*a
    vs_scaled = [(vs[i]*a + b) for i = 1:(length(vs)-1)]

    # mat"
    # figure
    # hold on
    # plot($vs_scaled, $σ̇)
    # hold off"

    # now we do some least squares business
    l = 20
    n = length(σ̇)
    θ = zeros(l)
    A = zeros(n,l)
    for i = 1:n
        # row =  collectPl(vs_scaled[i], lmax = (length(θ)-1))'
        # @show row
        A[i,:] = collectPl(vs_scaled[i], lmax = (length(θ)-1))'
    end
    θ .= A\σ̇

    σ̇_approx = A*θ
    mat"
    figure
    hold on
    plot($vs_scaled, $σ̇,'o')
    plot($vs_scaled, $σ̇_approx)
    legend('true','approx')
    hold off"

    @show θ

    @show norm(σ̇_approx - σ̇)
end

# function legendre(x::T, θ::SVector{N,T}) where {N,T}
#
#     P = SA[1,x]
#     for n = 3:N
#         Pnew = ((2*n - 1)/n)*x*P[n-1] - ((n-1)/n)*P[n-2]
#         # Pnew = (1/(n + 1))*((2*n + 1)*x*P[n-1] + n*P[n-2])
#         P = [P; Pnew]
#         # push!(P,Pnew)
#     end
#     P
# end
#
# let
#
#     θ = @SVector randn(5)
#
#     x = 0.32
#     P = legendre(x,θ)
#
#     # @show P
#
#     mat"$P2=legendreP(0:7,$x)'"
#
#     @show P2
#
#     P3 = [1,x,.5*(3*x^2-1),0.5*(5*x^3 - 3*x),(1/8)*(35*x^4 - 30*x^2 + 3),
#     (1/8)*(63*x^5 - 70*x^3 + 15*x),
#     (1/16)*(231*x^6 - 315*x^4 + 105*x^2 - 5),
#     (1/16)*(429*x^7 - 693*x^5 + 315*x^3 - 35*x),
#
#     ]
#     @show P3
#
#     P4 = collectPl(x, lmax = 7)
#
#     # x = zeros(8)
#     # @btime $x .= collectPl($x, lmax = 7)
#
#     @btime Pl($x,7)
#     # Pl(x,3)
#     @show P4
# end
