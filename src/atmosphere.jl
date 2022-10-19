
struct CPEGDensityParameters
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    e::Float64
    f::Float64
    g::Float64
    h::Float64
    i::Float64
    function CPEGDensityParameters()
        # default values from a marsgram sample
        p = [ -4.001833776317166
              -0.015696377977412
              -0.129412788692219
               0.000199820058253
               0.010377518080309
               0.000043652882189
              -0.000539682362508
              -0.000000205086106
               0.000000874980179]
        new(p...)
    end
end

function density(p::CPEGDensityParameters, h::T) where T
   h = h / 1000

   # clamp in a forward diff friendly way
   if h > 125.0
       h = one(h)*125.0
   elseif h < 0.2
       h = one(h)*0.2
   end
   num = @evalpoly(h, p.a, p.c, p.e, p.g, p.i)
   den = @evalpoly(h, 1.0, p.b, p.d, p.f, p.h)
   exp(num/den)
end

function wind(p::CPEGDensityParameters, h::T) where T
    h = h / 1000
    return [0.0, 0.0, 0.0]
end

function density_spline(p::CPEGDensityParameters, h::T) where T
   h = h / 1000

   #mean coefficients taken from spline where breaks are breaks = [10:5:60,62:2.:70,71:1.:120];
   coeff_a = [-1.85730963217633e-06
                5.63671194166011e-07
                -9.23854326127944e-07
                1.81373774887554e-07
                -1.06114739964355e-07
                -1.76851603918793e-07
                5.11294589789323e-08
                -6.90105898557052e-08
                1.92076685119273e-08
                -5.20516720253663e-08
                1.46602318132720e-07
                -1.96583893102288e-07
                8.32868772635157e-08
                -2.70293760024797e-08
                9.88876925185573e-08
                -6.77540403855885e-07
                6.83574806302507e-07
                -8.79679323492162e-08
                -2.20656877796438e-07
                1.13309216591852e-07
                1.61216545192674e-07
                -1.45082831650365e-07
                3.60361278137278e-08
                -2.77620653994685e-08
                2.08113190188334e-09
                -3.03450155831708e-08
                1.02620035127127e-08
                1.11332754570756e-08
                6.94346769244075e-09
                -2.45386822474158e-08
                9.76913967884423e-09
                4.81039642030914e-09
                8.43988718699577e-09
                2.22576077622844e-09
                7.12942864748372e-10
                -1.41430005990776e-08
                -7.65579322266507e-10
                7.13048327827783e-09
                -5.43753134977055e-09
                2.60487842848216e-09
                -1.05242604464062e-08
                6.04203045827173e-09
                8.18381527352796e-09
                -8.36460121687679e-09
                -2.12089892738481e-09
                6.90993114342309e-09
                -4.37117605828736e-09
                3.35426852280265e-10
                2.55707740092969e-09
                -2.10175731616554e-09
                1.65568885533589e-10
                -3.48016491726963e-10
                9.30130013034231e-10
                -1.52554498638333e-09
                1.21651990319952e-09
                -9.24688555784992e-11
                -4.76650182144385e-10
                6.24889193126397e-10
                -1.89958888138123e-10
                -5.24452837320624e-10
                1.00214148452797e-10
                1.13947453046397e-10
                8.04637497102590e-11
                -1.33754407803436e-10
                3.38632561519288e-10
                0]
    coeff_b = [3.54484471476105e-05
            7.58880266496562e-06
            1.60438705774558e-05
            2.18605568553662e-06
            4.90666230884993e-06
            3.31494120938461e-06
            6.62167150602710e-07
            1.42910903528669e-06
            3.93950187451117e-07
            6.82065215130026e-07
            -9.87098652504667e-08
            7.80904043545853e-07
            -3.98599315067877e-07
            1.01121948513218e-07
            -6.10543075016612e-08
            5.32271847609683e-07
            -1.50034936395797e-06
            5.50375054949550e-07
            2.86471257901901e-07
            -3.75499375487414e-07
            -3.55717257118584e-08
            4.48077909866164e-07
            1.28294149150701e-08
            1.20937798356253e-07
            3.76516021578478e-08
            4.38949978634978e-08
            -4.71400488860146e-08
            -1.63540383478765e-08
            1.70457880233504e-08
            3.78761911006726e-08
            -3.57398556415749e-08
            -6.43243660504225e-09
            7.99875265588512e-09
            3.33184142168724e-08
            3.99956965455577e-08
            4.21345251398028e-08
            -2.94476657430090e-10
            -2.59121462422962e-09
            1.88002352106039e-08
            2.48764116129220e-09
            1.03022764467387e-08
            -2.12705048924800e-08
            -3.14441351766479e-09
            2.14070323029191e-08
            -3.68677134771131e-09
            -1.00494681298658e-08
            1.06803253004035e-08
            -2.43320287445860e-09
            -1.42692231761780e-09
            6.24430988517127e-09
            -6.09620633253436e-11
            4.35744593275423e-10
            -6.08304881905466e-10
            2.18208515719723e-09
            -2.39454980195277e-09
            1.25500990764578e-09
            9.77603340910279e-10
            -4.52347205522876e-10
            1.42232037385632e-09
            8.52443709441947e-10
            -7.20914802519925e-10
            -4.20272357161534e-10
            -7.84299980223429e-11
            1.62961251108434e-10
            -2.38301972301875e-10
            0]
    coeff_c = [-0.000551292656877884
            -0.000336106407815003
            -0.000217943041602896
            -0.000126793410287934
            -9.13298203160016e-05
            -5.02218027248289e-05
            -3.03362609248923e-05
            -1.98798799954453e-05
            -1.07645838817563e-05
            -5.38450686885056e-06
            -2.46773011945276e-06
            -1.10334176286199e-06
            -3.38732305906035e-07
            -9.33687039015353e-07
            -8.53551756992239e-07
            8.88833232238064e-08
            -8.79194193124482e-07
            -1.82916850213290e-06
            -9.92322189281454e-07
            -1.08135030686697e-06
            -1.49242140806624e-06
            -1.07991522391193e-06
            -6.19007899130699e-07
            -4.85240685859376e-07
            -3.26651285345275e-07
            -2.45104685323929e-07
            -2.48349736346446e-07
            -3.11843823580337e-07
            -3.11152073904863e-07
            -2.56230094780840e-07
            -2.54093759321742e-07
            -2.96266051568360e-07
            -2.94699735517517e-07
            -2.53382568644759e-07
            -1.80068457882329e-07
            -9.79382361969685e-08
            -5.60981877145958e-08
            -5.89838789962555e-08
            -4.27748584098813e-08
            -2.14869820379852e-08
            -8.69706442995435e-09
            -1.96652928756957e-08
            -4.40802112858405e-08
            -2.58175925005862e-08
            -8.09733154537843e-09
            -2.18335710229555e-08
            -2.12027138524177e-08
            -1.29555914264728e-08
            -1.68157166185492e-08
            -1.19983290509958e-08
            -5.81498122914986e-09
            -5.44019869919978e-09
            -5.61275898782982e-09
            -4.03897871253806e-09
            -4.25144335729360e-09
            -5.39098325160060e-09
            -3.15837000304454e-09
            -2.63311386765714e-09
            -1.66314069932370e-09
            6.11623383974567e-10
            7.43152290896589e-10
            -3.98034868784870e-10
            -8.96737223968747e-10
            -8.12205970882656e-10
            -8.87546692076097e-10
            0]
    coeff_d = [0.00570116219576596
            0.00359874638604477
            0.00217839331286464
            0.00137429307852056
            0.000817649141080248
            0.000470402254725944
            0.000280060320846566
            0.000151324377359538
            7.90263795325156e-05
            3.74531733740031e-05
            2.10758104048302e-05
            1.69183292499845e-05
            1.62625907536257e-05
            1.46570238996502e-05
            1.29779026076526e-05
            1.18176834038099e-05
            1.17612981707875e-05
            1.00653294200076e-05
            8.69856804047498e-06
            7.77206023129899e-06
            6.42851976553646e-06
            5.06174317695104e-06
            4.28482303125491e-06
            3.71468067485300e-06
            3.32261572195041e-06
            3.03569717066487e-06
            2.80414246762127e-06
            2.51891468590152e-06
            2.20185009943038e-06
            1.91468728124131e-06
            1.67179469531373e-06
            1.39173022002925e-06
            1.09384212827616e-06
            8.15581032601525e-07
            5.97742638949866e-07
            4.58382820477843e-07
            3.88436108821600e-07
            3.31277865127308e-07
            2.76833254785100e-07
            2.47421100236052e-07
            2.31026637787841e-07
            2.22107589358220e-07
            1.87213822048316e-07
            1.48173012518338e-07
            1.35397851103794e-07
            1.21492849283320e-07
            9.65197412739216e-08
            8.16261766636200e-08
            6.65728092149688e-08
            5.08872476797315e-08
            4.30314711977414e-08
            3.73210967907998e-08
            3.19686261931485e-08
            2.66776923364475e-08
            2.32952537947233e-08
            1.78657805386764e-08
            1.36373383391431e-08
            1.09799214948645e-08
            8.51934961481086e-09
            8.08857040120535e-09
            9.02818465730124e-09
            9.15063629413070e-09
            8.44627652123069e-09
            7.55157304894986e-09
            6.76857392137221e-09
            0]

    breaks = [10:5:60; 62:2.:70; 71:1.:120]
    # show(breaks)
    if h >= 120
        h = 119.99
    elseif h < 10
        h = 10.0
    end
    for jj in range(1, length(breaks)-1)
        if (h < breaks[jj+1] && h >= breaks[jj])
            # show(h)
            return coeff_d[jj] + coeff_c[jj]*(h-breaks[jj]) +coeff_b[jj]*(h-breaks[jj])^2 +coeff_a[jj]*(h-breaks[jj])^3
        end
    end
    # println(h)
    # return 0.0
end


function wind_spline(p::CPEGDensityParameters, h::T) where T
   h = h / 1000

   #mean coefficients taken from spline where breaks are breaks = [10:5:60,62:2.:70,71:1.:120];
   EW_coeff_a = [-0.0354549285880261
            0.0486040302194431
            -0.0141907476382252
            -0.0236625460568965
            0.0410498561614765
            -0.0363480898352835
            -0.0119575893918287
            0.0788295871354854
            -0.167054075883303
            0.308570793520024
            -0.299913958910675
            0.141833632605584
            0.0603806437080357
            -0.469146018810774
            0.298983699730519
            -0.130320018337172
            -2.73241720667914
            2.77522656523175
            -0.0222119732036061
            0.0156947737366848
            -0.0253085936956157
            1.08903794760448
            -1.08218736014896
            0.00808966162103531
            0.00821554975719341
            -0.0234605444854195
            1.32774368782484
            -1.32341696100991
            0.0133651814387921
            0.00226546903231029
            -0.0106053063342183
            0.132230811465158
            -0.135733447741380
            0.0209901610178944
            -0.0253874486535697
            0.0438484312089180
            -1.78667984594558
            1.77578501577454
            -0.0206088060436893
            8.22785011864724e-05
            0.00961416696587847
            -0.0212473833167648
            0.0412028316782713
            -0.0820284808185738
            0.213555410055084
            0.422541748693198
            -0.678607797806183
            0.139032237461623
            -0.0593047979540082
            0.0184083193618215
            -0.0515644450569612
            0.0599768746368482
            -0.0127239804653407
            0.00581065782979129
            -0.00499147907614228
            -0.00284448113860891
            0.00641977428810292
            -0.000208960243368139
            -0.00318500957472589
            0.0104099189120915
            0.0173529549428424
            -0.0286332689610599
            0.00724260309748881
            -0.0104901778584017
            0.0225371445045151
            0.0485639667539219
            -0.0797629533544236
            0.0268817130768000
            -0.0236271844786795
            0.0421937580702216
            0.0779189829082493
            -0.128112380248717
            0.0325802744226682
            -0.0243375394925576
            0.0353316462372792
            0.0512261072072146
            -0.0862571011398666
            0.0150547758424663
            -0.00349017406518715
            -0.00542770934957426
            -0.0404343424682538
            0.0756221138153460]
    EW_coeff_b = [0.128752337379133
            -0.403071591441259
            0.325988861850388
            0.113127647277010
            -0.241810543576437
            0.373937298845708
            0.155848759834006
            0.0841032234830319
            0.557080746295944
            -0.445243709003870
            1.40618105211627
            -0.393302701347777
            0.457699094285732
            0.819982956533945
            -0.587455099898377
            0.309495999293182
            -0.0814640557183335
            -8.27871567575575
            0.0469640199395132
            -0.0196718996713057
            0.0274124215387488
            -0.0485133595480978
            3.21860048326533
            -0.0279615971815623
            -0.00369261231845464
            0.0209540369531247
            -0.0494275965031310
            3.93380346697139
            -0.0364474160583495
            0.00364812825802741
            0.0104445353549600
            -0.0213713836476944
            0.375321050747780
            -0.0318792924763596
            0.0310911905773228
            -0.0450711553833845
            0.0864741382433714
            -5.27356539959337
            0.0537896477302464
            -0.00803677040082462
            -0.00778993489726432
            0.0210525660003702
            -0.0426895839499260
            0.0809189110848871
            -0.165166531370834
            0.475499698794418
            1.74312494487401
            -0.292698448544537
            0.124398263840332
            -0.0535161300216929
            0.00170882806377171
            -0.152984507107112
            0.0269461168034325
            -0.0112258245925898
            0.00620614889678406
            -0.00876828833164267
            -0.0173017317474693
            0.00195759111683946
            0.00133071038673505
            -0.00822431833744286
            0.0230054383988314
            0.0750643032273586
            -0.0108355036558208
            0.0108923056366450
            -0.0205782279385600
            0.0470332055749854
            0.192725105836751
            -0.0465637542265198
            0.0340813850038801
            -0.0368001684321582
            0.0897811057785063
            0.323538054503254
            -0.0607990862428958
            0.0369417370251091
            -0.0360708814525636
            0.0699240572592742
            0.223602378880918
            -0.0351689245386817
            0.00999540298871704
            -0.000475119206844621
            -0.0167582472555683
            -0.138061274660330]
    EW_coeff_c = [-1.96297992791941
            -3.33457619823004
            -3.71998984618439
            -1.52440730054740
            -2.16782178204454
            -1.50718800569818
            -0.447615888338746
            0.0322880782953341
            1.31465601785329
            1.53833009243743
            3.46020477866223
            5.48596148019922
            5.61475426607513
            8.17011836771448
            8.40264622435005
            8.12468712374485
            8.35271906731970
            -0.00746066415437952
            -8.23921231997061
            -8.21192019970240
            -8.20417967783496
            -8.22528061584431
            -5.05519349212708
            -1.86455460604331
            -1.89620881554333
            -1.87894739090866
            -1.90742095045867
            1.97695492000960
            5.87431097092264
            5.84151168312232
            5.85560434673531
            5.84467749844257
            6.19862716554266
            6.54206892381408
            6.54128082191504
            6.52730085710898
            6.56870383996897
            1.38161257861897
            -3.83816317324415
            -3.79241029591473
            -3.80823700121282
            -3.79497437010971
            -3.81661138805927
            -3.77838206092431
            -3.86262968121025
            -3.55229651378667
            -1.33367187011824
            0.116754626211238
            -0.0515455584929676
            0.0193365753256712
            -0.0324707266322499
            -0.183746405675590
            -0.309784795979270
            -0.294064503768427
            -0.299084179464233
            -0.301646318899091
            -0.327716338978203
            -0.343060479608833
            -0.339772178105259
            -0.346665786055966
            -0.331884665994578
            -0.233814924368388
            -0.169586124796850
            -0.169529322816026
            -0.179215245117941
            -0.152760267481516
            0.0869980439302207
            0.233159395540452
            0.220677026317812
            0.217958242889534
            0.270939180235883
            0.684258340517643
            0.946997308778002
            0.923139959560215
            0.924010815132761
            0.957863990939471
            1.25139042707966
            1.43982388142190
            1.41465035987193
            1.42417064365381
            1.40693727719139
            1.25211775527550]
    EW_coeff_d = [11.2089383020535
            0.180981023431508
            -20.4931859763198
            -32.7172571157602
            -40.4689206936840
            -52.2220611731331
            -54.0314727078289
            -54.3989701603050
            -53.3673444126983
            -49.8461419988743
            -46.0818903018548
            -35.9360682073506
            -25.4026869914986
            -11.8593369325411
            -3.33838162710348
            4.77579319707871
            13.0796563017796
            18.6184941067018
            13.1075443320234
            4.89308405878872
            -3.32281326684830
            -11.5248891168401
            -18.7096451446281
            -21.6284255136388
            -23.5128520552426
            -25.4045379333472
            -27.2859918317882
            -27.9150966909251
            -23.3277552649540
            -17.4765265286510
            -11.6291012482383
            -5.77365767248225
            0.181879253777783
            6.62009402232684
            13.1512738146825
            19.6982583785213
            26.2243365114558
            31.0928346437225
            28.9766668385227
            25.1716845069651
            21.3713197191507
            17.5649069500065
            13.7697377625804
            9.95163962224948
            6.17214799159148
            2.35790718906548
            -0.296347877233571
            -0.565502600283978
            -0.602414185155655
            -0.588866277762299
            -0.604637513096499
            -0.686963856721938
            -0.963717894867792
            -1.25928055450897
            -1.55876022504020
            -1.85662973468379
            -2.16988882305313
            -2.50848711949070
            -2.84979896822606
            -3.19142544551931
            -3.53590563100063
            -3.82743190365353
            -4.01481579375562
            -4.18799481911080
            -4.35712201414859
            -4.53437834270057
            -4.59154143785318
            -4.39158124144063
            -4.17810388704990
            -3.94697266020689
            -3.72362082767929
            -3.28498155875665
            -2.40529754398447
            -1.48651904702670
            -0.550774889933929
            0.372496689983547
            1.45151084538951
            2.84024655021022
            4.25995628293591
            5.68111187173137
            7.09937968682876
            8.44912437429633]

    NW_coeff_a = [-0.00978550412653108
                0.0155164069949459
                -0.00625236517485601
                -0.00730007157592896
                0.0180055173975034
                0.0155457629448077
                -0.136667579671069
                0.00708827884700691
                0.139335519047827
                -0.268072756289426
                0.259436542046510
                -0.105951089405064
                -0.0319572114113139
                0.297614367806318
                -0.199730338191551
                0.110117341828799
                0.343170760447865
                -0.367378601016162
                -0.0252988874556355
                0.0303330320720718
                -0.0452459298930013
                1.83284842396475
                -1.82833177740320
                0.0377988963538432
                -0.0291505480560508
                0.0328113257069496
                -1.09855998026203
                1.08986266324768
                -0.0126698794893321
                0.00780994058117521
                -0.0143311029812377
                0.469235698193027
                -0.468532974766259
                0.0166500624649526
                -0.0183502911119438
                0.0298181610805974
                -1.24929744000824
                1.24581085214542
                -0.0217837530851304
                0.00514385166985143
                0.00886854769659706
                -0.0219765781312461
                0.0393857703733955
                -0.0733877263859444
                0.182067620105006
                0.355632533523485
                -0.574366836471508
                0.124369154109432
                -0.0648938771320280
                0.0545862524606726
                0.0304766123394193
                -0.0689141482471063
                0.0158278659976765
                -0.0146097478450998
                0.0270872531847745
                0.0526816748973984
                -0.0841154167885572
                0.0189988987900221
                -0.0126693145629034
                0.0181872332979757
                0.0288001917773757
                -0.0492016269113703
                0.0149668391354760
                -0.0122088974148299
                0.0156504693165462
                0.0192339941142480
                -0.0328833087608795
                0.00668775287361290
                -0.00669886700400113
                0.00708069243457743
                -0.0126644900633210
                0.0136512895993066
                -0.0101753438848671
                0.0138035677183250
                -0.0304183922364309
                -0.0615979410567231
                0.100073942625904
                -0.0249790099014300
                0.0172532722479866
                -0.0350124222050319
                -0.104930793041082
                0.203230656136642]
    NW_coeff_b = [0.0602818250083182
                -0.0865007368896480
                0.146245368034540
                0.0524598904117003
                -0.0570411832272341
                0.213041577735317
                0.306316155404163
                -0.513689322622251
                -0.471159649540210
                0.364853464746754
                -1.24358307298980
                0.313036179289253
                -0.322670357141134
                -0.514413625609016
                0.378429477809940
                -0.220761536764712
                0.109590488721684
                1.13910277006528
                0.0369669670167987
                -0.0389296953501059
                0.0520694008661060
                -0.0836683888128995
                5.41487688308134
                -0.0701184491282643
                0.0432782399332670
                -0.0441734042348836
                0.0542605728859651
                -3.24141936790013
                0.0281686218428950
                -0.00984101662510120
                0.0135888051184245
                -0.0294045038252886
                1.37830259075379
                -0.0272963335449852
                0.0226538538498735
                -0.0323970194859573
                0.0570574637558341
                -3.69083485626889
                0.0465977001673581
                -0.0187535590880330
                -0.00332200407847960
                0.0232836390113116
                -0.0426460953824266
                0.0755112157377600
                -0.144651963420073
                0.401550896894945
                1.46844849746540
                -0.254652011949125
                0.118455450379172
                -0.0762261810169118
                0.0875325763651063
                0.178962413383364
                -0.0277800313579548
                0.0197035666350738
                -0.0241256769002258
                0.0571360826540976
                0.215181107346293
                -0.0371651430193780
                0.0198315533506888
                -0.0181763903380214
                0.0363853095559057
                0.122785884888033
                -0.0248189958460777
                0.0200815215603504
                -0.0165451706841393
                0.0304062372654998
                0.0881082196082437
                -0.0105417066743945
                0.00952155194644488
                -0.0105750490655590
                0.0106670282381733
                -0.0273264419517902
                0.0136274268461305
                -0.0168986048084694
                0.0245120983465048
                -0.0667430783627889
                -0.251536901532957
                0.0486849263447535
                -0.0262521033595373
                0.0255077133844228
                -0.0795295532306728
                -0.394321932353921]
    NW_coeff_c = [-0.358472613146661
                -0.489567172553311
                -0.190844016828849
                0.802682275402354
                0.779775811324685
                1.55977778386510
                2.59849325014406
                2.18374691570788
                0.214048971382964
                0.00143660179605298
                -1.75602261469005
                -3.61711640209115
                -3.63638475779491
                -5.31055272329521
                -5.44653687109429
                -5.28886893004906
                -5.40003997809208
                -4.15134671930512
                -2.97527698222303
                -2.97723971055634
                -2.96410000504034
                -2.99569899298713
                2.33550950128131
                7.68026793523439
                7.65342772603940
                7.65253256173778
                7.66261973038886
                4.47546093537470
                1.26221018931746
                1.28053779453525
                1.28428558302858
                1.26846988432171
                2.61736797125022
                3.96837422845902
                3.96373174876391
                3.95398858312783
                3.97864902739771
                0.344871634884649
                -3.29936552121688
                -3.27152138013756
                -3.29359694330407
                -3.27363530837124
                -3.29299776474235
                -3.26013264438702
                -3.32927339206933
                -3.07237445859446
                -1.20237506423412
                0.0114214212821553
                -0.124775140287798
                -0.0825458709255384
                -0.0712394755773440
                0.195255514171127
                0.346437896196536
                0.338361431473655
                0.333939321208503
                0.366949726962375
                0.639266916962765
                0.817282881289680
                0.799949291620991
                0.801604454633659
                0.819813373851543
                0.978984568295481
                1.07695145733744
                1.07221398305171
                1.07575033392792
                1.08961140050928
                1.20812585738302
                1.28569237031687
                1.28467221558892
                1.28361871846981
                1.28371069764242
                1.26705128392881
                1.25335226882315
                1.25008109086081
                1.25769458439884
                1.21546360438256
                0.897183624486816
                0.694331649298613
                0.716764472283829
                0.716020082308715
                0.661998242462465
                0.188146756877872]
    NW_coeff_d = [-1.70400812852056
                -3.21251358486230
                -5.88331699550182
                -3.96294852563956
                0.449451164673600
                5.17299031530410
                9.26907829753403
                14.5979887820702
                16.9674315537731
                16.6255750507608
                15.9432800630245
                9.53239487805726
                2.70269807579146
                -6.11641055965341
                -11.6437625407513
                -16.9116002722272
                -22.3111133972122
                -27.2583921261347
                -30.6380146763907
                -33.6016235790526
                -36.5874599528869
                -39.5447364869542
                -40.7912554447895
                -34.8692008378300
                -27.2212524553700
                -19.5536970374534
                -11.9125265542436
                -5.29420623123079
                -2.97030200050855
                -1.69259306883753
                -0.414086350346197
                0.869456934819566
                2.57775801350902
                6.10489560074677
                10.0626235581258
                14.0306588696276
                17.9820685943501
                20.7684776454954
                18.6683252762565
                15.3937737021219
                12.1086426145661
                8.82059221488019
                5.54826396738901
                2.25200587763763
                -1.00600327739758
                -4.29786101278198
                -6.61305204095801
                -6.92134544419824
                -7.04020688075578
                -7.11142044779644
                -7.21560624727821
                -7.16883653415103
                -6.86353275484365
                -6.52904702400739
                -6.18559177374376
                -5.84869087625071
                -5.37192339173684
                -4.60159078421634
                -3.80247414715601
                -2.99536261674723
                -2.19374731915362
                -1.30874844396880
                -0.256179617696654
                0.810919682930182
                1.89100629012741
                2.96586192268774
                4.10511355457677
                5.36846432280716
                6.65030273932325
                7.93779763985462
                9.21792200169344
                10.4996352375107
                11.7530113690870
                13.0098157208715
                14.2568017746421
                15.5085900651510
                16.5957126501141
                17.3414333156939
                18.0594708814358
                18.7672365226081
                19.4737518960962
                19.9512897922869]

    breaks = [10:5:35; 37:2.:50; 51:1.:120]
    # show(breaks)
    if h >= 120
        h = 119.99
    elseif h < 10
        h = 10.0
    end
    for jj in range(1, length(breaks)-1)
        if (h < breaks[jj+1] && h >= breaks[jj])
            # show(h)
            EW = EW_coeff_d[jj] + EW_coeff_c[jj]*(h-breaks[jj]) +EW_coeff_b[jj]*(h-breaks[jj])^2 +EW_coeff_a[jj]*(h-breaks[jj])^3
            NW = NW_coeff_d[jj] + NW_coeff_c[jj]*(h-breaks[jj]) +NW_coeff_b[jj]*(h-breaks[jj])^2 +NW_coeff_a[jj]*(h-breaks[jj])^3
            return [EW, NW, 0.0]
        end
    end
    # println(h)
    # return 0.0
end
