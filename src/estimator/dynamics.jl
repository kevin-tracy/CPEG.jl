function esdensity(model,h,H)
    ρ0 = 5.25*1e-7
    ρ = ρ0*exp(-(h)/H)
    return ρ
end

function esdensity_spline(model,h)#::T) where T
   h = h / 1000

   # print(h)

   #mean coefficients taken from spline where breaks are breaks = [10:5:60,62:2.:70,71:1.:120];
   coeff_a = [-1.92328741603145e-06
               5.70233021940267e-07
               -9.17634382734206e-07
               1.67046258151837e-07
               -9.60410379784292e-08
               -1.81809912234820e-07
               5.11537551403809e-08
               -6.81539714546325e-08
               1.80097174994589e-08
               -5.05516359272255e-08
               1.38531963939899e-07
               -1.83695690335794e-07
               7.13797067291749e-08
               -2.16631049351496e-08
               9.23302684147096e-08
               -5.89008290453308e-07
               4.98544463788549e-07
               7.40865309407130e-08
               -2.71018743353236e-07
               9.31948854820336e-08
               1.64797562968432e-07
               -1.07482645644253e-07
               -8.49300539147749e-09
               6.15462595873603e-09
               -2.18873639382704e-08
               -1.16441154512268e-08
               -5.75636723797975e-09
               2.05888710662881e-08
               1.05612819522476e-09
               -1.64251652604314e-08
               6.45502558359313e-10
               1.08075846320204e-08
               4.40599598450801e-09
               5.33820713489058e-09
               -1.13138875343691e-09
               -1.11764466787285e-08
               -3.75714873386387e-09
               7.43674333277705e-09
               -3.72271651350473e-09
               2.57269134125480e-10
               -7.26069727181459e-09
               2.23737679537661e-09
               9.79234163687156e-09
               -7.20616671467939e-09
               -3.30938137344359e-09
               6.24497992365985e-09
               -2.54844100784435e-09
               -1.18517353969706e-09
               3.01282134446159e-09
               -1.70146102118281e-09
               -3.96117357150445e-10
               1.02809818331298e-10
               4.34515119912689e-10
               -8.70467948013050e-10
               4.76671477029097e-10
               4.81392756075857e-10
               -7.35781440946678e-10
               6.27706734918959e-10
               -1.04925183199405e-10
               -5.13921202665040e-10
               8.91815914556842e-12
               1.76987380763810e-10
               2.32085782723335e-11
               -2.70136194157839e-11
                  8.50068840616483e-11]
    coeff_b = [3.63834917838628e-05
              7.53418054339119e-06
              1.60876758724952e-05
              2.32316013148208e-06
              4.82885400375964e-06
              3.38823843408320e-06
              6.61089750560905e-07
              1.42839607766662e-06
              4.06086505847134e-07
              6.76232268339018e-07
              -8.20422705693636e-08
              7.49149513070029e-07
              -3.53024628944734e-07
              7.52536114303157e-08
              -5.47250181805823e-08
              4.99256592307676e-07
              -1.26776827905225e-06
              2.27865112313400e-07
              4.50124705135538e-07
              -3.62931524924171e-07
              -8.33468684780695e-08
              4.11045820427227e-07
              8.85978834944662e-08
              6.31188673200334e-08
              8.15827451962414e-08
              1.59206533814306e-08
              -1.90116929722503e-08
              -3.62807946861896e-08
              2.54858185126746e-08
              2.86542030983488e-08
              -2.06212926829454e-08
              -1.86847850078676e-08
              1.37379688881936e-08
              2.69559568417176e-08
              4.29705782463893e-08
              3.95764119860785e-08
              6.04707194989297e-09
              -5.22437425169862e-09
              1.70858557466325e-08
              5.91770620611833e-09
              6.68951360849474e-09
              -1.50925782069490e-08
              -8.38044782081920e-09
              2.09965770897955e-08
              -6.21923054242695e-10
              -1.05500671745735e-08
              8.18487259640609e-09
              5.39549572873028e-10
              -3.01597104621815e-09
              6.02249298716664e-09
              9.18109923618209e-10
              -2.70242147833126e-10
              3.81873071607671e-11
              1.34173266689883e-09
              -1.26967117714032e-09
              1.60343253946976e-10
              1.60452152217455e-09
              -6.02822800665489e-10
              1.28029740409139e-09
              9.65521854493172e-10
              -5.76241753501948e-10
              -5.49487276065244e-10
              -1.85251337738150e-11
              5.11006010431853e-11
              -2.99402572041671e-11]
       coeff_c = [-0.000557324802363520
                  -0.000337736440727249
                  -0.000219627158647817
                  -0.000127572978627931
                  -9.18129079517223e-05
                  -5.07274457625080e-05
                  -3.04808048392875e-05
                  -2.00333756981498e-05
                  -1.08609627805811e-05
                  -5.44936890965032e-06
                  -2.47841892080205e-06
                  -1.14420443580072e-06
                  -3.51954667550125e-07
                  -9.07496702578961e-07
                  -8.66439516079492e-07
                  2.26236321746961e-08
                  -7.45888054569875e-07
                  -1.78579122130872e-06
                  -1.10780140385978e-06
                  -1.02060822364841e-06
                  -1.46688661705065e-06
                  -1.13918766510150e-06
                  -6.39543961179805e-07
                  -4.87827210365305e-07
                  -3.43125597849030e-07
                  -2.45622199271358e-07
                  -2.48713238862178e-07
                  -3.04005726520618e-07
                  -3.14800702694133e-07
                  -2.60660681083110e-07
                  -2.52627770667706e-07
                  -2.91933848358519e-07
                  -2.96880664478193e-07
                  -2.56186738748282e-07
                  -1.86260203660175e-07
                  -1.03713213427707e-07
                  -5.80897294917358e-08
                  -5.72670317935414e-08
                  -4.54055502986075e-08
                  -2.24019883458567e-08
                  -9.79476853124359e-09
                  -1.81978331296979e-08
                  -4.16708591574661e-08
                  -2.90547298884899e-08
                  -8.68007585293709e-09
                  -1.98520660817533e-08
                  -2.22172606599206e-08
                  -1.34928384906415e-08
                  -1.59692599639867e-08
                  -1.29627380230382e-08
                  -6.02213511225332e-09
                  -5.37426733646823e-09
                  -5.60632217714059e-09
                  -4.22640220308099e-09
                  -4.15434071332248e-09
                  -5.26366863651582e-09
                  -3.49880386039430e-09
                  -2.49710513888524e-09
                  -1.81963053545934e-09
                  4.26188723125220e-10
                  8.15468824116444e-10
                  -3.10260205450748e-10
                  -8.78272615289807e-10
                  -8.45697148020437e-10
                  -8.24536804181419e-10]
    coeff_d = [0.00574028413295644
               0.00362283648873149
               0.00219378792642255
               0.00138313973215407
               0.000824234624570449
               0.000473886305158525
               0.000282228798168712
               0.000152746237128845
               7.97700141479325e-05
               3.78685775786379e-05
               2.12085852479586e-05
               1.70318340355962e-05
               1.62704576935885e-05
               1.47254874965428e-05
               1.30382036976249e-05
               1.18250667400613e-05
               1.17579386740903e-05
               1.02428268042568e-05
               8.75898722620216e-06
               7.83029178412468e-06
               6.53994692103413e-06
               5.15451099847383e-06
               4.31888650815531e-06
               3.75944742507849e-06
               3.34089370799196e-06
               3.05746349140090e-06
               2.81611783005974e-06
               2.54263653098734e-06
               2.22293888084682e-06
               1.93468012486058e-06
               1.68624848161539e-06
               1.41364492082310e-06
               1.11383387208873e-06
               8.35097172483240e-07
               6.11204597711566e-07
               4.66783583544344e-07
               3.91470335423986e-07
               3.35670529148280e-07
               2.80615866435817e-07
               2.48573455370337e-07
               2.32346442364724e-07
               2.21980490170161e-07
               1.90927455628890e-07
               1.50668490287477e-07
               1.35404170774103e-07
               1.22792790493479e-07
               9.86356371608125e-08
               8.20548080894536e-08
               6.79163456319880e-08
               5.19439359662448e-08
               4.33022299091905e-08
               3.78020873634049e-08
               3.22603876974349e-08
               2.71267679473677e-08
               2.33716304631725e-08
               1.84242900497388e-08
               1.38023574232458e-08
               1.11722936440794e-08
               8.70007243944765e-09
               8.05581412488029e-09
               8.93360349983364e-09
               9.18174872959370e-09
               8.49898862884152e-09
               7.62539945805023e-09
               6.80378929165720e-09]

    breaks = [10:5:60; 62:2.:70; 71:1.:120]
    # show(breaks)
    if h >=120
        h = 119.9999
    elseif h < 10
        h = 10
    end
    # show(length(breaks))
    for jj in range(0, length(breaks)-1)
        if (h < breaks[jj+1] && h >= breaks[jj])
            # show(jj)
            return coeff_d[jj] + coeff_c[jj]*(h-breaks[jj]) +coeff_b[jj]*(h-breaks[jj])^2 +coeff_a[jj]*(h-breaks[jj])^3
        end
    end
end


function eswind_spline(p::CPEGDensityParameters, h::T) where T
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


function esdynamics(ev::CPEGWorkspace, x::SVector{10,T}, u::SVector{1,W}) where {T,W}
    # scaled variables
    r_scaled = x[SA[1,2,3]]
    v_scaled = x[SA[4,5,6]]
    σ = x[7]
    kρ = x[8]
    kwE = x[9]
    kwN = x[10]

    # unscale
    r, v = unscale_rv(ev.scale,r_scaled,v_scaled)

    # altitude
    h,lat,lon = altitude(ev.params.gravity, r)

    # Compute NED basis unit vectors for wind
    uD,uN,uE = latlongtoNED(lat, lon)

    # density
    # ρ = density(ev.params.density, h)
    ρ = esdensity_spline(ev,h)*(1+kρ)#esdensity(ev, h, 7.295*1e3)*(1+kρ)
    wE, wN, wU = eswind_spline(ev.params.density,h)

    wE = wE*(kwE)
    wN = wN*(kwN)
    # wE positive to the east , m / s, wN positive to the north , m / s, wU positive up , m / s

    wind_pp = wN * uN + wE * uE - wU * uD  # wind velocity in pp frame , m / s
    v_rw = v + wind_pp  # relative wind vector , m / s # if wind == 0, the velocity = v
    v_rw_hat = v / norm(v_rw)  # relative wind unit vector , nd

    # lift and drag magnitudes
    L, D = LD_mags(ev.params.aero,ρ,r,v_rw)


    # basis for e frame
    e1, e2 = e_frame(r,v_rw)

    # drag and lift accelerations
    D_a = -(D/norm(v_rw))*v_rw
    L_a = L*sin(σ)*e1 + L*cos(σ)*e2

    # gravity
    g = gravity(ev.params.gravity,r)

    # acceleration
    ω = ev.planet.ω
    a = D_a + L_a + g - 2*cross(ω,v) - cross(ω,cross(ω,r))
    # println("alt ",h)
    # rescale units
    v,a = scale_va(ev.scale,v,a)

    return SA[v[1],v[2],v[3],a[1],a[2],a[3],u[1]*ev.scale.uscale,0,0,0]
end

function rk4_est(
    ev::CPEGWorkspace,
    x_n::SVector{10,T},
    u::SVector{1,W},
    dt_s::Float64) where {T,W}

    k1 = dt_s*esdynamics(ev,x_n,u)
    k2 = dt_s*esdynamics(ev,x_n+k1/2,u)
    k3 = dt_s*esdynamics(ev,x_n+k2/2,u)
    k4 = dt_s*esdynamics(ev,x_n+k3,u)
    return (x_n + (1/6)*(k1 + 2*k2 + 2*k3 + k4))
end

function get_discdyn(model,X,U,dt)
    # dt_s = dt/ev.scale.tscale
    A= ForwardDiff.jacobian(_x -> rk4(model,_x,U,dt),X)
    B= ForwardDiff.jacobian(_u -> rk4(model,X,_u,dt),U)
    return A,B
end
