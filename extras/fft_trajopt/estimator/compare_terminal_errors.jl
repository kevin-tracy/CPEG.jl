
function get_error_stats(dr_errors, cr_errors)
    err_vec = [[dr_errors[i],cr_errors[i]]/1000 for i = 1:length(dr_errors)]
    errors = [dr_errors/1000 cr_errors/1000]

    μ = vec(mean(errors; dims = 1))
    Σ = cov(errors)
    F = eigen(Σ)
    R = F.vectors
    a,b=sqrt.(F.values)
    P = [(3*R*[a*cos(t), b*sin(t)] + μ) for t in range(0,2*pi,length = 100)]
    P = hcat(P...)
end

let

    dfilter = load("mc_1000_v2.jld2")
    no_filter = load("mc_1000_nofilter.jld2")#; alts, drs, crs, σs, t_vecs, σ̇s, dr_errors, cr_errors, qp_iters, alt_g, dr_g, cr_g)

    # P_filter = get_error_stats(filter["dr_errors"],filter["cr_errors"])
    # P_nofilter = get_error_stats(no_filter["dr_errors"],no_filter["cr_errors"])
    #
    # dr_errors_f = filter["dr_errors"]
    # cr_errors_f = filter["cr_errors"]
    # dr_errors_nf = no_filter["dr_errors"]
    # cr_errors_nf = no_filter["cr_errors"]

    qp_iters = dfilter["qp_iters"]
    qp_iters = vcat(qp_iters...)
    # qp_iters = [d[1] for d in qp_iters]
    # @show typeof(qp_iters)
    # @show qp_iters
    # @show filter(isone,qp_iters)
    @show length(filter(isone,qp_iters))/length(qp_iters)

    # mat"
    # figure
    # hold on
    # p1=plot(round($dr_errors_f/1000,3), round($cr_errors_f/1000,3),'r+','MarkerSize',2);
    # p1.Color = [1,0,0,.2];
    # plot(round($P_filter(1,:),3),round($P_filter(2,:),3),'r')
    # p2=plot(round($dr_errors_nf/1000,3), round($cr_errors_nf/1000,3),'bo','MarkerSize',2);
    # p2.Color = [0,0,1,.2];
    # plot(round($P_nofilter(1,:),3),round($P_nofilter(2,:),3),'b')
    # xlabel('Downrange Error, km')
    # ylabel('Crossrange Error, km')
    # %axis equal
    # p1 = plot(0,0,'r');
    # p2 = plot(0,0,'b');
    # grid on
    # plot([0,0],[-5,5],'k','linewidth',0.5)
    # plot([-5,5],[0,0],'k','linewidth',0.5)
    # legend([p1,p2],'Filter On','Filter Off')
    # %legend boxoff
    # hold off
    # xlim([-1.5,1.5])
    # ylim([-.5,.9])
    # addpath('/Users/kevintracy/devel/GravNav/matlab2tikz-master/src')
    # matlab2tikz('terminal_errors_comparison.tikz')
    # "
end
