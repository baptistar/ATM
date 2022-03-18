function fig = push_forward_marginals(S, X, components)
%  Function plots qq plot for each marginal to test how it differs from
%  Gaussian marginals

    % define quantiles
    Nquant = 1e4;
    pd = ((1:Nquant)-0.5) ./ Nquant;
    
    % evaluate standard Gaussian quantiles
    qstd = norminv(pd);
    
    % evaluate push-forward map
    Sx = S.evaluate(X, components);
    
    fig = figure('position',[0,0,1200,1200]);
    hold on
    % plot quantiles of each push-forward marginal
    for i=1:size(Sx,2)
        qi = quantile(Sx(:,i),pd);        
        plot(qstd,qi,'+','MarkerSize',5,'DisplayName',['Marginal ' num2str(i)])
    end
    plot(qstd,qstd, '--r','linewidth',4,'DisplayName','Reference')
    legend('show','location','southeast')
    xlabel('Standard normal quantiles','FontSize',16)
    ylabel('$\hat{S}_\#\pi$ quantiles','FontSize',16)
    xlim([-4,4])
    ylim([-4,4])
    hold off

end

% -- END OF FILE --
