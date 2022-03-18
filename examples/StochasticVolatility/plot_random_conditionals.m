function plot_random_conditionals(S, X)

% define parameters for plotting
n_random_conditionals = 10;
N_plot = 100;
N = size(X,1);

% check dimensions of map and X
if S.dim ~= size(X,2)
    error('Dimensions of map and X don''t match')
end

% define plotting domain
xx = linspace(-4,4,N_plot);

figure('position',[0,0,1200,600])
for i=1:S.dim
    
    subplot(1,S.dim,i)
    hold on
    % plot random conditionals
    for j=1:n_random_conditionals
        rand_Xc = X(randi(N,1),:);
        Xj = repmat(rand_Xc,N_plot,1);
        Xj(:,i) = xx';
        SXc = S.evaluate(Xj);
        plot(xx, SXc)
    end
    % plot true map
    if i<S.dim-1
        plot(xx, zeros(N_plot,1), '--r', 'LineWidth',5)
    elseif i==S.dim-1
        plot(xx, -1/0.25*xx, '--r', 'LineWidth',5)
    else 
        plot(xx, 1/0.25*xx, '--r', 'LineWidth',5)
    end
    % plot samples
    histogram(X(:,i),'normalization','pdf')
    xlabel(['$x_' num2str(i) '$'],'FontSize',16)
    ylabel('$S(x)$','FontSize',16)
    hold off
    
end
