clear; close all; clc
addpath(genpath('../../src'))
sd = 1; rng(sd);

%% Plot joint density

% define prior and likelihood
prior_x = @(x) normpdf(x,0,0.5);
likelihood = @(y,x) normpdf(y - x.^2,0,0.1);

% define sampling function
x_sample = @(N) 0.5*randn(N,1);
y_sample = @(x) x.^2 + 0.1*randn(size(x,1),1);

% define domain
dom_x = [-2,2];
dom_y = [-0.5,2.5];
N = 10000;
xx = linspace(dom_x(1),dom_x(2),N);
yy = linspace(dom_y(1),dom_y(2),N);

%% Define non-Gaussian and Gaussian approximation

%load('TM_results.mat','CM','YX','d')

% sample data
d = 2;
N = 10000;
X = x_sample(N);
Y = y_sample(X);

% compose samples
YX = [Y,X];

% set max_terms
max_terms = 20;

% find the Gaussian approximation
G = GaussianPullbackDensity(d, true);
G = G.optimize(YX);
Z = G.S.evaluate(YX);

% find the non-Gaussian approximation (rescale variance to improve fit)
basis = ProbabilistHermiteFunction();
ref = IndependentProductDistribution(repmat({Normal()},1,d));
S = identity_map(1:d, basis);
PB = PullbackDensity(S, ref);
PB = PB.greedy_optimize(Z/2, [], max_terms, 'Split');
G.S.S{1}.L = G.S.S{1}.L/2;
G.S.S{2}.L = G.S.S{2}.L/2;
G.S.S{1}.c = G.S.S{1}.c/2;
G.S.S{2}.c = G.S.S{2}.c/2;

% compose maps
CM = ComposedPullbackDensity({G, PB}, ref);

% find non-diagonal linear map
TM_linear = GaussianPullbackDensity(d, false);
TM_linear = TM_linear.optimize(YX);

%% Plot conditionals

yst_vect = [0, 0.25, 0.5];
yst_str = {'$y^* = 0$','$y^* = 0.25$','$y^* = 0.5$'};
for i=1:length(yst_vect)

    % extract yst
    yst = yst_vect(i);
    
    figure
    hold on
    plot(xx,yst*ones(1,N),'-','LineWidth',2);
    plot(YX(:,2), YX(:,1), '.k', 'MarkerSize',5)
    set(gca,'FontSize',20)
    xlabel('$x$','FontSize',24)
    ylabel('$y$','FontSize',24)
    xlim(dom_x)
    ylim(dom_y)
    legend(yst_str{i},'FontSize',20,'location','northeast')
    hold off
    %print('-depsc',['presentation_figures/joint_y' num2str(i)])
    close all

    % evaluate true density
    post_tilde = @(x) prior_x(x) .* likelihood(yst,x);
    norm_const = trapz(xx, post_tilde(xx'));
    post = @(x) post_tilde(x)/norm_const;
    pi_post = post(xx');
    
    % sample Gaussian and non-Gaussian approximate density
    Sxx = CM.evaluate([yst*ones(N,1), xx'],2);
    Sxx_linear = TM_linear.evaluate([yst*ones(10000,1), xx'],2);

    Npost = 10000;
    Xpost = interp1(Sxx,xx,randn(Npost,1));
    Xpost_linear = interp1(Sxx_linear,xx,randn(Npost,1));
    
    figure
    hold on
    histogram(Xpost,40,'normalization','pdf')
    histogram(Xpost_linear,40,'normalization','pdf')
    plot(xx,pi_post,'-k');
    xlim(dom_x)
    yl = get(gca,'YLim');
    yl(2) = yl(2) + 0.6;
    set(gca,'YLim',yl);
    set(gca,'FontSize',20)
    xlabel('$x$','FontSize',24)
    legend({'Nonlinear map samples','Linear map samples',['Posterior at ' yst_str{i}]},'FontSize',20)
    hold off
    %print('-depsc',['presentation_figures/post_y' num2str(i)])
    close all

end

% - END OF FILE -