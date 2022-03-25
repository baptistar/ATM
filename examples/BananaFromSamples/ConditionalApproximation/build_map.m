clear; close all; clc
addpath(genpath('../../../src'))
sd = 2; rng(sd);

%% Generate data

x_sample = @(N) 0.5*randn(N,1);
y_sample = @(x) x.^2 + 0.1*randn(size(x,1),1);

prior_x = @(x) normpdf(x,0,0.5);
likelihood = @(y,x) normpdf(y - x.^2,0,0.1);

d = 2;
N = 10000;
X = x_sample(N);
Y = y_sample(X);

%% Build maps

% set max_terms
max_terms = 30;

% compose samples
YX = [Y,X];

% find the Gaussian approximation
G = GaussianPullbackDensity(d, true);
G = G.S.optimize(YX);
Z = G.S.evaluate(YX);

% define map 
basis = ProbabilistHermiteFunction();
S = identity_map(1:d, basis);
ref = IndependentProductDistribution(repmat({Normal()},1,d));
PB = PullbackDensity(S, ref);

% optimize map using Monte Carlo points
%PB = PB.greedy_optimize(Z, [], max_terms, 'Split');
PB = PB.greedy_optimize(Z/2, [], max_terms, 'Split');
G.S.S{1}.L = G.S.S{1}.L/2;
G.S.S{2}.L = G.S.S{2}.L/2;
G.S.S{1}.c = G.S.S{1}.c/2;
G.S.S{2}.c = G.S.S{2}.c/2;

% optimize map using quadrature points
%ZW.X=Z;
%ZW.W=(1/size(Z,1))*ones(size(Z,1),1);
%PB=PB.greedy_optimize(Z, [], max_terms, 'Split');

% compose map
CM = ComposedPullbackDensity({G, PB}, ref);

%% Plot joint approximation

% define grid
x_dom = [-3,3];
y_dom = [-1,3];
xx = linspace(-4,4,100);
yy = linspace(-1,5,100);
[Xr,Yr] = meshgrid(xx,yy);
logpi_r = CM.log_pdf([Yr(:), Xr(:)]);
logpi_r = reshape(logpi_r, size(Xr,1), size(Xr,2));

figure
contourf(Xr, Yr, exp(logpi_r), 20)
hold on
plot(X(1:4000,:), Y(1:4000,:), '.r','MarkerSize',1)
xlim(x_dom)
ylim(y_dom)
xlabel('$x$')
ylabel('$y$')
print('-depsc','figures/joint_xy')

%% Plot conditionals

yst_vect = [0,0.25,0.5];
yst_str = {'y^* = 0','y^* = 0.25','y^* = 0.5'};
x_dom = [-2,2];

blue_color = brewermap(256,'Blues'); blue_color = blue_color(200,:);

for i=1:length(yst_vect)

    % assign yst
    yst = yst_vect(i);
        
    % define true posterior
    post_tilde = @(x) prior_x(x) .* likelihood(yst,x);
    xx = linspace(x_dom(1),x_dom(2),10000);
    norm_const = trapz(xx, post_tilde(xx'));
    post = @(x) post_tilde(x)/norm_const;
    pi_post = post(xx');
    
    % define true map
    C_pi_x = cumtrapz(xx, post_tilde(xx')/norm_const);
	S_true = sqrt(2)*erfinv(2*C_pi_x - 1);
    S_true_orig = S_true;
    if i==1
        S_true(S_true == -inf) = -sqrt(2)*erfcinv(2*C_pi_x(S_true == -inf));
        S_true(xx > 0.84 ) = -1*interp1(xx, S_true, -1*xx(xx > 0.84));
    end

    % define approximate map
    pi_approx = exp(CM.log_pdf([repmat(yst,length(xx),1), xx'],2));
    S_approx = CM.evaluate([repmat(yst,length(xx),1), xx'],2);

    figure
    hold on
    plot(xx,pi_post,'--k','LineWidth',4);
    plot(xx,pi_approx,'-','Color',blue_color,'LineWidth',4);
    xlim([-1,1])
    xticks([-1,0,1])
    set(gca,'FontSize',24)
    xlabel('$x$','FontSize',30)
    ylabel(['$\pi_{X|Y}(x|' yst_str{i} ')$'],'FontSize',30)
    set(gca,'Ylim',[0,3.2])
    if i==3
        legend({'True posterior','Approximation'},'Location','north','FontSize',24)
    end
    hold off
    %print('-depsc',['figures/post_y' num2str(i)])
    %close all

    figure
    hold on
    plot(xx,S_true,'--k','LineWidth',4);
    plot(xx,S_approx,'-','Color',blue_color,'LineWidth',4);
    xlim([-1,1])
    xticks([-1,0,1])
    set(gca,'FontSize',24)
    xlabel('$x$','FontSize',30)
    ylabel(['$S^{\mathcal{X}}(' yst_str{i} ',x)$'],'FontSize',30)
    if i==3
        legend({'True KR map','Approximation'},'Location','north','FontSize',24)
    end
    hold off
    %print('-depsc',['figures/map_y' num2str(i)])
    %close all

end

% -- END OF FILE --