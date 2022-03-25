clear; clc; close all;
sd = 1; rng(sd);

% add paths
addpath(genpath('../../../src'))

% define parameters 
d = 2;     % dimension of unknown parameters
M = 1000;  % number of samples

%% -- STANDARDIZE SAMPLES --

% generate samples
X = sample_banana(M);
Xtest = sample_banana(M);

% standardize samples with a Gaussian linear diagonal map
G = GaussianPullbackDensity(d, true);
G = G.optimize(X);
Xnorm = G.S.evaluate(X);
Xnormtest = G.S.evaluate(Xtest);

%% -- LEARN TRANSPORT MAP --

% define reference distribution
ref = IndependentProductDistribution({Normal(), Normal()});

% setup map with greedy basis selection (start from S(x) = Id(x))
basis = HermiteProbabilistPolyWithLinearization();
TM = identity_map(1:d, basis);
PB = PullbackDensity(TM, ref);

% specify a maximum number of terms (5 for S^1, 40 for S^2)
[PB, ~] = PB.greedy_optimize(Xnorm, Xnormtest, [5,5], 'max_terms');

% compose map with linear transformation for pre-conditioning
CM = ComposedPullbackDensity({G, PB}, ref);

%% -- PLOT FULL DENSITY --

% check approximation
xdom = [-3,3];
ydom = [-3,5];
xx = linspace(xdom(1),xdom(2),100);
yy = linspace(ydom(1),ydom(2),100);
[Xg, Yg] = meshgrid(xx, yy);

% evaluate approximate and true density
true_pi   = exp(log_pdf_banana([Xg(:), Yg(:)]));
approx_pi = exp(CM.log_pdf([Xg(:), Yg(:)]));

true_pi   = reshape(true_pi, size(Xg,1), size(Xg,2));
approx_pi = reshape(approx_pi, size(Xg,1), size(Xg,2));

% plot densities and samples
figure()
hold on
contourf(Xg, Yg, true_pi)
plot(X(:,1), X(:,2), '.r', 'MarkerSize',8)
xlim(xdom)
ylim(ydom)
lim = caxis;
set(gca,'FontSize',18)
xlabel('$x_1$','FontSize',24)
ylabel('$x_2$','FontSize',24)
set(gca,'LineWidth',2)
title('True PDF')
hold off
print('-dpng','true_pdf')

figure()
contourf(Xg, Yg, approx_pi)
xlim(xdom)
ylim(ydom)
caxis(lim)
hold on
set(gca,'FontSize',18)
xlabel('$x_1$','FontSize',24)
ylabel('$x_2$','FontSize',24)
set(gca,'LineWidth',2)
title('Approximate PDF')
print('-dpng','approx_pdf')

%% -- PLOT CONDITIONAL DENSITY --

% check approximation
xst = 1;
yy = linspace(ydom(1), ydom(2), 100);

% evaluate approximate and true density
true_cond_pi_tilde = exp(log_pdf_banana([xst*ones(length(yy),1), yy.']));
true_cond_pi_norm_const = trapz(yy, true_cond_pi_tilde);
true_cond_pi = true_cond_pi_tilde/true_cond_pi_norm_const;
approx_pi = exp(CM.log_pdf([xst*ones(length(yy),1), yy.'],2));

% plot densities and samples
figure('position',[0,0,600,300])

subplot(1,2,1)
contourf(Xg, Yg, true_pi)
hold on
plot(xst*ones(length(xx),1), yy, '-r')
xlim(xdom)
ylim(ydom)
legend('PDF','$y^*$','location','northwest')
xlabel('$x$')
ylabel('$y$')
title('Joint PDF')
hold off

subplot(1,2,2)
hold on
plot(xx, true_cond_pi, '-k')
plot(xx, approx_pi)
xlim(xdom)
legend('Truth','Approximation','location','south')
xlabel('$x$')
ylabel('$\pi(x|y^*)$')
title('Approximate PDF')
hold off

%% -- DEFINE MODEL --

function X = sample_banana(N)
    x1 = randn(N,1);
    x2 = x1.^2 + randn(N,1);
    X = [x1, x2];
end

function log_pi = log_pdf_banana(X)
    log_pi_x1 = log(normpdf(X(:,1)));
    log_pi_x2 = log(normpdf(X(:,2) - X(:,1).^2));
    log_pi = log_pi_x1 + log_pi_x2;
end

% -- END OF FILE --