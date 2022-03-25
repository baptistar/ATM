clear; clc; close all;
sd = 1; rng(sd);
addpath(genpath('../../../src'))

d  = 2;         % dimension of unknown parameters
Ntrain = 1000;  % number of samples
Nvalid = 500;   % number of validation samples 

%% -- RUN TMAP FILTER --

% generate training and validation samples
Xtrain = sample_banana(Ntrain);
Xvalid = sample_banana(Nvalid);

% standardize samples with a Gaussian linear diagonal map
G = GaussianPullbackDensity(d, true);
G = G.optimize(Xtrain);
Ztrain = G.S.evaluate(Xtrain);
Zvalid = G.S.evaluate(Xvalid);

% define reference distribution
ref = IndependentProductDistribution({Normal(), Normal()});

% learn map with total-order basis
basis = HermiteProbabilistPolyWithLinearization();
TM = identity_map(1:d, basis);
PB = PullbackDensity(TM, ref);

% specify the maximum number of terms in each component's expansion
max_terms = 15;

% run optimization with 5-fold cross-validation for maximum number of terms
[PB, ~] = PB.greedy_optimize(Ztrain, Zvalid, max_terms, 'kFold');

% compose non-linear and linear maps
CM = ComposedPullbackDensity({G, PB}, ref);

%% Plot results

% check approximation
xdom = [-3,3];
ydom = [-3,5];
xx = linspace(xdom(1),xdom(2),100);
yy = linspace(ydom(1),ydom(2),100);
[Xg, Yg] = meshgrid(xx, yy);

% evaluate approximate and true density
approx_pi = exp(CM.log_pdf([Xg(:), Yg(:)]));
true_pi   = exp(log_pdf_banana([Xg(:), Yg(:)]));

approx_pi = reshape(approx_pi, size(Xg,1), size(Xg,2));
true_pi   = reshape(true_pi, size(Xg,1), size(Xg,2));

% generate samples
Zref = randn(1000,d);
Xnew = CM.inverse(Zref);

% plot densities and samples
figure('position',[0,0,600,300])

subplot(1,2,1)
contourf(Xg, Yg, true_pi)
hold on
plot(Xtrain(:,1), Xtrain(:,2), '.r','MarkerSize',6)
xlim(xdom)
ylim(ydom)
lim = caxis;
title('True PDF')

subplot(1,2,2)
contourf(Xg, Yg, approx_pi)
hold on
plot(Xnew(:,1), Xnew(:,2), '.r', 'MarkerSize',6) 
xlim(xdom)
ylim(ydom)
caxis(lim)
hold off
title('Approximate PDF')

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