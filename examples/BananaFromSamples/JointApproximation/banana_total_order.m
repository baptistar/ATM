clear; clc; close all;
sd = 1; rng(sd);

% add paths
addpath(genpath('../src'))

% define parameters 
d  = 2;     % dimension of unknown parameters
M  = 1000;  % number of samples

%% -- RUN TMAP FILTER --

% generate samples
Xtrain = sample_banana(M);

% standardize samples with a Gaussian linear diagonal map
G = GaussianPullbackDensity(d, true);
G = G.optimize(Xtrain);
Ztrain = G.evaluate(Xtrain);

% define reference distribution
ref = IndependentProductDitribution({Normal(), Normal()});

% learn map with total-order basis
order = 2;
TM = total_order_map(1:d, ProbabilistHermiteFunction(), order);
PB = PullbackDensity(TM, ref);
PB = PB.optimize(Ztrain);

% compose non-linear and linear maps
CM = ComposedPullbackDensity({G, PB}, ref);

%% Plot results

% check approximation
xx = linspace(-4,4,50);
[X, Y] = meshgrid(xx, xx);

% evaluate approximate and true density
approx_pi = exp(CM.log_pdf([X(:), Y(:)]));
true_pi   = exp(log_pdf_banana([X(:), Y(:)]));

approx_pi = reshape(approx_pi, size(X,1), size(X,2));
true_pi   = reshape(true_pi, size(X,1), size(X,2));

% plot densities and samples
figure('position',[0,0,600,300])

subplot(1,2,1)
contourf(X, Y, true_pi)
hold on
plot(Xtrain(:,1), Xtrain(:,2), '.r','MarkerSize',6)
axis([-4,4,-4,4])
lim = caxis;
title('True PDF')

subplot(1,2,2)
contourf(X, Y, approx_pi)
axis([-4,4,-4,4])
caxis(lim)
hold on
%plot(Xi_new(:,1), Xi_new(:,2), '.r', 'MarkerSize',6)
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