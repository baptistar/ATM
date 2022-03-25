clear; close all; clc
addpath(genpath('../../src'))
sd = 2; rng(sd);

% define target
pi = Banana();
d = 2;

%% Build map adaptively

% set max_terms
max_terms = 20;

% define reference and samples
N = 2000;
Ztrain = randn(N,d);
Zvalid = randn(N,d);

% define total-order identity map using Hermite functions
basis = HermiteProbabilistPolyWithLinearization();
T = identity_map(1:d, basis);
T = TriangularTransportMap(T);

% define and optimize pullback-density
[Topt, output] = adaptive_transport_map(T, pi, Ztrain, Zvalid, max_terms);

%% Plot approximation

% samples from push-forward density
Zeval = randn(1e4,2);
X_approx = Topt.evaluate(Zeval);

% define grid
xx = linspace(-3,3,100);
yy = linspace(-1,4,100);
[X1,X2] = meshgrid(xx,yy);
logpi_true = pi.log_pdf([X1(:),X2(:)]);
logpi_true = reshape(logpi_true, size(X1,1), size(X2,2));

% plot approximation
figure
hold on
contourf(X1, X2, exp(logpi_true), 20)
plot(X_approx(:,1), X_approx(:,2), '.r','MarkerSize',2)
xlabel('$x_1$')
ylabel('$x_2$')
xlim([-3,3])
ylim([-1,4])
title('ATM approximation')
set(gca,'FontSize',16)
hold off

% plot errors vs. iteration
figure
hold on
plot(1:length(output.train_err), output.train_err, '-o')
plot(1:length(output.valid_err), output.valid_err, '-o')
xlabel('Iterations')
ylabel('Negative log-likelihood')
legend('Training error','Test error')
set(gca,'FontSize',16)
set(gca,'YScale','log')
hold off

% % -- END OF FILE --