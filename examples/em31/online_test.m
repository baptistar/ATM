clear; close all; clc
addpath(genpath('../../src'))
addpath(genpath('./simplexquad'))
sd = 2; rng(sd);

d=2;

% c = parcluster('local')
% c.NumWorkers = 12;
% parpool(12)

for k=1
    X=load(['./offline_samples/samples_analyticalem31_',num2str(k-1),'.txt']);

    % find the Gaussian approximation
    G = GaussianPullbackDensity(d, true);
    G = G.optimize(X);
    Z = G.S.evaluate(X);

    % find the order k approximation
    basis = ProbabilistHermiteFunction();
    S = identity_map(1:d, basis);

    ref = IndependentProductDitribution(repmat({Normal()},1,d));
    PB = PullbackDensity(S, ref);
    max_terms=[1,30];
    PB = PB.greedy_optimize(Z, [], max_terms, 'Split');

    % compose map
    CM = ComposedPullbackDensity({G, PB}, ref);

    %parsave(['./offline_maps/offline_map',num2str(k),'.mat'],CM)

end

