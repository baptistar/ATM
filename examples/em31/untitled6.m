clear; close all; clc
addpath(genpath('../../src'))
addpath(genpath('./simplexquad'))
sd = 2; rng(sd);

d=5;

% c = parcluster('local')
% c.NumWorkers = 12;
% parpool(12)

for k=1
    XX=randn(10000,d);
    
    X=XX;
    
    X(:,2)=XX(:,1)+XX(:,2).*XX(:,3);
    X(:,3)=XX(:,2)+XX(:,3).*XX(:,3);
    X(:,4)=XX(:,5)+XX(:,1).*XX(:,3);
    X(:,5)=XX(:,1)+XX(:,4).*XX(:,3);
    
    % find the Gaussian approximation
    G = GaussianPullbackDensity(d, true);
    G = G.optimize(X);
    Z = G.S.evaluate(X);

    % find the order k approximation
    basis = ProbabilistHermiteFunction();
    S = identity_map(1:d, basis);

    ref = IndependentProductDitribution(repmat({Normal()},1,d));
    PB = PullbackDensity(S, ref);
    max_terms=[1,1,1,1,100];
    PB = PB.greedy_optimize(Z, [], max_terms, 'Split');

    % compose map
    CM = ComposedPullbackDensity({G, PB}, ref);

    %parsave(['./offline_maps/offline_map',num2str(k),'.mat'],CM)

end

