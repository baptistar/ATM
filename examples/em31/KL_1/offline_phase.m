clear; close all; clc
addpath(genpath('../../../src'))
sd = 2; rng(sd);


d=3;
L_S=load('SAMP_mes7_i_1mode_2.txt');

% c = parcluster('local');
% c.NumWorkers = 12;
% parpool(12)


parfor k=1:50
    X=zeros(10000,d);
    X(:,1)=L_S(:,51);
    X(:,2)=L_S(:,52);
    X(:,3)=L_S(:,k);
    
    % find the Gaussian approximation
    G = GaussianPullbackDensity(d, true);
    G = G.optimize(X);
    Z = G.S.evaluate(X);

    % find the order k approximation
    %basis = ProbabilistHermiteFunction();
    %basis=HermiteProbabilistPoly();
    basis=HermiteProbabilistPolyWithLinearization()
    S = identity_map(1:d, basis);

    ref = IndependentProductDistribution(repmat({Normal()},1,d));
    PB = PullbackDensity(S, ref);
    max_terms=[1,1,30];
    PB = PB.greedy_optimize(Z, [], max_terms, 'Split');

    % compose map
    CM = ComposedPullbackDensity({G, PB}, ref);

    parsave(['./offline_maps/offline_map_poly_linear2_',num2str(k),'.mat'],CM)

end

