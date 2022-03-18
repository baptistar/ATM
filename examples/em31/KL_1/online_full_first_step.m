clear; close all; clc
addpath(genpath('../../../src'))
sd = 2; rng(sd);


d=3;
L_S=load('SAMP_mes7_i_1mode_neat.txt');
List_obs=load('obs_wihtout_noise_neat.txt')+7*randn(50,1);


% c = parcluster('local');
% c.NumWorkers = 12;
% parpool(12)


k=1;

yobs=List_obs(k);

X=zeros(10000,d);
X(:,3)=L_S(:,51);
X(:,2)=L_S(:,52);
X(:,1)=L_S(:,k);

% find the Gaussian approximation
G = GaussianPullbackDensity(d, true);
G = G.optimize(X);
Z = G.S.evaluate(X);

% find the order k approximation
%basis = ProbabilistHermiteFunction();
%basis=HermiteProbabilistPoly();
basis=HermiteProbabilistPolyWithLinearization();
S = identity_map(1:d, basis);

ref = IndependentProductDistribution(repmat({Normal()},1,d));
PB = PullbackDensity(S, ref);
max_terms=[1,30,30];
PB = PB.greedy_optimize(Z, [], max_terms, 'Split');

% compose map
CM = ComposedPullbackDensity({G.S, PB.S}, ref);



x2=linspace(0.8,3.8,50);
x1=linspace(-3,3,50);
[X,Y]=meshgrid(x1,x2);
Z=exp(CM.log_pdf([yobs*ones(size(X(:))) Y(:) X(:)],2:3));

figure
surf(X,Y,reshape(Z,50,50))
title(['posterior full on (poly_linear), yobs=',num2str(yobs)])



