clear; close all; clc
addpath(genpath('../../../src'))

sd = 2; rng(sd);

X=load('X_2d_7.txt');
W=load('W_2d_7.txt');

XW.X=X;
XW.W=W;

%XW=randn(20000,2);
List_obs=load('obs_wihtout_noise_neat.txt')%+7*randn(50,1);
%List_obs=load('List_obs_neat.txt');

x=linspace(-5,5,100)';

d=2;
ref2 = IndependentProductDistribution(repmat({Normal()},1,d+1));

L_lkl=[];

for k=1:50
yobs=List_obs(1);
CM=load(['./offline_maps/offline_map_poly_linear_neat2_',num2str(k),'.mat']);
CM=CM.x;
L_M={CM.S{1}.S CM.S{2}.S};
CM=ComposedPullbackDensity(L_M,ref2);
lkl = LikelihoodFunction(CM, yobs);
L_lkl=[L_lkl,lkl];
end

mu_0=[0.5 3];
sigma_0=[0.64 0; 0 0.25];

prior=MultivariateGaussian(d,mu_0,sigma_0);
post=BayesianDistribution_L(d,prior,L_lkl);

x2=linspace(0.8,3.8,50);
x1=linspace(-3,3,50);
[X,Y]=meshgrid(x1,x2);
Z=exp(post.log_pdf([X(:) Y(:)]));

figure
surf(X,Y,reshape(Z,50,50))
title('posterior')

x0=mu_0;
[mu,Sigma] = optimize_laplace(post,x0);

Ls=chol(Sigma);

G_lap=GaussianPullbackDensity(d,false);

for k=1:d
    G_lap.S.S{k}.c=mu(k);
    G_lap.S.S{k}.L=Ls(k,1:k);
end

pi=PullbackDensity(G_lap.S,post);

x2=linspace(-4,4,50);
x1=x2;
[X,Y]=meshgrid(x1,x2);
Z=exp(pi.log_pdf([X(:) Y(:)]));

figure
pcolor(X,Y,reshape(Z,50,50))
title('target')
shading interp 

% x2=linspace(0.8,3.8,50);
% x1=linspace(-3,3,50);
% [X,Y]=meshgrid(x1,x2);
% Z=exp(lkl.log([X(:) Y(:)]));
% figure
% surf(X,Y,reshape(Z,50,50))
% title('likelihood')

% define reference and samples
N = 2000;
Zvalid = randn(N,d);
ref = IndependentProductDistribution(repmat({Normal()},1,d));
max_terms=[30];
max_patience=[];
var_tol=-3;

% define total-order identity map using Hermite functions
%basis = ProbabilistHermiteFunction();
basis=HermiteProbabilistPoly();
%basis=ConstExtProbabilistHermiteFunction();
%basis=HermiteProbabilistPolyWithLinearization();
T = identity_map(1:d, basis);
T = TriangularTransportMap(T);

% define and optimize pullback-density
[Topt, output] = adaptive_transport_map(T, pi, XW, Zvalid, max_terms,max_patience,var_tol);

CM_post=ComposedPullbackDensity({Topt, G_lap.S}, ref);

Z=CM_post.evaluate(randn(50000,d));

L_mu_post=[];
L_std_post=[];

mu_post=mean(Z);
std_post=std(Z);




