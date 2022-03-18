clear; close all; clc
addpath(genpath('../../src'))
addpath(genpath('./simplexquad'))
sd = 2; rng(sd);

X=load('./offline_samples/samples_analytical_em31.txt');

d=2;

% find the Gaussian approximation
G = GaussianPullbackDensity(d, true);
G = G.optimize(X);
Z = G.evaluate(X);

% find the order k approximation
basis = ProbabilistHermiteFunction();
S = identity_map(1:d, basis);

ref = IndependentProductDitribution(repmat({Normal()},1,d));
PB = PullbackDensity(S, ref);
max_terms=[1,15];
PB = PB.greedy_optimize(Z, [], max_terms, 'Split');

% compose map
CM = ComposedPullbackDensity({G, PB}, ref);

yobs=511.5718133;

lkl = LikelihoodFunction(CM, yobs);
mu_0=1.8;
sigma_0=0.3;

prior=MultivariateGaussian(1,mu_0,sigma_0);
post=BayesianDistribution(1,prior,lkl);

X=linspace(1.8,3,200)';

Y=post.log_pdf(X);
DY=post.grad_x_log_pdf(X);
DY2=gradient(Y,X);

figure
plot(X,DY)
hold on
plot(X,DY2);

x0=mu_0;
[mu,Sigma] = optimize_laplace(post,x0);


d=1;

G_lap=GaussianPullbackDensity(d,false);
G_lap.S{1}.c=mu;
G_lap.S{1}.L=Sigma;
 
pi=PullbackDensity(G_lap.S,post);


% define reference and samples
N = 2000;
d=1;
Ztrain = randn(N,d);
Zvalid = randn(N,d);

X=load('10_x.txt');
W=load('10_w.txt');

XW.X=X;
XW.W=W;

max_terms=10;

% define total-order identity map using Hermite functions
basis = ProbabilistHermiteFunction();
T = identity_map(1:d, basis);
T = TriangularTransportMap(T);

% define and optimize pullback-density
[Topt, output] = adaptive_transport_map(T, pi, XW, Zvalid, max_terms);

CM_post=ComposedPullbackDensity({Topt, G_lap}, ref);

Z=CM_post.evaluate(randn(50000,1));

figure
ksdensity(Z)
