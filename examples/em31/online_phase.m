clear; close all; clc
addpath(genpath('../../src'))
addpath(genpath('./simplexquad'))
sd = 2; rng(sd);


X=load('10_x.txt');
W=load('10_w.txt');

XW.X=X;
XW.W=W;

List_obs=load('List_dobs1d.txt');
x=linspace(-5,5,100)';

yobs=List_obs(1);

d=1;
ref2 = IndependentProductDistribution(repmat({Normal()},1,2));

CM=load('./offline_maps/offline_map1.mat');
CM=CM.x;
L_M={CM.S{1}.S CM.S{2}.S};
CM=ComposedPullbackDensity(L_M,ref2);

lkl = LikelihoodFunction(CM, yobs);
mu_0=1.8;
sigma_0=0.3;

prior=MultivariateGaussian(1,mu_0,sigma_0);
post=BayesianDistribution(1,prior,lkl);

x0=mu_0;
[mu,Sigma] = optimize_laplace(post,x0);

G_lap=GaussianPullbackDensity(d,false);
G_lap.S.S{1}.c=mu;
G_lap.S.S{1}.L=Sigma;

% G_lap.S.S{1}.c=0.01;
% G_lap.S.S{1}.L=1.01;
 
pi=PullbackDensity(G_lap.S,post);

% define reference and samples
N = 2000;
d=1;
Zvalid = randn(N,d);
ref = IndependentProductDitribution(repmat({Normal()},1,d));
max_terms=10;
max_patience=2;
var_tol=-3;

% define total-order identity map using Hermite functions
%basis = ProbabilistHermiteFunction();
%basis=HermiteProbabilistPoly();
%basis=ConstExtProbabilistHermiteFunction();
basis=HermiteProbabilistPolyWithLinearization();
T = identity_map(1:d, basis);
T = TriangularTransportMap(T);

% define and optimize pullback-density
[Topt, output] = adaptive_transport_map(T, pi, XW, Zvalid, max_terms,max_patience,var_tol);

% x=linspace(-5,5,100)';
% c=trapz(x,exp(pi.log_pdf(x)));
% figure
% plot(x,exp(pi.log_pdf(x))./c)
% hold on
% plot(x,normpdf(x))
% 
% Y=pi.log_pdf(x);
% DY=gradient(Y,x);
% figure
% plot(x,pi.grad_x_log_pdf(x))
% hold on
% plot(x,DY)

figure
plot(x,Topt.evaluate(x))


CM_post=ComposedPullbackDensity({Topt, G_lap.S}, ref);

Z=CM_post.evaluate(randn(50000,1));

L_mu_post=[];
L_std_post=[];

mu_post=mean(Z);
std_post=std(Z);

L_mu_post=[L_mu_post, mu_post];
L_std_post=[L_std_post, std_post];

L_idxs=[length(Topt.coeff)];

return
for k=2
disp(['------ Iteration : ',num2str(k),'-----------'])
PB_off=load(['./offline_maps/offline_map',num2str(k),'.mat']);
PB_off=PB_off.x;
L_M={PB_off.S{1}.S PB_off.S{2}.S};
PB_off=ComposedPullbackDensity(L_M,ref2);

yobs=List_obs(k);

lkl_seq=LikelihoodFunction_seq(PB_off,CM_post, yobs);

post_seq=BayesianDistribution(1,ref,lkl_seq);

max_terms=10;
% define total-order identity map using Hermite functions
T = identity_map(1:d, basis);
T = TriangularTransportMap(T);

[Topt_seq, output] = adaptive_transport_map(T, post_seq, XW, Zvalid, max_terms,max_patience,var_tol);

L_S=[{Topt_seq} CM_post.S];

CM_post=ComposedPullbackDensity(L_S, ref);

Z=CM_post.evaluate(randn(50000,1));

mu_post=mean(Z);
std_post=std(Z);

L_mu_post=[L_mu_post, mu_post];
L_std_post=[L_std_post, std_post];
L_idxs=[L_idxs,length(Topt_seq.coeff)];
end

L_S2=L_S;
L_S2(end)=[];
CM_2=ComposedPullbackDensity(L_S2, ref);

disp(['------ Regression :',num2str(k),' -----------'])
T = identity_map(1:d, basis);
T = TriangularTransportMap(T);
tol=1e-5;
[Treg, output] = adaptive_regression(T, CM_2, XW, Zvalid, ...
    7, 1,tol);

figure
plot(x,Treg.evaluate(x),'o')
hold on
plot(x,CM_2.evaluate(x))

