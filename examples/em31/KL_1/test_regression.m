clear; close all; clc
addpath(genpath('../../../src'))

sd = 2; rng(sd);

X=load('X_2d_7.txt');
W=load('W_2d_7.txt');

XW.X=X;
XW.W=W;

%XW=randn(20000,2);
List_obs=load('obs_wihtout_noise_neat.txt')%+0*randn(50,1);
%List_obs=load('obs_wihtout_noise_neat.txt')+7*randn(50,1);
%List_obs=load('List_obs_neat.txt');

x=linspace(-5,5,100)';

d=2;
ref2 = IndependentProductDistribution(repmat({Normal()},1,d+1));

yobs=List_obs(1);
CM=load('./offline_maps/offline_map_poly_linear_neat2_1.mat');
CM=CM.x;
L_M={CM.S{1}.S CM.S{2}.S};


CM=ComposedPullbackDensity(L_M,ref2);

lkl = LikelihoodFunction(CM, yobs);

mu_0=[0.5 3];
sigma_0=[0.64 0; 0 0.25];

prior=MultivariateGaussian(d,mu_0,sigma_0);
post=BayesianDistribution(d,prior,lkl);

x0=mu_0;
[mu,Sigma] = optimize_laplace(post,x0);

Ls=chol(Sigma);

G_lap=GaussianPullbackDensity(d,false);

for k=1:d
    G_lap.S.S{k}.c=mu(k);
    G_lap.S.S{k}.L=Ls(k,1:k);
end

pi=PullbackDensity(G_lap.S,post);


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



% x2=linspace(-4,4,50);
% x1=linspace(-4,4,50);
% [X,Y]=meshgrid(x1,x2);
% Z=exp(pi.log_pdf([X(:) Y(:)]));
% 
% figure
% surf(X,Y,reshape(Z,50,50))


CM_post=ComposedPullbackDensity({Topt, G_lap.S}, ref);

L_mu_post=[];
L_std_post=[];

Z=CM_post.evaluate(randn(50000,2));
mu_post=mean(Z);
std_post=std(Z);

L_mu_post=[L_mu_post, mu_post];
L_std_post=[L_std_post, std_post];
L_best_var=[output.best_var_d];

L_idxs=[length(Topt.coeff)];

r=0;

N=3;
for k=2:N
    disp(['------ Iteration : ',num2str(k),'-----------'])
    PB_off=load(['./offline_maps/offline_map_poly_linear_neat2_',num2str(k),'.mat']);
    PB_off=PB_off.x;
    L_M={PB_off.S{1}.S PB_off.S{2}.S};
    PB_off=ComposedPullbackDensity(L_M,ref2);
    
    yobs=List_obs(k);
    
    lkl_seq=LikelihoodFunction_seq(PB_off,CM_post, yobs);
    
    post_seq=BayesianDistribution(d,ref,lkl_seq);
    
    max_terms=25;

    % define total-order identity map using Hermite functions
    T = identity_map(1:d, basis);
    T = TriangularTransportMap(T);
    
    [Topt_seq, output] = adaptive_transport_map(T, post_seq, XW, Zvalid, max_terms,max_patience,var_tol);
    
    L_S=[{Topt_seq} CM_post.S];
    
    CM_post=ComposedPullbackDensity(L_S, ref);
    
    L_best_var=[L_best_var,output.best_var_d];
end


%basis = ProbabilistHermiteFunction();
basis=HermiteProbabilistPoly();
%basis=HermiteProbabilistPolyWithLinearization();
S = identity_map(1:2, basis);

ref = IndependentProductDistribution(repmat({Normal()},1,2));
PB = PullbackDensity(S, ref);
max_terms=[10,70];
tol=1e-6;
PB=PB.greedy_optimize_regression(CM_post,XW, ...
                                    max_terms, tol);

Xx=linspace(-4,4,100);
Xtest=[Xx(:),zeros(100,1)];
figure
subplot(1,2,1)
plot(Xx,PB.S.evaluate(Xtest,1))
hold on
plot(Xx,CM_post.evaluate(Xtest,1))
Y1=PB.S.evaluate(Xtest,2);
Y2=CM_post.evaluate(Xtest,2);
subplot(1,2,2)
plot(Xx,Y1)
hold on
plot(Xx,Y2)
title(num2str(tol))
