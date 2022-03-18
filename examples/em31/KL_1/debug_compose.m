clear; close all; clc
addpath(genpath('../../../src'))

sd = 2; rng(sd);

X=load('X_2d_7.txt');
W=load('W_2d_7.txt');

XW.X=X;
XW.W=W;

%XW=randn(20000,2);
%List_obs=load('obs_wihtout_noise_neat.txt')%+7*randn(50,1);
List_obs=load('List_obs_neat.txt');

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

x2=linspace(0.8,3.8,50);
x1=linspace(-3,3,50);
[X,Y]=meshgrid(x1,x2);
Z=exp(post.log_pdf([X(:) Y(:)]));

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


%%%%%%%% START ITERATION 2 %%%%%%%%%%%%%%%%%%%%%%%%
CM_post=ComposedPullbackDensity({Topt, G_lap.S}, ref);


%% VERIF GRAD MAP
% 21 (dT_2/dx_1)
Xtest=linspace(-4,4,100);
X=[Xtest(:) zeros(100,1)];

TX=CM_post.evaluate(X);

grad_TX_num=gradient(TX(:,2),Xtest);


G1=CM_post.S{1}.grad_x(X);

for k=1
    Y1=CM_post.S{2}.evaluate(X);
    G1=CM_post.S{1}.grad_x(Y1);
    G2=CM_post.S{2}.grad_x(X);
    
    G=mtimes(reshape(G2(k,:,:),2,2)',reshape(G1(k,:,:),2,2));
end





% dxS=self.S{1}.grad_x(X);
% L_S{1}=self.S{1};
% 
% for k=2:length(self.S)
%     if k==2
%         CM_temp_X=L_S{1}.evaluate(X);
%     else
%         CM_temp=ComposedPullbackDensity(L_S, self.ref);
%         CM_temp_X=CM_temp.evaluate(X);
%     end
%     dxS=self.S{k}.grad_x(CM_temp_X).*dxS;
%     L_S{k}=self.S{k};
% end
