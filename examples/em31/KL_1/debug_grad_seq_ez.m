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

PB_off=load(['./offline_maps/offline_map_poly_linear_neat2_',num2str(k),'.mat']);
PB_off=PB_off.x;
L_M={PB_off.S{1}.S PB_off.S{2}.S};
PB_off=ComposedPullbackDensity(L_M,ref2);
yobs=List_obs(2);

%% VERIF GRAD_X_LKL
%1
Xtest=linspace(-4,4,100);
X=[Xtest(:) zeros(100,1)];

x_ty=[X, repmat(yobs,size(X,1),1)];

log_lkl=PB_off.log_pdf(x_ty,3);

grad_x_log_lkl_num=gradient(log_lkl,Xtest);
grad_x_log_lkl_ana=PB_off.grad_x_log_pdf(x_ty,1:2,3);

figure
hold on
plot(Xtest,grad_x_log_lkl_num)
plot(Xtest,grad_x_log_lkl_ana(:,1),'o')
title('grad_x_log_lkl 1')

%2
Xtest=linspace(-4,4,100);
X=[zeros(100,1) Xtest(:)];

x_ty=[X, repmat(yobs,size(X,1),1)];

log_lkl=PB_off.log_pdf(x_ty,3);

grad_x_log_lkl_num=gradient(log_lkl,Xtest);
grad_x_log_lkl_ana=PB_off.grad_x_log_pdf(x_ty,1:2,3);

figure
hold on
plot(Xtest,grad_x_log_lkl_num)
plot(Xtest,grad_x_log_lkl_ana(:,2),'o')
title('grad_x_log_lkl 2')

%% VERIF GRAD MAP
% 21 (dT_2/dx_1)
X=[Xtest(:) zeros(100,1)];

TX=CM_post.evaluate(X);

grad_TX_num=gradient(TX(:,2),Xtest);

grad_TX_ana=CM_post.grad_x(X);

figure
hold on
plot(Xtest, grad_TX_num)
plot(Xtest,grad_TX_ana(:,2,1),'o')


%%GRAD_LKL

Xtest=linspace(-4,4,100);
X=[Xtest(:) zeros(100,1)];

lkl_seq=LikelihoodFunction_seq(PB_off,CM_post, yobs);

log_lkl=lkl_seq.log(X);

grad_x_log_lkl_num=gradient(log_lkl,Xtest);
grad_x_log_lkl_ana=lkl_seq.grad_x_log(X);

figure
hold on
plot(Xtest,grad_x_log_lkl_num)
plot(Xtest,grad_x_log_lkl_ana(:,1),'o')
title('grad_x_log_lkl 1')



return
TX=CM_post.evaluate(X);
grad_TX_ana=CM_post.grad_x(X);

grad_TX_num=gradient(TX(:,1),Xtest);

x_jt2=[TX, repmat(yobs,size(X,1),1)];
x_jt=[X, repmat(yobs,size(X,1),1)];

logx=PB_off.log_pdf(x_jt,3);
logx2=PB_off.log_pdf(x_jt2,3);

DY=gradient(logx,Xtest);
DY2=gradient(logx2,Xtest);

grad_PB_post=CM_post.grad_x(X);

lkl_grad=PB_off.grad_x_log_pdf(x_jt,1:2,3);
lkl_grad2=PB_off.grad_x_log_pdf(x_jt2,1:2,3);

Gtest1=zeros(100,2);
for k=1:100
Gtest1(k,:)=mtimes(reshape(grad_PB_post(k,:,:),2,2)',lkl_grad(k,:)');
end


figure
plot(Xtest,DY)
hold on
plot(Xtest,Gtest1(:,1),'*')


X=[zeros(100,1) Xtest(:)];

x2=CM_post.evaluate(X);
x_jt=[x2, repmat(yobs,size(X,1),1)];

logx=PB_off.log_pdf(x_jt,3);
DY=gradient(logx,Xtest);

grad_PB_post=CM_post.grad_x(X);
lkl_grad=PB_off.grad_x_log_pdf(x_jt,1:2,3);

Gtest1=zeros(100,2);
for k=1:100
Gtest1(k,:)=mtimes(reshape(grad_PB_post(k,:,:),2,2)',lkl_grad(k,:)');
end

figure
plot(Xtest,DY)
hold on
plot(Xtest,Gtest1(:,2),'*')

