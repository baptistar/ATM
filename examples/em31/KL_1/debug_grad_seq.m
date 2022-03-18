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

CM_post=ComposedPullbackDensity({Topt, G_lap.S}, ref);

r=0;
disp(['------ Iteration : ',num2str(k),'-----------'])
PB_off=load(['./offline_maps/offline_map_poly_linear_neat2_',num2str(k),'.mat']);
PB_off=PB_off.x;
L_M={PB_off.S{1}.S PB_off.S{2}.S};
PB_off=ComposedPullbackDensity(L_M,ref2);
yobs=List_obs(k);
Xtest=linspace(-4,4,100);

X=[Xtest(:) zeros(100,1)];

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




grad_PB_post=permute(grad_PB_post,[3 2 1]);
lkl_grad=reshape(lkl_grad,100,1,2);
lkl_grad=permute(lkl_grad,[2 3 1]);
lkl_grad=reshape(lkl_grad,2,1,100);

Gtest=pagemtimes(grad_PB_post,lkl_grad);
Gtest=permute(Gtest,[3 1 2]);

figure
plot(Xtest,DY)
hold on
plot(Xtest,Gtest(:,1),'o')
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

grad_PB_post=permute(grad_PB_post,[3 2 1]);
lkl_grad=reshape(lkl_grad,100,1,2);
lkl_grad=permute(lkl_grad,[2 3 1]);
lkl_grad=reshape(lkl_grad,2,1,100);

Gtest=pagemtimes(grad_PB_post,lkl_grad);
Gtest=permute(Gtest,[3 1 2]);

figure
plot(Xtest,DY)
hold on
plot(Xtest,Gtest(:,2),'o')
plot(Xtest,Gtest1(:,2),'*')

