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
Z=exp(prior.log_pdf([X(:) Y(:)]));

figure
for k=1:50
    subplot(5,10,k)
    yobs=List_obs(k);
    CM=load(['./offline_maps/offline_map_poly_linear_neat2_',num2str(k),'.mat']);
    CM=CM.x;
    L_M={CM.S{1}.S CM.S{2}.S};

    CM=ComposedPullbackDensity(L_M,ref2);

    lkl = LikelihoodFunction(CM, yobs);

    Z=Z.*exp(lkl.log([X(:) Y(:)]));
    pcolor(X,Y,reshape(Z,50,50))
    title(['TM post. perf.',num2str(k)])
    shading interp
end


x2=linspace(2.3,2.8,100);
x1=linspace(0.3,1.6,100);
[X,Y]=meshgrid(x1,x2);
Z=exp(prior.log_pdf([X(:) Y(:)]));

% x2=linspace(0.8,3.8,50);
% x1=linspace(-3,3,50);
% [X,Y]=meshgrid(x1,x2);
% Z=exp(prior.log_pdf([X(:) Y(:)]));


for k=1:47
    yobs=List_obs(k);
    CM=load(['./offline_maps/offline_map_poly_linear_neat2_',num2str(k),'.mat']);
    CM=CM.x;
    L_M={CM.S{1}.S CM.S{2}.S};

    CM=ComposedPullbackDensity(L_M,ref2);

    lkl = LikelihoodFunction(CM, yobs);

    Z=Z.*exp(lkl.log([X(:) Y(:)]));
end

figure
surf(X,Y,reshape(Z,100,100))
title('TM final post. perf.')

X=randn(100000,2);

CM_post_nois=load('CM_post_noise_10.mat');
CM_post_noise=CM_post_nois.CM_post;

CM_post_perf=load('CM_post_perf.mat');
CM_post_perf=CM_post_perf.CM_post;

Y_noise=CM_post_noise.evaluate(X);

% hold on
% ksdensity(Y_noise,'PlotFcn','contour');
% xlim([0.3 2])
% ylim([2.1 2.8])

L_best_var1=load('L_best_var_noise_10.mat');
L_best_var1=L_best_var1.L_best_var;
L_best_var2=load('L_best_var_perf.mat');
L_best_var2=L_best_var2.L_best_var;
L_terms2=load('L_terms_perf.mat');
L_terms2=L_terms2.L_idxs;
L_terms1=load('L_terms_10.mat');
L_terms1=L_terms1.L_idxs;


figure
plot(1:50,L_best_var1)
hold on
plot(1:50,L_best_var2)

figure
plot(1:50,L_terms1)
hold on
plot(1:50,L_terms2)
