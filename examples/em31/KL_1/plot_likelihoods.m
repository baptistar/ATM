clear; close all; clc
addpath(genpath('../../../src'))

sd = 2; rng(sd);


X=load('X_2d_7.txt');
W=load('W_2d_7.txt');

XW.X=X;
XW.W=W;

%XW=randn(20000,2);
%List_obs=load('obs_wihtout_noise.txt')+7*randn(50,1);
List_obs=load('List_obs_neat.txt');
x=linspace(-5,5,100)';

d=2;
ref2 = IndependentProductDistribution(repmat({Normal()},1,d+1));
x2=linspace(0.8,3.8,50);
x1=linspace(-3,3,50);
[X,Y]=meshgrid(x1,x2);
figure
for k=1:50
    yobs=List_obs(k);
    CM=load(['./offline_maps/offline_map_poly_linear_neat2_',num2str(k),'.mat']);
    CM=CM.x;
    L_M={CM.S{1}.S CM.S{2}.S};

    CM=ComposedPullbackDensity(L_M,ref2);

    lkl = LikelihoodFunction(CM, yobs);

    Z=exp(lkl.log([X(:) Y(:)]));
    subplot(5,10,k)
    pcolor(X,Y,reshape(Z,50,50))
    title(['likelihood ', num2str(k)])
    shading interp
end





