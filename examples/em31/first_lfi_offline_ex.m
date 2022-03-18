clear all; close all;

sd = 1; rng(sd);

% add paths
addpath(genpath('../src'))

% define parameters
d  = 3;     % dimension of unknown parameters
% define reference distribution
ref = IndependentProductDitribution({Normal(), Normal(),Normal()});
% setup map with greedy basis selection (start from S(x) = Id(x))

basis = ProbabilistHermiteFunction();
TM_off=total_order_map(1:d, ProbabilistHermiteFunction(), 2);
TM_off{1}.f.coeff=ones(1,length(TM_off{1}.f.coeff));
TM_off{2}.f.coeff=ones(1,length(TM_off{2}.f.coeff));
TM_off{3}.f.coeff=ones(1,length(TM_off{3}.f.coeff));

PB_off = PullbackDensity(TM_off, ref);

yobs=2;

lkl=LikelihoodFunction(PB_off, yobs);

mean=[0 0];
cov=[1 0;0 1];

prior=MultivariateGaussian(2, mean, cov);
pi=BayesianDistribution(prior, lkl);

X1=linspace(-5,5,100)';
X2=ones(100,1);

X=[X2 X1];

grad_pi=pi.grad_x_log_pdf(X);

figure
plot(X1,exp(pi.log_pdf(X)))

Y=pi.log_pdf(X);

DY=gradient(Y,X1);


figure
plot(X1,grad_pi(:,1))
hold on
plot(X1,grad_pi(:,2))
plot(X1,DY)


