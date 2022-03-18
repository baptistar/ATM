clear; clc; close all;

sd = 1; rng(sd);

% add paths
addpath(genpath('../src'))

% define parameters
d  = 3;     % dimension of unknown parameters
% define reference distribution
ref = IndependentProductDitribution({Normal(), Normal(),Normal()});
% setup map with greedy basis selection (start from S(x) = Id(x))

basis = ProbabilistHermiteFunction();
TM=total_order_map(1:d, ProbabilistHermiteFunction(), 2);

TM{1}.f.coeff=ones(1,length(TM{1}.f.coeff));
TM{2}.f.coeff=ones(1,length(TM{2}.f.coeff));
TM{3}.f.coeff=ones(1,length(TM{3}.f.coeff));

PB = PullbackDensity(TM, ref);

X=linspace(-5,5,100)';

yobs=2;

%Example 1
x_jt=[ones(length(X),1), X, repmat(yobs,length(X),1)];

logx=PB.log_pdf(x_jt,3);
grad_logpi=PB.grad_x_log_pdf(x_jt,2,3);
grad_logpi2=gradient(logx,X);
grad_logpi3=PB.grad_x_log_pdf(x_jt,1:2,3);


figure
plot(X,grad_logpi)
hold on
plot(X,grad_logpi2)
plot(X,grad_logpi3(:,2))

