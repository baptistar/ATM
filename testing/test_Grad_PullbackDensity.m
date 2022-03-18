clear; clc; close all;
sd = 1; rng(sd);
% add paths
addpath(genpath('~/Documents/MATLAB/AdaptiveTransportMaps/src'))
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

%Example 1 for gradient of map
x_jt=[ones(length(X),1), X, repmat(yobs,length(X),1)];
Sx=PB.evaluate(x_jt,3);
dSx = PB.grad_x(x_jt,2,3);
dSx2=gradient(Sx,x_jt(:,2));
figure
plot(X,dSx)
hold on
plot(X,dSx2,'--')

%Example 1 for log-det-Jacobian of map
%x_jt=[X,ones(100,2)];
x_jt = [ones(length(X),1), X, repmat(yobs,length(X),1)];
logDJ=PB.logdet_Jacobian(x_jt,3);
dxlogDJ = PB.grad_x_logdet_Jacobian(x_jt,2,3);
dxlogDJ2 = gradient(logDJ,x_jt(:,2));
figure
plot(X,dxlogDJ)
hold on
plot(X,dxlogDJ2,'--')

%Example 1
x_jt=[ones(length(X),1), X, repmat(yobs,length(X),1)];
logx=PB.log_pdf(x_jt,3);
grad_logpi=PB.grad_x_log_pdf(x_jt,2,3);
grad_logpi2=gradient(logx,X);
figure
plot(X,grad_logpi)
hold on
plot(X,grad_logpi2,'--')

%Example 2
x_jt=[X, ones(length(X),1), repmat(yobs,length(X),1)];
logx=PB.log_pdf(x_jt,2:3);
grad_logpi=PB.grad_x_log_pdf(x_jt,1,2:3);
grad_logpi2=gradient(logx,X);
figure
plot(X,grad_logpi)
hold on
plot(X,grad_logpi2,'--')

%Example 3
X2 = X;%randn(length(X),1);
x_jt=[ones(length(X),1), X2, repmat(yobs,length(X),1)];
logx=PB.log_pdf(x_jt,2);
grad_logpi=PB.grad_x_log_pdf(x_jt,2,2);
grad_logpi2=PB.grad_x_log_pdf(x_jt,1:2,2);
grad_logpi2=grad_logpi2(:,2);
grad_logpi3=gradient(logx,X2);
figure
plot(X,grad_logpi)
hold on
plot(X,grad_logpi2,'--')
plot(X,grad_logpi3,':')

%Example 4 with reference given by general pullback density
PB2=PullbackDensity(PB.S,PB);
X=linspace(1.8,3,100)';
logpi = PB2.log_pdf([X ones(length(X),1) ones(length(X),1)],2);
grad_logpi=PB2.grad_x_log_pdf([X ones(length(X),1) ones(length(X),1)],1,2);
grad_logpi2=gradient(logpi,X);
figure
plot(X,grad_logpi)
hold on
plot(X,grad_logpi2,'--')