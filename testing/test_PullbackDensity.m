clear; close all; clc
addpath(genpath('../src'))

%% Test definition of TM with product reference 

% generate samples 
d = 2;              % dimension of unknown parameters
M = 1000;           % number of samples
X = randn(M,2);     

% define basis
basis_1D = ConstExtProbabilistHermiteFunction();
basis = {basis_1D, basis_1D};

% define transport map
TM = total_order_map(1:d, basis, 2);

% define reference
ref_1d = Normal();
ref = IndependentProductDitribution({ref_1d, ref_1d});

% define PB
PB  = PullbackDensity(TM, ref);

%% Test coefficient

nc = PB.n_coeff();
c = PB.coeff();
midx1 = PB.S{1}.multi_idxs();
midx2 = PB.S{2}.multi_idxs();

%% Test log_pdf

log_pi = PB.log_pdf(X);

% evaluate true log-likelihood
Sx = PB.evaluate(X);
dxSx = PB.grad_x(X);
log_pi_c = sum(log(normpdf(Sx)),2) + log(dxSx(:,1,1)) + log(dxSx(:,2,2));

assert(norm(log_pi(:) - log_pi_c(:)) < tol);

%% Test optimize with Gaussian reference

PB = PB.optimize(X);

comp = 2;
PB = PB.optimize(X, comp);

%% Test optimize with non-product reference

% define transport map
S2 = total_order_map(1:d, basis, 2);
TM2 = TriangularTransportMap(S2);

% define PB
ref = PB;
PB2 = PullbackDensity(TM2, ref); PB2.d = 2;

%% Test coefficient

nc2 = PB2.n_coeff();
c2 = PB2.coeff();
midx21 = PB2.S{1}.multi_idxs();
midx22 = PB2.S{2}.multi_idxs();

%% Test log_pdf

log_pi = PB2.log_pdf(X);

% evaluate true log-likelihood
Sx = PB2.evaluate(X);
dxSx = PB2.grad_x(X);
refx = PB.log_pdf(Sx);
log_pi_c = refx + log(dxSx(:,1,1)) + log(dxSx(:,2,2));

assert(norm(log_pi(:) - log_pi_c(:)) < tol);

%% Test optimize with Gaussian reference

PB2 = PB2.optimize(X);

% -- END OF TIME --