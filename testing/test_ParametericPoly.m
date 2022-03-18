clear; close all; clc
addpath(genpath('../src'))

% set tolerance
tol = 1e-10;

% generate samples 
d = 2;     % dimension of unknown parameters
M = 1000;  % number of samples
X = randn(M,2);

% define basis
basis_1D = ConstExtProbabilistHermiteFunction();
basis = {basis_1D, basis_1D};

% define multi-indices
multi_idxs = [0,0; 0,1; 1,0; 1,1; 1,2];
coeff = randn(size(multi_idxs,1),1);

% define object
PP = ParametericPoly(basis, multi_idxs);
PP = PP.set_coeff(coeff);

precomp = PPprecomp();
precomp.eval_offdiagbasis = PP.evaluate_offdiagbasis(X);
precomp.eval_diagbasis = PP.evaluate_diagbasis(X);
precomp.grad_x_offdiagbasis = PP.grad_x_offdiagbasis(X);
precomp.hess_x_offdiagbasis = PP.hess_x_offdiagbasis(X);
precomp.grad_xd_diagbasis = PP.grad_xd_diagbasis(X);
precomp.hess_xd_diagbasis = PP.hess_xd_diagbasis(X);

%% Test coefficients

assert(all(size(PP.coeff) == [1,PP.n_coeff],'all'),'Coefficient size doesn''t match')
assert(norm(PP.coeff - coeff.') < tol, 'Coefficients don''t match')
assert(size(multi_idxs,1) == PP.n_coeff, 'Number of coefficients is incorrect')

%% Test evaluate

% check PP
Px = PP.evaluate(X);

% check precomp
pre_Px  = PP.evaluate(X, precomp); 

% check evaluations
Psi_x1  = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 0, true);
Psi_x2  = basis{2}.grad_vandermonde(X(:,2), max(multi_idxs(:,2)), 0, true);
Psi     = Psi_x1(:,multi_idxs(:,1)+1) .* Psi_x2(:,multi_idxs(:,2)+1);
Px_true = Psi * coeff;

assert(norm(Px(:) - Px_true(:)) < tol);
assert(norm(pre_Px(:) - Px_true(:)) < tol);

%% Test grad_xd, hess_xd, grad_x_grad_xd

% check PP
dxPd   = PP.grad_xd(X);
d2xPd  = PP.hess_xd(X);
dxdxPd = PP.grad_x_grad_xd(X);

% check precomp
pre_dxPd   = PP.grad_xd(X, precomp);
pre_d2xPd  = PP.hess_xd(X, precomp);
pre_dxdxPd = PP.grad_x_grad_xd(X, [], precomp);

% check evaluations
Psi_x1     = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 0, true);
dxPsi_x1   = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 1, true);
dxPsi_x2   = basis{2}.grad_vandermonde(X(:,2), max(multi_idxs(:,2)), 1, true);
d2xPsi_x2  = basis{2}.grad_vandermonde(X(:,2), max(multi_idxs(:,2)), 2, true);
dxdPsi     = Psi_x1(:,multi_idxs(:,1)+1) .* dxPsi_x2(:,multi_idxs(:,2)+1);
d2xdPsi    = Psi_x1(:,multi_idxs(:,1)+1) .* d2xPsi_x2(:,multi_idxs(:,2)+1);
dxdxdPsi   = dxPsi_x1(:,multi_idxs(:,1)+1) .* dxPsi_x2(:,multi_idxs(:,2)+1);

dxPd_true  = dxdPsi * coeff;
d2xPd_true = d2xdPsi * coeff;

dxdxPd_true = zeros(size(X,1), PP.dim);
dxdxPd_true(:,1) = dxdxdPsi * coeff;
dxdxPd_true(:,2) = d2xPd_true;

assert(norm(dxPd(:) - dxPd_true(:)) < tol);
assert(norm(d2xPd(:) - d2xPd_true(:)) < tol);
assert(norm(dxdxPd(:) - dxdxPd_true(:)) < tol);
assert(norm(pre_dxPd(:) - dxPd_true(:)) < tol);
assert(norm(pre_d2xPd(:) - d2xPd_true(:)) < tol);
assert(norm(pre_dxdxPd(:) - dxdxPd_true(:)) < tol);

%% Test grad_x, hess_x

% check PP
dxP    = PP.grad_x(X);
d2xP   = PP.hess_x(X);

% check precomp
pre_dxP    = PP.grad_x(X, [], precomp);
pre_d2xP   = PP.hess_x(X, [], precomp);

% check evaluations
Psi_x1  = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 0, true);
Psi_x2  = basis{2}.grad_vandermonde(X(:,2), max(multi_idxs(:,2)), 0, true);
dxPsi_x1   = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 1, true);
dxPsi_x2  = basis{2}.grad_vandermonde(X(:,2), max(multi_idxs(:,2)), 1, true);
dx1Psi     = dxPsi_x1(:,multi_idxs(:,1)+1) .* Psi_x2(:,multi_idxs(:,2)+1);
dx2Psi     = Psi_x1(:,multi_idxs(:,1)+1) .* dxPsi_x2(:,multi_idxs(:,2)+1);
dxP_true   = [dx1Psi * coeff, dx2Psi * coeff];

d2xPsi_x1  = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 2, true);
d2xPsi_x2  = basis{1}.grad_vandermonde(X(:,2), max(multi_idxs(:,2)), 2, true);
d2x1Psi    = d2xPsi_x1(:,multi_idxs(:,1)+1) .* Psi_x2(:,multi_idxs(:,2)+1);
dx12Psi    = dxPsi_x1(:,multi_idxs(:,1)+1) .* dxPsi_x2(:,multi_idxs(:,2)+1);
d2x2Psi    = Psi_x1(:,multi_idxs(:,1)+1) .* d2xPsi_x2(:,multi_idxs(:,2)+1);
d2xP_true  = zeros(M, d, d);
d2xP_true(:,1,1) = d2x1Psi * coeff;
d2xP_true(:,1,2) = dx12Psi * coeff;
d2xP_true(:,2,1) = dx12Psi * coeff;
d2xP_true(:,2,2) = d2x2Psi * coeff;

assert(norm(dxP(:) - dxP_true(:)) < tol);
assert(norm(d2xP(:) - d2xP_true(:)) < tol);
assert(norm(pre_dxP(:) - dxP_true(:)) < tol);
assert(norm(pre_d2xP(:) - d2xP_true(:)) < tol);

%% Test grad_coeff, hess_coeff

test_coeffs = 1:PP.n_coeff;
test_coeffs = test_coeffs(1:2:end);

% check PP
dcP    = PP.grad_coeff(X, test_coeffs);
d2cP   = PP.hess_coeff(X, test_coeffs);

% check precomp
pre_dcP    = PP.grad_coeff(X, test_coeffs, precomp);
pre_d2cP   = PP.hess_coeff(X, test_coeffs, precomp);

% check evaluations
Psi_x1  = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 0, true);
Psi_x2  = basis{2}.grad_vandermonde(X(:,2), max(multi_idxs(:,2)), 0, true);
Psi     = Psi_x1(:,multi_idxs(:,1)+1) .* Psi_x2(:,multi_idxs(:,2)+1);
dcP_true = Psi(:,test_coeffs);
d2cP_true = zeros(M, length(test_coeffs), length(test_coeffs));

assert(norm(dcP(:) - dcP_true(:)) < tol);
assert(norm(d2cP(:) - d2cP_true(:)) < tol);
assert(norm(pre_dcP(:) - dcP_true(:)) < tol);
assert(norm(pre_d2cP(:) - d2cP_true(:)) < tol);

%% Test grad_coeff_grad_xd, hess_coeff_grad_xd

test_coeffs = 1:PP.n_coeff;
test_coeffs = test_coeffs(1:2:end);

% check PP
dcdxP    = PP.grad_coeff_grad_xd(X, test_coeffs);
d2cdxP   = PP.hess_coeff_grad_xd(X, test_coeffs);

% check precomp
pre_dcdxP    = PP.grad_coeff_grad_xd(X, test_coeffs, precomp);
pre_d2cdxP   = PP.hess_coeff_grad_xd(X, test_coeffs, precomp);

% check evaluations
Psi_x1  = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 0, true);
dxPsi_x2  = basis{2}.grad_vandermonde(X(:,2), max(multi_idxs(:,2)), 1, true);
Psi     = Psi_x1(:,multi_idxs(:,1)+1) .* dxPsi_x2(:,multi_idxs(:,2)+1);
dcdxP_true = Psi(:,test_coeffs);
d2cdxP_true = zeros(M, length(test_coeffs), length(test_coeffs));

assert(norm(dcdxP(:) - dcdxP_true(:)) < tol);
assert(norm(d2cdxP(:) - d2cdxP_true(:)) < tol);
assert(norm(pre_dcdxP(:) - dcdxP_true(:)) < tol);
assert(norm(pre_d2cdxP(:) - d2cdxP_true(:)) < tol);

%% evaluate_f0, grad_x_evaluate_f0, hess_x_evaluate_f0, hess_coeff_f0, grad_coeff_f0

test_coeffs = 1:PP.n_coeff;
test_coeffs = test_coeffs(1:2:end);

% check PP
P0 = PP.evaluate_f0(X);
dcP0 = PP.grad_coeff_f0(X, test_coeffs);
d2cP0 = PP.hess_coeff_f0(X, test_coeffs);
dxP0 = PP.grad_x_evaluate_f0(X);
d2xP0 = PP.hess_x_evaluate_f0(X);

% check precomp
pre_P0  = PP.evaluate_f0(X, precomp);
pre_dcP0 = PP.grad_coeff_f0(X, test_coeffs, precomp);
pre_d2cP0 = PP.hess_coeff_f0(X, test_coeffs, precomp);
pre_dxP0 = PP.grad_x_evaluate_f0(X, [], precomp);
pre_d2xP0 = PP.hess_x_evaluate_f0(X, [], precomp);

% check evaluations
Psi_x1 = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 0, true);
dxPsi_x1 = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 1, true);
d2xPsi_x1 = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 2, true);
Psi_x2 = basis{2}.grad_vandermonde(zeros(size(X,1),1), max(multi_idxs(:,2)), 0, true);
Psi0 = Psi_x1(:,multi_idxs(:,1)+1) .* Psi_x2(:,multi_idxs(:,2)+1);
dx1Psi0 = (dxPsi_x1(:,multi_idxs(:,1)+1) .* Psi_x2(:,multi_idxs(:,2)+1));
d2x1Psi0 = (d2xPsi_x1(:,multi_idxs(:,1)+1) .* Psi_x2(:,multi_idxs(:,2)+1));

P0_true = Psi0 * PP.coeff.';
dcP0_true = Psi0(:,test_coeffs);
d2cP0_true = zeros(size(X,1), length(test_coeffs), length(test_coeffs));
dxP0_true = zeros(size(X,1), PP.dim);
dxP0_true(:,1) = dx1Psi0 * PP.coeff.';
d2xP0_true = zeros(size(X,1), PP.dim, PP.dim);
d2xP0_true(:,1,1) = d2x1Psi0 * PP.coeff.';

assert(norm(P0(:) - P0_true(:)) < tol);
assert(norm(dcP0(:) - dcP0_true(:)) < tol);
assert(norm(d2cP0(:) - d2cP0_true(:)) < tol);
assert(norm(dxP0(:) - dxP0_true(:)) < tol);
assert(norm(d2xP0(:) - d2xP0_true(:)) < tol);

assert(norm(pre_P0(:) - P0_true(:)) < tol);
assert(norm(pre_dcP0(:) - dcP0_true(:)) < tol);
assert(norm(pre_d2cP0(:) - d2cP0_true(:)) < tol);
assert(norm(pre_dxP0(:) - dxP0_true(:)) < tol);
assert(norm(pre_d2xP0(:) - d2xP0_true(:)) < tol);

% -- END OF FILE --