clear; close all; clc
addpath(genpath('../../src'))

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
MP = MultivariatePoly(basis, multi_idxs);
MP = MP.set_coeff(coeff);

precomp = struct;
precomp.eval_basis    = MP.evaluate_basis(X);
precomp.grad_x_basis  = MP.grad_x_basis(X);
precomp.hess_x_basis  = MP.hess_x_basis(X);
precomp.grad_xd_basis = MP.grad_xk_basis(X, 1, MP.dim);
precomp.hess_xd_basis = MP.grad_xk_basis(X, 2, MP.dim);

%% Test coefficients

assert(all(size(MP.coeff) == [1,MP.n_coeff],'all'),'Coefficient size doesn''t match')
assert(norm(MP.coeff - coeff.') < tol, 'Coefficients don''t match')
assert(size(multi_idxs,1) == MP.n_coeff, 'Number of coefficients is incorrect')

%% Test evaluate

% check MP
Px = MP.evaluate(X);

% check precomp
pre_Px  = MP.evaluate(X, precomp); 

% check evaluations
Psi_x1  = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 0, true);
Psi_x2  = basis{2}.grad_vandermonde(X(:,2), max(multi_idxs(:,2)), 0, true);
Psi     = Psi_x1(:,multi_idxs(:,1)+1) .* Psi_x2(:,multi_idxs(:,2)+1);
Px_true = Psi * coeff;

assert(norm(Px(:) - Px_true(:)) < tol);
assert(norm(pre_Px(:) - Px_true(:)) < tol);

%% Test grad_xd, hess_xd

% check MP
dxPd   = MP.grad_xd(X);
d2xPd  = MP.hess_xd(X);

% check precomp
pre_dxPd   = MP.grad_xd(X, precomp);
pre_d2xPd  = MP.hess_xd(X, precomp);

% check evaluations
Psi_x1     = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 0, true);
dxPsi_x2   = basis{2}.grad_vandermonde(X(:,2), max(multi_idxs(:,2)), 1, true);
d2xPsi_x2  = basis{2}.grad_vandermonde(X(:,2), max(multi_idxs(:,2)), 2, true);
dxdPsi     = Psi_x1(:,multi_idxs(:,1)+1) .* dxPsi_x2(:,multi_idxs(:,2)+1);
d2xdPsi    = Psi_x1(:,multi_idxs(:,1)+1) .* d2xPsi_x2(:,multi_idxs(:,2)+1);
dxPd_true  = dxdPsi * coeff;
d2xPd_true = d2xdPsi * coeff;

assert(norm(dxPd(:) - dxPd_true(:)) < tol);
assert(norm(d2xPd(:) - d2xPd_true(:)) < tol);
assert(norm(pre_dxPd(:) - dxPd_true(:)) < tol);
assert(norm(pre_d2xPd(:) - d2xPd_true(:)) < tol);

%% Test grad_x, hess_x

% check MP
dxP    = MP.grad_x(X);
d2xP   = MP.hess_x(X);

% check precomp
pre_dxP    = MP.grad_x(X, [], precomp);
pre_d2xP   = MP.hess_x(X, [], precomp);

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

%% Test subset of grad_x, hess_x

% check MP
dxP    = MP.grad_x(X, 1);
d2xP   = MP.hess_x(X, 1);

% check precomp
pre_dxP    = MP.grad_x(X, 1, precomp);
pre_d2xP   = MP.hess_x(X, 1, precomp);

% check evaluations
Psi_x2    = basis{2}.grad_vandermonde(X(:,2), max(multi_idxs(:,2)), 0, true);
dxPsi_x1  = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 1, true);
dx1Psi    = dxPsi_x1(:,multi_idxs(:,1)+1) .* Psi_x2(:,multi_idxs(:,2)+1);
dxP_true  = dx1Psi * coeff;

d2xPsi_x1 = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 2, true);
d2x1Psi   = d2xPsi_x1(:,multi_idxs(:,1)+1) .* Psi_x2(:,multi_idxs(:,2)+1);
d2xP_true = d2x1Psi * coeff;

assert(norm(dxP(:) - dxP_true(:)) < tol);
assert(norm(d2xP(:) - d2xP_true(:)) < tol);
assert(norm(pre_dxP(:) - dxP_true(:)) < tol);
assert(norm(pre_d2xP(:) - d2xP_true(:)) < tol);

% check MP with empty gradient index
dxP      = MP.grad_x(X);
d2xP     = MP.hess_x(X);
dxP_e    = MP.grad_x(X, []);
d2xP_e   = MP.hess_x(X, []);

assert(norm(dxP(:) - dxP_e(:)) < tol);
assert(norm(d2xP(:) - d2xP_e(:)) < tol);

%% Test grad_x_grad_xd

% check MP
dxdxP   = MP.grad_x_grad_xd(X);
dxdxP_e = MP.grad_x_grad_xd(X, []);
dx2dxdP = MP.grad_x_grad_xd(X, 2);

% check precomp
pre_dxdxP   = MP.grad_x_grad_xd(X, [], precomp);
pre_dx2dxdP = MP.grad_x_grad_xd(X, 2, precomp);

% check evaluations
Psi_x1    = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 0, true);
dxPsi_x1  = basis{1}.grad_vandermonde(X(:,1), max(multi_idxs(:,1)), 1, true);
dxPsi_x2  = basis{2}.grad_vandermonde(X(:,2), max(multi_idxs(:,2)), 1, true);
d2xPsi_x2 = basis{2}.grad_vandermonde(X(:,2), max(multi_idxs(:,2)), 2, true);
dx12Psi   = dxPsi_x1(:,multi_idxs(:,1)+1) .* dxPsi_x2(:,multi_idxs(:,2)+1);
dx2Psi    = Psi_x1(:,multi_idxs(:,1)+1) .* d2xPsi_x2(:,multi_idxs(:,2)+1);
dxdxP_true = [dx12Psi * coeff, dx2Psi * coeff];
dx2dxdP_true = MP.hess_xd(X);

assert(norm(dxdxP(:) - dxdxP_true(:)) < tol);
assert(norm(dxdxP_e(:) - dxdxP_true(:)) < tol);
assert(norm(dx2dxdP(:) - dx2dxdP_true(:)) < tol);
assert(norm(pre_dxdxP(:) - dxdxP_true(:)) < tol);
assert(norm(pre_dx2dxdP(:) - dx2dxdP_true(:)) < tol);

%% Test grad_coeff, hess_coeff

test_coeffs = 1:MP.n_coeff;
test_coeffs = test_coeffs(1:2:end);

% check MP
dcP    = MP.grad_coeff(X, test_coeffs);
d2cP   = MP.hess_coeff(X, test_coeffs);

% check precomp
pre_dcP    = MP.grad_coeff(X, test_coeffs, precomp);
pre_d2cP   = MP.hess_coeff(X, test_coeffs, precomp);

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

test_coeffs = 1:MP.n_coeff;
test_coeffs = test_coeffs(1:2:end);

% check MP
dcdxP    = MP.grad_coeff_grad_xd(X, test_coeffs);
d2cdxP   = MP.hess_coeff_grad_xd(X, test_coeffs);

% check precomp
pre_dcdxP    = MP.grad_coeff_grad_xd(X, test_coeffs, precomp);
pre_d2cdxP   = MP.hess_coeff_grad_xd(X, test_coeffs, precomp);

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

% -- END OF FILE --