clear; close all; clc
addpath(genpath('../src'))

% define different basis objects
basis = HermiteProbabilistPolyWithLinearization();
basis_orig = HermiteProbabilistPoly();

% set bounds
basis.bounds = [-3;3.5];

% define order and normalization
order = 5;
normt = true;

% samples for evaluating model
x = linspace(-10,10,1000).';

%% Check basis

% check Psi
Psi = basis.grad_vandermonde(x, order, 0, normt);

Psi_t = basis_orig.grad_vandermonde(x, order, 0, normt);
Psix_left   = basis_orig.grad_vandermonde(basis.bounds(1), order, 0, normt);
Psix_right  = basis_orig.grad_vandermonde(basis.bounds(2), order, 0, normt);
slope_left  = basis_orig.grad_vandermonde(basis.bounds(1), order, 1, normt);
slope_right = basis_orig.grad_vandermonde(basis.bounds(2), order, 1, normt);
Psi_t(x < basis.bounds(1),:) = Psix_left + slope_left.*(x(x < basis.bounds(1)) - basis.bounds(1));
Psi_t(x > basis.bounds(2),:) = Psix_right + slope_right.*(x(x > basis.bounds(2)) - basis.bounds(2));

assert(norm(Psi(:) - Psi_t(:)) < 1e-10)

%% Check first derivatives

% check dxPsi
dxPsi = basis.grad_vandermonde(x, order, 1, normt);

dxPsi_t = basis_orig.grad_vandermonde(x, order, 1, normt);
slope_left  = basis_orig.grad_vandermonde(basis.bounds(1), order, 1, normt);
slope_right = basis_orig.grad_vandermonde(basis.bounds(2), order, 1, normt);
dxPsi_t(x < basis.bounds(1),:) = repmat(slope_left, nnz(x < basis.bounds(1)),1);
dxPsi_t(x > basis.bounds(2),:) = repmat(slope_right,nnz(x > basis.bounds(2)),1);

assert(norm(dxPsi(:) - dxPsi_t(:)) < 1e-10)

%% Check second derivatives

% check d2xPsi
d2xPsi = basis.grad_vandermonde(x, order, 2, normt);

d2xPsi_t = basis_orig.grad_vandermonde(x, order, 2, normt);
d2xPsi_t(x < basis.bounds(1),:) = 0;
d2xPsi_t(x > basis.bounds(2),:) = 0;

assert(norm(d2xPsi(:) - d2xPsi_t(:)) < 1e-10)

% -- END OF FILE --