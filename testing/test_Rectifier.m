clear; close all; clc
addpath(genpath('../src'))

% define rectifier
basis = PhysicistHermiteFunction();
basis_poly = HermitePhysicistPoly();

% define g
g = Rectifier('softplus');

% define true functions
gt_evaluate = @(x) log(1 + 2.^(x))/log(2);
gt_gradx    = @(x) 1./(1 + 2.^(-x));
gt_hessx    = @(x) log(2)*2.^(-x)./(1 + 2.^(-x)).^2;
gt_inverse  = @(x) log(2.^(x) - 1)/log(2);

% define tolerance
tol = 1e-6;

% evaluate model
x = linspace(-100,100,1000);

%% Check evaluate

% evaluate g(x)
gx = g.evaluate(x);

% evaluate true g
gx_t = gt_evaluate(x);

assert(norm(gx(:) - gx_t(:)) < tol)

%% Check first derivative

% evaluate g'(x)
dxgx = g.grad_x(x);

% evaluate true g'
dxgx_t = gt_gradx(x);

assert(norm(dxgx(:) - dxgx_t(:)) < tol)

%% Check second derivative

% evaluate g''(x)
d2xgx = g.hess_x(x);

% evaluate true g''
d2xgx_t = gt_hessx(x);

assert(norm(d2xgx(:) - d2xgx_t(:)) < tol)

%% Check inverse

% evaluate g and inverse
gx = g.evaluate(x);
invgx = g.inverse(gx);
invgx(invgx == -inf) = 0;

% evaluate true inverse
invgx_t = gt_inverse(gx);
invgx_t(invgx_t == -inf) = 0;

assert(norm(invgx(:) - invgx_t(:)) < tol)

% -- END OF FILE --