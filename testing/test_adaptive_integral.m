% testing adaptive_integral function

clear; close all; clc
addpath(genpath('../../src'))

% define integration tol 
tol = 1e-6;

%% Test simple function

% define function
f = @(x) [2*sin(10*x), 2*cos(10*x)];

% define pts
N = 1000;
a = zeros(N,1);
b = 2*randn(N,1);

% evaluate true integral
I_true = zeros(N,2);
for i=1:N
    I_true(i,:) = integral(f, a(i), b(i), 'ArrayValued', true);
end

% define function to evaluate integration points
int_pts = @(N) int_pts_simple(N, a, b);

% estimate integrals
[I1, ~] = adaptive_integral(f, int_pts);
assert(norm(I1(:) - I_true(:)) < tol)

%%
% check if cache is empty
[I2, cache_xi_e, ~, ~] = adaptive_integral(f, int_pts, [], [], []);
assert(norm(I2(:) - I_true(:)) < tol)

% check if cache is included as argument
cache_xi = cell(10,1);
cache_wi = cell(10,1);
cache_zi = cell(10,1);
[cache_xi{1}, cache_wi{1}, cache_zi{1}] = int_pts(6);
[I3, cache_xi, cache_wi, cache_zi] = adaptive_integral(f, int_pts, cache_xi, cache_wi, cache_zi);
assert(norm(I3(:) - I_true(:)) < tol)

% check number of points
assert(numel(cache_xi_e{2}) == 6*2)
assert(numel(cache_xi{2}) == 6*2)

% check if previous cache is included
[I4, ~] = adaptive_integral(f, int_pts, cache_xi, cache_wi, cache_zi);
assert(norm(I4(:) - I_true(:)) < tol)

% check if tolerance is reduced
[I5, cache] = adaptive_integral(f, int_pts, cache_xi, cache_wi, cache_zi, [10, 10]);
assert(norm(I5(:) - I_true(:)) < 1e-1)

%% Test function of polynomial

% define polynomial
d = 3;
basis = repmat({ProbabilistHermiteFunction()},d,1);
m_idx = TotalOrderMultiIndices(2*ones(1,d));
P = ParametericPoly(basis, m_idx);
P = P.set_coeff(randn(P.n_coeff,1));

% define nonlinear function
gf = @(fx) log(1 + exp(fx * P.coeff'));
g = @(x) log(1 + exp(x));

% define pts
N = 1000;
a = zeros(N,1);
X = 2*randn(N,3);

% define function for integration points
precomp = PPprecomp();
int_pts = @(N) precomp.evaluate_quadrature_Psi(P, X, N);

% evaluate integral
I = adaptive_integral(gf, int_pts, [], [], [], [1e-6, 1e-6]);

% compute integrals exactly
I_true = zeros(N, 1);
for i=1:N
    Xi = X(i,1:end-1);
    dxPi = @(x) P.grad_xd([repmat(Xi,length(x),1),x.']).';
    I_true(i) = integral(@(x) g(dxPi(x)), 0, X(i,end));
end
assert(norm(I(:) - I_true(:)) < tol)

% evaluate integral
I = adaptive_integral(g, int_pts, [], [], [], [1e-6, 1e-6]);

% evaluate integral with empty precomp
I_pre = adaptive_integral(g, int_pts, precomp.quad_dxPsii, precomp.quad_wi, precomp.quad_xi, [1e-6, 1e-6]);

% compute integrals exactly
I_true = zeros(N, P.n_coeff);
for i=1:N
    Xi = X(i,1:end-1);
    dxPi = @(x) P.evaluate_offdiagbasis([repmat(Xi,length(x),1),x.']) .* P.grad_xd_diagbasis([repmat(Xi,length(x),1),x.']);
    I_true(i,:) = integral(@(x) g(dxPi(x)), 0, X(i,end), 'ArrayValued', true);
end
assert(norm(I(:) - I_true(:)) < tol)
assert(norm(I_pre(:) - I_true(:)) < tol)

% evaluate precompute
precomp = precomp.evaluate(P, X);

% evaluate integral with precomp
[I, ~] = adaptive_integral(g, int_pts, precomp.quad_dxPsii, precomp.quad_wi, precomp.quad_xi, [1e-6, 1e-6]);
assert(norm(I(:) - I_true(:)) < tol)

%% --- Helper Functions ---

function [xi,wi,zi] = int_pts_simple(N, a, b)

    [xcc, wcc] = clenshaw_curtis(N);

    % sum contribution from each integration node
    xi = cell(N,1); wi = cell(N,1);
    for i=1:N
        [xi{i}, wi{i}] = rescale_pts(a, b, xcc(i), wcc(i));
    end
    
    % set zi = xi for simple function
    zi = xi;
    
end

function [fi,wi,xi] = int_pts_sotftplus_poly_xd(precomp, P, X, N)

    % use precomp function to evaluate quadrature points
     [fi,wi,xi] = precomp.evaluate_quadrature_Psi(P, X, N);
     
end

% -- END OF FILE --