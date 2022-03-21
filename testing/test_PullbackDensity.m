clear; close all; clc
addpath(genpath('../src'))

% generate samples 
d = 3;              % dimension of unknown parameters
M = 1000;           % number of samples
order = 2;
X = randn(M,d);     

% define tolerance
tol = 1e-5;
h = 1e-5;

% define basis
basis_1D = ConstExtProbabilistHermiteFunction();
basis = repmat({basis_1D},d,1);

% define transport map
TM = total_order_map(1:d, basis, order);

% define reference
ref_1d = Normal();
ref = IndependentProductDistribution(repmat({ref_1d},d,1));

% define PB
PB  = PullbackDensity(TM, ref);

% set coefficients
for k=1:d
    PB.S.S{k} = PB.S.S{k}.set_coeff(ones(1,PB.S.S{k}.n_coeff));
end

% Test optimize with Gaussian reference
PB = PB.optimize(X);
comp = 2;
PB = PB.optimize(X, comp);

% define transport map with non-product reference
S2 = total_order_map(1:d, basis, d);
TM2 = TriangularTransportMap(S2);

% define PB
ref = PB;
PB2 = PullbackDensity(TM2, ref);

% set coefficients
for k=1:d
    PB2.S.S{k} = PB2.S.S{k}.set_coeff(ones(1,PB2.S.S{k}.n_coeff));
end

%% Test log_pdf

log_pi = PB.log_pdf(X);

% evaluate true log-likelihood
Sx = PB.S.evaluate(X);
dxSx = PB.S.grad_x(X);
log_pi_c = sum(log(normpdf(Sx)),2);
for k=1:d
    log_pi_c = log_pi_c + log(dxSx(:,k,k));
end

assert(norm(log_pi(:) - log_pi_c(:)) < tol);

%% Test grad_x_log_pdf

log_pi = PB.log_pdf(X);
dx_log_pi = PB.grad_x_log_pdf(X);

% evaluate gradient using finite differences
dx_log_pi_t = zeros(size(X,1), d);
for k=1:d
    Xk = X; Xk(:,k) = Xk(:,k) + h;
    log_pik = PB.log_pdf(Xk);
    dx_log_pi_t(:,k) = (log_pik - log_pi)/h;
end

assert(norm(dx_log_pi(:) - dx_log_pi_t(:))/norm(dx_log_pi(:)) < tol);

%% Test hess_x_log_pdf

log_pi = PB.log_pdf(X);
d2x_log_pi = PB.hess_x_log_pdf(X);

% evaluate gradient using finite differences
d2x_log_pi_t = zeros(size(X,1), d, d);
for k=1:d
    for l=1:d
        Xpkl = X; Xpkl(:,k) = Xpkl(:,k) + h; Xpkl(:,l) = Xpkl(:,l) + h;
        log_pi_pkl = PB.log_pdf(Xpkl);
        Xpk = X; Xpk(:,k) = Xpk(:,k) + h;
        log_pi_pk = PB.log_pdf(Xpk);
        Xpl = X; Xpl(:,l) = Xpl(:,l) + h;
        log_pi_pl = PB.log_pdf(Xpl);
        Xmkl = X; Xmkl(:,k) = Xmkl(:,k) - h; Xmkl(:,l) = Xmkl(:,l) - h;
        log_pi_mkl = PB.log_pdf(Xmkl);
        Xmk = X; Xmk(:,k) = Xmk(:,k) - h;
        log_pi_mk = PB.log_pdf(Xmk);
        Xml = X; Xml(:,l) = Xml(:,l) - h;
        log_pi_ml = PB.log_pdf(Xml);
        d2x_log_pi_t(:,k,l) = (log_pi_pkl - log_pi_pk - log_pi_pl + 2*log_pi - log_pi_mk - log_pi_ml + log_pi_mkl)/(2*tol^2);
    end
end

assert(norm(d2x_log_pi(:) - d2x_log_pi_t(:))/norm(d2x_log_pi(:)) < 1e-4);

%% Test grad_x_log_pdf with multiple inputs

comp_idx = 2;

% define input
N = 10000;

for i=1:d
    % define input
    Xi = ones(N,d); Xi(:,i) = linspace(-5,5,N)';

    % compute gradients of log_pdf
    logx = PB.log_pdf(Xi, comp_idx);
    grad_x_logpi = PB.grad_x_log_pdf(Xi, i, comp_idx);
    grad_x_logpi_fd = gradient(logx, Xi(:,i));

    % assess error
    err = norm(grad_x_logpi(:) - grad_x_logpi_fd(:));
    if norm(grad_x_logpi(:)) ~= 0
        err = err/norm(grad_x_logpi(:));
    end
    assert(err < tol)
end

%% Test log_pdf

log_pi = PB2.log_pdf(X);

% evaluate true log-likelihood
Sx = PB2.S.evaluate(X);
dxdSx = PB2.S.grad_xd(X);
log_pi_c = PB2.ref.log_pdf(Sx) + sum(log(dxdSx),2);

assert(norm(log_pi(:) - log_pi_c(:)) < tol);

%% Test grad_x_log_pdf

log_pi = PB2.log_pdf(X);
dx_log_pi = PB2.grad_x_log_pdf(X);

% evaluate gradient using finite differences
dx_log_pi_t = zeros(size(X,1), d);
for k=1:d
    Xk = X; Xk(:,k) = Xk(:,k) + h;
    log_pik = PB2.log_pdf(Xk);
    dx_log_pi_t(:,k) = (log_pik - log_pi)/h;
end

assert(norm(dx_log_pi(:) - dx_log_pi_t(:))/norm(dx_log_pi(:)) < tol);

%% Test hess_x_log_pdf

log_pi = PB2.log_pdf(X);
d2x_log_pi = PB2.hess_x_log_pdf(X);

% evaluate gradient using finite differences
d2x_log_pi_t = zeros(size(X,1), d, d);
for k=1:d
    for l=1:d
        Xpkl = X; Xpkl(:,k) = Xpkl(:,k) + h; Xpkl(:,l) = Xpkl(:,l) + h;
        log_pi_pkl = PB2.log_pdf(Xpkl);
        Xpk = X; Xpk(:,k) = Xpk(:,k) + h;
        log_pi_pk = PB2.log_pdf(Xpk);
        Xpl = X; Xpl(:,l) = Xpl(:,l) + h;
        log_pi_pl = PB2.log_pdf(Xpl);
        Xmkl = X; Xmkl(:,k) = Xmkl(:,k) - h; Xmkl(:,l) = Xmkl(:,l) - h;
        log_pi_mkl = PB2.log_pdf(Xmkl);
        Xmk = X; Xmk(:,k) = Xmk(:,k) - h;
        log_pi_mk = PB2.log_pdf(Xmk);
        Xml = X; Xml(:,l) = Xml(:,l) - h;
        log_pi_ml = PB2.log_pdf(Xml);
        d2x_log_pi_t(:,k,l) = (log_pi_pkl - log_pi_pk - log_pi_pl + 2*log_pi - log_pi_mk - log_pi_ml + log_pi_mkl)/(2*tol^2);
    end
end

assert(norm(d2x_log_pi(:) - d2x_log_pi_t(:))/norm(d2x_log_pi(:)) < 1e-4);

%% Test grad_x_log_pdf with multiple inputs

comp_idx = 2;

% define input
N = 10000;

for i=1:d
    % define input
    Xi = ones(N,d); Xi(:,i) = linspace(-5,5,N)';

    % compute gradients of log_pdf
    logx = PB2.log_pdf(Xi, comp_idx);
    grad_x_logpi = PB2.grad_x_log_pdf(Xi, i, comp_idx);
    grad_x_logpi_fd = gradient(logx, Xi(:,i));

    % assess error
    err = norm(grad_x_logpi(:) - grad_x_logpi_fd(:));
    if norm(grad_x_logpi(:)) ~= 0
        err = err/norm(grad_x_logpi(:));
    end
    assert(err < tol)
end

%% Test optimize with Gaussian reference

PB2 = PB2.optimize(X);

% only optimizing the second component should fail
comp = 2;
PB2 = PB2.optimize(X, comp);

% -- END OF TIME --