clear; close all; clc
addpath(genpath('../src'))

% generate samples 
d = 3;              % dimension of unknown parameters
M = 1000;           % number of samples
order = 2;
X = randn(M,d);     

% define tolerance
tol = 1e-5;

% set fd tolerance
h = 1e-5;

% define basis
basis_1D = ConstExtProbabilistHermiteFunction();
basis = repmat({basis_1D},d,1);

% define transport map
S = total_order_map(1:d, basis, order);
TM = TriangularTransportMap(S);

% set coefficients
for k=1:d
    TM.S{k} = TM.S{k}.set_coeff(ones(1,TM.S{k}.n_coeff));
end

%% Test evaluate

% evaluate true map
Sx = TM.evaluate(X);

Sx_t = zeros(M,d);
for k=1:d
    Sx_t(:,k) = TM.S{k}.evaluate(X(:,1:k));
end

assert(norm(Sx(:) - Sx_t(:))/norm(Sx(:)) < tol)

%% Test inverse

% Compute inverse and compute to original inputs
Zref = randn(M,d);
Xref = TM.inverse(Zref);
Z = TM.evaluate(Xref);

assert(norm(Zref(:) - Z(:))/norm(Zref(:)) < tol)

%% Test grad_x

Sx = TM.evaluate(X);
dx_Sx = TM.grad_x(X);

% evaluate gradient using finite differences
dx_Sx_t = zeros(size(X,1), d, d);
for k=1:d
    Xk = X; Xk(:,k) = Xk(:,k) + h;
    Sx_pk = TM.evaluate(Xk);
    dx_Sx_t(:,:,k) = (Sx_pk - Sx)/h;
end

assert(norm(dx_Sx(:) - dx_Sx_t(:))/norm(dx_Sx(:)) < tol);

% check individual components
comp_idx = 2;

for i=1:d

    % define input
    N = 10000;
    Xg = ones(N,d); Xg(:,i) = linspace(-5,5,N)';

    % evaluate gradient on grid
    Sx   = TM.evaluate(Xg, comp_idx);
    dSx  = TM.grad_x(Xg, i, comp_idx);
    dSx_ti = gradient(Sx, Xg(:,i));
    
    % evaluate error
    err = norm(dSx(:) - dSx_ti(:));
    if norm(dSx(:)) > 0
        err = err/norm(dSx(:));
    end
    assert(err < tol)

end

%% Test hess_x

Sx = TM.evaluate(X);
d2x_Sx = TM.hess_x(X);

% set h
h = 1e-5;

% evaluate gradient using finite differences
d2x_Sx_t = zeros(size(X,1), d, d, d);
for k=1:d
    for l=1:k
        Xpkl = X; Xpkl(:,k) = Xpkl(:,k) + h; Xpkl(:,l) = Xpkl(:,l) + h;
        S_pkl = TM.evaluate(Xpkl);
        Xpk = X; Xpk(:,k) = Xpk(:,k) + h;
        S_pk = TM.evaluate(Xpk);
        Xpl = X; Xpl(:,l) = Xpl(:,l) + h;
        S_pl = TM.evaluate(Xpl);
        Xmkl = X; Xmkl(:,k) = Xmkl(:,k) - h; Xmkl(:,l) = Xmkl(:,l) - h;
        S_mkl = TM.evaluate(Xmkl);
        Xmk = X; Xmk(:,k) = Xmk(:,k) - h;
        S_mk = TM.evaluate(Xmk);
        Xml = X; Xml(:,l) = Xml(:,l) - h;
        S_ml = TM.evaluate(Xml);
        d2x_Sx_t(:,:,k,l) = (S_pkl - S_pk - S_pl + 2*Sx - S_mk - S_ml + S_mkl)/(2*h^2);
        d2x_Sx_t(:,:,l,k) = d2x_Sx_t(:,:,k,l);
    end
end
assert(norm(d2x_Sx(:) - d2x_Sx_t(:))/norm(d2x_Sx(:)) < 1e-4)

%% Test logdet_Jacobian

logDJ = TM.logdet_Jacobian(X);

% evaluate true map
logDJ_t = zeros(size(X,1),1);
for k=1:d
    logDJ_t = logDJ_t + log(TM.S{k}.grad_xd(X(:,1:k)));
end

assert(norm(logDJ(:) - logDJ_t(:))/norm(logDJ(:)) < tol)

%% Test grad_x_logdet_Jacobian

logDJ = TM.logdet_Jacobian(X);
grad_x_logDJ = TM.grad_x_logdet_Jacobian(X);

% evaluate using finite differences
grad_x_logDJ_t = zeros(size(X,1), d);
for k=1:d
    Xk = X; Xk(:,k) = Xk(:,k) + h;
    logDJ_pk = TM.logdet_Jacobian(Xk);
    grad_x_logDJ_t(:,k) = (logDJ_pk - logDJ)/h;
end

assert(norm(grad_x_logDJ(:) - grad_x_logDJ_t(:))/norm(grad_x_logDJ(:)) < tol)

%% Test hess_x_logdet_Jacobian

logDJ = TM.logdet_Jacobian(X);
hess_x_logDJ = TM.hess_x_logdet_Jacobian(X);

% evaluate using finite differences
hess_x_logDJ_t = zeros(size(X,1), d, d);
for k=1:d
    for l=1:d
        Xpkl = X; Xpkl(:,k) = Xpkl(:,k) + h; Xpkl(:,l) = Xpkl(:,l) + h;
        logDJ_pkl = TM.logdet_Jacobian(Xpkl);
        Xpk = X; Xpk(:,k) = Xpk(:,k) + h;
        logDJ_pk = TM.logdet_Jacobian(Xpk);
        Xpl = X; Xpl(:,l) = Xpl(:,l) + h;
        logDJ_pl = TM.logdet_Jacobian(Xpl);
        Xmkl = X; Xmkl(:,k) = Xmkl(:,k) - h; Xmkl(:,l) = Xmkl(:,l) - h;
        logDJ_mkl = TM.logdet_Jacobian(Xmkl);
        Xmk = X; Xmk(:,k) = Xmk(:,k) - h;
        logDJ_mk = TM.logdet_Jacobian(Xmk);
        Xml = X; Xml(:,l) = Xml(:,l) - h;
        logDJ_ml = TM.logdet_Jacobian(Xml);
        hess_x_logDJ_t(:,k,l) = (logDJ_pkl - logDJ_pk - logDJ_pl + 2*logDJ - logDJ_mk - logDJ_ml + logDJ_mkl)/(2*h^2);
    end
end

assert(norm(hess_x_logDJ(:) - hess_x_logDJ_t(:))/norm(hess_x_logDJ(:)) < tol)

%% Test grad_coeff

Sx = TM.evaluate(X);
dcdSx = TM.grad_coeff(X);

% extract coeffs
c0 = TM.coeff;

% evaluate using finite differences
grad_c_Sx_t = zeros(size(X,1), d, TM.n_coeff);
for i=1:TM.n_coeff
    ci = c0; ci(i) = ci(i) + h;
    TM = TM.set_coeff(ci);
    Sx_pi = TM.evaluate(X);
    grad_c_Sx_t(:,:,i) = (Sx_pi - Sx)/h;
end

assert(norm(dcdSx(:) - grad_c_Sx_t(:))/norm(dcdSx(:)) < tol)


%% Test grad_coeff_grad_xd

dxdSx = TM.grad_xd(X);
grad_c_dxdSx = TM.grad_coeff_grad_xd(X);

% extract coeffs
c0 = TM.coeff;

% evaluate using finite differences
grad_c_dxdSx_t = zeros(size(X,1), d, TM.n_coeff);
for i=1:TM.n_coeff
    ci = c0; ci(i) = ci(i) + h;
    TM = TM.set_coeff(ci);
    dxdSx_pi = TM.grad_xd(X);
    grad_c_dxdSx_t(:,:,i) = (dxdSx_pi - dxdSx)/h;
end

assert(norm(grad_c_dxdSx(:) - grad_c_dxdSx_t(:))/norm(grad_c_dxdSx(:)) < tol)

%% Test grad_coeff_logdet_Jacobian

logDJ = TM.logdet_Jacobian(X);
grad_c_logDJ = TM.grad_coeff_logdet_Jacobian(X);

% extract coeffs
c0 = TM.coeff;

% evaluate using finite differences
grad_c_logDJ_t = zeros(size(X,1), TM.n_coeff);
for i=1:TM.n_coeff
    ci = c0; ci(i) = ci(i) + h;
    TM = TM.set_coeff(ci);
    logDJ_pi = TM.logdet_Jacobian(X);
    grad_c_logDJ_t(:,i) = (logDJ_pi - logDJ)/h;
end

assert(norm(grad_c_logDJ(:) - grad_c_logDJ_t(:))/norm(grad_c_logDJ(:)) < tol)

% -- END OF TIME --