clear; close all; clc
addpath(genpath('../src'))

% test IntegratedPositiveFunction
d = 2;
N = 500;

% set tolerance
tol = 1e-6;

% define basis
basis = LinearExtProbabilistHermiteFunction();
basis = repmat({basis},1,d);

% define and sort multi_idxs
orders = [8,3];
m_idxs = TotalOrderMultiIndices(orders);
[~, sort_idx] = sort(m_idxs(:,2));
m_idxs = m_idxs(sort_idx, :);

% generate samples and coefficients
X = randn(N,d);
c = randn(1, size(m_idxs,1));

% define I using ParametericPolynomial
P = ParametericPoly(basis, m_idxs);
Ip = IntegratedPositiveFunction(P);
Ip = Ip.set_coeff(c);

% define precomp
precomp = PPprecomp();
precomp.evaluate(Ip.f, X);

%% check objects

assert(all(Ip.f.multi_idxs == m_idxs, 'all'), 'Multi-indices don''t match')
assert(all(Ip.n_coeff == length(c)), 'Number of coeffs doesn''t match')
assert(all(Ip.coeff == c),'Coefficients don''t match')

%% evaluate

disp('evaluate')
tic; Fp = Ip.evaluate(X); toc;
disp('evaluate with precomp')
tic; Fp_pre = Ip.evaluate(X, precomp); toc;

% evaluate first term
Psi1x = basis{1}.grad_vandermonde(X(:,1), max(m_idxs(:,1)), 0, true);
Psi20 = basis{2}.grad_vandermonde(zeros(N,1), max(m_idxs(:,2)), 0, true);
F0t = (Psi1x(:,m_idxs(:,1)+1) .* Psi20(:,m_idxs(:,2)+1)) * c';

% evaluate second term
Psi1x = basis{1}.grad_vandermonde(X(:,1), max(m_idxs(:,1)), 0, true);
IntgdPxt = zeros(size(X,1),1);
for i=1:size(X,1)
    x2 = linspace(0,X(i,2),1000)';
    dxPsi2 = basis{2}.grad_vandermonde(x2, max(m_idxs(:,1)), 1, true);
    Psi1_i = repmat(Psi1x(i,m_idxs(:,1)+1), length(x2), 1);
    dxPsi2_i = dxPsi2(:,m_idxs(:,2)+1);
    gdPx = Ip.rec.evaluate((Psi1_i .* dxPsi2_i) * c');
    IntgdPxt(i) = trapz(x2, gdPx);
end

% compute sum
Ft = F0t + IntgdPxt;

assert(norm(Fp(:) - Ft(:),2)/numel(Ft) < tol);
assert(norm(Fp_pre(:) - Ft(:),2)/numel(Ft) < tol);

%% grad_xd

disp('grad_xd')
tic; dxFp = Ip.grad_xd(X); toc;
disp('grad_xd with precomp')
tic; dxFp_pre = Ip.grad_xd(X, precomp); toc;

% evaluate second term
Psi1x   = basis{1}.grad_vandermonde(X(:,1), max(m_idxs(:,1)), 0, true);
dxPsi2x = basis{2}.grad_vandermonde(X(:,2), max(m_idxs(:,2)), 1, true);
gdPx    = Ip.rec.evaluate((Psi1x(:,m_idxs(:,1)+1) .* dxPsi2x(:,m_idxs(:,2)+1)) * c');
dxFt    = gdPx;

assert(norm(dxFp(:) - dxFt(:),2)/numel(dxFt) < tol);
assert(norm(dxFp_pre(:) - dxFt(:),2)/numel(dxFt) < tol);

%% hess_xd

disp('hess_xd')
tic; d2xFp = Ip.hess_xd(X); toc;
disp('hess_xd with precomp')
tic; d2xFp_pre = Ip.hess_xd(X, precomp); toc;

% evaluate second term
Psi1x    = basis{1}.grad_vandermonde(X(:,1), max(m_idxs(:,1)), 0, true);
dxPsi2x  = basis{2}.grad_vandermonde(X(:,2), max(m_idxs(:,2)), 1, true);
d2xPsi2x = basis{2}.grad_vandermonde(X(:,2), max(m_idxs(:,2)), 2, true);
d2xFt    = Ip.rec.grad_x((Psi1x(:,m_idxs(:,1)+1) .* dxPsi2x(:,m_idxs(:,2)+1)) * c') .* ...
            ((Psi1x(:,m_idxs(:,1)+1) .* d2xPsi2x(:,m_idxs(:,2)+1)) * c');

assert(norm(d2xFp(:) - d2xFt(:),2)/numel(d2xFt) < tol);
assert(norm(d2xFp_pre(:) - d2xFt(:),2)/numel(d2xFt) < tol);

%% grad_coeff

test_coeffs = 1:Ip.n_coeff;
test_coeffs = test_coeffs(1:2:end); 

F = Ip.evaluate(X);
disp('grad_coeff')
tic; daFp = Ip.grad_coeff(X, test_coeffs); toc;
disp('grad_coeff with precomp')
tic; daFp_pre = Ip.grad_coeff(X, test_coeffs, precomp); toc;

% tolerance
h = 1e-9;

% setup object
It = IntegratedPositiveFunction(P);

daFt = zeros(size(X,1), numel(c));
for j=1:numel(c)
    cp = c; cp(j) = cp(j) + h;
    It = It.set_coeff(cp);
    Fp = It.evaluate(X);
    daFt(:,j) = (Fp - F)/h;
end
daFt = daFt(:, test_coeffs);

assert(norm(daFt(:) - daFp(:),2)/numel(daFt) < tol);
assert(norm(daFt(:) - daFp_pre(:),2)/numel(daFt) < tol);

%% grad_coeff_grad_xd

test_coeffs = 1:Ip.n_coeff;
test_coeffs = test_coeffs(1:2:end);

dxF = Ip.grad_xd(X);
disp('grad_coeff_grad_xd')
tic; dadxFp = Ip.grad_coeff_grad_xd(X, test_coeffs); toc;
disp('grad_coeff_grad_xd with precomp')
tic; dadxFp_pre = Ip.grad_coeff_grad_xd(X, test_coeffs, precomp); toc;

% tolerance
h = 1e-9;

% setup object
It = IntegratedPositiveFunction(P);

dadxFt = zeros(size(X,1), numel(c));
for j=1:numel(c)
    cp = c; cp(j) = cp(j) + h;
    It = It.set_coeff(cp);
    dxFp = It.grad_xd(X);
    dadxFt(:,j) = (dxFp - dxF)/h;
end
dadxFt = dadxFt(:, test_coeffs);

assert(norm(dadxFt(:) - dadxFp(:))/numel(dadxFt) < tol);
assert(norm(dadxFt(:) - dadxFp_pre(:))/numel(dadxFt) < tol);

%% hess_coeff

test_coeffs = 1:Ip.n_coeff;
test_coeffs = test_coeffs(1:2:end);

disp('hess_coeff')
tic; d2aFp = Ip.hess_coeff(X, test_coeffs); toc;
disp('hess_coeff with precomp')
tic; d2aFp_pre = Ip.hess_coeff(X, test_coeffs, precomp); toc;

% tolerance
h = 1e-6;

% setup object
It = IntegratedPositiveFunction(P);

d2aFt = zeros(size(X,1), numel(c), numel(c));
for j=1:numel(c)
    for k=1:numel(c)
        
        cpp = c; cpp(j) = cpp(j) + h; cpp(k) = cpp(k) + h;
        It = It.set_coeff(cpp);
        Fpp = It.evaluate(X);

        cpm = c; cpm(j) = cpm(j) + h; cpm(k) = cpm(k) - h;
        It = It.set_coeff(cpm);
        Fpm = It.evaluate(X);

        cmp = c; cmp(j) = cmp(j) - h; cmp(k) = cmp(k) + h;
        It = It.set_coeff(cmp);
        Fmp = It.evaluate(X);

        cmm = c; cmm(j) = cmm(j) - h; cmm(k) = cmm(k) - h;
        It = It.set_coeff(cmm);
        Fmm = It.evaluate(X);
        
        d2aFt(:,j,k) = (Fpp - Fpm - Fmp + Fmm)/(4*h^2);

    end
end
d2aFt = d2aFt(:, test_coeffs, test_coeffs);

assert(norm(d2aFt(:) - d2aFp(:), 2)/numel(d2aFt) < tol);
assert(norm(d2aFt(:) - d2aFp_pre(:), 2)/numel(d2aFt) < tol);

%% hess_coeff_grad_xd

test_coeffs = 1:Ip.n_coeff;
test_coeffs = test_coeffs(1:2:end);

disp('hess_coeff_grad_xd')
tic; d2adxFp = Ip.hess_coeff_grad_xd(X, test_coeffs); toc;
disp('hess_coeff_grad_xd with precomp')
tic; d2adxFp_pre = Ip.hess_coeff_grad_xd(X, test_coeffs, precomp); toc;

% tolerance
h = 1e-6;

% setup object
It = IntegratedPositiveFunction(P);

d2adxFt = zeros(size(X,1), numel(c), numel(c));
for j=1:numel(c)
    for k=1:numel(c)
        
        cpp = c; cpp(j) = cpp(j) + h; cpp(k) = cpp(k) + h;
        It = It.set_coeff(cpp);
        dxFpp = It.grad_xd(X);

        cpm = c; cpm(j) = cpm(j) + h; cpm(k) = cpm(k) - h;
        It = It.set_coeff(cpm);
        dxFpm = It.grad_xd(X);

        cmp = c; cmp(j) = cmp(j) - h; cmp(k) = cmp(k) + h;
        It = It.set_coeff(cmp);
        dxFmp = It.grad_xd(X);

        cmm = c; cmm(j) = cmm(j) - h; cmm(k) = cmm(k) - h;
        It = It.set_coeff(cmm);
        dxFmm = It.grad_xd(X);
        
        d2adxFt(:,j,k) = (dxFpp - dxFpm - dxFmp + dxFmm)/(4*h^2);

    end
end
d2adxFt = d2adxFt(:, test_coeffs, test_coeffs);

assert(norm(d2adxFt(:) - d2adxFp(:), 2)/numel(d2adxFt) < tol);
assert(norm(d2adxFt(:) - d2adxFp_pre(:), 2)/numel(d2adxFt) < tol);

%% grad_x

disp('grad_x')
tic; dxFp = Ip.grad_x(X, []); toc;
disp('grad_x with precomp')
tic; dxFp_pre = Ip.grad_x(X, [], precomp); toc;

dxFt = zeros(size(X,1), d);

% evaluate derivatives of first and second term
Psi1x   = basis{1}.grad_vandermonde(X(:,1), max(m_idxs(:,1)), 0, true);
dxPsi1x = basis{1}.grad_vandermonde(X(:,1), max(m_idxs(:,1)), 1, true);
dxPsi2x = basis{2}.grad_vandermonde(X(:,2), max(m_idxs(:,2)), 1, true);
Psi20   = basis{2}.grad_vandermonde(zeros(N,1), max(m_idxs(:,2)), 0, true);

% evaluate \partial_x1 f(0) and \partial_x2 f(0)
dx1F0 = (dxPsi1x(:,m_idxs(:,1)+1) .* Psi20(:,m_idxs(:,2)+1) * c');
dx2F0 = zeros(size(X,1),1);

% evaluate \partial_x1 \int_0^x2 g(\partial_x2 f) dt
dx1IF = zeros(size(X,1),1);
for i=1:size(X,1)
    x2 = linspace(0,X(i,2),1000)';
    dxPsi2 = basis{2}.grad_vandermonde(x2, max(m_idxs(:,2)), 1, true);
    Psi1_i = repmat(Psi1x(i,m_idxs(:,1)+1), length(x2), 1);
    dx1Psi1_i = repmat(dxPsi1x(i,m_idxs(:,1)+1), length(x2), 1);
    dxPsi2_i = dxPsi2(:,m_idxs(:,2)+1);
    d1gdPx = Ip.rec.grad_x((Psi1_i .* dxPsi2_i) * c') .* ((dx1Psi1_i .* dxPsi2_i) * c');
    dx1IF(i) = trapz(x2, d1gdPx);
end

% evaluate \partial_x2 \int_0^x2 g(\partial_x2 f) dt = g(\partial_x2 f)
dx2IF = Ip.rec.evaluate((Psi1x(:,m_idxs(:,1)+1) .* dxPsi2x(:,m_idxs(:,2)+1)) * c');

% compute sums
dxFt(:,1) = dx1F0 + dx1IF;
dxFt(:,2) = dx2F0 + dx2IF; 

assert(norm(dxFp(:) - dxFt(:),2)/numel(dxFt) < tol);
assert(norm(dxFp_pre(:) - dxFt(:),2)/numel(dxFt) < tol);

%% grad_x_grad_xd

disp('grad_x_grad_xd')
tic; dxjdxFp = Ip.grad_x_grad_xd(X, []); toc;
disp('grad_x_grad_xd with precomp')
tic; dxjdxFp_pre = Ip.grad_x_grad_xd(X, [], precomp); toc;

dxjdxFt = zeros(size(X,1), d);

% evaluate derivatives of basis functions
Psi1x    = basis{1}.grad_vandermonde(X(:,1), max(m_idxs(:,1)), 0, true);
Psi2x    = basis{2}.grad_vandermonde(X(:,2), max(m_idxs(:,2)), 0, true);
dxPsi1x  = basis{1}.grad_vandermonde(X(:,1), max(m_idxs(:,1)), 1, true);
dxPsi2x  = basis{2}.grad_vandermonde(X(:,2), max(m_idxs(:,2)), 1, true);
d2xPsi2x = basis{2}.grad_vandermonde(X(:,2), max(m_idxs(:,2)), 2, true);

% evaluate \partial_x1,x2 \Psi, \partial_x2 \Psi, \partial^2_x2 \Psi
dx12Psi = dxPsi1x(:,m_idxs(:,1)+1) .* dxPsi2x(:,m_idxs(:,2)+1);
dx2Psi  = Psi1x(:,m_idxs(:,1)+1) .* dxPsi2x(:,m_idxs(:,2)+1);
d2x2Psi = Psi1x(:,m_idxs(:,1)+1) .* d2xPsi2x(:,m_idxs(:,2)+1);

% evaluate \partial_x1 \int_0^x2 g(\partial_x2 f) dt
dxjdxFt(:,1) = Ip.rec.grad_x(dx2Psi * c') .* (dx12Psi * c');
dxjdxFt(:,2) = Ip.rec.grad_x(dx2Psi * c') .* (d2x2Psi * c');

assert(norm(dxjdxFp(:) - dxjdxFt(:),2)/numel(dxjdxFt) < tol);
assert(norm(dxjdxFp_pre(:) - dxjdxFt(:),2)/numel(dxjdxFt) < tol);

%% hess_x

disp('hess_x')
tic; d2xFp = Ip.hess_x(X, []); toc;
disp('hess_x with precomp')
tic; d2xFp_pre = Ip.hess_x(X, [], precomp); toc;

% evaluate derivatives of first and second term
Psi1x    = basis{1}.grad_vandermonde(X(:,1), max(m_idxs(:,1)), 0, true);
dxPsi1x  = basis{1}.grad_vandermonde(X(:,1), max(m_idxs(:,1)), 1, true);
dxPsi2x  = basis{2}.grad_vandermonde(X(:,2), max(m_idxs(:,2)), 1, true);
d2xPsi1x = basis{1}.grad_vandermonde(X(:,1), max(m_idxs(:,1)), 2, true);
d2xPsi2x = basis{2}.grad_vandermonde(X(:,2), max(m_idxs(:,2)), 2, true);
Psi20    = basis{2}.grad_vandermonde(zeros(N,1), max(m_idxs(:,2)), 0, true);

% evaluate \partial_x1^2 f(0)
d2x1F0  = (d2xPsi1x(:,m_idxs(:,1)+1) .* Psi20(:,m_idxs(:,2)+1) * c');

% define \nabla^2_x f(0)
d2xF0 = zeros(size(X,1), d, d);
d2xF0(:,1,1) = d2x1F0;

d2xIF = zeros(size(X,1), d, d);
% evaluate \partial^2_x1 \int_0^x2 g(\partial_x2 f) dt
for i=1:size(X,1)
    x2 = linspace(0,X(i,2),2000)';
    dxPsi2 = basis{2}.grad_vandermonde(x2, max(m_idxs(:,2)), 1, true);
    Psi1_i = repmat(Psi1x(i,m_idxs(:,1)+1), length(x2), 1);
    dx1Psi1_i = repmat(dxPsi1x(i,m_idxs(:,1)+1), length(x2), 1);
    d2x1Psi1_i = repmat(d2xPsi1x(i,m_idxs(:,1)+1), length(x2), 1);
    dxPsi2_i = dxPsi2(:,m_idxs(:,2)+1);
    d2x1gdPx = Ip.rec.grad_x((Psi1_i .* dxPsi2_i) * c') .* ((d2x1Psi1_i .* dxPsi2_i) * c') + ...
             Ip.rec.hess_x((Psi1_i .* dxPsi2_i) * c') .* ((dx1Psi1_i .* dxPsi2_i) * c').^2;
    d2xIF(i,1,1) = trapz(x2, d2x1gdPx);
end
% evaluate \partial_x1\partial_x2 \int_0^x2 g(\partial_x2 f) dt
dxf    = (Psi1x(:,m_idxs(:,1)+1) .* dxPsi2x(:,m_idxs(:,2)+1)) * c';
dx1x2f = (dxPsi1x(:,m_idxs(:,1)+1) .* dxPsi2x(:,m_idxs(:,2)+1)) * c';
d2xIF(:,1,2) = Ip.rec.grad_x(dxf) .* dx1x2f;
d2xIF(:,2,1) = d2xIF(:,1,2);
% evaluate \partial_x2^2 \int_0^x2 g(\partial_x2 f) dt = g(\partial_x2 f)
d2xf   = (Psi1x(:,m_idxs(:,1)+1) .* d2xPsi2x(:,m_idxs(:,2)+1)) * c';
d2xIF(:,2,2) = Ip.rec.grad_x(dxf) .* d2xf;

% compute sums
d2xFt = d2xF0 + d2xIF;

assert(norm(d2xFp(:) - d2xFt(:),2)/numel(d2xFt) < tol);
assert(norm(d2xFp_pre(:) - d2xFt(:),2)/numel(d2xFt) < tol);

%% Test hess_x_grad_xd

disp('hess_x_grad_xd')
tic; d2xdxFp = Ip.hess_x_grad_xd(X, []); toc;
disp('hess_x_grad_xd with precomp')
tic; d2xdxFp_pre = Ip.hess_x_grad_xd(X, [], precomp); toc;

d2xdxFt = zeros(size(X,1), d, d);

% evaluate derivatives of basis functions
Psi1x    = basis{1}.grad_vandermonde(X(:,1), max(m_idxs(:,1)), 0, true);
Psi2x    = basis{2}.grad_vandermonde(X(:,2), max(m_idxs(:,2)), 0, true);
dxPsi1x  = basis{1}.grad_vandermonde(X(:,1), max(m_idxs(:,1)), 1, true);
dxPsi2x  = basis{2}.grad_vandermonde(X(:,2), max(m_idxs(:,2)), 1, true);
d2xPsi1x = basis{1}.grad_vandermonde(X(:,1), max(m_idxs(:,1)), 2, true);
d2xPsi2x = basis{2}.grad_vandermonde(X(:,2), max(m_idxs(:,2)), 2, true);
d3xPsi2x = basis{2}.grad_vandermonde(X(:,2), max(m_idxs(:,2)), 3, true);

% evaluate \partial_x1,x2 \Psi, \partial_x2 \Psi, \partial^2_x2 \Psi
dx12Psi    = dxPsi1x(:,m_idxs(:,1)+1) .* dxPsi2x(:,m_idxs(:,2)+1);
d2x1dx2Psi = d2xPsi1x(:,m_idxs(:,1)+1) .* dxPsi2x(:,m_idxs(:,2)+1);
dx1dx2Psi  = dxPsi1x(:,m_idxs(:,1)+1) .* d2xPsi2x(:,m_idxs(:,2)+1);
dx2Psi     = Psi1x(:,m_idxs(:,1)+1) .* dxPsi2x(:,m_idxs(:,2)+1);
d2x2Psi    = Psi1x(:,m_idxs(:,1)+1) .* d2xPsi2x(:,m_idxs(:,2)+1);
d3x2Psi    = Psi1x(:,m_idxs(:,1)+1) .* d3xPsi2x(:,m_idxs(:,2)+1);

% evaluate \partial_x1 \int_0^x2 g(\partial_x2 f) dt
d2xdxFt(:,1,1) = Ip.rec.hess_x(dx2Psi * c') .* (dx12Psi * c').^2 + Ip.rec.grad_x(dx2Psi * c') .* (d2x1dx2Psi * c');
d2xdxFt(:,1,2) = Ip.rec.hess_x(dx2Psi * c') .* (dx12Psi * c') .* (d2x2Psi * c') + Ip.rec.grad_x(dx2Psi * c') .* (dx1dx2Psi * c');
d2xdxFt(:,2,1) = d2xdxFt(:,1,2);
d2xdxFt(:,2,2) = Ip.rec.hess_x(dx2Psi * c') .* (d2x2Psi * c').^2 + Ip.rec.grad_x(dx2Psi * c') .* (d3x2Psi * c');

assert(norm(d2xdxFp(:) - d2xdxFt(:),2)/numel(d2xdxFt) < tol);
assert(norm(d2xdxFp_pre(:) - d2xdxFt(:),2)/numel(d2xdxFt) < tol);

%% Test identity function

% define domain
d = 2;
N = 100;
x = linspace(-3,3,N);

% define basis + multi_idx
basis = ConstExtProbabilistHermiteFunction();
basis = repmat({basis},1,d);
multi_idx = zeros(1,d);

% define map with Parametric Poly
f = ParametericPoly(basis, multi_idx);
S2 = IntegratedPositiveFunction(f);
S2 = S2.set_coeff(0);

% evaluate map
X = [randn(N,1), x'];
Sx = S2.evaluate(X);

assert(norm(Sx - x') < tol)

%% Test inverse

% invert samples
Z  = randn(size(X,1),1);
Xd = Ip.inverse(X(:,1:d-1),Z);

% evaluate map at Xd
Zt = Ip.evaluate([X(:,1:d-1),Xd]);

assert(norm(Z(:) - Zt(:))/norm(Z) < 1e-3)

% -- END OF TESTS --