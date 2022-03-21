clear; close all; clc
addpath(genpath('../src'))

% define different basis objects
basis = HermitePhysicistPoly();

% define order and normalization
order = 5;
normt = true;

% samples for evaluating model
x = sort([0, linspace(-4,4,100000)])';

%% Check basis

% check Psi
Psi = basis.grad_vandermonde(x, order, 0, normt);

Psi_t = zeros(length(x), order+1);
Psi_t(:,0+1) = ones(length(x),1);
Psi_t(:,1+1) = 2*x;
for i=2:order
    Psi_t(:,i+1) = 2*x.*Psi_t(:,i-1+1) - 2*(i-1)*Psi_t(:,i-2+1);
end
if normt == true
    for i=0:order
        Psi_t(:,i+1) = Psi_t(:,i+1) / sqrt(2^i * factorial(i));
    end
end

assert(norm(Psi(:) - Psi_t(:)) < 1e-10)

%% Check first derivatives

% check dxPsi
dxPsi = basis.grad_vandermonde(x, order, 1, normt);

dxPsi_t = zeros(length(x), order+1);
for i=1:order
    dxPsi_t(:,i+1) = 2*i*basis.evaluate(x, i-1, false);
    if normt == true
        dxPsi_t(:,i+1) = dxPsi_t(:,i+1) / sqrt(2^i * factorial(i));
    end
end

assert(norm(dxPsi(:) - dxPsi_t(:)) < 1e-10)

%% Check second derivatives

% check d2xPsi
d2xPsi = basis.grad_vandermonde(x, order, 2, normt);

d2xPsi_t = zeros(length(x), order+1);
for i=2:order
    d2xPsi_t(:,i+1) = 4*i*(i-1)*basis.evaluate(x, i-2, false);
    if normt == true
        d2xPsi_t(:,i+1) = d2xPsi_t(:,i+1) / sqrt(2^i * factorial(i));
    end
end

d2xPsi_m1 = d2xPsi; d2xPsi_m1(1,:) = zeros(1,order+1);

assert(norm(d2xPsi(:) - d2xPsi_t(:)) < 1e-10)

% -- END OF FILE --