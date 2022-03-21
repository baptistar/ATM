clear; close all; clc
addpath(genpath('../src'))

% define different basis objects
basis = PhysicistHermiteFunction();
basis_poly = HermitePhysicistPoly();

% define order
order = 5;
normt = true;

% define tolerance
tol = 1e-6;

% evaluate model
x = sort([0, linspace(-4,4,100000)])';

%% Check basis

% check Psi
Psi = basis.grad_vandermonde(x, order, 0, normt);

Psi_t = zeros(length(x), order+1);
Psi_t(:,0+1) = ones(length(x),1) .* exp(-x.^2/2);
Psi_t(:,1+1) = 2*x .* exp(-x.^2/2);
for i=2:order
    Psi_t(:,i+1) = 2*x.*Psi_t(:,i-1+1) - 2*(i-1)*Psi_t(:,i-2+1);
end
if normt == true
    for i=0:order
        Psi_t(:,i+1) = Psi_t(:,i+1) / sqrt(2^i * factorial(i));
    end
end

assert(norm(Psi(:) - Psi_t(:)) < tol)

%% Check first derivatives

% check dxPsi
dxPsi = basis.grad_vandermonde(x, order, 1, normt);

dxPsi_t = zeros(length(x), order+1);
for i=0:order
    dxPsi_t(:,i+1) = basis_poly.grad_x(x,i,1,false) .* exp(-x.^2/2) - ...
        x .* basis_poly.evaluate(x,i,false) .* exp(-x.^2/2);
    if normt == true
        dxPsi_t(:,i+1) = dxPsi_t(:,i+1) / sqrt(2^i * factorial(i));
    end
end

assert(norm(dxPsi(:) - dxPsi_t(:)) < tol)

%% Check second derivatives

% check d2xPsi
d2xPsi = basis.grad_vandermonde(x, order, 2, normt);

d2xPsi_t = zeros(length(x), order+1);
for i=0:order
    d2xPsi_t(:,i+1) = basis_poly.grad_x(x,i,2,false) .* exp(-x.^2/2) + ...
                      basis_poly.grad_x(x,i,1,false) .* exp(-x.^2/2) .* (-2*x) + ...
                      basis_poly.evaluate(x,i,false) .* exp(-x.^2/2) .* (x.^2-1);
    if normt == true
        d2xPsi_t(:,i+1) = d2xPsi_t(:,i+1) / sqrt(2^i * factorial(i));
    end
end

assert(norm(d2xPsi(:) - d2xPsi_t(:)) < tol)

%% Check third derivatives

% check d3xPsi
d3xPsi = basis.grad_vandermonde(x, order, 3, normt);

d3xPsi_t = zeros(length(x), order+1);
for i=0:order
    d3xPsi_t(:,i+1) = basis_poly.grad_x(x,i,3,false) .* exp(-x.^2/2) + ...
                      basis_poly.grad_x(x,i,2,false) .* exp(-x.^2/2) .* (-3*x) + ...
                      basis_poly.grad_x(x,i,1,false) .* exp(-x.^2/2) .* (3*x.^2-3) + ...
                      basis_poly.evaluate(x,i,false) .* exp(-x.^2/2) .* (-x.^3+3*x);
    if normt == true
        d3xPsi_t(:,i+1) = d3xPsi_t(:,i+1) / sqrt(2^i * factorial(i));
    end
end

assert(norm(d3xPsi(:) - d3xPsi_t(:)) < tol)

%% Check integral of basis

% check Int of basis
IntPsi = basis.grad_vandermonde(x, order, -1, normt);

IntPsi_t = zeros(length(x), order+1);
for i=0:order
    Intf = @(x) basis.evaluate(x, i, normt);
    IntPsi_t(:,i+1) = recursive_integral(Intf, zeros(size(x)), x, [1e-6, 1e-6], 12, 12);
end

assert(norm(IntPsi(:) - IntPsi_t(:)) < tol)

% -- END OF FILE --