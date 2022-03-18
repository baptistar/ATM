%% Examples file: 1D Random field representation using OSE method
%{
--------------------------------------------------------------------------
Created by:                       Date:           Comment:
Felipe Uribe                      Feb/2015        Comparison of methods
--------------------------------------------------------------------------
Based on:
1."Stochastic finite element methods and reliability"
   B. Sudret and A. Der Kiureghian. State of the art report. (2000)
--------------------------------------------------------------------------
%}
clear; clc; close all;

%% random field parameters
Mterms = 10;      % number of terms in the expansion
mu = 0;           % mean value of the random field
sig = 1;          % std of the random field
opc = 1;
switch opc
    case 1   % 1D exponential kernel example
        corr_leng = 5;    % correlation length x
        corr_func = @(x) exp(-abs(x(1)-x(2))/corr_leng);
        cov_func  = @(x) sig^2 * corr_func(x);
        dom_bound = {[0 10]};
        partition = 20;   %  partition in x
    case 2   % 1D square exponential kernel example
        corr_leng = 0.5;    % correlation length x
        corr_func  = @(x) exp(-(x(1)-x(2))^2/corr_leng^2);
        dom_bound = {[-1 1]};
        partition = 100;   %  partition in x
    case 3   % 1D Wiener process kernel example
        corr_leng = 1;    % correlation length x
        corr_func  = @(x) min(x(1),x(2));
        dom_bound = {[0 1]};
        partition = 1e2;   %  partition in x
    case 4   % 1D sin kernel example
        corr_leng = 0.1;    % correlation length x
        corr_func  = @(x) sin(abs(x(1)-x(2))/corr_leng)/(abs(x(1)-x(2))/corr_leng);
        dom_bound = {[-0.5 0.5]};
        partition = 20;   %  partition in x
end

%% solving using OSE method
[eigval, eigvec, Sigma, P] = OSE_method(cov_func, dom_bound, corr_leng, Mterms);

%% representation of the process: sample function
% legendre polynomials with shifting and scaling
xnod = linspace(dom_bound{1}(1), dom_bound{1}(2), partition);
h = cell(Mterms, 1);
T = (dom_bound{1}(2)+dom_bound{1}(1))/2;   % shift
a = (dom_bound{1}(2)-dom_bound{1}(1))/2;   % scale

% scaling polys
for i = 1:Mterms
    h{i} = polyval(P{i}, (xnod-T)/a)*sqrt((2*i-1)/(2*a)); % Eq. (2.29) Part II.
end
HH = cell2mat(h);

% procedure: Eq. (2.25) Part II.
set(0, 'defaultTextInterpreter', 'latex');
Nsim = 100;
H_hat = zeros(Nsim, partition);
figure;
for k = 1:Nsim
    xi = randn(1, Mterms);   % standard Gaussian random variables
    Hrep1 = zeros(Mterms, 1);
    Hrep2 = zeros(Mterms, 1);
    for l = 1:partition
        for i = 1:Mterms
            for j = 1:Mterms
                Hrep2(j) = eigvec(j, i)*HH(j, l);
            end
            Hrep1(i) = sig*sqrt(eigval(i))*xi(i)*sum(Hrep2);
        end
        H_hat(k, l) = mu + sum(Hrep1);
    end
    % plot realization
    plot(xnod, H_hat(k,:), 'color', [0.55, 0.55, 0.55]); hold on;
end
grid minor; axis tight;
H_hat_mu = mean(H_hat);
H_hat_std = std(H_hat);
plot(xnod, H_hat_mu, '-', 'color', [0 0 1], 'LineWidth', 2);
plot(xnod, H_hat_mu-H_hat_std, '--', 'color', [0 0 1], 'LineWidth', 2);
plot(xnod, H_hat_mu+H_hat_std, '--', 'color', [0 0 1], 'LineWidth', 2);
xlabel('$$x$$', 'FontSize',18);
ylabel('$$\tilde{H}(x)$$', 'FontSize', 18);
set(gca, 'FontSize', 18);

%% variance calculation
% finer partition
NN = 5e2;
xx = linspace(dom_bound{1}(1), dom_bound{1}(2), NN);
var_error = ones(NN,1);

% Legendre polynomials and Gauss points
nGp = 20;
[xi_gl, wi_gl, ~] = gauss_quad(nGp, 'legen');

% integration
for l = 1:NN
    hi = zeros(Mterms, 1);
    part2 = zeros(Mterms, 1);
    for i = 1:Mterms
        hi(i) = sqrt((2*i-1)/(2*a)) * polyval(P{i}, (xx(l)-T)/a);
        I = 0;
        for p = 1:nGp
            I = I + polyval(P{i}, xi_gl(p)) * (corr_func([xx(l), a*xi_gl(p)+T])) * wi_gl(p);
        end
        part2(i) = hi(i) * (a * sqrt((2*i-1)/(2*a)) * I);
    end
    var_error(l) = 1 + (hi'*(Sigma*hi)) - 2*sum(part2);
end
figure;
plot(xx, var_error, 'k:', 'LineWidth', 2);
grid; axis tight;
%%END