%% Examples file: Random field representation using EOLE method
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

%% solving using EOLE method
[eigval, eigvec, xnod] = EOLE_method(cov_func, dom_bound, partition, Mterms);

%% representation of the process: sample function
N = length(xnod);
NN = 5e2;
xx = linspace(dom_bound{1}(1), dom_bound{1}(2), NN);

% evaluating cov kernel
cov_mat = zeros(N,NN);
for j = 1:N
    for l = 1:NN
        cov_mat(j,l) = cov_func([xnod(j), xx(l)]);
    end
end

% evaluating EOLE variance formula
Hvar = zeros(1, NN);   % after taking var[]
for i = 1:Mterms
    Hvar = Hvar + (1/eigval(i))*(eigvec(:,i)'*cov_mat).^2;
end
var_error = 1 - Hvar;  % Part I Eq. (2.66) & (2.68)

figure;
plot(xx, var_error,'r-', 'LineWidth', 2); grid minor; axis tight;
title('Variance error');

%% representation of random field
set(0,'defaultTextInterpreter','latex');
Nsim = 50;   % number of samples
H_hat = zeros(Nsim, NN);

figure;
for k = 1:Nsim
    xi = randn(Mterms, 1);   % standard Gaussian random variables
    Hrep = zeros(1, NN);
    for i = 1:Mterms
        Hrep = Hrep + sig*(xi(i)/sqrt(eigval(i)))*eigvec(:,i)'*cov_mat;
    end
    H_hat(k,:) = mu + Hrep;
    plot(xx, H_hat(k,:), 'Color', [0.55 0.55 0.55]); hold on;
end
grid minor; axis tight;
H_hat_mu = mean(H_hat);
H_hat_std = std(H_hat);
plot(xx, H_hat_mu, 'Color', [0 0 1], 'LineWidth', 3);
plot(xx, H_hat_mu-H_hat_std, '--', 'Color', [0 0 1], 'LineWidth', 3);
plot(xx, H_hat_mu+H_hat_std, '--', 'Color', [0 0 1], 'LineWidth', 3);
xlabel('$$x$$', 'FontSize', 15); ylabel('$$\tilde{H}(x)$$', 'FontSize', 15);
set(gca, 'FontSize', 15);
%%END