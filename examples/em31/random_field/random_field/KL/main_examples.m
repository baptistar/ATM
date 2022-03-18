%% Examples file: Random field discretization
%{
--------------------------------------------------------------------------
Created by:                       Date:           Comment:
Felipe Uribe                      2016            Comparison of methods
--------------------------------------------------------------------------
Estimation of eigenvalues and eigenvectors of Wiener processes
See: http://venus.usc.edu/book/chp2/node14.html
See demo of equations 1.67 and 1.68 in WIKIPEDIA
http://en.wikipedia.org/wiki/Karhunen-Loeve_theorem#The_Wiener_process
--------------------------------------------------------------------------
%}
clear; clc; close all;

%% initial
d_KL = 25;        % number of terms in the expansion
mu = 1;           % mean of the field
sigma = 0.5;      % standard deviation of the field
l_c = 0.25;       % correlation length x

% domain of definition
a = 0;
b = 2;
D = b-a;

% example number
opc = 5;
switch opc
    case 1 % 1D brownian kernel example
        cov_kernel = @(x1,x2) sigma^2* min(x1,x2)-x1.*x2;
    case 2 % 1D modified exp kernel
        at = 2.726;
        cov_kernel  = @(x1,x2) sigma^2* exp(-abs(x1-x2)*at).*(1+at*abs(x1-x2));
    case 3 % squared exponential example
        cov_kernel = @(x1,x2) sigma^2* exp(-(x1-x2).^2/l_c^2);
    case 4 % 1D wiener
        cov_kernel = @(x1,x2) sigma^2* min(x1,x2);
    case 5 % 1D exp kernel
        cov_kernel = @(x1,x2) sigma^2* exp(-abs(x1-x2)/l_c);
end

%% methods
method = 1;
switch method
    case 1 % using Nystrom
        nGL = 100;   % number of Gauss points
        n = 2e2;
        xnod = linspace(a,b,n);
        [eigval, eigvec] = KL_Nystrom(d_KL, cov_kernel, nGL, xnod, 'true');
        
    case 2 % using Galerkin
        n = 3e2;
        xnod = linspace(a,b,n);
        [eigval, eigvec] = KL_Galerkin(d_KL, cov_kernel, xnod, 'true');
        
    case 3 % using discrete
        n = 5e2;
        xnod = linspace(a,b,n);
        [eigval, eigvec] = KL_discrete(d_KL, cov_kernel, xnod, 'true');
end

%% exact solution only for example 4 and 5
switch opc
    case 4
        eigval3 = zeros(d_KL,1);
        eigvec3 = zeros(n,d_KL);
        for i = 1:d_KL
            % calculation of the theoretical eigenvalue and eigenvector
            eigval3(i) = (4*b^2)/(pi^2 * (2*(i-1)+1)^2);
            eigvec3(:,i) = sqrt(2)*sin(xnod/sqrt(eigval3(i)));
        end
        eigval3 = sigma^2 *eigval3;
        figure(1); hold on;
        plot(1:d_KL,eigval3,'ks','LineWidth',1);
        xlabel('Index [n]','Interpreter','Latex','FontSize',18);
        ylabel('Eigenvalue [$\lambda_n$]','Interpreter','Latex','FontSize',18);
        set(gca,'FontSize',13);
        
        figure;
        plot(xnod,eigvec3,'Linewidth',1); grid minor;
        
    case 5
        [eigval3, eigvec3, ~] = KL_analytical(d_KL, l_c, sigma, xnod, 'true');
end

%% random field representation using K-L expansion
n = 1e2;                 % number of realizations
theta = randn(d_KL, n);      % standard Gaussian random variables
Phi = eigvec * diag(sqrt(eigval));
H_hat = mu + Phi*theta;
%
figure;
plot(xnod, H_hat, 'Color','b','LineWidth',1); hold on;
grid minor; axis tight;
title('Random field realizations');
xlabel('x','FontSize',15); ylabel('{H}(x)','FontSize',15);
set(gca,'FontSize',15);

%% covariance approximation
Z_hat = eigvec*(eigvec*diag(eigval))';
[X, Y] = meshgrid(xnod);
Z = cov_kernel(X,Y);   % target cov kernel
%
figure;
subplot(2,2,1); surfl(X, Y, Z); colormap bone; shading interp; box on; axis tight;
title('Process covariance');
subplot(2,2,2); surfl(X, Y, Z_hat); colormap bone; shading interp; box on; axis tight;
title('K-L approximation');
subplot(2,2,3:4); surfl(X, Y, abs(1-(Z_hat./Z))); colormap bone; box on;
title('Relative error surface');

%% averaged global variance error
eigval_sum_ex = cumsum(eigval);
var_error_ex  = abs(1 - (eigval_sum_ex)/(D*sigma^2));
tau = 0.1;   % 90% of the spatially averaged variability
%
k = 1;
while var_error_ex(k) >= tau
    k = k+1;
    if k >= d_KL
        fprintf('The approximation requires more terms. Increase the d_KL');
        break;
    end
end
fprintf('Truncation order is %g \n', k);
%%END