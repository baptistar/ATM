function [eigval, eigvec] = KL_discrete(d_KL, K, xnod, img)
%% Solve the homogeneous Fredholm equation of the second kind
%{
--------------------------------------------------------------------------
Using the discrete Karhunen-Loeve method proposed in C. A. Schenk and
G. I. Schueller. Uncertainty assessment of large finite element systems.
Springer. 2005, page 49
--------------------------------------------------------------------------
Created by:                       Date:           Comment:
Felipe Uribe                      Feb/2015        Comparison of methods
--------------------------------------------------------------------------
Input:
K(t,s) - kernel (function handle that can handle matrices as inputs)
a,b    - integration limits
m      - number of steps -- used in x = linspace(a,b,m);
d_KL   - number of terms in the expansion
--------------------------------------------------------------------------
%}

%% evaluating covariance matrix
if nargin == 4
    m = length(xnod);
    [x1, x2] = meshgrid(xnod, xnod);
    
    % eigenvalues of the covariance matrix
    A  = eye(m);
    [eigvec, eigval] = eigs(K(x1,x2), A, d_KL);
else
    error('Number of parameters not supported');
end

eigval = diag(eigval);
[eigval, idx] = sort(eigval, 'descend'); % sort the eigenvalues and
eigvec = eigvec(:,idx);          % eigenvectors in descending order

%% normalization of the eigenvalues
dx = xnod(2)-xnod(1);
eigval = eigval*dx;

%% normalization of the eigenfunctions
% So that \int_D phi_i(t)^2 dt = 1  (eq. 8.53) and note in page 57.
eigvec = eigvec./repmat(sqrt(trapz(xnod,eigvec.^2)),m,1);

%% plots
if strcmp(img,'true')
    % eigenvalues
    figure(1); hold on;
    plot(1:d_KL,eigval, 'ro-', 'LineWidth', 1);
    xlabel('Index [n]','Interpreter','Latex','FontSize',18);
    ylabel('Eigenvalue [$\lambda_n$]','Interpreter','Latex','FontSize',18);
    set(gca,'FontSize',13);
    
    % eigenfunctions
    figure;
    plot(xnod, eigvec(:,1:end), 'LineWidth', 1); axis tight;
    xlabel('Length [$x$]','Interpreter','Latex','FontSize',18);
    ylabel('Eigenfunction [$\phi(x)$]','Interpreter','Latex','FontSize',18);
    title('Nystrom','Interpreter','Latex','FontSize',18);
    set(gca,'FontSize',13); grid minor;
end

return;
%%END