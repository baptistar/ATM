function [eigval, eigvec] = KL_Nystrom(d_KL, cov_kernel, nGL, xnod, img)
%% Nystrom method for the Fredholm equation with 1D Matern covariance kernel
%{
--------------------------------------------------------------------------
Created by:                        Date:            Comment:
Felipe Uribe                       August 2016      Important algorithm
--------------------------------------------------------------------------
d_KL       ---> number of KL terms
cov_kernel ---> autocovariance kernel
lambda     ---> correlation length
nGL        ---> number of Gauss-Legendre points for Nystrom
xnod       ---> [xmin,xmax]
img        ---> 'true' if want to show the eigenpairs plots
--------------------------------------------------------------------------
Based on:
1. "Numerical Recipes: The art of scientific computing"
    Press et.al. - 3rd edition 2007 - Cambridge university press
--------------------------------------------------------------------------
%}

%% domain data
a = (xnod(end)-xnod(1))/2;

%% compute the Gauss-Legendre points and weights
[xi,w] = gauss_quad(nGL);

% transform nodes and weights to [0,a]
xt = a*xi + a;
wt = a*w;

%% compute matrices D and C
% compute diagonal matrix D
D = spdiags(sqrt(wt), 0, nGL, nGL);
S = repmat(sqrt(wt)', nGL, 1) .* repmat(sqrt(wt), 1, nGL);

% compute the covariance matrix
cov_mat = zeros(nGL);
for i = 1:nGL
    for j = 1:nGL
        cov_mat(i,j) = double(cov_kernel(xt(i), xt(j)));
    end
end

%% solve the eigenvalue problem
A = cov_mat.*S; %A = D.*cov_mat.*D;
rng(1);
opts.v0 = rand(nGL,1);
opts.tol = eps;
opts.maxit = 500;
opts.p = max(2*d_KL+10,20);
if issymmetric(A) && isreal(A)
    [h, L] = eigs(A, d_KL, 'LA', opts); % [h,L] = eigs(A,M);; % Eq.(19.1.8) Ref.[1]
else
    error('A is not symmetric/real');   %[h,L] = eig((A+A')./2);
end
[eigval,idx] = sort(real(diag(L)),'descend');
h = h(:,idx);
f = D\h;

%% Nystrom's interpolation formula
n = length(xnod);
cov_mat = zeros(n,nGL);
for i = 1:n
    for j = 1:nGL
        cov_mat(i,j) = cov_kernel(xnod(i),xt(j));
    end
end
eigvec = (cov_mat .* repmat(wt,1,n)')*f*diag(1./eigval);
% eigvec = zeros(n,M);
% for i = 1:M
%    for j = 1:nGL
%       eigvec(:,i) = eigvec(:,i) + (1/eigval(i))*wt(j)*f(j,i)*cov_mat(:,j);
%    end
% end

% normalize eigenvectors (just in case)
norm_fact = sqrt(trapz(xnod,eigvec.^2));    % sqrt(sum(eigvec.*eigvec.*repmat(wt,1,M),1));
eigvec = eigvec./repmat(norm_fact,n,1);

% check orthonormality
% So that \int_D phi_i(t)phi_j(t) dt = delta_ij
AA  = ones(d_KL);
for i = 1:d_KL
    for j = 1:d_KL
        AA(i,j) = trapz(xnod, eigvec(:,i).*eigvec(:,j));
    end
end
% figure(55); spy(round(AA,1));

%% plots
if strcmp(img,'true')
    % eigenvalues
    figure(1); hold on;
    plot(1:d_KL, eigval, 'ro-', 'LineWidth', 1);
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

return
%%END