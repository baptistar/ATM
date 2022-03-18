function [eval, evec, xnod] = EOLE_method(cov_func, dom_bound, partition, Nterms)
% Random field representation using EOLE method
%{
--------------------------------------------------------------------------
Expansion optimal linear estimator (EOLE) by Li & Der Kiureghian (1993)
--------------------------------------------------------------------------
Created by:                       Date:           Comment:
Felipe Uribe                      Feb/2015        Comparison of methods
--------------------------------------------------------------------------
Based on:
1."Stochastic finite element methods and reliability"
   B. Sudret and A. Der Kiureghian. State of the art report. (2000)
--------------------------------------------------------------------------
%}

%% initial
set(0,'defaultTextInterpreter','latex');
xnod = linspace(dom_bound{1}(1),dom_bound{1}(2),partition);
nnp = length(xnod);

figure;
plot(xnod,zeros(nnp),'b.');
title('Random field mesh');

%% computing covariance matrix
C_mat = zeros(nnp);
tic;
for i = 1:nnp
    for j = i:nnp
        C_mat(i,j) = cov_func([xnod(i) xnod(j)]);
        C_mat(j,i) = C_mat(i,j);
    end
end
t1 = toc;
fprintf('\nElapsed time assembling the covariance kernel matrix %g s', t1);

%% calculate the eigenpairs
A = eye(nnp);
tic;
[evec, eval] = eigs(C_mat, A, Nterms);   % Eq.(2.19) Ref.[1] Part 2.
t2 = toc;
[eval,idx] = sort(diag(eval), 'descend');
evec = evec(:,idx);
fprintf('\nElapsed time solving eigenvalue problem %g s\n\n', t2);

% normalize eigenvectors to obtain same solution as in the analytic case
%{
norm_fact = sqrt(trapz(xnod,evec.^2));
evec      = evec./repmat(norm_fact,nnp,1);
dt        = xnod(2)-xnod(1);
eval      = eval*dt;
%}

%% plots
% eigenvalues
figure;
plot(eval,'bo','Linewidth',2); grid minor;
xlabel('Index, $$n$$','FontSize',18); ylabel('$\lambda_n$','FontSize',18);
set(gca,'FontSize',16);

% eigenfunctions
figure;
plot(xnod,evec,'LineWidth',2); hold on; grid minor; axis tight;
xlabel('x'); ylabel('Eigenfunctions');

return
%%END