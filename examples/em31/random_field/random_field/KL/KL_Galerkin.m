function [eigval, eigvec] = KL_Galerkin(d_KL, cov_kernel, xnod, img)
% Solution of the Fredholm integral using Galerkin FEM procedure
%{
--------------------------------------------------------------------------
Created by:                       Date:           Comment:
Felipe Uribe                      Mar/2015        Comparison of methods
--------------------------------------------------------------------------
Based on:
1."Stochastic finite elements A spectral approach"
   R. Ghanem and P.D. Spanos. Rev edition 2012. Dover publications Inc.
2."Numerical methods for the discretization of random fields by means of
   the Karhunen-Loève expansion"
   Betz et al. (2014). In Comput. methods in appl. mech. and eng.
--------------------------------------------------------------------------
%}

%% FEM parameters
ned  = 1;                         % number of dof per node
nen  = 2;                         % number of nodes per element
nnp  = length(xnod);              % number of nodal points
nfe  = nnp - 1;                   % number of finite elements
neq  = ned*nen;                   % number of element equations
ndof = ned*nnp;                   % number of degrees of freedom
IEN  = [(1:(nnp-1))' (2:nnp)']';  % loc-glo
ID   = 1:ndof;

% localization matrix
LM = cell(nfe,1);
for e = 1:nfe
    LM{e} = [ID(IEN(1,e)); ID(IEN(2,e))];
end

%% two-node 1D elements: lagrangian shape functions
Nshape = @(xi) [ -(xi-1)/2, (xi+1)/2 ]';    % [N1, N2]

%% Gauss-Legendre parameters
nGL = 3;                 % Gauss-Legendre quadrature order
[x_gl, w_gl] = gauss_quad(nGL);

%% Computing matrix B
B = sparse(ndof, ndof);
tic;
for e = 1:nfe
    Be = zeros(neq);
    det_Je = (xnod(IEN(2,e))-xnod(IEN(1,e)))/2;
    
    % gauss-legendre quadrature integration
    for p = 1:nGL
        xi_gl = x_gl(p);
        NN = Nshape(xi_gl);    % shape functions on GL points
        
        % B matrix
        Be = Be + NN'*NN*det_Je*w_gl(p);
    end
    B(LM{e}, LM{e}) = B(LM{e}, LM{e}) + Be;
end
toc;

%% computing matrix C
C = sparse(ndof,ndof);
tic;
for e = 1:nfe
    xe = xnod(IEN(:,e));
    det_Je = (xnod(IEN(2,e))-xnod(IEN(1,e)))/2;
    for f = 1:nfe
        Cef = zeros(nen);
        xf = xnod(IEN(:,f));
        det_Jf = (xnod(IEN(2,f))-xnod(IEN(1,f)))/2;
        for p1 = 1:nGL
            xi_gl_e = x_gl(p1);
            NNe = Nshape(xi_gl_e);   % shape function of element e
            xp1 = sum(NNe.*xe);      % in global coordinates
            for p2 = 1:nGL
                xi_gl_f = x_gl(p2);
                NNf = Nshape(xi_gl_f);   % shape function of element f
                xp2 = sum(NNf.*xf);      % in global coordinates
                
                % element C matrix
                Cef = Cef + cov_kernel(xp1, xp2)*NNe'*NNf*det_Je*det_Jf*w_gl(p1)*w_gl(p2);
            end
        end
        C(LM{e},LM{f}) = C(LM{e},LM{f}) + Cef;
    end
end
toc;
%figure; subplot(121); spy(C); subplot(122); spy(B);

%% solve generalized eigenvalue problem: CA = BAD
rng(1);
opts.v0    = rand(ndof, 1);
opts.tol   = eps;
opts.maxit = 500;
opts.p     = max(2*d_KL+10, 20);
[D,A] = eigs(C, B, d_KL, 'LM', opts);  % or: [D,L] = eig(C,B);
%
[eigval,idx] = sort(real(diag(A)), 'descend');
eigvec = D(:, idx);

% function form
eigfun = cell(d_KL, 1);
for k = 1:d_KL
    eigfun{k} = @(xx) interp1(xnod, eigvec(:,k), xx);
end

% normalize eigenvectors (just in case)
norm_fact = sqrt(trapz(xnod, eigvec.^2));    % sqrt(sum(eigvec.*eigvec.*repmat(wt,1,M),1));
eigvec = eigvec./repmat(norm_fact, nnp, 1);

% check orthonormality
% So that \int_D phi_i(t)phi_j(t) dt = delta_ij
AA = ones(d_KL);
for i = 1:d_KL
    for j = 1:d_KL
        AA(i,j) = trapz(xnod, eigvec(:,i).*eigvec(:,j));
    end
end
figure(66); spy(round(AA, 1));

%% plots
if strcmp(img,'true')
    % eigenvalues
    figure(1); hold on;
    plot(1:d_KL, eigval, 'b*--', 'LineWidth', 1);
    xlabel('Index [n]','Interpreter','Latex','FontSize',18);
    ylabel('Eigenvalue [$\lambda_n$]','Interpreter','Latex','FontSize',18);
    set(gca,'FontSize',13);
    
    % eigenfunctions
    figure;
    plot(xnod, eigvec(:,1:end), 'LineWidth', 1); axis tight;
    xlabel('Length [$x$]','Interpreter','Latex','FontSize',18);
    ylabel('Eigenfunction [$\phi(x)$]','Interpreter','Latex','FontSize',18);
    title('Galerkin','Interpreter','Latex','FontSize',18);
    set(gca,'FontSize',13); grid minor;
end
%%END