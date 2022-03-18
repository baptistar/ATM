function [eval, evec, Sigma, P] = OSE_method(cov_func, dom_bound, corr_leng, Nterms)
% Random field representation using OSE method
%{
--------------------------------------------------------------------------
Orthogonal series expansion method (OSE) by Zhang & Ellingwood (1994)
--------------------------------------------------------------------------
Created by:                       Date:           Comment:
Felipe Uribe                      Feb/2015        Comparison of methods
--------------------------------------------------------------------------
Based on:
1."Stochastic finite element methods and reliability"
   B. Sudret and A. Der Kiureghian. State of the art report. (2000)
--------------------------------------------------------------------------
%}
%%
dim = length(corr_leng);
switch dim
    case 1
        a = (dom_bound{1}(2)-dom_bound{1}(1))/2;   % scale
        T = (dom_bound{1}(2)+dom_bound{1}(1))/2;   % shift
        
        % Legendre polynomials and Gauss points
        nGp = Nterms;
        [x_gl, w_gl, pol] = gauss_quad(nGp, 'legen');
        
        % extract Legendre polys
        P = cell(Nterms,1);
        for i = 1:Nterms+1
            P{i} = pol{i};
        end
        
        % covariance between Gauss points
        C = zeros(nGp);
        for i = 1:nGp
            xi_i = x_gl(i);
            for j = i:nGp
                xi_j = x_gl(j);
                C(i,j) = cov_func([a*xi_i+T, a*xi_j+T]);
                C(j,i) = C(i,j);
            end
        end
        
        % G-L integration of the covariance function: Eq. (2.22) or Eq. (2.34) Part II.
        Sigma = zeros(Nterms);
        for k = 1:Nterms
            for l = 1:Nterms
                integral = 0;
                for i = 1:nGp   % 1st integral
                    xi_i = x_gl(i);    w_i = w_gl(i);
                    Pk = polyval(P{k}, xi_i);
                    for j = 1:nGp   % 2nd integral
                        xi_j = x_gl(j);   w_j = w_gl(j);
                        Pl = polyval(P{l}, xi_j);
                        integral = integral + w_i*w_j * C(i,j) * Pk*Pl;
                    end
                end
                Sigma(k, l) = (a/2) * sqrt((2*k-1)*(2*l-1)) * integral;
                Sigma(l, k) = Sigma(k, l);
            end
        end
        
        % calculate the eigenpairs
        A = eye(Nterms);
        tic;
        [evec, eval] = eigs(Sigma, A, Nterms);   % Eq.(2.23) Ref.[1] Part 2.
        t2 = toc;
        [eval, idx] = sort(diag(eval), 'descend');
        evec = evec(:,idx);
        fprintf('\nElapsed time solving eigenvalue problem %g s\n\n', t2);
        
        % plot eigenvalues
        figure;
        plot(1:Nterms, eval,'ro'); grid minor; axis tight;
        title('Eigenvalues');
        
        % plot eigenvectors
        %{
        m      = 1e2;
        xnod   = linspace(dom_bound{1}(1),dom_bound{1}(2),nGp);
        xx     = linspace(dom_bound{1}(1),dom_bound{1}(2),m);
        eigfun = cell(Nterms,1);
        figure;
        for i = 1:Nterms  % eigfun: eigvec in function handle form
            eigfun{i} = @(xx) interp1(xnod,evec(:,i),xx,'spline');
            plot(xx,eigfun{i}(xx),'LineWidth',2); hold on; grid minor;
        end
        %}
    otherwise
        error('Non-supported dimension');
end

return
%%END