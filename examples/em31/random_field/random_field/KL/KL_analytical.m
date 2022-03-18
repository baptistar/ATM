function [eigval, eigvec, eigfun] = KL_analytical(d_KL, l_c, sigma, xnod, img)
%% Analytical solution of the Fredholm equation for the exponential kernel
%{
--------------------------------------------------------------------------
Created by:                        Date:            Comment:
Felipe Uribe                       August 2016      Important algorithm
--------------------------------------------------------------------------
d_KL   ---> number of KL terms
lambda ---> correlation length 
n      ---> partition of the domain
img    ---> 'true' if want to show the eigenpairs plots
--------------------------------------------------------------------------
Based on:
1."Stochastic finite elements A spectral approach"
   R. Ghanem and P.D. Spanos. Rev edition (2012). Dover publications Inc.
--------------------------------------------------------------------------
%}

%% domain data
a = (xnod(end)-xnod(1))/2;
[~, eigval, eigfun] = analytical_solution(d_KL, a, l_c);

% eigenvectors
T = (xnod(end) + xnod(1))/2;   % shift parameter (non-symmetric domain)
eigvec = zeros(length(xnod),d_KL);
for i = 1:d_KL
   eigvec(:,i) = eigfun{i}(xnod-T);
end

% scale eigenvalues to the autocovariance
eigval = sigma^2*eigval; 
   
%% plots
if strcmp(img,'true')
   % eigenvalues
   figure(1); hold on;
   plot(1:d_KL, eigval, 'ks', 'LineWidth', 1);
   xlabel('Index [n]','Interpreter','Latex','FontSize',18);
   ylabel('Eigenvalue [$\lambda_n$]','Interpreter','Latex','FontSize',18);
   set(gca,'FontSize',13);
   
   % eigenfunctions
   figure;
   plot(xnod, eigvec(:,1:d_KL), 'LineWidth', 1); axis tight;
   xlabel('Length [$x$]','Interpreter','Latex','FontSize',18);
   ylabel('Eigenfunction [$\phi(x)$]','Interpreter','Latex','FontSize',18);
   title('Analytical','Interpreter','Latex','FontSize',18);
   set(gca,'FontSize',13); grid minor;
end

return

%%========================================================================
function [wn, eigval, eigfun] = analytical_solution(M, a, l_c)

c = 1/l_c;
fun_o = @(ww) c - ww*tan(ww*a);
fun_e = @(ww) ww + c*tan(ww*a);

% constants for indexing the point of search
j = 0;   k = 0;
option = optimset('Display','off');
wn = zeros(M,1);
eigfun = cell(M,1);
%
for i = 0:ceil(M/2)
   % odd: compute data associated with equation : c - w*tan(a*w) = 0
   if ((i > 0) && (2*i-1 <= M))
      k = k+1;
      n = 2*i-1;
      wn(n)     = fsolve(fun_o, (k-1)*(pi/a)+1e-3, option);  
      alpha     = sqrt(a + (sin(2*wn(n)*a)./(2*wn(n))));
      eigfun{n} = @(x) cos(wn(n)*x)/alpha;
   end
   % even: compute data associated with equation : w + c*tan(a*w)
   if ((2*i+2) <= M)
      j = j+1;
      n = 2*i+2;
      wn(n)     = fsolve(fun_e, (j-0.5)*(pi/a)+1e-3, option); 
      alpha     = sqrt(a - (sin(2*wn(n)*a)./(2*wn(n))));
      eigfun{n} = @(x) sin(wn(n)*x)/alpha;
   end
end
eigval = (2*c)./(wn.^2 + c^2);

return
%%END