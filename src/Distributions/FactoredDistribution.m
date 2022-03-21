classdef FactoredDistribution
   
    % Defines a multivariate distribution as a product of conditional
    % factors: \pi(x) = \pi(x_1) \pi(x_2|x_1) ... \pi(x_d|x_{1:d-1})
	% 
    % Factors must contain property: log_pdf, 
    %
	% Methods: log_pdf, grad_log_pdf, hess_log_pdf, sample
	% Date:    August 2020

    properties
		factors  % factors in conditional factorization
	end
    methods 
        function FD = FactoredDistribution(factors)
            FD.factors = factors;
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function d = d(FD)
            d = length(FD.factors);
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function logpi = log_pdf(FD, X, comp_idx)
            if (nargin < 3)
                comp_idx = 1:FD.d;
            end
            logpi = zeros(size(X,1),1);
            for k=comp_idx
                logpi = logpi + FD.factors{k}.log_pdf(X(:,1:k));
            end
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function grad_logpi = grad_x_log_pdf(FD, X, grad_dim)
            grad_logpi = zeros(size(X,1), length(grad_dim));
            for k=1:length(grad_dim)
                d_k = grad_dim(k);
                dxk_logpi = FD.factors{d_k}.grad_x_log_pdf(X(:,1:d_k), grad_dim(1:k));
                grad_logpi(:,1:d_k) = grad_logpi(:,1:d_k) + dxk_logpi;
            end
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function hess_logpi = hess_x_log_pdf(FD, X, grad_dim)
            hess_logpi = zeros(size(X,1), length(grad_dim), length(grad_dim));
            for k=1:length(grad_dim)
                d_k = grad_dim(k);
                d2xk_logpi = FD.factors{d_k}.hess_x_log_pdf(X(:,1:d_k), grad_dim(1:k));
                hess_logpi(:,1:d_k,1:d_k) = hess_logpi(:,1:d_k,1:d_k) + d2xk_logpi;
            end
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
    end %endMethods
    
end %endClass