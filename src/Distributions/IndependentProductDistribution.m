classdef IndependentProductDistribution < FactoredDistribution
   
    % Defines a multivariate distribution as a product of independent
    % factors: \pi(x) = \pi(x_1) \pi(x_2) ... \pi(x_d)
	% 
    % Factors must contain property: log_pdf, 
    %
	% Methods: log_pdf, grad_log_pdf, hess_log_pdf, sample
	% Date:    August 2020

    properties
    end
    
    methods 
        function IP = IndependentProductDistribution(factors)
            IP@FactoredDistribution(factors);
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function d = dim(FD)
            d = length(FD.factors);
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function logpi = log_pdf(FD, X, comp_idx)
            if (nargin < 3) || isempty(comp_idx)
                comp_idx = 1:FD.dim;
            end
            logpi = zeros(size(X,1),1);
            for k=1:length(comp_idx)
                Ck = comp_idx(k);
                logpi = logpi + FD.factors{Ck}.log_pdf(X(:,Ck));
            end
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function grad_logpi = grad_x_log_pdf(FD, X, grad_dim, comp_idx)
        % compute gradient of log_pdf
        % Inputs: X - (N x d) array of samples
        %         grad_dim - dimensions for computing gradient
        %         comp_idx - components to compute gradients
        % Output: grad_logpi - (N x d) array where (i,j) is the
        %         derivative with respect to grad_dim(j) for sample i
            assert(size(X,2) == FD.dim)
            if nargin < 4 || isempty(comp_idx)
                comp_idx = 1:FD.dim;
            end
            if nargin < 3 || isempty(grad_dim)
                grad_dim = 1:FD.dim;
            end
            grad_logpi = zeros(size(X,1), length(grad_dim));
            for j=1:length(grad_dim)
                dj = grad_dim(j);
                if ~isempty(intersect(comp_idx,dj))
                    grad_logpi(:,j) = FD.factors{dj}.grad_x_log_pdf(X(:,dj));
                end
            end
            %grad_logpi = grad_logpi(:,:,grad_dim);
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function hess_logpi = hess_x_log_pdf(FD, X, grad_dim, comp_idx)
        % compute Hessian of log_pdf
        % Inputs: X - (N x d) array of samples
        %         grad_dim - dimensions for computing gradient
        %         comp_idx - components to compute gradients
        % Output: grad_logpi - (N x d x d) array where (i,j,k) is the
        %         derivative with respect to grad_dim(j) and grad_dim(k) for sample i
            assert(size(X,2) == FD.dim)
            if nargin < 4 || isempty(comp_idx)
                comp_idx = 1:FD.dim;
            end
            if nargin < 3 || isempty(grad_dim)
                grad_dim = 1:FD.dim;
            end
            hess_logpi = zeros(size(X,1), length(grad_dim), length(grad_dim));
            for j=1:length(grad_dim)
                dj = grad_dim(j);
                if ~isempty(intersect(comp_idx,dj))
                    d2xk_logpi = FD.factors{dj}.hess_x_log_pdf(X(:,dj));
                    hess_logpi(:,dj,dj) = d2xk_logpi;
                end
            end
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
    end %endMethods
    
end %endClass