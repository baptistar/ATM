classdef MultivariateGaussian
   
    % Defines a multivariate Gaussian distribution
	% 
	% Methods: log_pdf, grad_log_pdf, hess_log_pdf, sample
	% Date:    August 2020

    properties
        d      % dimension of density
		mean   % mean of density
		cov    % covariance of density
        R      % cholesky factor of cov
    end
    
    methods 
        function MG = MultivariateGaussian(d, mean, cov, comp_cholesky)
            
            % save dimension
            MG.d = d;
            
            % set default for comp_cholesky
            if (nargin < 4)
                comp_cholesky = true;
            end
            
            % define covariance and check dimensions
            if (nargin < 3)
                cov = eye(d);
            end
            if (size(cov,1) ~= size(cov,2)) || (size(cov,1) ~= d)
                error('Dimension mismatch of covariance')
            end

            % define mean and check dimensions
            if (nargin < 2)
                mean = zeros(1,d);
            end
            if (length(mean) ~= d)
                error('Dimension mismatch of mean')
            end
            if (size(mean,2) ~= d)
                mean = mean.';
            end
            
            % set mean and covariance
            MG.mean = mean;
            MG.cov = cov;
            if (comp_cholesky == true)
                MG.R = cholcov(MG.cov);
            else
                MG.R = [];
            end
            
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function logpi = log_pdf(MG, X,comp_idx)
            % compute cholesky factor
            if isempty(MG.R)
                MG.R = cholcov(MG.cov);
            end
            % center and whiten data
            X0_Rinv = ((X - MG.mean) / MG.R);
            % compute log pdf
            quad_form = sum( X0_Rinv.^2 , 2);
            logdetR = sum(log(diag(MG.R)));
            logpi = -MG.d/2*log(2*pi) - logdetR - 0.5*quad_form;
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function grad_logpi = grad_x_log_pdf(MG, X,grad_dim,comp_idx)
            grad_logpi = -1*(X - MG.mean) / MG.cov;
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function hess_logpi = hess_x_log_pdf(MG, X)
            hess_logpi = zeros(size(X,1), MG.d, MG.d);
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function Z = sample(MG, N)
            Z = mvnrnd(MG.mean, MG.cov, N);
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
    end %endMethods
    
end %endClass