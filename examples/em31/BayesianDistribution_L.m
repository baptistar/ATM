classdef BayesianDistribution_L
    
    % Defines a Bayesian (likelihood x prior) distribution
    %
    % Methods: log_pdf, grad_log_pdf, hess_log_pdf, sample
    % Date:    August 2020
    
    properties
        d       % dimension of density
        prior   % log-prior density
        L_lkl   % List of log-Likelihood functions (obs,x)
    end
    
    methods
        function BD = BayesianDistribution_L(d, prior, L_lkl)
            
            % save dimension
            BD.d = d;
            BD.prior=prior;
            BD.L_lkl=L_lkl;
            
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function logpi = log_pdf(BD, X,comp_idx)
            if (nargin < 3)
                comp_idx = 1:BD.d;
            end
            
            logpi=BD.prior.log_pdf(X);
            for k=1:length(BD.L_lkl)
                lkl=BD.L_lkl(k);
                logpi= logpi+lkl.log(X);
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function grad_logpi = grad_x_log_pdf(BD, X,grad_dim,comp_idx)
            % if not specified, evaluate all components of transport map
            if (nargin < 4)
                comp_idx = 1:BD.d;
            end
            % if not specified, compute gradients for all variables
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:BD.d;
            end
            
            grad_logpi = BD.prior.grad_x_log_pdf(X,grad_dim);
 
            for k=1:length(BD.L_lkl)
                lkl=BD.L_lkl(k);
                grad_logpi= grad_logpi+lkl.grad_x_log(X,grad_dim);
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
%         function hess_logpi = hess_x_log_pdf(BD, X)
%             hess_logpi = zeros(size(X,1));
%         end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        %         function Z = sample(MG, N)
        %             Z = mvnrnd(MG.mean, MG.cov, N);
        %         end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
    end %endMethods
end %endClass