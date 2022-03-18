classdef BayesianDistribution
    
    % Defines a Bayesian (likelihood x prior) distribution
    %
    % Methods: log_pdf, grad_log_pdf, hess_log_pdf, sample
    % Date:    August 2020
    
    properties
        d       % dimension of density
        prior   % log-prior density
        lkl     % log-Likelihood function (obs,x)
    end
    
    methods
        function BD = BayesianDistribution(d, prior, lkl)
            
            % save dimension
            BD.d = d;
            BD.prior=prior;
            BD.lkl=lkl;
            
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function logpi = log_pdf(BD, X,comp_idx)
            if (nargin < 3)
                comp_idx = 1:BD.d;
            end
            logpi= BD.lkl.log(X)+BD.prior.log_pdf(X);
            
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
            grad_logpi = BD.lkl.grad_x_log(X,grad_dim)+BD.prior.grad_x_log_pdf(X,grad_dim);
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