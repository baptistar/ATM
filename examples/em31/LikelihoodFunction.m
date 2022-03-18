classdef LikelihoodFunction
   
    % Defines a likelihood function defined by a TM
	% 
	% Methods: log, grad_x_log
	% Date:    August 2020

    properties
		PB       % log-prior density
		yobs     % obs
        dp       % nb of parameters
        dy       % nb of observations
    end
    
    methods 
        function lkl = LikelihoodFunction(PB, yobs)
            
            lkl.PB = PB;
            lkl.yobs=yobs;
            lkl.dp=PB.S{1}.d-length(yobs);
            lkl.dy=length(yobs);
            
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function logx = log(lkl, X,comp_idx)
            idx_comp=lkl.dp+1:lkl.dp+lkl.dy;
            
            if nargin < 3
                comp_idx=1:length(idx_comp);
            end
            %X size (N,dtheta)
            x_jt=[X, repmat(lkl.yobs,size(X,1),1)];
            logx=lkl.PB.log_pdf(x_jt,idx_comp(comp_idx));

        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function grad_logpi = grad_x_log(lkl, X,grad_dim,comp_idx)
            idx_grad=1:lkl.dp;
            idx_comp=lkl.dp+1:lkl.dp+lkl.dy;
            
            % if not specified, evaluate all components of transport map
            if (nargin < 4)
                comp_idx = 1:length(idx_comp);
            end
            % if not specified, compute gradients for all variables
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:length(idx_grad);
            end

            x_jt=[X, repmat(lkl.yobs,size(X,1),1)];
            
            
            grad_logpi=lkl.PB.grad_x_log_pdf(x_jt,idx_grad(grad_dim),idx_comp(comp_idx));
            
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
%         function hess_logpi = hess_x_log_pdf(lkl, X)
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