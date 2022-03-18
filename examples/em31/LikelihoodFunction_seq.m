classdef LikelihoodFunction_seq
   
    % Defines a likelihood function defined by a TM
	% 
	% Methods: log, grad_x_log
	% Date:    August 2020

    properties
		PB_off       % log-prior density
        PB_prec
		yobs     % obs
        dp       % nb of parameters
        dy       % nb of observations
    end
    
    methods 
        function lkl = LikelihoodFunction_seq(PB_off,PB_prec, yobs)
            lkl.PB_off = PB_off;
            lkl.PB_prec=PB_prec;
            lkl.yobs=yobs;
            
            lkl.dp=PB_off.S{1}.d-length(yobs);
            lkl.dy=length(yobs);
            
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function logx = log(lkl, X,comp_idx)
            idx_comp=lkl.dp+1:lkl.dp+lkl.dy;
            if nargin < 3
                comp_idx=1:length(idx_comp);
            end
            
            x2=lkl.PB_prec.evaluate(X);
            %X size (N,dtheta)
            x_jt=[x2, repmat(lkl.yobs,length(X),1)];
            logx=lkl.PB_off.log_pdf(x_jt,idx_comp(comp_idx));
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
            
            x2=lkl.PB_prec.evaluate(X);
            x_jt=[x2, repmat(lkl.yobs,size(X,1),1)];
            
            grad_PB_prec=lkl.PB_prec.grad_x(X);
            lkl_grad=lkl.PB_off.grad_x_log_pdf(x_jt,idx_grad(grad_dim),idx_comp(comp_idx));
            
            grad_PB_prec=permute(grad_PB_prec,[3 2 1]); %grad transpose
            lkl_grad=permute(lkl_grad,[2 3 1]);
            
            grad_logpi=pagemtimes(grad_PB_prec,lkl_grad);
            
            grad_logpi=permute(grad_logpi,[3 1 2]);

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