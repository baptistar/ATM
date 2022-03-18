classdef Uniform

	% Defines a univariate uniform distribution
	% 
	% Methods: log_pdf, grad_log_pdf, hess_log_pdf, sample
	% Date:    July 2019

	properties
        a         % left boundary
        b         % right boundary
        inf       % value for negative inf
	end

	methods 
		function U = Uniform(varargin)

			p = ImprovedInputParser;
			addParameter(p,'a',-1);
			addParameter(p,'b',1);
			parse(p,varargin{:});
			U = passMatchedArgsToProperties(p, U);
            
            U.inf = 1e3;

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function log_pi = log_pdf(U, X)
            log_pi = -U.inf * ones(size(X,1),1);
            log_pi(X > U.a & X < U.b) = -1*log(U.b - U.a);
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function grad_logpi = grad_x_log_pdf(~, X)
			grad_logpi = zeros(size(X,1),1);
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function hess_logpi = hess_x_log_pdf(~, X)
			hess_logpi = zeros(size(X,1),1,1);
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function Z = sample(U, N)
			Z = (U.b - U.a) * rand(N,1) + U.a;
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods
end %endClass
