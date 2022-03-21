classdef Laplace

	% Defines a univariate Laplace distribution
	% 
	% Methods: log_pdf, grad_log_pdf, hess_log_pdf, sample
	% Date:    July 2019

	properties
		d = 1     % dimension is fixed
		location  % location (i.e., mean) parameter of density
		scale     % scale parameter of density
	end

	methods 
		function Lp = Laplace(varargin)

			p = ImprovedInputParser;
			addParameter(p,'location',0);
			addParameter(p,'scale',1);
			parse(p,varargin{:});
			Lp = passMatchedArgsToProperties(p, Lp);

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function log_pi = log_pdf(Lp, X)
			log_pi = -1*log(2*Lp.scale) -1*abs(X - Lp.location)/Lp.scale;
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function grad_logpi = grad_x_log_pdf(Lp, X)
		% Note: ignoring the differentiability of the density at X
			grad_logpi = -1*sign(X - Lp.location)/Lp.scale;
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function hess_logpi = hess_x_log_pdf(~, X)
		% Note: ignoring the differentiability of the density at X
			hess_logpi = zeros(size(X,1),1,1);
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function Z = sample(Lp, N)
		% Note: X,Y ~ Unif[0,1], Z = log(X/Y) ~ Laplace(0,1)
			X = rand(N,2);
			Z = Lp.location + Lp.scale*log(X(:,1)./X(:,2));
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods
end %endClass
