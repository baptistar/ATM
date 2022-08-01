classdef Funnel

	% Defines a multivariate distribution with 2 dim that
	% have a funnel shape and d-2 dim Gaussian components
	%
	% Author:
	% Date:   September 2019

	properties
		d 			   % dimension of distribution
		sigma 		   % standard deviation of density (default: 2.0)
		limit_min      % parameter of distribution (default: 0.0)
		limit_max      % parameter of distribution (default: 10.0)
                name
	end

	methods
		function F = Funnel(d, varargin)

			% check dimension
			if (d < 2)
				error('F: dimension must be at least 2')
			end

			% declare F object
			p = ImprovedInputParser;
			addRequired(p,'d');
			addParameter(p,'sigma',2.0);
			addParameter(p,'limit_min',0.0);
			addParameter(p,'limit_max',10.0);
			parse(p,d,varargin{:});
			F = passMatchedArgsToProperties(p, F);

                        F.name = 'funnel';
 
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function X = sample(F, N)
			X = randn(N, F.d);
			X(:,1) = F.sigma*X(:,1);
			for i=2:F.d
				X(:,i) = X(:,i).*sqrt(F.threshold(X(:,1)));
			end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function v = threshold(F, x)
			v = exp(-x);
			v(v > F.limit_max) = F.limit_max;
			v(v < F.limit_min) = F.limit_min;
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function log_pi = log_pdf(F, X)

			% check inputs
			if size(X,2) ~= F.d
				error('F: dimension mismatch for inputs samples')
			end

			% compute log_pi for first dimension
			log_pi = F.norm_log_pdf(X(:,1), 0, F.sigma);

			% evaluate log_pi for remaining dimensions
			for i=2:F.d
				v = F.threshold(X(:,1));
				log_pi = log_pi + F.norm_log_pdf(X(:,i), 0, sqrt(v));
			end

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function log_pi = norm_log_pdf(~, X, mean, sigma)
			log_pi = -0.5*log(2*pi*sigma.^2) - 0.5./sigma.^2.*(X - mean).^2;
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods
end %endClass
