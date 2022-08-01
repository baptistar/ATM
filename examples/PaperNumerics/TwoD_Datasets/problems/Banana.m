classdef Banana

	% Defines a multivariate distribution with 2 dim banana
	% and d-2 dim Gaussian components
	% 
	% Methods: log_pdf, sample
	% Date:    September 2019

	properties
		d 			   % dimension of distribution
		sigma 		   % standard deviation of density (default: 2)
		bananicity     % parameter of distribution (default: 0.2)
		ban_comps      % number of banana components
        name
	end

	methods
		function B = Banana(d, varargin)

			% check dimension
			if (d < 2)
				error('B: dimension must be at least 2')
			end

			% declare B object
			p = ImprovedInputParser;
			addRequired(p,'d');
			addParameter(p,'sigma',2);
			addParameter(p,'bananicity',0.2);
			addParameter(p,'ban_comps',2);
			parse(p,d,varargin{:});
			B = passMatchedArgsToProperties(p, B);

            B.name = 'banana';

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function X = sample(B, N)
			X = randn(N, B.d);
			X(:,1) = B.sigma*X(:,1);
			for i=B.ban_comps
				X(:,i) = X(:,i) + B.bananicity*(X(:,1).^2 - B.sigma^2);
			end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function log_pi = log_pdf(B, X)

			% check inputs
			if size(X,2) ~= B.d
				error('B: dimension mismatch for inputs samples')
			end

			% compute log_pi for first dimension
			log_pi = B.norm_log_pdf(X(:,1), 0, B.sigma);

			% compute log_pi for banana dimensions
			for i=B.ban_comps
				log_pi = log_pi + B.norm_log_pdf(X(:,i), ...
					B.bananicity*(X(:,1).^2 - B.sigma^2), 1);
			end

			% append log_pdf for remaining dimensions
			non_ban_comps = setdiff(2:B.d, B.ban_comps);
			for i=non_ban_comps
				log_pi = log_pi + B.norm_log_pdf(X(:,i),0,1);
			end

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function log_pi = norm_log_pdf(B, X, mean, sigma)
			log_pi = -0.5*log(2*pi*sigma^2) - 0.5/sigma^2*(X - mean).^2;
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods
end %endClass
