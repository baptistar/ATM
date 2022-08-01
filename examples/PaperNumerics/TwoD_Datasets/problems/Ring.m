classdef Ring

	% Defines an d-dimensional distribution with
	% two components lying along a ring and
	% (d-2) Gaussian components.
	%
	% Author:
	% Date:   September 2019

	properties
		d 			   % dimension of distribution
		sigma 		   % standard deviation of density (default: 2)
		radia          % parameter of distribution (default: 5.0)
                name
	end

	methods
		function R = Ring(d, varargin)

			% check dimension
			if (d < 2)
				error('R: dimension must be at least 2')
			end

			% declare R object
			p = ImprovedInputParser;
			addRequired(p,'d');
			addParameter(p,'sigma',0.2);
			addParameter(p,'radia',5.0);
			parse(p,d,varargin{:});
			R = passMatchedArgsToProperties(p, R);

                        R.name = 'ring';

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function X = sample(R, N)

			% draw angles and noise
			angles = rand(N,1)*2*pi;
			noise  = randn(N,1)*R.sigma;

			% compute weights
			weights = 2*pi*R.radia; 
			weights = weights/sum(weights);

			% sample from possible radia vector
			radia_idx = randsample(length(R.radia), N, true, weights);
			radius_samples = R.radia(radia_idx) + noise;

			% compute points and append into X
			xs = radius_samples.*sin(angles);
			ys = radius_samples.*cos(angles);
			X = [xs, ys];

			% append independent Gaussian directions
			if R.d > 2
				X = [X, R.sigma*randn(N, R.d-2)];
			end

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function log_pi = log_pdf(R, X)

			% check inputs
			if size(X,2) ~= R.d
				error('R: dimension mismatch for inputs samples')
			end

			% compute weights
			weights = 2*pi*R.radia;
			weights = weights/sum(weights);

			% compute norms of samples in first two directions
			norms = sqrt(sum(X(:,1:2).^2,2));

			% compute log_pi by summing value under rings for each sample
			log_pi = zeros(size(X,1),1);
			for i=1:size(X,1)
				log_pdf_ring_comps = -0.5*(norms(i) - R.radia)^2/(R.sigma^2) - ...
					0.5*log(2*pi*R.sigma^2) - log(2*pi*R.radia);
				log_pi(i) = R.log_sum_exp(log_pdf_ring_comps + log(weights));
			end

			% compute log_pdf for remaining Gaussian directions
			for i=3:R.d
				log_pi = log_pi + R.norm_log_pdf(X(:,i), 0, R.sigma);
			end

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function s = log_sum_exp(~, X)
			% evaluate log(\int f(x)p(x)dx) given p(x) and log f(x) samples

			% compute minimum value and remove from x
			[X_min, min_idx] = min(X);
			X_without_min = X; X_without_min(min_idx) = [];
            
			% compute log(\sum_i exp(x_i))
			s = X_min + log(1 + sum(exp(X_without_min - X_min)));

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function log_pi = norm_log_pdf(~, X, mean, sigma)
			log_pi = -0.5*log(2*pi*sigma^2) - 0.5/sigma^2*(X - mean).^2;
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods
end %endClass
