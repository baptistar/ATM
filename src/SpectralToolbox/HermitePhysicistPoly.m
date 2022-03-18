classdef HermitePhysicistPoly

	% HermitePhysicistPoly defines Physicist Hermite polynomials 
	%
	% Un-normalized:
	%  H_0(x) = 1
	%  H_1(x) = 2*x
	%  H_2(x) = 4x^2 - 2
	%
	% Normalized:
	%  \tilde{H}_n(x) = H_n(x)/\sqrt(n!*2^n*\sqrt(pi))
	%
	% Methods include: evaluate, grad_x, grad_vandermonde
	%
	% Author: Ricardo Baptista
	% Date:   September 2019

	properties

	end %endProperties

	methods
		function HP = HermitePhysicistPoly(varargin)
			
			% Define HP object
			p = ImprovedInputParser;
			parse(p, varargin{:});
			HP = passMatchedArgsToProperties(p, HP);

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function c = normalization_const(~, m)
            %c = sqrt(gamma(m+1) .* 2.^m * sqrt(pi));
            c = sqrt(gamma(m+1) .* 2.^m);  %sqrt(pi) is already accounted for by sampling
        end %endFunction
        %------------------------------------------------------------------
		%------------------------------------------------------------------
		function c = monomial_coeffs(HP, m, norm)
		% Return the monomial polynomial coefficients
		% normalized coefficients divide by \sqrt(\sqrt(pi) * 2^m * m!)
		%
		% Inputs:  m - order of Hermite functions
		% 		   norm - True/False bool for normalization
		% Outputs: c - (ncoeff x 1) coefficients
			c = HP.Hermite_coeffs(m);
			if norm == true
				c = c / HP.normalization_const(m);
			end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function c = Hermite_coeffs(~, N)
		% Compute the coefficients of the Hermite polynomials
		% using a recursive algorithm. Input: N = polynomial order
		%
		% Example: [1], H(0,x) = 1
		%		   [0, 2], P(1,x) = 2*x
		% 		   [-2, 0, 4], P(2,x) = 4x^2 - 2; 
		%
		% Reference: adapted from John Burkardt (2010)

			% N<0, N=0, and N=1 cases
			if N < 0
				c = [];
				return
			elseif N == 0
				c = 1;
				return
            elseif N == 1
				c = [0 2];
				return
			end

			% compute N>=2 by recursion
			c = [1, 0; 0, 2];
			for i=2:N
				c(i+1,1)     =  -2.0 * ( i - 1 ) * c(i-1,1);
				c(i+1,2:i-1) =   2.0             * c(i  ,1:i-2)...
								-2.0 * ( i - 1 ) * c(i-1,2:i-1);
				c(i+1,  i  ) =   2.0             * c(i  ,  i-1);
				c(i+1,  i+1) =   2.0             * c(i  ,  i  );
			end
			c = c(end,:);
		
		end % endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------ 
		function p = evaluate(HP, x, m, norm)
		% Evaluate the m-th order Hermite physicist polynomial
		% at N=size(x,1) points using the recursion relation 
		% H_0(x) = 1 and H_{n}(x) = 2xH_{n-1}(x) - H_{n-1}'(x)
		% where H_{n-1}'(x) = 2(n-1)H_{n-2}(x)
		%
		% Inputs:  x - (N x 1) inputs
		% 		   m - order of Hermite functions
		% 		   norm - True/False bool for normalization
		% Outputs: p - (N x 1) poly evaluations

			% set initial condition H_0(x) for recursion
			N = size(x,1);
			p = ones(N,1);

			% run recursion by tracking H_{n-1}, H_{n}, H_{n+1}
			if m > 0

				% initialize p_jm1, p_jm2
				p_jm1 = p;
				p_jm2 = zeros(N,1);

				% evaluate H_{j} by recursion and
				% saving H_{j-1}(x) and H_{j}(x)
				for j=1:m
					p = 2*x.*p_jm1 - 2*(j-1)*p_jm2;
					p_jm2 = p_jm1;
					p_jm1 = p;
				end
                
			end

			% normalize polynomials
			if norm == true
				p = p / HP.normalization_const(m);
			end

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function dp = grad_x(HP, x, m, k, norm)
		% Evaluate the k-th derivative of the m-th order Hermite 
		% physicist polynomial at m=size(x,1) points using the
		% relation H_{n}^(k)(x) = 2^{k} n!/(n-k)! H_{n-k}(x)
		%
		% Inputs:  x - (N x 1) input samples
		% 		   m - order of Hermite functions
		% 		   norm - True/False bool for normalization
		% Outputs: dp - (N x 1) poly evaluations

			if m >= k
				fact = 2^k*exp(gammaln(m+1) - gammaln(m-k+1));
				dp = fact*HP.evaluate(x, m-k, norm);
                % multiply by ratio of normalization constants c_(m-k)/c_m
                if norm == true
                    dp = dp/sqrt(fact);
                end
			else
				dp = zeros(size(x));
			end

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------ 
        function dV = grad_vandermonde(HP, x, m, k, norm)
		% Generate matrix with the k-th derivative of all m-order
		% Hermite Physicists' polynomials evaluated at N points in x
		%
		% Inputs:  x - (N x 1) input samples
		% 		   m - order of Hermite functions
		%  		   k - k-th derivative (k=0 just evaluates poly)
		% 		   norm - True/False bool for normalization
		% Outputs: dV (N x m+1) matrix of polynomial evaluations
			dV = zeros(size(x,1), m+1);
			for i=0:m
				dV(:,i+1) = HP.grad_x(x, i, k, norm);
			end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods

end %endClass
