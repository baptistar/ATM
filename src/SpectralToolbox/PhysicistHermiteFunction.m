classdef PhysicistHermiteFunction

	% PhysicistHermiteFunction defines the physicist Hermite functions
	%   \psi_n(x) = H_{n}(x)*exp(-x^2/2)
	%
	% Methods include: evaluate, grad_x, grad_vandermonde
	% 
	% Author: Ricardo Baptista
	% Date: June 2018 

	properties

		H   % Physicist Hermite polynomial object
		He  % Probabilist Hermite polynomial object

	end

	methods
		function HF = PhysicistHermiteFunction(varargin)

			% Define HF object
			p = ImprovedInputParser;
			parse(p, varargin{:});
			HF = passMatchedArgsToProperties(p, HF);

			% Set Hermite polynomial properties
			HF.H  = HermitePhysicistPoly();
			HF.He = HermiteProbabilistPoly();

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function p = evaluate(HF, x, m, norm)
		% evaluate the m-th order Hermite Physicists' functions
			p = HF.H.evaluate(x, m, false).*exp(-x.^2/2);
            if norm == true
                p = p / HF.H.normalization_const(m);
            end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function dp = grad_x(HF, x, m, k, norm)
		% evaluate the k-th derivative of the m-th order Hermite functions
            if k==0
                dp = HF.evaluate(x, m, norm);
            elseif k>0
                % use the recurrence relation for the k derivative
                % \psi_m^k(x) = \sum_{i=0}^{k} (k choose i) (-1)^i * 
                %   2^((k-i)/2) * \sqrt{m!/(m-k+i)!} \psi_{m-k+i}(x) He_i(x)
                dp = zeros(size(x));
                for i=max(0,k-m):k
                    Fi  = nchoosek(k,i) * (-1)^i * 2^(k-i);
                    Fi  = Fi * exp(gammaln(m+1) - gammaln(m-k+i+1));
                    psi = HF.evaluate(x, m-k+i, false);
                    dp  = dp + Fi * psi .* HF.He.evaluate(x, i, false);
                end
                % normalize by the Hermite physicist polynomial constant
                if norm == true
                    dp = dp / HF.H.normalization_const(m);
                end
            elseif k==-1
                % extract monomial coefficients
                c = HF.H.monomial_coeffs(m, norm);
                % compute integral using monomial formula
                const    = 0;
                pi_erf_c = 0;  		     % * sqrt(pi/2) * erf(x/sqrt(2))
                exp_c    = zeros(m,1);   % * exp(-x**2/2)
                % compute coeffs for each term
                for nn=0:m
                    if mod(nn,2) == 0
                        pi_erf_c = pi_erf_c + (c(nn+1)*fact2(nn-1));
                        for k=1:(nn/2)
                            exp_c(nn-2*k+1+1) = exp_c(nn-2*k+1+1) + ...
                                (c(nn+1) * fact2(nn-1) / fact2(nn-2*k+1));
                        end
                    else
                        const = const + (c(nn+1) * fact2(nn-1));
                        for k=0:((nn-1)/2)
                            exp_c(nn-2*k-1+1) = exp_c(nn-2*k-1+1) + ...
                                (c(nn+1) * fact2(nn-1) / fact2(nn-2*k-1));
                        end
                    end
                end
                % evaluate exponential terms
                ev_pi_erf = sqrt(pi/2)*erf(x/sqrt(2));
                ev_exp = exp(-x.^2/2);
				% evaluate x^N matrix
				ntot = size(exp_c,1);
				X = ones(size(x,1),ntot);
				for i=2:ntot
					X(:,i) = X(:,i-1).*x;
				end
				% compute integral
				dp = const + pi_erf_c .* ev_pi_erf - (X*exp_c) .* ev_exp;
            else
                error('HF: k < -1 is not implemented')
            end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function dV = grad_vandermonde(HF, x, m, k, norm)
		% Generate matrix with the k-th derivative of all m-order
		% Hermite physicist functions evaluated at N points in x
		%
		% Inputs:  x - (N x 1) input samples
		% 		   m - order of Hermite functions
		%  		   k - k-th derivative (k=0 just evaluates poly)
		% 		   norm - True/False bool for normalization
		% Outputs: dV (N x m+1) matrix of polynomial evaluations
			dV = zeros(size(x,1), m+1);
			for i=0:m
				dV(:,i+1) = HF.grad_x(x, i, k, norm);
			end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods

end %endClass