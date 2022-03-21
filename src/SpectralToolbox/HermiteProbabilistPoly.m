classdef HermiteProbabilistPoly

	% HermiteProbabilistPoly defines probabilist Hermite polynomials
	% using the rescaling of physicist Hermite polynomials
	%
	% Un-normalized:
	%  He_0(x) = 1
	%  He_1(x) = x
	%  He_2(x) = x^2 - 1
	%
	% Normalized:
	%  \tilde{He}_n(x) = He_n(x)/\sqrt(\sqrt(2*pi)*n!)
	%
	% Methods include: evaluate, grad_x, grad_vandermonde
	% 
	% Author: Ricardo Baptista
	% Date:   September 2019

	properties

		HPhy  % Physicist Hermite polynomial object

	end

	methods
		function HP = HermiteProbabilistPoly(varargin)

			% Define HP object
			p = ImprovedInputParser;
			parse(p, varargin{:});
			HP = passMatchedArgsToProperties(p, HP);

			% Set Physicist polynomial property
			HP.HPhy = HermitePhysicistPoly();

		end %endFunction
        %------------------------------------------------------------------
		%------------------------------------------------------------------
        function c = normalization_const(~, m)
            % sqrt(2*pi) is already accounted for by sampling from N(0,1)
            % using m! = gamma(m+1)
            c = sqrt(gamma(m+1));
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------ 
		function p = evaluate(HP, x, m, norm)
		% Evaluate the m-th order Hermite probabilists' polynomial
		% using the relation He_n(x) = 2^(-n/2)*H_n(x/sqrt(2))
		% accounting for the correct scaling factors
		%
		% Inputs:  x - (N x 1) inputs
		% 		   m - order of Hermite functions
		% 		   norm - True/False bool for normalization
		% Outputs: p - (N x 1) poly evaluations
			p = 1/sqrt(2^m)*HP.HPhy.evaluate(x/sqrt(2), m, false);
            if norm == true
                p = p / HP.normalization_const(m);
            end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------ 
		function dp = grad_x(HP, x, m, k, norm)
		% Evaluate the k-th derivative of the m-th order Hermite 
		% probabilists' polynomial at m=size(x,1) points using the
		% relation He_n(x) = 2^(-n/2)*H_n(x/sqrt(2)) and accounting
		% for the correct scaling factors
		%
		% Inputs:  x - (N x 1) input samples
		% 		   m - order of Hermite functions
		% 		   norm - True/False bool for normalization
		% Outputs: dp - (N x 1) poly evaluations
			dp = 1/sqrt(2^m)*HP.HPhy.grad_x(x/sqrt(2), m, k, false);
			dp = dp*(1/sqrt(2))^k;
            if norm == true
                dp = dp / HP.normalization_const(m);
            end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function dV = grad_vandermonde(HP, x, m, k, norm)
		% Generate matrix with the k-th derivative of all m-order
		% Hermite Probablist' polynomials evaluated at N points in x
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