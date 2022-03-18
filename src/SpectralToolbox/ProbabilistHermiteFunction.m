classdef ProbabilistHermiteFunction

	% ProbabilistHermiteFunction defines the probabilist Hermite functions
	%   \psi_en(x) = H_{en}(x)*exp(-x^2/4)
	%
	% Methods include: evaluate, grad_x, grad_vandermonde
	% 
	% Author: Ricardo Baptista
	% Date: June 2018 

	properties

		Hf  % PhysicistHermiteFunction
        He  % HermiteProbabilistPoly

	end

	methods
		function HF = ProbabilistHermiteFunction(varargin)

			% Define HF object
			p = ImprovedInputParser;
			parse(p, varargin{:});
			HF = passMatchedArgsToProperties(p, HF);

			% Set Hermite function and polynomial properties
			HF.Hf = PhysicistHermiteFunction();
			HF.He = HermiteProbabilistPoly();

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function p = evaluate(HF, x, m, norm)
		% evaluate the m-th order Hermite probabilist function using
		% \psi_{em}(x) = 1/sqrt(2^{m})*\psi_m(x/\sqrt(2))
            p = 1/sqrt(2^m)*HF.Hf.evaluate(x/sqrt(2), m, false);
            if norm == true
                p = p / HF.He.normalization_const(m);
            end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function dp = grad_x(HF, x, m, k, norm)
		% Evaluate the k-th derivative of the m-th order Hermite functions
        % Note: this also holds for k == -1
			dp = 1/sqrt(2^m)*HF.Hf.grad_x(x/sqrt(2), m, k, false);
            dp = dp*(1/sqrt(2))^(k);
			if norm == true
				dp = dp / HF.He.normalization_const(m);
			end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function dV = grad_vandermonde(HF, x, m, k, norm)
		% Generate matrix with the k-th derivative of all m-order
		% Hermite probabilist functions evaluated at N points in x
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