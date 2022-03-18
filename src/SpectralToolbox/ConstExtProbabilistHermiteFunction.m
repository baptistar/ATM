classdef ConstExtProbabilistHermiteFunction

	% ConstExtProbabilistHermiteFunction class defines the 
	% probabilist Hermite functions given by 
	% \psi_n(x) = H_{n-1}(x)*exp(-x^2/2) for n > 1 
	% extended by the constant function \psi_0(x) = 1
	%
	% Methods include: evaluate, grad_x, GradVandermonde
	% 
	% Author: Ricardo Baptista
	% Date: June 2018 

	properties

		Hf  % ProbabilistHermiteFunction

	end

	methods
		function CHF = ConstExtProbabilistHermiteFunction(varargin)

			% Define CHF object
			p = ImprovedInputParser;
			parse(p, varargin{:});
			CHF = passMatchedArgsToProperties(p, CHF);

			% Set Hermite polynomial properties
			CHF.Hf = ProbabilistHermiteFunction();

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function p = evaluate(CHF, x, m, norm)
		% evaluate the m-th order Hermite probabilist functions by
		% by adding that \psi_0 = 1
		%
		% Inputs:  x - (N x 1) input samples
		% 		   m - order of Hermite functions
		% 		   norm - True/False bool for normalization
		% Outputs: p - (N x 1) evaluations
			if m < 0
				error('CHF: poly order must be non-negative')
			elseif m > 0
				p = CHF.Hf.evaluate(x, m-1, norm);
			else
				p = ones(size(x));
			end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function dp = grad_x(CHF, x, m, k, norm)
		% evaluate the k-th derivative of the m-th order Hermite 
		% probabilist functions extended by the constant function
		%
		% Inputs:  x - (N x 1) input samples
		% 		   m - order of Hermite functions
		%  		   k - k-th derivative (k=0 just evaluates poly)
		% 		   norm - True/False bool for normalization
		% Outputs: dp - (N x 1) evaluations
			if (m < 0) || (k < 0)
				error('CHF: order and derivative must be non-negative')
			elseif m > 0
				dp = CHF.Hf.grad_x(x, m-1, k, norm);
			elseif m == 0 && k == 0   % constant evaluation
				dp = ones(size(x));
			else
				dp = zeros(size(x));
			end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function dV = grad_vandermonde(CHF, x, m, k, norm)
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
				dV(:,i+1) = CHF.grad_x(x, i, k, norm);
			end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods

end %endClass