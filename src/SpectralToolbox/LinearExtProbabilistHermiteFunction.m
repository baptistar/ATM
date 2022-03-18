classdef LinearExtProbabilistHermiteFunction

	% LinearExtProbabilistHermiteFunction class defines the 
	% probabilist Hermite functions given by 
	% \psi_n(x) = H_{n-1}(x)*exp(-x^2/2) for n > 1 
	% extended by the constant function \psi_0(x) = 1
    % and the linear function \psi_1(x) = x
	%
	% Methods include: evaluate, grad_x, grad_vandermonde
	% 
	% Author: Ricardo Baptista
	% Date: January 2020 

	properties

		Hf  % ProbabilistHermiteFunction

	end

	methods
		function LHF = LinearExtProbabilistHermiteFunction(varargin)

			% Define CHF object
			p = ImprovedInputParser;
			parse(p, varargin{:});
			LHF = passMatchedArgsToProperties(p, LHF);

			% Set Hermite polynomial properties
			LHF.Hf = ProbabilistHermiteFunction();

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function p = evaluate(LHF, x, m, norm)
		% evaluate the m-th order Hermite probabilist functions by
		% by adding that \psi_0 = 1 and \psi_1 = x
		%
		% Inputs:  x - (N x 1) input samples
		% 		   m - order of Hermite functions
		% 		   norm - True/False bool for normalization
		% Outputs: p - (N x 1) evaluations
			if m < 0
				error('LHF: poly order must be non-negative')
            elseif m == 0
                p = ones(size(x));
            elseif m == 1
                p = x;
            else
                p = LHF.Hf.evaluate(x, m-2, norm);
			end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function dp = grad_x(LHF, x, m, k, norm)
		% evaluate the k-th derivative of the m-th order Hermite 
		% probabilist functions extended by the constant & linear function
		%
		% Inputs:  x - (N x 1) input samples
		% 		   m - order of Hermite functions
		%  		   k - k-th derivative (k=0 just evaluates poly)
		% 		   norm - True/False bool for normalization
		% Outputs: dp - (N x 1) evaluations
            if (m < 0) || (k < 0)
                error('LHF: order and derivative must be non-negative')    
            elseif (k == 0)
                dp = LHF.evaluate(x, m, norm);
            elseif (m == 1 && k == 1)
                dp = ones(size(x));
            elseif (m == 0 && k > 0) || (m == 1 && k > 1)
                dp = zeros(size(x));
            else
                dp = LHF.Hf.grad_x(x, m-2, k, norm);
            end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function dV = grad_vandermonde(LHF, x, m, k, norm)
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
				dV(:,i+1) = LHF.grad_x(x, i, k, norm);
			end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods

end %endClass
