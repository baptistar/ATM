classdef HermiteProbabilistPolyWithLinearization

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
	% Date:   January 2021

	properties
        
		HP      % Hermite polynomial object
        bounds  % interval for polynomials
        
	end

	methods
		function HPL = HermiteProbabilistPolyWithLinearization(varargin)

			% Define HP object
			p = ImprovedInputParser;
			parse(p, varargin{:});
			HPL = passMatchedArgsToProperties(p, HPL);

			% Set Physicist polynomial property
			HPL.HP = HermiteProbabilistPoly();
            HPL.bounds = [-3,3].';

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------ 
		function p = evaluate(HPL, x, m, norm)
		% Evaluate the m-th order Hermite probabilists' polynomial
		% using the relation He_n(x) = 2^(-n/2)*H_n(x/sqrt(2))
		% accounting for the correct scaling factors
		%
		% Inputs:  x - (N x 1) inputs
		% 		   m - order of Hermite functions
		% 		   norm - True/False bool for normalization
		% Outputs: p - (N x 1) poly evaluations
            inner_pts = (x >= HPL.bounds(1) & x <= HPL.bounds(2));
            outer_pts_left  = x < HPL.bounds(1);
            outer_pts_right = x > HPL.bounds(2);
            p = zeros(length(x),1);
            p(inner_pts) = HPL.HP.evaluate(x(inner_pts), m, norm);
            bd_values = HPL.HP.evaluate(HPL.bounds, m, norm);
            bd_derivatives = HPL.HP.grad_x(HPL.bounds, m, 1, norm);
            p(outer_pts_left) = bd_derivatives(1)*(x(outer_pts_left) - HPL.bounds(1)) + bd_values(1);
            p(outer_pts_right) = bd_derivatives(2)*(x(outer_pts_right) - HPL.bounds(2)) + bd_values(2);
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------ 
		function dp = grad_x(HPL, x, m, k, norm)
		% Evaluate the k-th derivative of the m-th order Hermite 
		% probabilists' polynomial at m=size(x,1) points using the
		% relation He_n(x) = 2^(-n/2)*H_n(x/sqrt(2)) and accounting
		% for the correct scaling factors
		%
		% Inputs:  x - (N x 1) input samples
		% 		   m - order of Hermite functions
		% 		   norm - True/False bool for normalization
		% Outputs: dp - (N x 1) poly evaluations
            if k == 0
                dp = HPL.evaluate(x, m, norm);
            else
                inner_pts = (x >= HPL.bounds(1) & x <= HPL.bounds(2));
                outer_pts_left  = x < HPL.bounds(1);
                outer_pts_right = x > HPL.bounds(2);
                dp = zeros(length(x),1);
                dp(inner_pts) = HPL.HP.grad_x(x(inner_pts), m, k, norm);
                % the derivatives of the linear part are non-zero for k=1
                if k == 1
                    bd_derivatives = HPL.HP.grad_x(HPL.bounds, m, 1, norm);
                    dp(outer_pts_left) = bd_derivatives(1);
                    dp(outer_pts_right) = bd_derivatives(2);
                end
            end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
        function dV = grad_vandermonde(HPL, x, m, k, norm)
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
				dV(:,i+1) = HPL.grad_x(x, i, k, norm);
			end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods

end %endClass