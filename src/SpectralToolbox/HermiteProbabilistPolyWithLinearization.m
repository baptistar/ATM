classdef HermiteProbabilistPolyWithLinearization

	% HermiteProbabilistPolyWithLinearization defines probabilist 
    % Hermite polynomials that are linearized outside of an connected
    % interval defined by bounds.
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
		function HPL = HermiteProbabilistPolyWithLinearization(bounds, varargin)

			% Define HP object
			p = ImprovedInputParser;
			parse(p, varargin{:});
			HPL = passMatchedArgsToProperties(p, HPL);

			% Set Physicist polynomial property
			HPL.HP = HermiteProbabilistPoly();
            % set and check bounds
            if (nargin < 1)
                bounds = [-3,3].';
            end
            HPL.bounds = bounds;

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------ 
        function HPL = set.bounds(HPL, bounds)
            if length(bounds) ~= 2 || bounds(1) >= bounds(2)
                error('Bounds do not define connected interval') 
            else
                HPL.bounds = bounds;
            end
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
            % determine inner and outer domain
            inner_pts = (x >= HPL.bounds(1) & x <= HPL.bounds(2));
            outer_pts_left  = x < HPL.bounds(1);
            outer_pts_right = x > HPL.bounds(2);
            % evaluate polynomials on inner domain
            p = zeros(length(x),1);
            p(inner_pts) = HPL.HP.evaluate(x(inner_pts), m, norm);
            % evaluate linear functions on outer domain
            bd_value_left       = HPL.HP.evaluate(HPL.bounds(1), m, norm);
            bd_value_right      = HPL.HP.evaluate(HPL.bounds(2), m, norm);
            bd_derivative_left  = HPL.HP.grad_x(HPL.bounds(1), m, 1, norm);
            bd_derivative_right = HPL.HP.grad_x(HPL.bounds(2), m, 1, norm);
            p(outer_pts_left) = bd_derivative_left*(x(outer_pts_left) - HPL.bounds(1)) + bd_value_left;
            p(outer_pts_right) = bd_derivative_right*(x(outer_pts_right) - HPL.bounds(2)) + bd_value_right;
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
                % determine inner and outer domain
                inner_pts = (x >= HPL.bounds(1) & x <= HPL.bounds(2));
                outer_pts_left  = x < HPL.bounds(1);
                outer_pts_right = x > HPL.bounds(2);
                % evaluate polynomials on inner domain
                dp = zeros(length(x),1);
                dp(inner_pts) = HPL.HP.grad_x(x(inner_pts), m, k, norm);
                % evaluate linear functions on outer domain
                % the derivatives of the linear part are zero for k>1
                if k == 1
                    bd_derivative_left  = HPL.HP.grad_x(HPL.bounds(1), m, 1, norm);
                    bd_derivative_right = HPL.HP.grad_x(HPL.bounds(2), m, 1, norm);
                    dp(outer_pts_left)  = bd_derivative_left;
                    dp(outer_pts_right) = bd_derivative_right;
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