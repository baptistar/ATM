classdef Wavelet

	% Wavelet class 
	%
	% Methods include: evaluate, grad_x, grad_vandermonde
	%
	% Author: Ricardo Baptista
	% Date:   April 2021

	properties
        f_dom
        psi_dom
	end %endProperties

	methods
		function HP = Wavelet(varargin)
			
			% Define HP object
			p = ImprovedInputParser;
            addOptional(p, 'f_dom', [0,1]);
            addOptional(p, 'psi_dom', [0,1]);
			parse(p, varargin{:});
			HP = passMatchedArgsToProperties(p, HP);

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------ 
        function z = f_translate(W, x)
            z = (x - W.f_dom(1))/range(W.f_dom);
        end
		%------------------------------------------------------------------
		%------------------------------------------------------------------ 
        function z = psi_translate(W, x)
            z = range(W.psi_dom)*x + W.psi_dom(1);
        end
		%------------------------------------------------------------------
		%------------------------------------------------------------------ 
		function p = evaluate(W, x, j, m)
		% Evaluate the m-th order Hermite physicist polynomial
		% at N=size(x,1) points using the recursion relation 
		% H_0(x) = 1 and H_{n}(x) = 2xH_{n-1}(x) - H_{n-1}'(x)
		% where H_{n-1}'(x) = 2(n-1)H_{n-2}(x)
		%
		% Inputs:  x - (N x 1) inputs
		% 		   j - scale of dilation
		% 		   m - scale of translation
		% Outputs: p - (N x 1) evaluations
            z = W.f_translate(x);
            z = W.psi_translate(2^j * z - m);
            %z = (2^j * z - m);
            p = 2^(j/2) * W.mother(z);
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function dp = grad_x(W, x, j, m, k)
		% Evaluate the k-th derivative of the m-th order Hermite 
		% physicist polynomial at m=size(x,1) points using the
		% relation H_{n}^(k)(x) = 2^{k} n!/(n-k)! H_{n-k}(x)
		%
		% Inputs:  x - (N x 1) input samples
		% 		   m - order of Hermite functions
		% 		   norm - True/False bool for normalization
		% Outputs: dp - (N x 1) poly evaluations
			if k==0
				dp = W.evaluate(x, j, m);
			elseif k==1
                z1 = W.f_translate(x);
                %z2 = (2^j * z1 - m);
                z2 = W.psi_translate(2^j * z1 - m);
				%dp = 2^(j/2) * W.grad_x_mother(z2) * 2^j / range(W.f_dom); 
                dp = 2^(j/2) * W.grad_x_mother(z2) * range(W.psi_dom) * 2^j / range(W.f_dom);
			else
				error('Not implemented!')
			end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------ 
        function dV = grad_vandermonde(W, x, j, m, k)
		% Generate matrix with the k-th derivative of all wavelet functions
		% 
		%
		% Inputs:  x - (N x 1) input samples
		% 		   j - scale of dilation
		% 		   m - maximum scale of translation
		%  		   k - k-th derivative (k=0 just evaluates wavelet)
		% Outputs: dV (N x m+1) matrix of evaluations
			assert(length(j) == length(m))
			dV = zeros(size(x,1), length(m));
			for i=1:length(m)
				dV(:,i) = W.grad_x(x, j(i), m(i), k);
			end
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods

end %endClass
