classdef RickerWavelet < Wavelet

	properties
	    sigma = 4;
        %shift = 0.5;
	end

	methods
		function RW = RickerWavelet(varargin)
			% Define RW object
			p = ImprovedInputParser;
			parse(p, varargin{:});
			RW = passMatchedArgsToProperties(p, RW);
		end
		function psi = mother(RW, x)
            %x = x - RW.shift;
		    const = 2/(sqrt(3*RW.sigma)*pi^(1/4));
		    psi = const * (1 - (x/RW.sigma).^2) .* exp(-x.^2/(2*RW.sigma^2));
		end
		function dxpsi = grad_x_mother(RW, x)
            %x = x - RW.shift;
		    const  = 2/(sqrt(3*RW.sigma)*pi^(1/4));
		    dxpsi1 = (-2*x/RW.sigma.^2) .* exp(-x.^2/(2*RW.sigma^2));
		    dxpsi2 = (1 - (x/RW.sigma).^2) .* exp(-x.^2/(2*RW.sigma^2)) .* (-x/RW.sigma^2);
		    dxpsi  = const*(dxpsi1 + dxpsi2);
		end
	end
end

