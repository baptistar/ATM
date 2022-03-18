function [x,w] = rescale_pts(a, b, x, w)
	% Rescale the quadratures nodes x and weights w for
	% integrating a function with respect to [a,b]
	%
	% Date: November 2019

    % check if a and b have the same size
    if any(size(a) ~= size(b))
        error('a and b must be have the same size in intCC')
    end
    
	% rescale nodes
	x = 0.5*(b + a) + 0.5*(b - a)*x;
	% rescale weights
	w = 0.5*(b - a)*w;

end 

