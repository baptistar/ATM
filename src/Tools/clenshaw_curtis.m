function [x,w] = clenshaw_curtis(N)
	% Compute the nodes x and weights w for integrating a
	% continuous functions from [-1,1] using the Clenshaw-Curtis
	% integration rule with order N
	%
	% Date: November 2019

	if N<1
		error('CC: Order must be greater than 0')
	end

	% initialize w and x
	w = zeros(N, 1);
	x = zeros(N, 1);

	% return results for N == 1 
	if N == 1
		x(1) = 0.0;
		w(1) = 2.0;
		return
	end

	% compute x values
	for i=1:N
		x(i) = cos( (N - i)*pi/(N - 1) );
	end

	n = N;
	x(1) = -1.0;
	if ( mod ( n, 2 ) == 1 )
	x((n+1)/2) = 0.0;
	end
	x(n) = +1.0;

	w(1:n) = 1.0;

	for i = 1 : n

	theta = ( i - 1 ) * pi / ( n - 1 );

	for j = 1 : ( n - 1 ) / 2

	  if ( 2 * j == ( n - 1 ) )
	    b = 1.0;
	  else
	    b = 2.0;
	  end

	  w(i) = w(i) - b * cos ( 2.0 * j * theta ) / ( 4 * j * j - 1 );

	end

	end

	w(1)     =       w(1)     / ( n - 1 );
	w(2:n-1) = 2.0 * w(2:n-1) / ( n - 1 );
	w(n)     =       w(n)     / ( n - 1 );

end 
