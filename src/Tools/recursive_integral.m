function I = recursive_integral(fun, a, b, tol, num_IntPts, max_level)

    % set defaults
    if (nargin < 6)
        num_IntPts = 4;
    end
    if (nargin < 5)
        max_level = 12;
    end
    if (nargin < 4)
        tol = [1e-3,1e-3];
    else
        assert(length(tol) == 2)
    end
       
    % call adaptive quadrature method
    I = adaptiveCC(fun, a, b, num_IntPts, tol, max_level, 0,0);%1, 0);

end

function I = adaptiveCC(fun, a, b, num_IntPts, tol, max_level, level, I1)
% adaptiveCC: adaptive Clenshaw-Curtis integration rule using recursion%
% Inputs: fun
%         (a,b) - limits of integration
%         num_IntPts - number of integration points on which to evaluate
%         tol - tolerance
% 
% Author: Ricardo Baptista, Olivier Zahm
% Date:   December 2019
% 

% update current refinement level
level = level+1;

% evaluate I1 on first call of function
if level == 1
    I1 = intCC(fun, a, b, num_IntPts);
end

% evluate CC integral on each sub-interval
c=a+0.5*(b-a);
I2_left  = intCC(fun, a, c, num_IntPts);
I2_right = intCC(fun, c, b, num_IntPts);

% compute total integral
I2 = I2_left + I2_right;

% Issue a warning if recursion limit is exceeded
if (level > max_level)
    warning('In adaptiveCC: level = %d exceeds the specified limit\n', level);
    % Stop the recursion if recursion limit is exceeded by 2
    if level > (max_level+2)
        fprintf('Refinement is halted, recursion limit is exceeded by 2\n');
        I=I2;
        return;
    end
end

% If the difference between levels is lower than tolerance, stop the
% recursion and accept the fine level result, else subdivide further
% Note: condition must be satisfied for all outputs of function
if all(abs(I2-I1) < max(tol(1)*abs(I1),tol(2)))
    I = I2;
    %fprintf('min level = %d\n', level);
    return;
else
    % compute I_left
    I_left = adaptiveCC(fun, a, c, num_IntPts, [tol(1);tol(2)/2], max_level, level, I2_left);
    % compute I_right
    I_right = adaptiveCC(fun, c, b, num_IntPts, [tol(1);tol(2)/2], max_level, level, I2_right);
    % compute sum
    I = I_left + I_right;
end

end

function I = intCC(fun, a, b, N)
    % Integrate a univariate function using the Clenshaw-Curtis
    % integration rule.
    %
    % Date: November 2019
    
    % extract nodes and weights
    [x, w] = clenshaw_curtis(N);
    
    % initialize I
    [x1, w1] = rescale_pts(a, b, x(1), w(1));
    I = w1.*fun(x1);
    
    % sum contribution from each integration node
    n_pts = numel(x);
    for i=2:n_pts
        [xi, wi] = rescale_pts(a, b, x(i), w(i));
        I = I + wi.*fun(xi);
    end

end

% -- END OF FILE --
