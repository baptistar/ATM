function [X,fX,gX] = hybridRootFindingSolver(f, a, b, delta_x, delta_f, max_iter)
% Combine bisection and Newton method for guaranteed convergence. Function
% can minimize for x \in R^d as long as a,b \in R^d
% Inputs: f - Function f(x) and its gradient f'(x)
%         a - left point of interval
%         interval b - right point of interval
%         delta tolerance

% set default inputs
if (nargin < 7)
    max_iter = 10000;
end
if (nargin < 6)
    delta_f = 1e-6;
end
if (nargin < 5)
    delta_x = 1e-6;
end

% check inputs
N = size(a,1);
assert(size(b,1) == N)

% define initial vector and evaluate function/gradient
dX = (b-a);
dXold = dX;
X = (a + b)/2;
[fX, gX] = f(X);

for j=1:max_iter
    for i=1:N
        % Bisect if Newton out of range, or not decreasing fast enough.
        if ((X(i) - b(i))*gX(i) - fX(i))*((X(i) - a(i))*gX(i) - fX(i)) > 0.0 || ...
            (abs(2.0*fX(i)) > abs(dXold(i) * gX(i)) && fX(i) > delta_f)
            dXold(i) = dX(i);
            dX(i) = 0.5*(b(i)-a(i));
            X(i)  = a(i) + dX(i);
        % Newton step is acceptable
        else
            dXold(i) = dX(i);
            dX(i) = fX(i)/gX(i);
            X(i) = X(i) - dX(i);
        end
    end

    % evaluate function and its gradient at new iterate
    [fX, gX] = f(X);

    % check convergence criterion
    if norm(fX) < delta_f || norm(dX) < delta_x
        break
    end

    % update the bracket on the root
    for i=1:N
        if (fX(i) < 0.0)
            a(i) = X(i);
        else
            b(i) = X(i);
        end
    end

end

% check that max_iter has not been reached
if j==max_iter
    warning('Hit maximum number of iterations: check solver')
end

end