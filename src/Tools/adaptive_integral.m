function [I_l, cache_xi, cache_zi, cache_wi] = adaptive_integral(f, ...
            int_pts, cache_xi, cache_zi, cache_wi, tol, pts_per_level, max_level)
% adaptive_integral: approximate \int_{a}^{b} f(x) using the points 
% specified in the int_pts function. The number of points is increased 
% adaptively in each level until the total error is no longer changing.
%
% Inputs: f     - function to be integrated
%         [x,w] = int_pts(N)
%         cache - cells with points x,w that have been evaluated
% 
% Author: Ricardo Baptista, Olivier Zahm
% Date:   August 2020

% set defaults
if (nargin < 8)
    max_level = 12;
end
if (nargin < 7)
    pts_per_level = 6;
end
if (nargin < 6)
    rel_tol = 1e-3;
    abs_tol = 1e-3;
else
    rel_tol = tol(1);
    abs_tol = tol(2);
end
if (nargin == 3) || (nargin == 4)
    error('Specify all cached (xi, wi, zi) points')
end
if (nargin < 3) || isempty(cache_xi) || isempty(cache_wi) || isempty(cache_zi)
    cache_xi = cell(max_level,1);
    cache_wi = cell(max_level,1);
    cache_zi = cell(max_level,1);
end

%% Estimate integral using pre-computed points

% estimate initial integral on the first level
if isempty(cache_xi{1}) || isempty(cache_wi{1})
    n_pts = pts_per_level;
    [cache_xi{1}, cache_wi{1}, cache_zi{1}] = int_pts(n_pts);
end
I_l = estimate_integral(f, cache_xi{1}, cache_wi{1});
I_lm1 = I_l;

% define initial error and counter
err = Inf*ones(size(I_l));
level = 2;

while(any(any(err > max(rel_tol*abs(I_l), abs_tol))) && level <= max_level)

    % collect points for next level
    if isempty(cache_xi{level}) || isempty(cache_wi{level})
        n_pts = pts_per_level*2^(level-1);
        [cache_xi{level}, cache_wi{level}, cache_zi{level}] = int_pts(n_pts);
    end
    
    % estimate integral with points
    I_l = estimate_integral(f, cache_xi{level}, cache_wi{level});
    
    % update maximum error and I_lm1
    err = abs(I_l - I_lm1);
    I_lm1 = I_l;
    
    % update level counter
    level = level + 1;
    %fprintf('Level %d, max err %f\n', level, max(max(err)))
end
%fprintf('Max = %d, ',level-1);
%if level >= max_level
%    warning('Refinement is halted, max level is reached\n');
%    return
%end

end

function I = estimate_integral(f, x, w)
    % Compute integral \int f(x) \approx \sum_{i=1}^{n_pts} f(x{i})w{i}
    % Inputs: f - function to be integrated that acts elementwise
    %         x - cell with multi-dimensional arrays of size (N x d)
    %         w - cell with arrays of size (N x 1)

    % evaluate first term because the dimensions are unknown
    I = f(x{1}) .* w{1};
    % add terms for remaining points
    n_pts = numel(x);
    for i=2:n_pts
        I = I + f(x{i}) .* w{i};
    end

end

% -- END OF FILE --
