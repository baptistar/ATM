function [S, exit_flag] = optimize_map(S, ref, a0, XW, precomp, alpha)
    % set default inputs
    
    X=XW.X;
    if (nargin < 7)
        alpha = 0;
    end
    if (nargin < 6) || isempty(precomp)
        % assign precomp
        precomp = cell(S.d,1);
        for k=1:S.d
           precomp{k} = PPprecomp();
           precomp{k} = precomp{k}.evaluate(S.S{k}.f, X(:,1:k));
        end
    end
    % check coefficients are a column vector
    if size(a0,2) == length(a0)
        a0 = a0.';
    end

    % define regularized objective
    obj = @(a) objective(S, ref, a, XW, precomp, alpha);
    
    
    % set options and run optimization
    options = optimoptions('fminunc','SpecifyObjectiveGradient', true, 'Display', 'off');
    [a_opt, ~, exit_flag] = fminunc(obj, a0, options);
        
    % check exit_flag
    if exit_flag <= -1
        disp('Warning: Optimization failed')
    end
    
    % set coefficients in S
    S = S.set_coeff(a_opt);

end

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

function [L, dcL] = objective(S, ref, a, XW, precomp, alpha)
    
    X=XW.X;
    W=XW.W;
    
    % set coefficients in all map components
    S = S.set_coeff(a);
     
    % define delta (regularization term) - add small diagonal term
    delta = 1e-9;

    % evaluate objective
    Sx = S.evaluate(X, [], precomp) + delta*X;
    dxdS = S.grad_xd(X, [], precomp) + delta;

    % evaluate log_pi(x)
    L = ref.log_pdf(Sx) + sum(log(dxdS),2);
   
    %L=max(L,min(L(~isinf(L))));
    L = -1 * sum(W.*L,1);
    
    % evaluate gradient
    if (nargout > 1)
        % evaluate \nabla_c S, \nabla_c_xd S
        dcS = S.grad_coeff(X, precomp);
        dcdxdS = S.grad_coeff_grad_xd(X, precomp);
        % evaluate \nabla_c log_pi(x)
        dcL = squeeze(sum(ref.grad_x_log_pdf(Sx) .* dcS + dcdxdS ./ dxdS, 2));
        dcL = -1 * sum(W.*dcL,1);
    end
        
    % add L2 regularization term
    D = alpha*eye(length(a));
    L = L + a'*D*a;
    if (nargout > 1)
        dcL = dcL + (2*D*a)';
    end
    
end %endFunction
