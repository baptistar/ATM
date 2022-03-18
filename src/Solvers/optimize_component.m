function [S, exit_flag] = optimize_component(S, ref, a0, XW, precomp, alpha, coeff_idx)
    
    X=XW.X;
    
    % set default inputs
    if (nargin < 7)
        coeff_idx = 1:S.n_coeff;
    end
    if (nargin < 6)
        alpha = 0;
    end
    if (nargin < 5) || isempty(precomp)
        precomp = PPprecomp();
        precomp = precomp.evaluate(S.f, X);
    end
    % check coefficients are a column vector
    if size(a0,2) == length(a0)
        a0 = a0.';
    end

    % find normalization using QR factorization of eval_basis
    Psi_norm = sqrt(sum(precomp.eval_basis.^2,1));
    Psi_norm = Psi_norm(coeff_idx);
    Dinv = diag(1./Psi_norm);
    Psi = precomp.eval_basis(:,coeff_idx) * Dinv;
    [~,R] = qr(Psi,0);
    
    % check condition number of basis evaluations: cond(Psi) = cond(R)
    if cond(R) > 1e9 || size(R,1) < size(R,2)
        exit_flag = -1;
        S = S.set_coeff(nan*ones(length(a0),1));
        disp('Warning: Maximum condition number reached')
        return;
    end

    % define inner-preconditioning matrix: A = Dinv * inv(R)
    % for new objective over ct: f(c) = f(A*ct) = g(ct)
    A = Dinv / R;

    % define regularized objective
    obj = @(a) objective(S, ref, a, A, XW, precomp, alpha, coeff_idx);

    % re-scale initial condition: a0p = inv(A) * a0;
    a0 = a0(coeff_idx);
    a0p = R * (Psi_norm.' .* a0);
    
    % set options and run optimization
    options = optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient', true, 'Display', 'off');
    [a_opt, ~, exit_flag] = fminunc(obj, a0p, options);
        
    % check exit_flag
    if exit_flag <= -1
        disp('Warning: Optimization failed')
    end
    
    % rescale coefficients by input-preconditioner
    a_opt = (A * a_opt);
    a_new = S.coeff; a_new(coeff_idx) = a_opt;
    
    % set coefficients in S
    S = S.set_coeff(a_new);

end %endFunction

% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

function [L, dcL, d2cL] = objective(S, ref, a, A, XW, precomp, alpha, coeff_idx)

    % extract current coeffs
    a_old = S.coeff;
    % pre-condition inputs using matrix A
    ap = (A * a);
    % assign a_new
    a_new = a_old; a_new(coeff_idx) = ap;
    % set coefficients in map component
    S = S.set_coeff(a_new);

    % call negative_log_likelihood
    if (nargout == 1)
        L = negative_log_likelihood(S, ref, XW, precomp, coeff_idx);
    elseif (nargout == 2)
        [L, dcL] = negative_log_likelihood(S, ref, XW, precomp, coeff_idx);
    else
        [L, dcL, d2cL] = negative_log_likelihood(S, ref, XW, precomp, coeff_idx);
    end

    % apply pre-conditioner to outputs (gradients and Hessian)
    if (nargout > 1)
        dcL = (dcL * A);
    end
    if (nargout > 2)
        d2cL = (A.' * d2cL * A);
    end
    
    % add L2 regularization term
    D = alpha*(A.'*A);
    %D = alpha*eye(length(a));
    L = L + a'*D*a;
    if (nargout > 1)
        dcL = dcL + (2*D*a)';
    end
    if (nargout > 2)
        d2cL = d2cL + 2*D;
    end

end

% -- END OF FILE --