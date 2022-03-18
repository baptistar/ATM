function [L, dcL, d2cL] = negative_log_likelihood(S, ref, XW, precomp, coeff_idx)
    % define default inputs
    if (nargin < 5)
        coeff_idx = 1:S.n_coeff;
    end
    if (nargin < 4)
        precomp = PPprecomp();
    end

    % define delta (regularization term) - add small diagonal term
    delta = 1e-9;
    
    X=XW.X;
    W=XW.W;
    
    % evaluate objective
    Sk = S.evaluate(X, precomp) + delta*X(:,end);
    dxdSk = S.grad_xd(X, precomp) + delta;
    % evaluate log_pi(x)
    L = ref.log_pdf(Sk) + log(dxdSk);
    L = -1 * sum(W.*L,1);
    
    % evaluate gradient
    if (nargout > 1)
        % evaluate \nabla_c S, \nabla_c_xd S
        dcSk = S.grad_coeff(X, coeff_idx, precomp);
        dcdxSk = S.grad_coeff_grad_xd(X, coeff_idx, precomp);
        % evaluate \nabla_c log_pi(x)
        dcL = ref.grad_x_log_pdf(Sk) .* dcSk + dcdxSk ./ dxdSk;
        dcL = -1 * sum(W.*dcL,1);
    end

    % evaluate Hessian
    if (nargout > 2)
        % evaluate \nabla^2_c S, \nabla^2c_xd S
        d2cSk   = S.hess_coeff(X, coeff_idx, precomp);
        d2cdxSk = S.hess_coeff_grad_xd(X, coeff_idx, precomp);
        % evaluate  \nabla^2_c log_pi(x)
        d2cL = dcSk' * diag(ref.hess_x_log_pdf(Sk)) * dcSk / size(X,1) + ...
               squeeze(mean(ref.grad_x_log_pdf(Sk) .* d2cSk , 1)) + ...
               dcdxSk' * diag(-1./(dxdSk).^2) * dcdxSk / size(X,1) + ...
               squeeze(mean(1./dxdSk .* d2cdxSk, 1));
        d2cL = -1 * d2cL;
    end
    
end
