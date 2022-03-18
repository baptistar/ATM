function [L, dcL] = regression_objective(S, Sref, XW, precomp, coeff_idx)
    % define default inputs
    if (nargin < 5)
        coeff_idx = 1:S.n_coeff;
    end
    if (nargin < 4)
        precomp = PPprecomp();
    end

    W=XW.W;
    X=XW.X;
    
    
    DSk=(Sref-S.evaluate(X,precomp));
    L=0.5*sum(W.*(DSk.^2),1);
    
    if (nargout>1)
        dcL=-sum(W.*DSk.*S.grad_coeff(X,coeff_idx,precomp),1);
    end
    
    
end
