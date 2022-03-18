function [S1,exit_flag] = optimize_regression2(S1,S2,a0,XW,precomp)
%component regression
    X=XW.X;
    if (nargin < 5) || isempty(precomp)
        % assign precomp
        precomp = cell(S1.d,1);
        for k=1:S1.d
           precomp{k} = PPprecomp();
           precomp{k} = precomp{k}.evaluate(S1.S{k}.f, X(:,1:k));
        end
    end

    obj = @(a) regression_objective(S1,S2,XW,a,precomp);
        
    options = optimoptions('fminunc','SpecifyObjectiveGradient', true, 'Display', 'off');
        
    [a_opt, ~, exit_flag] = fminunc(obj, a0, options);
    
    
    S1=S1.set_coeff(a_opt);

end


function [f,g] = regression_objective(S1,S2,XW,a,precomp)
%objective function for regression

    W=XW.W;
    X=XW.X;

    S1=S1.set_coeff(a);
    
    DSk=(S2.evaluate(X)-S1.evaluate(X,[],precomp));
    f=sum(W.*(DSk.^2),1);
    
    g=-sum(W.*DSk.*S1.grad_coeff(X,precomp),1);
    

end

