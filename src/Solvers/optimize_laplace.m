function [mu,Sigma] = optimize_laplace(pi,x0)

    %component regression
    if (nargin < 2)
        x0 = zeros(1,pi.d);
    end

    obj = @(x) laplace_objective(pi,x);
    
    options = optimoptions('fminunc','SpecifyObjectiveGradient', true, 'Display', 'off');

    [x_opt,~,~,~,~,A] = fminunc(obj, x0, options);
    
   
    mu=x_opt;
    
    Sigma=sqrtm(inv(A)); % To change!!!

end


function [f,g] = laplace_objective(pi,x)
%objective function for regression


f=-pi.log_pdf(x);

g=-pi.grad_x_log_pdf(x);
g=g(1,:);


end

