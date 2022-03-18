function SV = stoc_volatility_model(N,d)

    % sample mu, phi
    mu = randn(N,1);
    phis = randn(N,1)+3;
    phi = 2*exp(phis)./(1+exp(phis)) - 1;
    
    % set sigma
    sigma = 1;
    
    % sample Z0
    Z = sqrt(1./(1-phi.^2)) .* randn(N,1) + mu;
    
    % sample auto-regressively
    for i=1:d-3
        Zi = mu + phi .* (Z(:,end) - mu) + sigma*randn(N,1);
        Z = [Z, Zi];
    end
    
    % append parameters
    SV = [mu, phi, Z];

end