classdef stoc_volatility
    % Stochastic volatility model for tracking log-returns of financial 
    % assets: 
    
    properties
        params
        sigma 
        d
    end
    
    methods
        % -----------------------------------------------------------------
        function SV = stoc_volatility(d, sigma, hyper_parameters)
            % set hyper-parameters
            if nargin < 3
                SV.params = [];
            elseif length(hyper_parameters) == 2
                SV.params = hyper_parameters;
            else
                error('Specify two parameters')
            end
            % set sigma
            if nargin < 2
                sigma = 0.25;
            end
            SV.sigma = sigma;
            % set d
            SV.d = d;
        end
        % -----------------------------------------------------------------
        function X = sample(SV,N)
            
            % set or sample hyper-params
            if ~isempty(SV.params)
                mu = SV.params(1)*ones(N,1);
                phi = SV.params(2)*ones(N,1);
            else
                mu = randn(N,1);
                phis = randn(N,1)+3;
                phi = 2*exp(phis)./(1+exp(phis)) - 1;
            end

            % sample Z0
            Z = sqrt(1./(1-phi.^2)) .* randn(N,1) + mu;

            % sample auto-regressively
            for i=1:SV.d-3
                Zi = mu + phi .* (Z(:,end) - mu) + SV.sigma*randn(N,1);
                Z = [Z, Zi];
            end

            % append parameters
            X = [mu, phi, Z];

        end
        % -----------------------------------------------------------------
        function logpi = log_pdf(SV, X)

            % define vector to store density evaluations
            logpi = zeros(size(X,1),0);

            % compute density for hyper-parameters if they are random
            if isempty(SV.params)
                
                % extract variables mu, phi, and states
                mu = X(:,1);
                phi = X(:,2);
                Z = X(:,3:end);

                % compute density for \mu
                logpi_mu = log(normpdf(mu));
                % compute density for \phi
                phi_ref = log((1 + phi)./(1 - phi));
                dphi_ref = 2./(1 - phi.^2);
                logpi_phi = log(normpdf(phi_ref,3,1)) + log(dphi_ref);
                % add pi_mu, pi_phi to density
                logpi = [logpi, logpi_mu, logpi_phi];

            % set hyper-parameters
            else 
                mu = SV.params(1);
                phi = SV.params(2);
                Z = X;
            end

            % determine number of time-steps
            dz = size(Z,2);

            % conditional density for Z_0
            mu_Z0 = mu;
            std_Z0 = sqrt(1./(1-phi.^2));
            logpi_Z0 = log(normpdf(Z(:,1), mu_Z0, std_Z0));
            logpi = [logpi, logpi_Z0];

            % compute auto-regressive conditional densities for Z_i|Z_{1:i-1}
            for i=2:dz
                mean_Zi = mu + phi .* (Z(:,i-1) - mu);
                std_Zi = SV.sigma;
                logpi_Zi = log(normpdf(Z(:,i), mean_Zi, std_Zi));
                logpi = [logpi, logpi_Zi];
            end
            
        end
        % -----------------------------------------------------------------

    end
end
