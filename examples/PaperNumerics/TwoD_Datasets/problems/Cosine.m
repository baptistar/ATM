classdef Cosine

    % Defines a multivariate distribution with 2 dim that
    % have a cosine mean and uniform additive noise and
    % d-2 dim Gaussian components
    % 
    % Methods: log_pdf, sample
    % Date:    September 2019

    properties
        d
        sigma
        x_lim
        A
        omega
        name
    end

    methods
        function C = Cosine(d, varargin)

            % check dimension
            if (d < 2)
                error('C: dimension must be at least 2')
            end

            % declare C object
            p = ImprovedInputParser;
            addRequired(p,'d');
            addParameter(p,'sigma',1.0);
            addParameter(p,'x_lim',4.0);
            addParameter(p,'A',3.0);
            addParameter(p,'omega',2.0);
            parse(p,d,varargin{:});
            C = passMatchedArgsToProperties(p, C);

            C.name = 'cosine';
 
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function log_pi = log_pdf(C, X)
          
            % check inputs
            if size(X,2) ~= C.d
                error('C: dimension mismatch for inputs samples')
            end

            % extract x1 and compute mean of x(:,2)
            x1 = X(:,1);
            x2_mu = C.A*cos(C.omega*x1);
        
            % log_pdf for first two directions (uniform, Gaussian)
            log_pi = -1*log(2*C.x_lim)*ones(size(X,1),1);
            log_pi = log_pi + C.norm_log_pdf(X(:,2), x2_mu, C.sigma);

            % compute log_pdf for remaining Gaussian directions
            for i=3:C.d
                log_pi = log_pi + C.norm_log_pdf(X(:,i), 0, 1);
            end

            % assign values of x1 > x_lim to -Inf
            log_pi(abs(x1) > C.x_lim) = -Inf;

        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function X = sample(C, N)

            % sample first two directions 
            x1 = (2*C.x_lim)*rand(N,1) - C.x_lim;
            x2 = C.A*cos(C.omega*x1);

            % sample remaining Gaussian directions
            X = randn(N, C.d);

            % append x1 and x2
            X(:,1) = x1;
            X(:,2) = C.sigma*X(:,2) + x2;

        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function log_pi = norm_log_pdf(~, X, mean, sigma)
			log_pi = -0.5*log(2*pi*sigma^2) - 0.5/sigma^2*(X - mean).^2;
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
    end %endMethods
end %endClass
