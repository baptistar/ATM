classdef Banana

    properties
        X1 = Normal(0,0.5)
        X2_noise = Normal(0,0.1)
    end
    
    methods
        % -----------------------------------------------------------------
        function check_inputs(~, x)
            assert(size(x,2) == 2)
        end
        % -----------------------------------------------------------------
        function logpi = log_pdf(self, x)
            self.check_inputs(x)
            logpi_1 = self.X1.log_pdf(x(:,1));
            logpi_2 = self.X2_noise.log_pdf(x(:,2) - x(:,1).^2);
            logpi = logpi_1 + logpi_2;
        end %endFunction
        % -----------------------------------------------------------------
        function grad_logpi = grad_x_log_pdf(self, x, grad_dim)
            self.check_inputs(x)
            if nargin < 3
                grad_dim = 1:2;
            end
            % evaluate conditional x2 samples
            x2_noise = x(:,2) - x(:,1).^2;
            grad_logpi = zeros(size(x,1), 2);
            if ~isempty(intersect(grad_dim,1))
                grad_logpi(:,1) = self.X1.grad_x_log_pdf(x(:,1)) + ...
                    self.X2_noise.grad_x_log_pdf(x2_noise) .* (-2*x(:,1));
            end
            if ~isempty(intersect(grad_dim,2))
                grad_logpi(:,2) = self.X2_noise.grad_x_log_pdf(x2_noise);
            end
            grad_logpi = grad_logpi(:,grad_dim);
        end %endFunction
        % -----------------------------------------------------------------
        function x = sample(self, N)
            x1 = self.X1.sample(N);
            x2 = x1.^2 + self.X2_noise.sample(N);
            x = [x1, x2];
        end %endFunction
        % -----------------------------------------------------------------
	end
    
end   
