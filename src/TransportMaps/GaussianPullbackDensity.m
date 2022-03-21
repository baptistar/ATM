classdef GaussianPullbackDensity < PullbackDensity

	% Define the pullback of a linear transport map that standardizes 
    % a random variables X ~ \pi by subtracting the mean and
    % scaling by the inverse Cholesky factor of the empirical covariance.
	%
	% Methods include: optimize, sample
    %                  log_pdf is defined in PullbackDensity
    %
    % Author: Ricardo Baptista
    % Date:   June 2021

	properties
    end
    
	methods
        %------------------------------------------------------------------
		function self = GaussianPullbackDensity(d, diagonal)
            % define default properties
            if (nargin < 2)
                diagonal = true;
            end
            % assign reference
            ref = IndependentProductDistribution(repmat({Normal()},d,1));
            % assign S
            S = LinearTriangularTransportMap(d, diagonal);
            % define PullbackDensity object 
            self@PullbackDensity(S, ref);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function self = optimize(self, X, ~)
        % Function finds optimal mean and covariance for all dimensions
        % ignoring the third comp_idx input.
            % check dimensions of inputs
            if size(X,2) ~= self.d
                error('Dimension mismatch of inputs and dimension')
            end
            % compute covariance or diagonal scaling of data
            if self.S.diagonal == true
                std_data = std(X,[],1);
                L = diag(1./std_data);
            else
                C = cov(X);
                L = inv(chol(C,'lower'));
            end
            % compute constant based on mean of data
            c = -1 * L * mean(X,1).';
            % assign to components
            for k=1:self.d
                self.S.S{k} = self.S.S{k}.set_coeff(c(k), L(k,1:k));
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function X = sample(self, N)
            % sample from the reference
            Z = self.ref.sample(N);
            X = self.L() \ Z + self.c;
        end %endFunction
        %------------------------------------------------------------------
    end %endMethods

end %endClass