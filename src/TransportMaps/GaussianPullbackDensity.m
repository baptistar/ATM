classdef GaussianPullbackDensity < PullbackDensity

	% Define a linear transport map of the form S(x) = L(x - c) that
    % standardizes a random variables X ~ \pi by subtracting the mean and
    % scaling by the inverse Cholesky factor of the empirical covariance.
	%
	% Methods include: evaluate, inverse, optimize, log_pdf

	properties
        %S         % map objects
        %d         % dimension of the map
        %diagonal  % flag for using diagonal or dense transformation
        %ref       % reference distribution (default: Gaussian)
    end
    
	methods
		function self = GaussianPullbackDensity(d, diagonal)
            % assign dimension
            %self.d    = d;

%             % define default properties
%             if (nargin < 2)
%                 diagonal = true;
%             end
%             self.diagonal = diagonal;

            % assign reference
            ref = IndependentProductDistribution(repmat({Normal()},d,1));

%             % initialize to identity map
%             default_c    = zeros(1,d);
%             default_L    = eye(d);
%             S    = cell(d,1);
%             for k=1:d
%                 S{k} = LinearFunction(default_c(k), default_L(k,1:k));
%             end
%             self.S  = S;

            S = LinearTriangularTransportMap(d, diagonal);
            % define PullbackDensity object 
            self@PullbackDensity(S, ref);
            
        end
%         %------------------------------------------------------------------
%         %------------------------------------------------------------------
%         function nc = ncoeff(self)
%             nc = 0;
%             for k=1:self.d
%                 nc = nc + self.S{k}.ncoeff();
%             end
%         end %endFunction
%         %------------------------------------------------------------------
%         %------------------------------------------------------------------
%         function L = L(self)
%             % extract linear map from LinearMap components
%             L = zeros(self.d, self.d);
%             for k=1:self.d
%                 L(k,1:k) = self.S{k}.L;
%             end
%         end
%         %------------------------------------------------------------------
%         %------------------------------------------------------------------
%         function c = c(self)
%             % extract constant from LinearMap components
%             neg_Lc = zeros(self.d,1);
%             for k=1:self.d
%                 neg_Lc(k) = self.S{k}.c;
%             end
%             L = self.L();
%             c = -1*L\neg_Lc;
%         end
%         %------------------------------------------------------------------
%         %------------------------------------------------------------------
%         function Z = evaluate(self, X, comp_idx)
%             % if not specified, evaluate all components of transport map
%             if (nargin < 3)
%                 comp_idx = 1:self.d;
%             end
%             % check dimensions of input samples
%             if size(X,2) ~= self.d
%                 error('LM: dimension mismatch for input samples X')
%             end
%             % evaluate each component
%             Z = zeros(size(X,1), length(comp_idx));
%             for k=1:length(comp_idx)
%                 Ck = comp_idx(k);
%                 Z(:,k) = self.S{Ck}.evaluate(X(:,1:Ck));
%             end
%         end %endFunction
%         %------------------------------------------------------------------
%         %------------------------------------------------------------------	
%         function X = inverse(self, Z, Xp, comp_idx)
%         % invert map S(Xp,X) = Z for X given (Xp,Z). The inverse only holds
%         % if comp_idx is the lower part of the map.
%             % if not specified, invert all components of transport map
%             if (nargin < 4) || isempty(Xp)
%                 comp_idx = 1:self.d;
%                 Xp = zeros(size(Z,1),0);
%             end
%             % check dimensions of input samples
%             if size(Z,2) ~= length(comp_idx)
%                 error('LM: dimension mismatch for input samples Z')
%             end
%             if size(Xp,2) ~= (self.d - length(comp_idx))
%                 error('LM: dimension mismatch for input samples X')
%             end
%             if size(Xp,1) ~= size(Z,1)
%                 error('LM: dimension mismatch for input samples X and Z')
%             end
%             % generate matrix to store input samples
%             X = zeros(size(Z,1), self.d);
%             X(:,1:size(Xp,2)) = Xp;
%             % invert each map component independently
%             for k=1:length(comp_idx)
%                 Ck = comp_idx(k);
%                 X(:,Ck) = self.S{Ck}.inverse(X(:,1:Ck-1), Z(:,k));
%             end
%             X = X(:, comp_idx);
%             % evalaute map at conditioned inputs if map is non-diagonal
%             % and subtract evaluations from right-hand side Z
%             %if self.diagonal ~= true
%             %    input_idx = setdiff(1:self.d, comp_idx);
%             %    L_ll = self.L(comp_idx, input_idx);
%             %    Zp = (Xp - self.c(input_idx)) * L_ll.';
%             %    Z = Z - Zp;
%             %end
%             % apply inverse of lower-right block of map
%             %if self.diagonal == true
%             %    Linv_diag = diag(self.Linv(comp_idx, comp_idx)).';
%             %    Xc = Linv_diag .* Z;
%             %else
%             %    Xc = Z * self.Linv(comp_idx, comp_idx).';
%             %end
%             % re-center samples using c
%             %X = Xc + self.c(comp_idx);
%         end %endFunction
%         %------------------------------------------------------------------
%         %------------------------------------------------------------------	        
%         function dJ = logdet_Jacobian(self, X, comp_idx)
%             % evaluate each component
%             dJ = zeros(size(X,1),1);
%             for k=1:length(comp_idx)
%                 Ck = comp_idx(k);
%                 dJ = dJ + log(self.S{Ck}.grad_xd(X(:,1:Ck)));
%             end
%             %dxS_idx = diag(self.L(comp_idx,comp_idx));
%             %dJ = sum(log(dxS_idx))*ones(size(X,1),1);
%         end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------	        
        function [log_pi, Sx] = log_pdf(self, X, comp_idx)
            % if not specified, evaluate all components of transport map
            if (nargin < 3)
                comp_idx = 1:self.d;
            end
            % evalute map components and log-determinant
            Sx = self.evaluate(X, comp_idx);
            dJ = self.logdet_Jacobian(X, comp_idx);
        	% evaluate log_pdf with transformed inputs
            log_pi = zeros(size(X,1),1);
            for k=1:length(comp_idx)
                Ck = comp_idx(k);
                log_pi  = log_pi + self.ref.factors{Ck}.log_pdf(Sx(:,k));
            end
            log_pi = log_pi + dJ;
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
        %------------------------------------------------------------------
    end %endMethods

end %endClass