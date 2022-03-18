% Author: Ricardo Baptista and Olivier Zahm and Youssef Marzouk
% Date:   Feb 2021
%
% See LICENSE.md for copyright information
%

classdef ConditionalTriangularTransportMap < TriangularTransportMap
    % Define a transport map S(y,x) of dimension m+d

    properties
        m   % total number of conditioned variables
    end

    methods  
        %------------------------------------------------------------------
        function CTM = ConditionalTriangularTransportMap(S, m)
            % assign d and S
            CTM = CTM@TriangularTransportMap(S);
            CTM.m = m;
            CTM.d = CTM.d - CTM.m;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function check_inputs(self, X, Y)
            assert(size(X,1) == size(Y,1))
            assert(size(X,2) == self.d)
            assert(size(Y,2) == self.m)
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function nc = n_coeff(self)
            % Extract total number of coefficients in all map components.
            nc = 0;
            for k=1:self.d
                nc = nc + self.S{self.m+k}.n_coeff;
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function c = coeff(self)
            % Extract coefficients in all map components as a vector.
            c = [];
            for k=1:self.d
                c = [c, self.S{self.m+k}.coeff];
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function self = set_coeff(self, coeff)
            % Set coefficients in all map components.
            c = 1;
            for k=1:self.d
                ck = self.m+k;
                nc = self.S{ck}.n_coeff;
                self.S{ck} = self.S{ck}.set_coeff(coeff(c:c+nc-1));
                c = c+nc;
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function precomp = initialize_precomp(self)
            precomp = cell(self.d+self.m,1);
            for kk=(self.m+1):(self.m+self.d)
                precomp{kk} = PPprecomp();
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function Z = evaluate(self, X, Y, precomp)
        % evaluate: apply map to samples X~\pi. If S(x) is true map,
        % function maps samples from \pi to samples from \rho
            if nargin < 4
                precomp = self.initialize_precomp();
            end
            self.check_inputs(X, Y)
            % evaluate each component Sk and save samples in Z(:,k)
            Z = zeros(size(X,1), self.d);
            for k=1:self.d
                ck = self.m + k;
                Z(:,k) = self.S{ck}.evaluate([Y,X(:,1:k)], precomp{ck});
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function X = inverse(self, Z, Y, precomp)
        % inverse: apply map to samples Z~\rho. If S(x) is true map,
        % function maps samples from reference to samples from \pi (target)
            if nargin < 4
                precomp = self.initialize_precomp();
            end
            self.check_inputs(Z, Y)
            % invert each map component Sk independently
            X = zeros(size(Z,1), self.d);
            for k=1:self.d
                ck = self.m + k;
                X(:,k) = self.S{ck}.inverse([Y,X(:,1:k-1)], Z(:,k), precomp{ck});
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxdS = grad_xd(self, X, Y, precomp)
        % evaluate gradient of map components with respect to last variable
            if nargin < 4
                precomp = self.initialize_precomp();
            end
            self.check_inputs(X, Y)
            dxdS = zeros(size(X,1), self.d);
            for k=1:self.d
                ck = self.m + k;
                dxdS(:,k) = self.S{ck}.grad_xd([Y, X(:,1:k)], precomp{ck});
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function DJ = logdet_Jacobian(self, X, Y, precomp)
        % evaluate log-determinant of Jacobian of map: log(det(\nabla S(x)))
        % Output: (N x 1) array
            if nargin < 4
                precomp = self.initialize_precomp();
            end
            self.check_inputs(X, Y)
            % evaluate gradient of last variable for each component
            DJ = sum(log(self.grad_xd(X, Y, precomp)),2);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dadS = grad_coeff(self, X, Y, precomp)
        % evaluate gradient of map components with respect to coefficients
        % Output: (N x d x nc) array
            if nargin < 4
                precomp = self.initialize_precomp();
            end
            self.check_inputs(X, Y)
            dadS = zeros(size(X,1), self.d, self.n_coeff);
            % evaluate gradient of map components with respect to coeffs
            c = 1;
            for k=1:self.d
                ck = self.m + k;
                nc = self.S{ck}.n_coeff;
                dadS(:,k,c:c+nc-1) = self.S{ck}.grad_coeff([Y, X(:,1:k)], [], precomp{ck});
                c = c+nc;
            end
        end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dadxdS = grad_coeff_grad_xd(self, X, Y, precomp)
        % evaluate gradient of map components with respect to last variable
        % and coefficients in each component
            if nargin < 4
                precomp = self.initialize_precomp();
            end
            self.check_inputs(X, Y)
            dadxdS = zeros(size(X,1), self.d, self.n_coeff);
            c = 1;
            for k=1:self.d
                ck = self.m + k;
                nc = self.S{ck}.n_coeff;
                dadxdS(:,k,c:c+nc-1) = self.S{ck}.grad_coeff_grad_xd([Y, X(:,1:k)], [], precomp{ck});
                c = c+nc;
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function daDJ = grad_coeff_logdet_Jacobian(self, X, Y, precomp)
        % evaluate gradient of log determinant with respect to coefficients
        % Output: (N x nc) array
            if nargin < 4
                precomp = self.initialize_precomp();
            end
            self.check_inputs(X, Y)
            daDJ = zeros(size(X,1), self.n_coeff);
            % evaluate gradient of log-gradient with respect to coeffs
            c = 1;
            for k=1:self.d
                ck = self.m + k;
                nc = self.S{ck}.n_coeff;
                dxdS = self.S{ck}.grad_xd([Y, X(:,1:k)], precomp{ck});
                dadxdS = self.S{ck}.grad_coeff_grad_xd([Y, X(:,1:k)], [], precomp{ck});
                daDJ(:,c:c+nc-1) = 1./dxdS .* dadxdS;
                c = c+nc;
            end
        end
        %------------------------------------------------------------------
    end %endMethods

end %endClass