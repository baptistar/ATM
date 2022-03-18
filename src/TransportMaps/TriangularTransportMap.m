% Author: Ricardo Baptista
% Date:   Feb 2021
%
% See LICENSE.md for copyright information
%

classdef TriangularTransportMap
    % Define a transport map S of dimension d
    
    properties
        d  % total number of total dimensions
        S  % {d x 1} cell of components type IntegratedPositiveFunction
    end
    
    methods
        %------------------------------------------------------------------
        function TM = TriangularTransportMap(S)
            % assign d and S
            TM.d = length(S);
            TM.S = S;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function check_inputs(self, X)
            assert(size(X,2) == self.d)
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function precomp = initialize_precomp(self)
            precomp = cell(self.d,1);
            for k=1:self.d
                precomp{k} = PPprecomp();
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function nc = n_coeff(self)
            % Extract total number of coefficients in all map components.
            nc = 0;
            for k=1:self.d
                nc = nc + self.S{k}.n_coeff;
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function c = coeff(self)
            % Extract coefficients in all map components as a vector.
            c = [];
            for k=1:self.d
                c = [c, self.S{k}.coeff];
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function self = set_coeff(self, coeff)
            % Set coefficients in all map components.
            c = 1;
            for k=1:self.d
                nc = self.S{k}.n_coeff;
                self.S{k} = self.S{k}.set_coeff(coeff(c:c+nc-1));
                c = c+nc;
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function Z = evaluate(self, X, comp_idx, precomp)
            % Apply map to samples X. Output has the same dimension as X.
            % Inputs:  X - (N x d) matrix
            %          comp_idx - list of components
            %          precomp - cell of objects with precomputed terms
            % Output:  Z - (N x |comp_idx|) matrix
            if (nargin < 4)
                precomp = self.initialize_precomp();
            end
            if (nargin < 3) || isempty(comp_idx)
                comp_idx = 1:self.d;
            end
            self.check_inputs(X)
            % evaluate Sk and save samples in Z(:,k)
            Z = zeros(size(X,1), self.d);
            for Ck = comp_idx
                Z(:, Ck) = self.S{Ck}.evaluate( X(:,1:Ck), precomp{Ck} );
            end
            % extract components in comp_idx
            Z = Z(:, comp_idx);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function X = inverse(self, Z, Xp, comp_idx, precomp)
            % Invert map S(x) = z for inputs x given outputs z.
            % Output has the same dimension as Z.
            if (nargin < 5)
                precomp = self.initialize_precomp();
            end
            if (nargin < 4) || isempty(comp_idx)
                comp_idx = 1:self.d;
                Xp = zeros(size(Z,1),0);
            end
            % check inputs
            if size(Z,2) ~= length(comp_idx)
                error('TM: dimension mismatch for input samples Z')
            end
            if size(Xp,2) ~= (self.d - length(comp_idx))
                error('TM: dimension mismatch for input samples X')
            end
            % store available samples
            X = zeros(size(Z,1), self.d);
            X(:,1:size(Xp,2)) = Xp;
            % invert each map component independently
            for Ck = comp_idx
                X(:,Ck) = self.S{Ck}.inverse( X(:,1:Ck-1), Z(:,Ck), precomp{Ck});
            end
            % extract components in comp_idx
            X = X(:, comp_idx);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxS = grad_x(self, X, grad_dim, comp_idx, precomp)
            % evaluate the gradients of each component with respect to inputs x
            % Output is (N x d x d) matrix. The (i,j,k) entry is the derivative
            % of component j with respect to input k for sample i.
            if (nargin < 5)
                precomp = self.initialize_precomp();
            end
            if (nargin < 4) || isempty(comp_idx)
                comp_idx = 1:self.d;
            end
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:self.d;
            end
            self.check_inputs(X)
            % compute gradients for all components in comp_idx
            dxS = zeros(size(X,1), self.d, self.d);
            for Ck = comp_idx
                grad_dim_Ck = intersect(1:Ck, grad_dim);
                if ~isempty(grad_dim_Ck)
                    dxS(:, Ck, grad_dim_Ck) = ...
                        self.S{Ck}.grad_x( X(:,1:Ck), grad_dim_Ck, precomp{Ck});
                end
            end
            % extract components in comp_idx and gradients in grad_dim
            dxS = dxS(:, comp_idx, grad_dim);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxdS = grad_xd(self, X, comp_idx, precomp)
            % evaluate gradient of diagonal terms of triangular map.
            % Inputs:   X - (N x d) array
            %           comp_idx - list of components
            %           precomp - cell of objects with precomputed terms
            % Outputs:  dxdS - (N x |comp_idx|) array
            if (nargin < 4)
                precomp = self.initialize_precomp();
            end
            if (nargin < 3) || isempty(comp_idx)
                comp_idx = 1:self.d;
            end
            
            self.check_inputs(X)
            % compute gradients for all components in comp_idx
            dxS = zeros(size(X,1), self.d, self.d);
            for Ck = comp_idx
                dxdS(:,Ck) = self.S{Ck}.grad_xd( X(:,1:Ck), precomp{Ck} );
            end
            % extract components in comp_idx
            dxdS = dxdS(:, comp_idx);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function DJ = logdet_Jacobian(self, X, comp_idx, precomp)
            % evaluate log-determinant of Jacobian of triangular map.
            % Output: DJ - (N x 1) array
            if (nargin < 4)
                precomp = self.initialize_precomp();
            end
            if (nargin < 3) || isempty(comp_idx)
                comp_idx = 1:self.d;
            end
            DJ = sum( log( self.grad_xd(X, comp_idx, precomp) ), 2);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxDJ = grad_x_logdet_Jacobian(self, X, grad_dim, comp_idx, precomp)
            % evaluate gradient of log-determinant of map Jacobian.
            % Output: dxDJ - (N x d) matrix
            if (nargin < 5)
                precomp = self.initialize_precomp();
            end
            if (nargin < 4) || isempty(comp_idx)
                comp_idx = 1:self.d;
            end
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:self.d;
            end
            self.check_inputs(X)
            % evaluate gradients for each component in comp_idx
            dxDJ = zeros(size(X,1), self.d);
            for Ck = comp_idx
                grad_dim_Ck = intersect(1:Ck, grad_dim);
                dxSk = self.S{Ck}.grad_xd( X(:,1:Ck), precomp{Ck} );
                dxdxSk = self.S{Ck}.grad_x_grad_xd( X(:,1:Ck), grad_dim_Ck, precomp{Ck} );
                dxDJ(:,grad_dim_Ck) = dxDJ(:,grad_dim_Ck) + 1./dxSk .* dxdxSk;
            end
            % extract gradients in grad_dim
            dxDJ = dxDJ(:, grad_dim);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        %         function d2xDJ = hess_x_logdet_Jacobian(self, X, grad_dim, comp_idx, precomp)
        %         % evaluate Hessian of log-determinant of map Jacobian.
        %         % Output: dxDJ - (N x d) matrix
        %             if (nargin < 4)
        %                 comp_idx = 1:self.d;
        %             end
        %             if (nargin < 3)
        %                 grad_dim = 1:self.d;
        %             end
        %             % check dimensions of inputs
        %             if size(X,2) ~= self.d
        %                 error('PB: dimension mismatch for input samples')
        %             end
        %
        %             % compute and sum derivative for each component in comp_idx
        %             d2xDJ = zeros(size(X,1), length(grad_dim), length(grad_dim));
        %             for Ck = comp_idx
        %                 grad_dim_Ck = intersect(grad_dim, 1:Ck);
        %                 dxSk = self.S{k}.grad_xd( X(:,1:Ck) );
        %                 dxdxSk = self.S{k}.grad_x_grad_xd( X(:,1:Ck), grad_dim_Ck );
        %                 d2xdxSk = self.S{k}.hess_x_grad_xd( X(:,1:Ck), grad_dim_Ck );
        %                 d2xJk = 1./(dxSk).^2 .* OuterProd(dxdxSk, dxdxSk) + ...
        %                         1./(dxSk) .* d2xdxSk;
        %                 d2xDJ(:,grad_dim_Ck, grad_dim_Ck) = ...
        %                     d2xDJ(:,grad_dim_Ck, grad_dim_Ck) + d2xJk;
        %             end
        %
        %         end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dadS = grad_coeff(self, X, precomp)
            % Compute gradients of map component wrt to coeffs \nabla_a S(x)
            % Output is a (N x d x nc) matrix. Matrix is sparse because
            % coefficients aren't shared across components.
            if nargin < 3
                precomp = self.initialize_precomp();
            end
            self.check_inputs(X)
            dadS = zeros(size(X,1), self.d, self.n_coeff);
            c = 1;
            for k=1:self.d
                nc = self.S{k}.n_coeff;
                dadS(:,k,c:c+nc-1) = self.S{k}.grad_coeff(X(:,1:k), [], precomp{k});
                c = c+nc;
            end
        end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dadxdS = grad_coeff_grad_xd(self, X, precomp)
            % evaluate gradient of map components with respect to last variable
            % and coefficients in each component
            if nargin < 3
                precomp = self.initialize_precomp();
            end
            self.check_inputs(X)
            dadxdS = zeros(size(X,1), self.d, self.n_coeff);
            c = 1;
            for k=1:self.d
                nc = self.S{k}.n_coeff;
                dadxdS(:,k,c:c+nc-1) = self.S{k}.grad_coeff_grad_xd( X(:,1:k), [], precomp{k} );
                c = c+nc;
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function daDJ = grad_coeff_logdet_Jacobian(self, X, precomp)
            % Compute gradient \nabla_a log(det(\nabla_x S(x))).
            % Output: (N x nc) matrix
            if nargin < 3
                precomp = self.initialize_precomp();
            end
            self.check_inputs(X)
            daDJ = zeros(N, self.n_coeff);
            c = 1;
            for k = 1 : self.d
                nc = self.S{k}.n_coeff;
                dxdS = self.S{k}.grad_xd( X(:,1:k), precomp{k} );
                dadxdS = self.S{k}.grad_coeff_grad_xd( X(:,1:k), [], precomp{k} );
                daDJ(:,c:c+nc-1) = 1./dxdS .* dadxdS;
                c = c+nc;
            end
        end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
    end %endMethods
end %endClass