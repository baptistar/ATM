classdef ParametericPoly < MultivariatePoly

    % ParametericPoly defines a class of polynomial expansions
    % where the coefficients are parameterized as functions of other
    % variables and the basis functions are univariate functions of x_d
    %   S(x) = \sum_{i} c_{i}(x_{1:d-1}) \Psi_{i}(x_d)
    %
    % Methods: n_coeff, coeff, set_coeff, multi_idxs
    %          evaluate, grad_x, hess_x,
    %          grad_xd, hess_xd, grad_x_grad_xd,
    %          grad_coeff, hess_coeff, 
    %          grad_coeff_grad_xd, grad_coeff_hess_xd
    %
    % Author: Ricardo Baptista
    % Date:   February 2020

    methods
        %------------------------------------------------------------------
        function self = ParametericPoly(basis, multi_idxs, is_orth)

            % assemble superclass constructor arguments
            MP_args{1} = basis;
            MP_args{2} = multi_idxs;
            if (nargin == 3)
                MP_args{3} = is_orth;
            end
                            
            % load parent class object
            self = self@MultivariatePoly(MP_args{:});
            
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function P = evaluate(self, X, precomp)
            % Evaluate f(x_{1:d})
            % pre-compute evaluations of off-diagonal basis functions
            if (nargin == 3) && ~isempty(precomp.eval_basis)
                Psi = precomp.eval_basis;
            elseif (nargin == 3) && ~isempty(precomp.eval_offdiagbasis)
                Psid = self.evaluate_diagbasis(X);
                Psi = precomp.eval_offdiagbasis .* Psid;
            else
                Psi = self.evaluate_basis(X);
            end
            % compute inner product of basis derivatives and coefficients
            P = (Psi * self.coeff.');
        end %endFunction        
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function P = evaluate_f0(self, X, precomp)
            % evaluate f(x_{1:d-1},0)
            % pre-compute evaluations of off-diagonal basis functions
            if (nargin == 3) && ~isempty(precomp.eval_basis0)
                Psi = precomp.eval_basis0;
            elseif (nargin == 3) && ~isempty(precomp.eval_offdiagbasis)
                X0 = [X(:,1:end-1), zeros(size(X,1),1)];
                Psi = precomp.eval_offdiagbasis .* self.evaluate_diagbasis(X0);
            else
                X0 = [X(:,1:end-1), zeros(size(X,1),1)];
                Psi = self.evaluate_basis(X0);
            end
            % compute inner product of basis derivatives and coefficients
            P = (Psi * self.coeff.');
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxP0 = grad_x_evaluate_f0(self, X, grad_dim, precomp)
            % set grad_dim to all dimensions if it is not specified
            if ~exist('grad_dim', 'var') || isempty(grad_dim)
                grad_dim = 1:self.dim;
            end
            % declare empty array to store gradients of basis
            dxPsi = zeros(size(X,1), self.n_coeff, length(grad_dim));
            % get index of self.dim in grad_dim
            dim_idx = (grad_dim == self.dim);
            % if function contains any off-diagonal x dependence
            if any(~dim_idx)
                % pre-compute derivatives of basis functions
                if (nargin == 4) && ~isempty(precomp.grad_x_offdiagbasis)
                    dxPsio = precomp.grad_x_offdiagbasis;
                    dxPsio = dxPsio(:,:,grad_dim(~dim_idx));
                else
                    dxPsio = self.grad_x_offdiagbasis(X, grad_dim(~dim_idx));
                end
                % pre-compute evaluations of f0
                if (nargin == 4) && ~isempty(precomp.eval_diagbasis0)
                    Psid0 = precomp.eval_diagbasis0;
                else
                    X0 = [X(:,1:end-1), zeros(size(X,1),1)];
                    Psid0 = self.evaluate_diagbasis(X0);
                end
                % add evaluations to all rows except for dim_idx
                dxPsi(:,:,~dim_idx) = dxPsio .* Psid0;
            end
            % compute inner product of basis and coefficients
            dxP0 = reshape(sum( dxPsi .* self.coeff, 2), size(dxPsi,1), length(grad_dim));
            %dxP0 = InnerProd(dxPsi, self.coeff, 2);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2xf0 = hess_x_evaluate_f0(self, X, grad_dim, precomp)
            % set grad_dim to all dimensions if it is not specified
            if ~exist('grad_dim', 'var') || isempty(grad_dim)
                grad_dim = 1:self.dim;
            end
            % get index of self.dim in grad_dim
            dim_idx = (grad_dim == self.dim);
            % pre-compute derivatives of basis functions
            if (nargin == 4) && ~isempty(precomp.hess_x_offdiagbasis)
                d2xPsio = precomp.hess_x_offdiagbasis;
                d2xPsio = d2xPsio(:,:,grad_dim(~dim_idx),grad_dim(~dim_idx));
            else
                d2xPsio = self.hess_x_offdiagbasis(X, grad_dim(~dim_idx));
            end
            % pre-compute evaluations of f0
            if (nargin == 4) && ~isempty(precomp.eval_diagbasis0)
                Psid0 = precomp.eval_diagbasis0;
            else
                X0 = [X(:,1:end-1), zeros(size(X,1),1)];
                Psid0 = self.evaluate_diagbasis(X0);
            end
            % add evaluations to all rows except for dim_idx
            d2xPsi = zeros(size(X,1), self.n_coeff, length(grad_dim),length(grad_dim));
            d2xPsi(:,:,~dim_idx,~dim_idx) = d2xPsio .* Psid0;
            % compute inner product of basis functions and coefficients
            d2xf0 = InnerProd(d2xPsi, self.coeff, 2);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxdxdP = grad_x_grad_xd(self, X, grad_dim, precomp)
            % set grad_dim to all dimensions if it is not specified
            if ~exist('grad_dim', 'var') || isempty(grad_dim)
                grad_dim = 1:self.dim;
            end
            % pre-compute using hessian
            if (nargin == 4) && ~isempty(precomp.hess_x_basis)
                dxdxdPsi = precomp.hess_x_basis(:,:,grad_dim,end);
                dxdxdP = InnerProd(dxdxdPsi, self.coeff, 2);
                return
            end
            % declare array to store result
            dxdxdP = zeros(size(X,1), length(grad_dim));
            % get index of self.dim in grad_dim
            dim_idx = (grad_dim == self.dim);
            % evaluate \nabla_xj \nabla_xd f(x) for j \neq d
            if any(~dim_idx)
                if (nargin == 4) && ~isempty(precomp.grad_x_offdiagbasis)
                    dxPsio = precomp.grad_x_offdiagbasis;
                    dxPsio = dxPsio(:,:,grad_dim(~dim_idx));
                else
                    dxPsio = self.grad_x_offdiagbasis(X, grad_dim(~dim_idx));
                end
                % pre-compute diagonal basis
                if (nargin == 4) && ~isempty(precomp.grad_xd_diagbasis)
                    dxPsid = precomp.grad_xd_diagbasis;
                else
                    dxPsid = self.grad_xd_diagbasis(X);
                end
                dxdxdP(:,~dim_idx) = InnerProd(dxPsio .* dxPsid, self.coeff, 2);
            end
            % evaluate \nabla^2_xd f(x)
            if any(dim_idx)
                % pre-compute off-diagonal basis
                if (nargin == 4) && ~isempty(precomp.eval_offdiagbasis)
                    Psid = precomp.eval_offdiagbasis;
                else
                    Psid = self.evaluate_offdiagbasis(X);
                end
                % pre-compute diagonal basis
                if (nargin == 4) && ~isempty(precomp.hess_xd_diagbasis)
                    d2xPsid = precomp.hess_xd_diagbasis;
                else
                    d2xPsid = self.hess_xd_diagbasis(X);
                end
                dxdxdP(:,dim_idx) = ((Psid .* d2xPsid) * self.coeff.');
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dcP0 = grad_coeff_f0(self, X, coeff_idx, precomp)
            % if coeff_idx is not specified/empty, set to entire basis
            if ~exist('coeff_idx', 'var') || isempty(coeff_idx)
                coeff_idx = 1:self.n_coeff;
            end
            % evaluate \nabla_c f(x_{1:d-1},0)
            if (nargin == 4) && ~isempty(precomp.eval_basis0)
                Psi0 = precomp.eval_basis0(:, coeff_idx);
            elseif (nargin == 4) &&  ~isempty(precomp.eval_offdiagbasis)
                X0 = [X(:,1:end-1), zeros(size(X,1),1)];
                multi_idxs = self.multi_idxs(coeff_idx,:);
                Psid0 = self.evaluate_diagbasis(X0, multi_idxs);
                Psi0 = (precomp.eval_offdiagbasis(:,coeff_idx) .* Psid0);
            else
                X0 = [X(:,1:end-1), zeros(size(X,1),1)];
            	Psi0 = self.grad_coeff(X0, coeff_idx);
            end
            dcP0 = Psi0;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2cP0 = hess_coeff_f0(~, X, coeff_idx, ~)
            d2cP0 = zeros(size(X,1), length(coeff_idx), length(coeff_idx));
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxP = grad_xd(self, X, precomp)
            % evaluate \partial_xd f(x_{1:d})
            % evaluate basis function derivatives of last variable
            if (nargin == 3) && ~isempty(precomp.grad_xd_basis)
                dxPsi = precomp.grad_xd_basis; 
            elseif (nargin == 3) && ~isempty(precomp.eval_offdiagbasis)
                dxPsi = precomp.eval_offdiagbasis .* self.grad_xd_diagbasis(X);
            else
                dxPsi = self.grad_xk_basis(X, 1, self.dim);
            end
            % compute inner product of basis derivatives and coefficients
            dxP = (dxPsi * self.coeff.');
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2xP = hess_xd(self, X, precomp)
            % evaluate \partial^2_xd f(x_{1:d})
            % evaluate basis function derivatives of last variable
            if (nargin == 3) && ~isempty(precomp.hess_xd_basis)
                d2xPsi = precomp.eval_offdiagbasis;                
            elseif (nargin == 3) && ~isempty(precomp.eval_offdiagbasis)
                d2xPsi = precomp.eval_offdiagbasis .* self.hess_xd_diagbasis(X);
            else
                d2xPsi = self.grad_xk_basis(X, 2, self.dim);
            end
            % compute inner product of basis derivatives and coefficients
            d2xP = (d2xPsi * self.coeff.');
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dcP = grad_coeff(self, X, coeff_idx, precomp)
            % if coeff_idx is not specified/empty, set to entire basis
            if ~exist('coeff_idx', 'var') || isempty(coeff_idx)
                coeff_idx = 1:self.n_coeff;
            end
            % load pre-computed evaluations of basis functions
            if (nargin == 4) && ~isempty(precomp.eval_basis)
                Psi = precomp.eval_basis;
                Psi = Psi(:, coeff_idx);
            elseif (nargin == 4) && ~isempty(precomp.eval_offdiagbasis)
                multi_idx = self.multi_idxs(coeff_idx,:);
                Psid = self.evaluate_diagbasis(X, multi_idx);
                Psi = precomp.eval_offdiagbasis(:, coeff_idx) .* Psid;
            else
                multi_idxs = self.multi_idxs(coeff_idx,:);
                Psi = self.evaluate_basis(X, 1:self.dim, multi_idxs);
            end
            % return basis function evaluations
            dcP = Psi;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dcdxdP = grad_coeff_grad_xd(self, X, coeff_idx, precomp)
            % if coeff_idx is not specified/empty, set to entire basis
            if ~exist('coeff_idx', 'var') || isempty(coeff_idx)
                coeff_idx = 1:self.n_coeff;
            end
            % load pre-computed evaluations of basis functions
            if (nargin == 4) && ~isempty(precomp.grad_xd_basis)
                dxPsi = precomp.grad_xd_basis;
                dxPsi = dxPsi(:, coeff_idx);
            elseif (nargin == 4) && ~isempty(precomp.eval_offdiagbasis) 
                % evaluate derivatives with respect to last variable
                multi_idx = self.multi_idxs(coeff_idx,:);
                dxPsid = self.grad_xd_diagbasis(X, multi_idx);
                dxPsi = precomp.eval_offdiagbasis(:, coeff_idx) .* dxPsid;
            else
                multi_idx = self.multi_idxs(coeff_idx,:);
                dxPsi = self.grad_xk_basis(X, 1, self.dim, 1:self.dim, multi_idx);
            end
            % return basis function evaluations
            dcdxdP = dxPsi;
        end %endFunction
        %------------------------------------------------------------------
    end
    %
    methods %(Access == protected)
        %------------------------------------------------------------------
        function Psi = evaluate_offdiagbasis(self, X, multi_idxs)
        % Inputs:   X - (N x dim) sample matrix
        %           multi_idxs - indices of basis functions
            if ~exist('multi_idxs','var') || isempty(multi_idxs)
                multi_idxs = self.multi_idxs;
            end
            % evaluate basis of x_{1:d-1}
            Psi = self.evaluate_basis(X, 1:self.dim-1, multi_idxs);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxPsi = grad_x_offdiagbasis(self, X, grad_dim, multi_idxs)
        % Inputs:   X - (N x dim) sample matrix
        %           grad_dim - variables for which derivatives are computed
        %                      e.g., grad_dim = dim for gradients of x_d
        %           multi_idxs - indices of basis functions
            if ~exist('multi_idxs','var') || isempty(multi_idxs)
                multi_idxs = self.multi_idxs;
            end
            if ~exist('grad_dim','var')
                grad_dim = 1:self.dim;
            end
            % evaluate derivatives of basis of x_{1:d-1}
            dxPsi = self.grad_x_basis(X, grad_dim, 1:self.dim-1, multi_idxs);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2xPsi = hess_x_offdiagbasis(self, X, grad_dim, multi_idxs)
        % Inputs:   X - (N x dim) sample matrix
        %           grad_dim - variables for which derivatives are computed
        %                      e.g., grad_dim = dim for gradients of x_d
        %           multi_idxs - indices of basis functions
            if ~exist('multi_idxs','var') || isempty(multi_idxs)
                multi_idxs = self.multi_idxs;
            end
            if ~exist('grad_dim','var')
                grad_dim = 1:self.dim;
            end
            % evaluate derivatives of basis of x_{1:d-1}
            d2xPsi = self.hess_x_basis(X, grad_dim, 1:self.dim-1, multi_idxs);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function Psi = evaluate_diagbasis(self, X, multi_idxs)
        % Inputs:   X - (N x dim) sample matrix
        %           multi_idxs - indices of basis functions
            if ~exist('multi_idxs','var') || isempty(multi_idxs)
                multi_idxs = self.multi_idxs;
            end
            % evaluate basis of x_{d}
            Psi = self.evaluate_basis(X, self.dim, multi_idxs);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function Psi = grad_xd_diagbasis(self, X, multi_idxs)
        % Inputs:   X - (N x dim) sample matrix
        %           multi_idxs - indices of basis functions
            if ~exist('multi_idxs','var') || isempty(multi_idxs)
                multi_idxs = self.multi_idxs;
            end
            % evaluate derivatives of basis of x_{d}
            Psi = self.grad_xk_basis(X, 1, self.dim, self.dim, multi_idxs);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function Psi = hess_xd_diagbasis(self, X, multi_idxs)
        % Inputs:   X - (N x dim) sample matrix
        %           multi_idxs - indices of basis functions
            if ~exist('multi_idxs','var') || isempty(multi_idxs)
                multi_idxs = self.multi_idxs;
            end
            % evaluate derivatives of basis of x_{d}
            Psi = self.grad_xk_basis(X, 2, self.dim, self.dim, multi_idxs);
        end %endFunction
        %------------------------------------------------------------------
    end %endMethods
    
end %endClass