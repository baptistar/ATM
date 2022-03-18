classdef MultivariatePoly

    % MultivariatePoly defines a multivariate expansion
    %   P(x) = \sum_{\alpha} c_{\alpha}*\Psi_{\alpha}(x)
    %
    % Methods: evaluate, grad_x, hess_x, 
    %          grad_xd, hess_xd, grad_x_grad_xd, 
    %          grad_coeff, hess_coeff, 
    %          grad_coeff_grad_xd, hess_coeff_grad_xd
    %
    % Author: Ricardo Baptista, Olivier Zahm
    % Date:   February 2020
    
    properties (SetAccess = immutable)
        dim           % input dimension
        basis         % (dim x 1) cell of univariate basis objects
        is_orth       % normalization of monomials (default: true) 
    end
    properties
        coeff         % expansion coefficients
        multi_idxs    % (n_coeff x d) multi-index array
    end

    methods
        %------------------------------------------------------------------
        function self = MultivariatePoly(basis, multi_idxs, is_orth)
            
            % set default
            if (nargin < 3)
                is_orth = true;
            end  

            % Define MP object
            self.basis = basis;
            self.dim = length(basis);
            self.is_orth = is_orth;

            % check and assign multi_indices
            if size(multi_idxs,2) ~= self.dim
               error('MP: multi_index should be an array of multi-indices')
            end
            self.multi_idxs = multi_idxs;

        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function nc = n_coeff(self)
            nc = size(self.multi_idxs, 1);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function self = set_coeff(self, c)
            % check dimension of coefficients
            if length(c) ~= self.n_coeff
                error('MP: number of coefficients don''t match basis')
            end
            % set to a row-vector if c is a column vector
            if size(c,2) == 1
                c = c';
            end
            self.coeff = c;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function P = evaluate(self, X, precomp)
            % pre-compute evaluations of basis functions
            if (nargin == 3) && ~isempty(precomp.eval_basis)
                Psi = precomp.eval_basis;
            else
                Psi = self.evaluate_basis(X);
            end
            % compute inner product of basis evaluations and coefficients
            P = (Psi * self.coeff.');
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxP = grad_x(self, X, grad_dim, precomp)
            % if grad_dim is not specified/empty, set to all dimensions
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:self.dim;
            end
            % pre-compute derivatives of basis functions
            if (nargin == 4) && ~isempty(precomp.grad_x_basis)
                dxPsi = precomp.grad_x_basis;
                dxPsi = dxPsi(:,:,grad_dim);
            else
                dxPsi = self.grad_x_basis(X, grad_dim);
            end
            % compute inner product of basis derivatives and coefficients
            dxP = InnerProd(dxPsi, self.coeff, 2);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2xP = hess_x(self, X, grad_dim, precomp)
            % if grad_dim is not specified/empty, set to all dimensions
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:self.dim;
            end
            % pre-compute derivatives of basis functions
            if (nargin == 4) && ~isempty(precomp.hess_x_basis)
                d2xPsi = precomp.hess_x_basis;
                d2xPsi = d2xPsi(:,:,grad_dim,grad_dim);
            else
                d2xPsi = self.hess_x_basis(X, grad_dim);
            end
            % compute inner product of basis derivatives and coefficients
            d2xP = InnerProd(d2xPsi, self.coeff, 2);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxdP = grad_xd(self, X, precomp)
            % pre-compute derivatives of basis functions
            if (nargin == 3) && ~isempty(precomp.grad_xd_basis)
                dxdPsi = precomp.grad_xd_basis;
            else
                dxdPsi = self.grad_xk_basis(X, 1, self.dim);
            end
            % compute inner product of basis derivatives and coefficients
            dxdP = (dxdPsi * self.coeff.');
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2xdP = hess_xd(self, X, precomp)
            % load pre-computed derivatives of basis functions
            if (nargin == 3) && ~isempty(precomp.hess_xd_basis) 
                d2xdPsi = precomp.hess_xd_basis;
            else
                d2xdPsi = self.grad_xk_basis(X, 2, self.dim);
            end
            % compute inner product of basis derivatives and coefficients
            d2xdP = (d2xdPsi * self.coeff.');
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxjdP = grad_x_grad_xd(self, X, grad_dim, precomp)
            % if grad_dim is not specified/empty, set to all dimensions
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:self.dim;
            end
            % load pre-computed derivatives of basis functions
            if (nargin == 4) && ~isempty(precomp.hess_x_basis)
                dxjdPsi = precomp.hess_x_basis;
                dxjdPsi = dxjdPsi(:,:,grad_dim,self.dim);
            else            
                dxjdPsi = zeros(size(X,1), self.n_coeff, length(grad_dim));
                for i = 1:length(grad_dim)
                    if grad_dim(i) == self.dim
                        dxjdPsi(:,:,i) = self.grad_xk_basis(X, 2, self.dim, 1:self.dim);
                    else
                        d_i = [grad_dim(i), self.dim];
                        dxjdPsi(:,:,i) = self.grad_xk_basis(X, 1, d_i, 1:self.dim);
                    end
                end
            end
            % compute inner product of basis derivatives and coefficients
            dxjdP = InnerProd(dxjdPsi, self.coeff, 2);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dcP = grad_coeff(self, X, coeff_idx, precomp)
            % if coeff_idx is not specified/empty, set to entire basis
            if (nargin < 3) || isempty(coeff_idx)
                coeff_idx = 1:self.n_coeff;
            elseif any(coeff_idx > self.n_coeff) || any(coeff_idx <= 0)
                error('coeff_idx is outside allowable range')
            end
            % load pre-computed evaluations of basis functions
            if (nargin == 4) && ~isempty(precomp.eval_basis)
                dcP = precomp.eval_basis;
                dcP = dcP(:, coeff_idx);
            else
                multi_idx = self.multi_idxs(coeff_idx,:);
                dcP = self.evaluate_basis(X, 1:self.dim, multi_idx);
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2cP = hess_coeff(self, X, coeff_idx, ~)
            % if coeff_idx is not specified/empty, set to entire basis
            if (nargin < 3) || isempty(coeff_idx)
                coeff_idx = 1:self.n_coeff;
            elseif any(coeff_idx > self.n_coeff) || any(coeff_idx <= 0)
                error('coeff_idx is outside allowable range')
            end
            % evaluate second derivatives
            n_coeff_idx = length(coeff_idx);
            d2cP = zeros(size(X,1), n_coeff_idx, n_coeff_idx);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dcdxdP = grad_coeff_grad_xd(self, X, coeff_idx, precomp)
            % if coeff_idx is not specified/empty, set to entire basis
            if (nargin < 3) || isempty(coeff_idx)
                coeff_idx = 1:self.n_coeff;
            elseif any(coeff_idx > self.n_coeff) || any(coeff_idx <= 0)
                error('coeff_idx is outside allowable range')
            end
            % load pre-computed evaluations of basis functions
            if (nargin == 4) && ~isempty(precomp.grad_xd_basis) 
                dcdxdP = precomp.grad_xd_basis;
                dcdxdP = dcdxdP(:, coeff_idx);
            else
                multi_idx = self.multi_idxs(coeff_idx,:);
                dcdxdP = self.grad_xk_basis(X, 1, self.dim, 1:self.dim, multi_idx);
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2cdxdP = hess_coeff_grad_xd(self, X, coeff_idx, ~)
            % if coeff_idx is not specified/empty, set to entire basis
            if (nargin < 3) || isempty(coeff_idx)
                coeff_idx = 1:self.n_coeff;
            end
            % evaluate second derivatives
            n_coeff_idx = length(coeff_idx);
            d2cdxdP = zeros(size(X,1), n_coeff_idx, n_coeff_idx);
        end %endFunction
        %------------------------------------------------------------------
    end %endMethods
    %
    methods %(Access = protected)
        %------------------------------------------------------------------
        function Psi = evaluate_basis(self, X, dims, multi_idxs)
        % Compute products of basis functions of the base polynomials 
        % with respect to each dimension
        %
        % Inputs:   X - (N x dim) sample matrix
        %           dims - basis functions included in product
        %                  e.g., dims = 1:self.dim-1 for off-diagonals
        %           multi_idxs - indices of basis functions
        % Outputs:  Psi - (N x n_coeff) basis matrix
        
            % if coeff_idx is not specified/empty, set to entire basis
            if ~exist('multi_idxs','var') || isempty(multi_idxs)
                multi_idxs = self.multi_idxs;
            end
            if size(multi_idxs,2) ~= self.dim
                error('multi_idxs are not correctly specified')
            end
            % if dims is not specified, set to all dimensions
            if ~exist('dims','var')
                dims = 1:self.dim;
            end
            
            % compute product of 1D basis functions for each dimension
            Psi = ones(size(X,1), size(multi_idxs,1));
            for j = dims
                m_idxs_j = multi_idxs(:,j);
                Psi_j = self.basis{j}.grad_vandermonde(X(:,j), ...
                    max(m_idxs_j), 0, self.is_orth);
                Psi = Psi .* Psi_j(:,m_idxs_j+1);
            end
            
        end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxkPsi = grad_xk_basis(self, X, k, grad_dim, dims, multi_idxs)
        % Concatenate basis and gradient evaluations of the base polynomial 
        % with respect to each dimension
        %
        % Inputs:   X - (N x dim) sample matrix
        %           k - the derivative order
        %           grad_dim - variables for which derivatives are computed
        %                      e.g., grad_dim = dim for gradients of x_d
        %           dims - basis functions included in product
        %                  e.g., dims = 1:self.dim-1 for off-diagonals
        %           multi_idxs - indices of basis functions
        % Outputs:  dxkPsi - (N x n_coeff) basis matrix
        
            % if coeff_idx is not specified/empty, set to entire basis
            if ~exist('multi_idxs','var') || isempty(multi_idxs)
                multi_idxs = self.multi_idxs;
            end
            % if dims is not specified, set to all dimensions
            if ~exist('dims','var')
                dims = 1:self.dim;
            end
            % check inputs
            if any(grad_dim < 1) || any(grad_dim > self.dim)
                error('MP: grad_dim must be one of dimensions')
            end
            if k <= 0
                error('MP: derivative order must be greater than 0')
            end
            
            % compute product of 1D basis functions for each dimension
            dxkPsi = ones(size(X,1), size(multi_idxs,1));
            for j = dims
                m_idxs_j = multi_idxs(:,j);
                if any(j==grad_dim)
                    Psi_j = self.basis{j}.grad_vandermonde(X(:,j), ...
                        max(m_idxs_j), k, self.is_orth);
                else
                    Psi_j = self.basis{j}.grad_vandermonde(X(:,j), ...
                        max(m_idxs_j), 0, self.is_orth);
                end
                dxkPsi = dxkPsi .* Psi_j(:,m_idxs_j+1);
            end
            
        end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxPsi = grad_x_basis(self, X, grad_dim, dims, multi_idxs)
        % Inputs:   X - (N x dim) sample matrix
        %           grad_dim - variables for which derivatives are computed
        %                      e.g., grad_dim = dim for gradients of x_d
        %           dims - basis functions included in product
        %                  e.g., dims = 1:self.dim-1 for off-diagonals
        %           multi_idxs - indices of basis functions
        % Outputs:  dxPsi - (N x len(dims) x n_coeff) matrix where
        %                   dxPsi(:,l,j) = \partial_{x_{j}} \Psi_{l} 
            
            % if multi_idxs is not specified/empty - set to entire basis
            if ~exist('multi_idxs','var') || isempty(multi_idxs)
                multi_idxs = self.multi_idxs;
            end
            % if dims is not specified, set to all dimensions
            if ~exist('dims','var')
                dims = 1:self.dim;
            end
            % if dims is not specified, set to all dimensions
            if ~exist('grad_dim','var') || isempty(grad_dim)
                grad_dim = 1:self.dim;
            end
            
            % evaluate gradients with respect to each dimension
            dxPsi = zeros(size(X,1), size(multi_idxs,1), length(grad_dim));
            for i = 1:length(grad_dim)
                d_i = grad_dim(i);
                dxPsi(:,:,i) = self.grad_xk_basis(X, 1, d_i, dims, multi_idxs);
            end
            
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2xPsi = hess_x_basis(self, X, grad_dim, dims, multi_idxs)
        % Inputs:   X - (N x dim) sample matrix
        %           grad_dim - variables for which derivatives are computed
        %                      e.g., grad_dim = dim for gradients of x_d
        %           dims - basis functions included in product
        %                  e.g., dims = 1:self.dim-1 for off-diagonals
        %           multi_idxs - indices of basis functions
        % Outputs:  d2xPsi - (N x len(dims) x n_coeff) matrix where
        %                   d2xPsi(:,l,k,j) = \nabla_x_j \nabla_x_k \Psi_l            

            % if coeff_idx is not specified/empty - set to entire basis
            if ~exist('multi_idxs','var') || isempty(multi_idxs)
                multi_idxs = self.multi_idxs;
            end
            % if dims is not specified, set to all dimensions
            if ~exist('dims','var')
                dims = 1:self.dim;
            end
            % if dims is not specified, set to all dimensions
            if ~exist('grad_dim','var') || isempty(grad_dim)
                grad_dim = 1:self.dim;
            end

            % evaluate second derivatives with respect to each dimension
            d2xPsi = zeros(size(X,1), size(multi_idxs,1), length(grad_dim), length(grad_dim));
            % fill in diagonal (second derivatives)
            for i = 1:length(grad_dim)
                d_i = grad_dim(i);
                d2xPsi(:,:,i,i) = self.grad_xk_basis(X, 2, d_i, dims, multi_idxs);
            end
            % fill upper/lower part with mixed partial derivatives
            for i = 1:length(grad_dim)
                d_i = grad_dim(i);
                for j = i+1:length(grad_dim)
                    d_j = grad_dim(j);
                    d2xPsi(:,:,i,j) = self.grad_xk_basis(X, 1, [d_i,d_j], dims, multi_idxs);
                    d2xPsi(:,:,j,i) = d2xPsi(:,:,i,j);
                end
            end
            
        end %endFunction
        %------------------------------------------------------------------
    end %endMethods

end %endClass

% -- END OF FILE --