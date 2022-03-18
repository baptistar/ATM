classdef LinearFunction

    % Class defines the linear function S(x) = c + \sum_i^d L_i x_i
    % Note: class does not impose that L_d is positive
    %
    % Methods: evaluate, grad_x, hess_x, 
    %          grad_xd, hess_xd, grad_x_grad_xd, 
    %          grad_coeff, hess_coeff, 
    %          grad_coeff_grad_xd, hess_coeff_grad_xd,
    %          inverse
    %
    % Author: Ricardo Baptista
    % Date:   June 2021

    properties
        dim      % dimension
        L        % vector of coefficients
        c        % mean shift
    end

    methods (Access = public)
        function self = LinearFunction(c, L)
            self.dim = length(L);
            self = self.set_coeff(c, L);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function nc = n_coeff(self)
            nc = self.dim + 1;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function c = coeff(self)
            c = [self.c, self.L];
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function self = set_coeff(self, c, L)
            % check input dimensions
            assert(length(c) == 1)
            assert(length(L) == self.dim)
            % set constant
            self.c = c;
            % set L as a row vector if specified as a column
            if size(L,2) == 1
                L = L.';
            end
            self.L = L;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function check_inputs(self, X)
            assert(size(X,2) == self.dim);
        end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function S = evaluate(self, X, ~)
            self.check_inputs(X)
            S = [ones(size(X,1),1), X]*(self.coeff).';
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxS = grad_x(self, X, grad_dim, ~)
            self.check_inputs(X)
            % if grad_dim is not specified/empty, set to all dimensions
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:self.dim;
            end
            dxS = repmat(self.L(grad_dim), size(X,1), 1);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2xS = hess_x(self, X, grad_dim, ~)
            self.check_inputs(X)
            % if grad_dim is not specified/empty, set to all dimensions
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:self.dim;
            end
            % evaluate \nabla^2_x S(x)
            d2xS = zeros(size(X,1), length(grad_dim));
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxdS = grad_xd(self, X, ~)
            self.check_inputs(X)
            dxdS = self.L(self.dim) * ones(size(X,1), 1);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2xdS = hess_xd(self, X, ~)
            self.check_inputs(X)
            d2xdS = zeros(size(X,1),1);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxdxdS = grad_x_grad_xd(self, X, grad_dim, ~)
            % if grad_dim is not specified/empty, set to all dimensions
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:self.dim;
            end
            self.check_inputs(X)
            dxdxdS = zeros(size(X,1), length(grad_dim));
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dcS = grad_coeff(self, X, coeff_idx, ~)
            self.check_inputs(X)
            % set coeff_idx to all indices if it is not specified
            if (nargin < 3) || isempty(coeff_idx)
                coeff_idx = 1:self.n_coeff;
            elseif any(coeff_idx <= 0) || any(coeff_idx > self.n_coeff)
                error('coeff_idx is not in allowable range')
            end
            dcS = [ones(size(X,1),1), X];
            dcS = dcS(:,coeff_idx);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dcdxS = grad_coeff_grad_xd(self, X, coeff_idx, ~)
            % set coeff_idx to all coefficients if it is not specified
            if (nargin < 3) || isempty(coeff_idx)
                coeff_idx = 1:self.n_coeff;
            elseif any(coeff_idx <= 0) || any(coeff_idx > self.n_coeff)
                error('coeff_idx is not in allowable range')
            end
            self.check_inputs(X)
            dcdxS = zeros(size(X,1), length(coeff_idx));
            % grad_xd S = Ld where Ld = coeff(end)
            % check if coeff_idx includes the last variable
            if ismember(coeff_idx, self.dim+1)
                dcdxS(:, self.dim+1) = ones(size(X,1), 1);
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function Xd = inverse(self, Xdm1, Z, ~)
            % check dimension of input samples X and Z
            if size(Xdm1,1) ~= size(Z,1)
                error('Different sample sizes provided to X and Z')
            end
            if size(Xdm1,2) ~= (self.dim-1)
                error('Xdm1 is not of the correct dimension')
            end
            if size(Z,2) ~= 1
                error('Too many dimensions in output samples')
            end
            % compute inverse analytically
            Xd = Z - self.c - Xdm1 * self.L(1:self.dim-1).';
            Xd = Xd / self.L(self.dim);
        end %endFunction
        %------------------------------------------------------------------
    end %endMethods
        
end %endClass