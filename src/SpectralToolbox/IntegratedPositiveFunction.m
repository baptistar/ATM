classdef IntegratedPositiveFunction

    % Class defines the monotonic function in d-dimensions:
    %   x -> R(f)(x) = f(x_{1:d-1},0) + \int_0^x_d g(\partial_{x_d} f) dt
    % where f is a linear expansion  (i.e., ParametrizedPoly), and
    %       g is a positive function (i.e., RectifierOfPoly)
    %
    % Methods: evaluate, grad_x, hess_x, 
    %          grad_xd, hess_xd, grad_x_grad_xd, 
    %          grad_coeff, hess_coeff, 
    %          grad_coeff_grad_xd, hess_coeff_grad_xd,
    %          inverse
    %
    % Author: Ricardo Baptista, Olivier Zahm
    % Date:   August 2020

    properties
        f        % object for linear expansion
        rec      % object for rectifier
    end

    methods (Access = public)
        function self = IntegratedPositiveFunction(f, rec_type)%, n_quad, n_levels)
                        
            % check f 
            if ~isa(f, 'ParametericPoly') && ~isa(f, 'NonParametericPoly')
                error('f must be ParametericPoly or NonParametericPoly object')
            end
            
            % assign f and rec to object
            self.f = f;
            if nargin < 2
                 rec_type = 'softplus';
            end
            self.rec = Rectifier(rec_type);

            % assign integration properties
            % if nargin < 3
            %     n_quad = 12;
            % end
            % self.n_quad = n_quad;
            % if nargin < 4
            %     n_levels = 1;
            % end
            % self.n_levels = n_levels;

        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d = dim(self)
            d = self.f.dim;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function nc = n_coeff(self)
            nc = self.f.n_coeff;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function c = coeff(self)
            c = self.f.coeff;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function self = set_coeff(self, c)
            self.f = self.f.set_coeff(c);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function m_idxs = multi_idxs(self)
            m_idxs = self.f.multi_idxs();
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function self = set_multi_idxs(self, m_idxs)
            self.f.multi_idxs = m_idxs;
        end %endFunction
        function check_inputs(self, X)
            assert(size(X,2) == self.f.dim);
        end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function S = evaluate(self, X, precomp)
            if (nargin < 3)
                precomp = PPprecomp();
            end
            self.check_inputs(X)
            % evaluate f(x_{1:d-1},0) & \int_0^x_d g(\partial_x_d f) dt
            f0 = self.f.evaluate_f0(X, precomp);
            If = self.integrate_xd(X, precomp);
            % evaluate sum of terms
            S = f0 + If;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxS = grad_x(self, X, grad_dim, precomp)
            if (nargin < 4)
                precomp = PPprecomp();
            end
            self.check_inputs(X)
            % if grad_dim is not specified/empty, set to all dimensions
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:self.dim;
            end
            % evaluate \nabla_x f(x_{1:d-1},0) & 
            %          \nabla_x \int_0^xd g(\partial_x_d f) dt
            dxf0 = self.f.grad_x_evaluate_f0(X, grad_dim, precomp);
            dxIf = self.grad_x_integrate_xd(X, grad_dim, precomp);
            % evaluate sum of terms
            dxS = dxf0 + dxIf;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2xS = hess_x(self, X, grad_dim, precomp)
            if (nargin < 4)
                precomp = PPprecomp();
            end
            self.check_inputs(X)
            % if grad_dim is not specified/empty, set to all dimensions
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:self.dim;
            end
            % evaluate \nabla^2_x f(x_{1:d-1},0) & 
            %          \nabla^2_x \int_0^xd g(\partial_x_d f) dt
            d2xf0 = self.f.hess_x_evaluate_f0(X, grad_dim, precomp);
            d2xIf = self.hess_x_integrate_xd(X, grad_dim, precomp);
            % evaluate sum of terms
            d2xS = d2xf0 + d2xIf;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxdS = grad_xd(self, X, precomp)
            if (nargin < 3)
                precomp = PPprecomp();
            end
            self.check_inputs(X)
            % evaluate g(\partial_x_d f)
            dxf = self.f.grad_xd(X, precomp);
            dxdS = self.rec.evaluate(dxf);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2xdS = hess_xd(self, X, precomp)
            if (nargin < 3)
                precomp = PPprecomp();
            end
            self.check_inputs(X)
            % evaluate \partial_x_d f and \partial^2_x_d f
            dxdf = self.f.grad_xd(X, precomp);
            d2xdf = self.f.hess_xd(X, precomp);
            % evaluate g'(\partial_x_d f) * \partial^2_x_d f
            d2xdS = self.rec.grad_x(dxdf) .* d2xdf;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxdxdS = grad_x_grad_xd(self, X, grad_dim, precomp)
            if (nargin < 4)
                precomp = PPprecomp();
            end
            % if grad_dim is not specified/empty, set to all dimensions
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:self.dim;
            end
            self.check_inputs(X)
            % evaluate \partial_x_d f & \nabla_x \partial_x_d f
            dxdf = self.f.grad_xd(X, precomp);
            dxdxdf = self.f.grad_x_grad_xd(X, grad_dim, precomp);
            % evaluate \nabla_x_j g(\partial_x_d f)
            dxdxdS = self.rec.grad_x(dxdf) .* dxdxdf;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2xdxdS = hess_x_grad_xd(self, X, grad_dim, precomp)
            if (nargin < 4)
                precomp = PPprecomp();
            end
            % if grad_dim is not specified/empty, set to all dimensions
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:self.dim;
            end
            self.check_inputs(X)
            % evaluate \partial_x_d f & \nabla_x \partial_x_d f
            dxdf = self.f.grad_xd(X, precomp);
            dxdxdf = self.f.grad_x_grad_xd(X, grad_dim, precomp);
            d2xdxdf = self.f.hess_x_grad_xd(X, grad_dim, precomp);
            % evaluate \nabla_x_j g(\partial_x_d f)
            d2xdxdS = self.rec.grad_x(dxdf) .* d2xdxdf + ...
                self.rec.hess_x(dxdf) .* OuterProd(dxdxdf, dxdxdf);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dcS = grad_coeff(self, X, coeff_idx, precomp)
            if (nargin < 4)
                precomp = PPprecomp();
            end
            self.check_inputs(X)
            % set coeff_idx to all indices if it is not specified
            if (nargin < 3) || isempty(coeff_idx)
                coeff_idx = 1:self.n_coeff;
            elseif any(coeff_idx <= 0) || any(coeff_idx > self.n_coeff)
                error('coeff_idx is not in allowable range')
            end
            % evaluate \partial_c of f0 & \int_0^x_d g(\partial_xd f) dt
            dcf0 = self.f.grad_coeff_f0(X, coeff_idx, precomp);
            dcIf = self.grad_coeff_integrate_xd(X, coeff_idx, precomp);
            % evaluate sum of terms
            dcS  = dcf0 + dcIf;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2cS = hess_coeff(self, X, coeff_idx, precomp)
            if (nargin < 4)
                precomp = PPprecomp();
            end
            self.check_inputs(X)
            % set coeff_idx to all coefficients if it is not specified
            if (nargin < 3) || isempty(coeff_idx)
                coeff_idx = 1:self.n_coeff;
            elseif any(coeff_idx <= 0) || any(coeff_idx > self.n_coeff)
                error('coeff_idx is not in allowable range')
            end
            % evaluate \nabla^2_c f0 & \int_0^x_d g(\partial_xd f) dt
            d2cf0 = self.f.hess_coeff_f0(X, coeff_idx, precomp);
            d2cIf = self.hess_coeff_integrate_xd(X, coeff_idx, precomp);
            % evaluate sum of terms
            d2cS  = d2cf0 + d2cIf;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dcdxS = grad_coeff_grad_xd(self, X, coeff_idx, precomp)
            if (nargin < 4)
                precomp = PPprecomp();
            end
            % set coeff_idx to all coefficients if it is not specified
            if (nargin < 3) || isempty(coeff_idx)
                coeff_idx = 1:self.n_coeff;
            elseif any(coeff_idx <= 0) || any(coeff_idx > self.n_coeff)
                error('coeff_idx is not in allowable range')
            end
            self.check_inputs(X)
            % evaluate \partial_xd f & \nabla_c \partial_xd f
            dxf = self.f.grad_xd(X, precomp);
            dcdxf = self.f.grad_coeff_grad_xd(X, coeff_idx, precomp);
            % evaluate g'(\partial_xd f) * \nabla_c \partial_xd f
            dcdxS = self.rec.grad_x(dxf) .* dcdxf;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2cS = hess_coeff_grad_xd(self, X, coeff_idx, precomp)
            if (nargin < 4)
                precomp = PPprecomp();
            end
            % set coeff_idx to all coefficients if it is not specified
            if (nargin < 3) || isempty(coeff_idx)
                coeff_idx = 1:self.n_coeff;
            elseif any(coeff_idx <= 0) || any(coeff_idx > self.n_coeff)
                error('coeff_idx is not in allowable range')
            end
            self.check_inputs(X)
            % evaluate \partial_xd f & \nabla_c \partial_xd f
            dxf = self.f.grad_xd(X, precomp);
            dcdxf = self.f.grad_coeff_grad_xd(X, coeff_idx, precomp);
            % evaluate g''(d_xd f) (\nabla_c d_xd f )( \nabla_c d_xd f)^T
            d2cS = OuterProd( self.rec.hess_x(dxf) .* dcdxf, dcdxf);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function X = inverse(self, Xdm1, Z, precomp)
            % invert function S(x_{1:d-1},X) = Z for each sample
            if (nargin < 4)
                precomp = PPprecomp();
            end
            % check dimension of input samples X and Z
            if size(Xdm1,1) ~= size(Z,1)
                error('Different sample sizes provided to X and Z')
            end
            if size(Z,2) ~= 1
                error('Too many dimensions in output samples')
            end
            % define objective
            Zs = Z - self.f.evaluate_f0([Xdm1, zeros(size(Xdm1,1),1)], precomp);
            fun = @(X) self.inverse_fn(X, Xdm1, Zs, precomp);
            % determine bounds for inversion
            a = -ones(size(Z,1),1);
            b = ones(size(Z,1),1);
            fa = fun(a);
            fb = fun(b);
            while(any(fa .* fb > 0.0))
                for i=1:size(Z,1)
                    if fa(i) * fb(i) > 0.0
                        deltai = 0.5*(b(i) - a(i));
                        if fa(i) > 0
                            a(i) = a(i) - deltai;
                        elseif fb(i) < 0
                            b(i) = b(i) + deltai;
                        end
                    end
                end
                fa = fun(a);
                fb = fun(b);
            end
            % call hybrid Newton-bisection solver
            [X,fXk,~] = hybridRootFindingSolver(fun, a, b);
            % check error
            if any(fXk > 1e-6)
                warning('Inversion did not converge: Max error %f\n', max(fXk));
            end
        end %endFunction
        %------------------------------------------------------------------
    end %endMethods
        
    methods (Access = private)
        %------------------------------------------------------------------
        function IntF = integrate_xd(self, X, precomp)
            % evaluate off-diagonal basis
            if isempty(precomp.eval_offdiagbasis)
                precomp.eval_offdiagbasis = self.f.evaluate_offdiagbasis(X);
            end
            % define quadrature points
            quad_pts = @(N) precomp.evaluate_quadrature_Psi(self.f, X, N);
            % define function to be integrated
            dcdxdS = @(dxPsi) self.rec.evaluate( dxPsi * self.coeff.' );
            % evaluate \int_0^x_d g(\partial_x_d f) dt
            [IntF, precomp.quad_dxPsii, precomp.quad_xi, precomp.quad_wi] = ...
                adaptive_integral(dcdxdS, quad_pts, ...
                precomp.quad_dxPsii, precomp.quad_xi, precomp.quad_wi, ...
                precomp.tol, precomp.pts_per_level, precomp.max_levels);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxIntF = grad_x_integrate_xd(self, X, grad_dim, precomp)
            % get index of self.dim in grad_dim
            dim_idx = (grad_dim == self.dim);
            grad_dim_md = grad_dim(~dim_idx);
            % evaluate off-diagonal basis evaluations and 
            if ~isempty(precomp.eval_offdiagbasis)
                Psio = precomp.eval_offdiagbasis;
            else
                Psio = self.f.evaluate_offdiagbasis(X);
            end
            if ~isempty(precomp.grad_x_offdiagbasis)
                dxPsio = precomp.grad_x_offdiagbasis;
            else
                dxPsio = self.f.grad_x_offdiagbasis(X);
            end
            % define quadrature points
            quad_pts = @(N) precomp.evaluate_quadrature_Psid(self.f, X, N);
            % evaluate \int_0^x_d \nabla_x g(\partial_x_d f) dt
            dxIntF = zeros(size(X,1), length(grad_dim));
            for j = 1:length(grad_dim_md)
                d_j = grad_dim_md(j);
                % define function to be integrated
                dxf = @(dxPsid) (Psio .* dxPsid) * self.coeff.';
                dxjdf = @(dxPsid) (dxPsio(:,:,d_j) .* dxPsid) * self.coeff.';
                dxjS = @(dxPsid) self.rec.grad_x( dxf(dxPsid) ) .* dxjdf(dxPsid);
                % evaluate \int_0^x_d g(\partial_x_d f) dt
                [dxIntF(:,j), precomp.quad_dxPsidi, precomp.quad_xi, precomp.quad_wi] = ...
                    adaptive_integral(dxjS, quad_pts, ...
                    precomp.quad_dxPsidi, precomp.quad_xi, precomp.quad_wi, ...
                    precomp.tol, precomp.pts_per_level, precomp.max_levels);
            end
            % add term for  \nabla^2_x \int_0^x_d g(\partial_x_d) f
            if any(dim_idx)
                dxIntF(:,dim_idx) = self.grad_xd(X, precomp);
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2xIntF = hess_x_integrate_xd(self, X, grad_dim, precomp)
            % get index of self.dim in grad_dim
            dim_idx = (grad_dim == self.dim);
            grad_dim_md = grad_dim(~dim_idx);
            % evaluate off-diagonal basis evaluations and 
            if ~isempty(precomp.eval_offdiagbasis)
                Psio = precomp.eval_offdiagbasis;
            else
                Psio = self.f.evaluate_offdiagbasis(X);
            end
            if ~isempty(precomp.grad_x_offdiagbasis)
                dxPsio = precomp.grad_x_offdiagbasis;
            else
                dxPsio = self.f.grad_x_offdiagbasis(X);
            end
            if ~isempty(precomp.hess_x_offdiagbasis)
                d2xPsio = precomp.hess_x_offdiagbasis;
            else
                d2xPsio = self.f.hess_x_offdiagbasis(X);
            end
            % define quadrature points
            quad_pts = @(N) precomp.evaluate_quadrature_Psid(self.f, X, N);
            % evaluate \int_0^x_d \nabla_x g(\partial_x_d f) dt
            d2xIntF = zeros(size(X,1), length(grad_dim), length(grad_dim));
            for i = 1:length(grad_dim_md)
                for j = 1:length(grad_dim_md)
                    d_i = grad_dim_md(i);
                    d_j = grad_dim_md(j);
                    % define function to be integrated
                    dxf = @(dxPsid) (Psio .* dxPsid) * self.coeff.';
                    dxidf = @(dxPsid) (dxPsio(:,:,d_i) .* dxPsid) * self.coeff.';
                    dxjdf = @(dxPsid) (dxPsio(:,:,d_j) .* dxPsid) * self.coeff.';
                    dxixjdf = @(dxPsid) (d2xPsio(:,:,d_i,d_j) .* dxPsid) * self.coeff.';
                    dxixjS = @(dxPsid) self.rec.grad_x( dxf(dxPsid) ) .* dxixjdf(dxPsid) + ...
                        self.rec.hess_x( dxf(dxPsid) ) .* dxidf(dxPsid) .* dxjdf(dxPsid);
                    % evaluate \int_0^x_d g(\partial_x_d f) dt
                    [d2xIntF(:,i,j), precomp.quad_dxPsidi, precomp.quad_xi, precomp.quad_wi] = ...
                        adaptive_integral(dxixjS, quad_pts, ...
                        precomp.quad_dxPsidi, precomp.quad_xi, precomp.quad_wi, ...
                        precomp.tol, precomp.pts_per_level, precomp.max_levels);
                    d2xIntF(:,j,i) = d2xIntF(:,i,j);
                end
            end
            % add term for  \nabla^2_x \int_0^x_d g(\partial_x_d) f
            if any(dim_idx)
                for i = 1:length(grad_dim_md)
                    d_i = grad_dim_md(i);
                    d2xIntF(:,dim_idx,i) = self.grad_x_grad_xd(X, d_i, precomp);
                    d2xIntF(:,i,dim_idx) = d2xIntF(:,dim_idx,i);
                end
                d2xIntF(:,dim_idx,dim_idx) = self.hess_xd(X, precomp);
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dcIntF = grad_coeff_integrate_xd(self, X, coeff_idx, precomp)
            % evaluate off-diagonal basis evaluations
            if isempty(precomp.eval_offdiagbasis)
                precomp.eval_offdiagbasis = self.f.evaluate_offdiagbasis(X);
            end
            % define quadrature points
            quad_pts = @(N) precomp.evaluate_quadrature_Psi(self.f, X, N);
            % define function to be integrated
            dcdxdS = @(dxPsi) self.rec.grad_x( dxPsi * self.coeff.' ) .* dxPsi(:, coeff_idx);
            % evaluate \int_0^x_d g(\partial_x_d f) dt
            [dcIntF, precomp.quad_dxPsii, precomp.quad_xi, precomp.quad_wi] = ...
                adaptive_integral(dcdxdS, quad_pts, ...
                precomp.quad_dxPsii, precomp.quad_xi, precomp.quad_wi, ...
                precomp.tol, precomp.pts_per_level, precomp.max_levels);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2cIntF = hess_coeff_integrate_xd(self, X, coeff_idx, precomp)
            % evaluate off-diagonal basis evaluations 
            if isempty(precomp.eval_offdiagbasis)
                precomp.eval_offdiagbasis = self.f.evaluate_offdiagbasis(X);
            end
            % define quadrature points
            quad_pts = @(N) precomp.evaluate_quadrature_Psi(self.f, X, N);
            % define function to be integrated
            dcdxdS = @(dxPsi) self.rec.hess_x( dxPsi * self.coeff.' ) .* ...
                        OuterProd(dxPsi(:,coeff_idx), dxPsi(:,coeff_idx));
            % evaluate \int_0^x_d g(\partial_x_d f) dt
            [d2cIntF, precomp.quad_dxPsii, precomp.quad_xi, precomp.quad_wi] = ...
                adaptive_integral(dcdxdS, quad_pts, ...
                precomp.quad_dxPsii, precomp.quad_xi, precomp.quad_wi, ...
                precomp.tol, precomp.pts_per_level, precomp.max_levels);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function [f,g] = inverse_fn(self, X, Xdm1, Zs, precomp)
            % discard quadrature points
            precomp.quad_dxPsidi = [];
            precomp.quad_wi = [];
            % evaluate integrals
            f = self.integrate_xd([Xdm1, X], precomp) - Zs; 
            if nargin > 1
                g = self.grad_xd([Xdm1, X], precomp);
            end
        end
        %------------------------------------------------------------------
    end %endMethods

end %endClass
