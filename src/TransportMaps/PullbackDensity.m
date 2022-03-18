% Author: Ricardo Baptista and Olivier Zahm and Youssef Marzouk
% Date:   May 2020
%
% See LICENSE.md for copyright information
%

classdef PullbackDensity
    % Define a transport map S of dimension d that couples a target
    % and reference density by learning the transport map from samples

    properties
        S   % {d x 1} cell of map components
        ref % global reference density
    end

    methods  
        %------------------------------------------------------------------
        function PB = PullbackDensity(S, ref)
            % For cell S, redefine using TriangularTransportMap
            if iscell(S)
                S = TriangularTransportMap(S);
            elseif ~isa(S, 'TriangularTransportMap')
		error('Define S as TriangularTransportMap or cell')
	    end
            % assign S and ref
            PB.S = S;
            PB.ref = ref;
            % check dimension of S and ref
            %if PB.S.d ~= PB.ref.d
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d = d(self)
            d = self.S.d;
        end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function log_pi = log_pdf(self, X, comp_idx)
        % evaluate: log pullback of density defined by transport map S(x)
        %
        % Inputs:  X - (N x d) sample matrix
        %          comp_idx - (optional) list of k components to evaluate 
        % Outputs: log_pi - (N x 1) matrix of density evaluations

            % if not specified, evaluate all components of transport map
            if (nargin < 3)
                comp_idx = 1:self.S.d;
            end
            % check dimensions of inputs
            if size(X,2) ~= self.S.d
                error('PB: dimension mismatch for input samples')
            end
                        
            % evaluate S(x) and \partial_xd S(x)
            Sx = self.S.evaluate(X);
            log_dxSx = self.S.logdet_Jacobian(X, comp_idx);

            % evaluate pullback density log_pi(x)
            log_pi = self.ref.log_pdf(Sx, comp_idx) + log_dxSx;

        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dx_log_pi = grad_x_log_pdf(self, X, grad_dim, comp_idx)
            % if not specified, evaluate all components of transport map
            if (nargin < 4)
                comp_idx = 1:self.S.d;
            end
            % if not specified, compute gradients for all variables
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:self.S.d;
            end
            % check dimensions of inputs
            if size(X,2) ~= self.S.d
                error('PB: dimension mismatch for input samples')
            end
            
            % evaluate S, \nabla_x S, \nabla^2_x S, extract \partial_xd S
            Sx = self.S.evaluate(X);
            dxSx = self.S.grad_x(X, grad_dim);
            dx_logeta = self.ref.grad_x_log_pdf(Sx, [], comp_idx);
            dxlogdS = self.S.grad_x_logdet_Jacobian(X, grad_dim, comp_idx);
            
            % evaluate gradient
            dx_logetaS = dx_logeta .* dxSx;
            dx_log_pi = squeeze(sum(dx_logetaS,2)) + dxlogdS;

        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
%         function d2xDJ = hess_x_logdet_Jacobian(self, X, grad_dim, comp_idx)
%             % if not specified, evaluate all components of transport map
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
%         %------------------------------------------------------------------
%         %------------------------------------------------------------------
%         function d2x_log_pi = hess_x_log_pdf(self, X, grad_dim, comp_idx)
%             %error('hess_x_log_pdf is not yet implemented!')
%             % if not specified, evaluate all components of transport map
%             if (nargin < 4)
%                 comp_idx = 1:self.d;
%             end
%             % if not specified, compute gradients for all variables
%             if (nargin < 3)
%                 grad_dim = 1:self.d;
%             end
%             % check dimensions of inputs
%             if size(X,2) ~= self.d
%                 error('PB: dimension mismatch for input samples')
%             end
%             
%             % evaluate S, \nabla_x S, \nabla^2_x S, extract \partial_xd S
%             Sx = self.evaluate(X, comp_idx);
%             dxSx = self.grad_x(X, grad_dim, comp_idx);
%             d2xSx = self.hess_x(X, grad_dim, comp_idx);
%             dx_logeta = self.ref.grad_x_log_pdf(Sx, grad_dim, comp_idx);
%             d2x_logeta = self.ref.hess_x_log_pdf(Sx, grad_dim, comp_idx);
%             d2xlogdS = self.hess_x_logdet_Jacobian(self, X, grad_dim, comp_idx);
%             
%             % evaluate gradient
%             d2x_log_pi = d2x_logeta .* OuterProd(dxSx, dxSx) + ...
%                         dx_logeta .* d2xSx + d2xlogdS;
% 
%         end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function [self, output] = optimize(self, XW, comp_idx)
            % check if XW is a structure containing (points, weights), 
            % otherwise set weights to 1/N
            if isa(XW,'struct')
                X=XW.X;
                W=XW.W;
            else
                X=XW;
                W=(1/size(XW,1))*ones(size(XW,1),1);
            end
            % if not specified, optimize all components of transport map
            if (nargin < 3)
                comp_idx = 1:self.S.d;
            end
            % check dimensions of input samples
            if size(X,2) ~= self.S.d
                error('PB: dimension mismatch for input samples X')
            end
            
            % optimize components seperately if reference is product measure
            if isa(self.ref, 'IndependentProductDitribution')
                output = cell(self.S.d, 1);
                for k=comp_idx
                    XW_k.X=X(:,1:k);
                    XW_k.W=W;
                    % extract initial condition if coefficients are set
                    if ~isempty(self.S.S{k}.coeff)
                        a0 = (self.S.S{k}.coeff).';
                    else
                        a0 = zeros(self.S.S{k}.n_coeff, 1);
                    end
                    % optimize variables
                    [self.S.S{k}, output{k}] = optimize_component( ...
                        self.S.S{k}, self.ref.factors{k}, a0, XW_k);
                end
            else
                if ~isa(self.S, 'TriangularTransportMap')
                    error('Define S using TriangularTransportMap')
                end
                a0 = self.S.coeff;
                self.S = optimize_map(self.S, self.ref, a0, XW);
            end
            
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function [self, output] = greedy_optimize(self, XW, XWvalid, ...
                                    max_terms, stopping, comp_idx)
            % check if XW is a structure containing (points, weights), 
            % otherwise set weights to 1/N
            if isa(XW,'struct')
                X=XW.X;
                W=XW.W;
            else
                X=XW;
                W=(1/size(XW,1))*ones(size(XW,1),1);
            end
            % if not specified, optimize all components of transport map
            if (nargin < 6)
                comp_idx = 1:self.S.d;
            end
            if (nargin < 5)
                stopping = 'max_terms';
            end
            % check dimensions of input samples
            if size(X,2) ~= self.S.d
                error('PB: dimension mismatch for input samples X')
            end
            % check if max_terms is a vector
            if isscalar(max_terms)
                max_terms = max_terms*ones(self.S.d,1);
            end
            % check if XWvalid is a structure containing (points, weights), 
            % otherwise set weights to 1/N
            if (nargin >= 3) && isa(XWvalid,'struct')
                Xvalid=XWvalid.X;
                Wvalid=XWvalid.W;
            elseif (nargin >= 3) && ~isempty(XWvalid)
                Xvalid=XWvalid;
                Wvalid=(1/size(XWvalid,1))*ones(size(XWvalid,1),1);
            else
                Xvalid = zeros(0, self.S.d);
                Wvalid=(1/size(Xvalid,1))*ones(size(Xvalid,1),1);
            end
            
            % optimize components seperately if reference is product measure
            if isa(self.ref, 'IndependentProductDistribution')
                output = cell(self.S.d, 1);
                for k=comp_idx
                    % subselect points for component k                    
                    XW_k.X=X(:,1:k);
                    XW_k.W=W;
                    XWvalid_k.X=Xvalid(:,1:k);
                    XWvalid_k.W=Wvalid;                   
                    fprintf('Optimizing component %d\n',k)
                    [self.S.S{k}, output{k}] = greedy_basis_selection( ...
                        self.S.S{k}, self.ref.factors{k}, XW_k, ...
                        XWvalid_k, max_terms(k),  stopping);
                end
            else
                %[self.S{1}, output] = greedy_basis_selection( ...
                %        self.S{1}, self.ref, X, Xvalid, max_terms, stopping);
                error('Global greedy optimization is not yet implemented')
            end
            
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function [self, output] = greedy_optimize_regression(self,Sref,XW, ...
                                    max_terms, tol)
            % check if XW is a structure containing (points, weights), 
            % otherwise set weights to 1/N
            if isa(XW,'struct')
                X=XW.X;
                W=XW.W;
            else
                X=XW;
                W=(1/size(XW,1))*ones(size(XW,1),1);
            end
            % if not specified, optimize all components of transport map
            comp_idx = 1:self.S.d;
            % check dimensions of input samples
            if size(X,2) ~= self.S.d
                error('PB: dimension mismatch for input samples X')
            end
            % check if max_terms is a vector
            if isscalar(max_terms)
                max_terms = max_terms*ones(self.S.d,1);
            end
            if isscalar(tol)
                tol = tol*ones(self.S.d,1);
            end
            
            % optimize components seperately if reference is product measure
            if isa(self.ref, 'IndependentProductDistribution')
                output = cell(self.S.d, 1);
                for k=comp_idx
                    % subselect points for component k                    
                    XW_k.X=X(:,1:k);
                    XW_k.W=W;
                    
                    Sref_k=Sref.evaluate(X,k); 
                    
                    fprintf('Optimizing component %d\n',k)
                    [self.S.S{k}, output{k}] = greedy_regression( ...
                        self.S.S{k}, Sref_k, XW_k, ...
                         max_terms(k), tol(k));
                end
            else
                %[self.S{1}, output] = greedy_basis_selection( ...
                %        self.S{1}, self.ref, X, Xvalid, max_terms, stopping);
                error('Global greedy optimization is not yet implemented')
            end
            
        end %endFunction
        %------------------------------------------------------------------
    end %endMethods
end %endClass
