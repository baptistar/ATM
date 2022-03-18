classdef PPprecomp < matlab.mixin.Copyable%handle
    
    % Define struct to store pre-computed values for MultivariatePoly, 
    % ParametericPoly, and IntegratedPositiveFunction classes
    %
    % Author: Ricardo Baptista
    % Date: May 2020
    
    properties

        X                    % input evaluations

        eval_diagbasis0      % diagonal basis evaluations at x_d = 0
        eval_basis0          % basis evaluations at x_d = 0
        grad_x_basis0        % derivatives of all variables at x_d = 0

        eval_offdiagbasis    % off-diagonal basis evalutions
        eval_diagbasis       % diagonal basis evaluations
        eval_basis           % basis evaluations

        grad_xd_diagbasis    % derivatives of diagonal basis
        hess_xd_diagbasis    % second derivatives of diagonal basis
        grad_xd_basis        % derivatives of basis wrt x_d
        hess_xd_basis        % seond derivatives of basis wrt x_d
        
        grad_x_offdiagbasis  % derivatives of all variables
        hess_x_offdiagbasis  % second derivatives of all variables
        grad_x_basis         % derivatives of all variables
        hess_x_basis         % seond derivatives of all variables

        quad_dxPsidi         % evaluation quad points for Psid on [0,x_d]
        quad_dxPsii          % evaluation quad points for Psi on [0,x_d]
        quad_xi              % evaluations of d_{xd} diagbasis on [0,x_d]
        quad_wi              % weights for integration

        pts_per_level        % points per integration level
        max_levels           % maximum number of levels
        num_levels           % number of pre-computed levels
        tol
        
    end
    
    methods
        % -----------------------------------------------------------------
        function precomp = PPprecomp()
            
            % set defauts for integration
            precomp.pts_per_level = 4;
            precomp.max_levels = 12;
            precomp.num_levels = 2;
            precomp.tol = [1e-3, 1e-3];
            
            precomp.quad_dxPsidi = cell(precomp.max_levels,1);
            precomp.quad_xi      = cell(precomp.max_levels,1);
            precomp.quad_wi      = cell(precomp.max_levels,1);

            precomp.quad_dxPsii  = cell(precomp.max_levels,1);
            
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function self = evaluate(self, f, X)

            % save inputs
            self.X = X;
            
            % evaluate off-diagonal and diagonal basis
            self.eval_offdiagbasis = f.evaluate_offdiagbasis(X);
            self.eval_diagbasis = f.evaluate_diagbasis(X);
            self.eval_basis = (self.eval_offdiagbasis .* self.eval_diagbasis); 
            
            % evaluate gradient of diagonal basis
            self.grad_xd_diagbasis = f.grad_xd_diagbasis(X);
            self.grad_xd_basis = (self.eval_offdiagbasis .* self.grad_xd_diagbasis); 
            
            % evaluate basis at X0
            X0 = [X(:,1:end-1), zeros(size(X,1),1)];
            self.eval_diagbasis0 = f.evaluate_diagbasis(X0);
            self.eval_basis0 = (self.eval_offdiagbasis .* self.eval_diagbasis0);
            
            % evaluate integral points for the first two levels
            for l=1:self.num_levels
                n_pts = self.pts_per_level * 2^(l-1);
                [self.quad_dxPsii{l}, self.quad_wi{l}, self.quad_xi{l}] = ...
                    self.evaluate_quadrature_Psi(f, X, n_pts);
            end

        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function [dxdfdi, wi, zi] = evaluate_quadrature_Psid(~, f, X, N)

            % extract CC nodes and weights
            [xcc, wcc] = clenshaw_curtis(N);

            % sum contribution from each integration node
            zi = cell(N,1); dxdfdi = cell(N,1); wi = cell(N,1);
            for j=1:N
                % extract a,b endpoints and rescale points to interval
                a = zeros(size(X,1),1);
                b = X(:,end);
                [zi{j}, wi{j}] = rescale_pts(a, b, xcc(j), wcc(j));
                % evaluate basis functions
                max_degree = max(f.multi_idxs(:,end));
                dxdPsi_j = f.basis{end}.grad_vandermonde(zi{j}, max_degree, 1, f.is_orth);
                dxdfdi{j} = dxdPsi_j(:, f.multi_idxs(:,end)+1);
            end

        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function [dxdfi, wi, zi] = evaluate_quadrature_Psi(self, f, X, N)
            
            % check if off-diagonal basis is empty
            if isempty(self.eval_offdiagbasis)
                self.eval_offdiagbasis = f.evaluate_offdiagbasis(X);
            end

            % compute diagonal basis functions
            [dxdfdi, wi, zi] = self.evaluate_quadrature_Psid(f, X, N);
            
            % compute products with off-diagonal basis functions
            dxdfi = cell(N,1);
            for j=1:N
                dxdfi{j} = self.eval_offdiagbasis .* dxdfdi{j};
            end
            
        end %endFunction
        % -----------------------------------------------------------------
        % -----------------------------------------------------------------
        function self = update_precomp(self, f, X, new_multi_idxs)
            
            % evalaute new off-diagonal and diagonal basis functions
            new_offdiagbasis = f.evaluate_offdiagbasis(X, new_multi_idxs);
            new_diagbasis = f.evaluate_diagbasis(X, new_multi_idxs);

            % append new basis functions
            self.eval_offdiagbasis = [self.eval_offdiagbasis, new_offdiagbasis];
            self.eval_diagbasis = [self.eval_diagbasis, new_diagbasis];
            self.eval_basis = [self.eval_basis, new_offdiagbasis .* new_diagbasis]; 

            % evaluate gradient of new diagonal basis functions
            new_grad_xd_diagbasis = f.grad_xd_diagbasis(X, new_multi_idxs);
            self.grad_xd_diagbasis = [self.grad_xd_diagbasis, new_grad_xd_diagbasis];
            self.grad_xd_basis = [self.grad_xd_basis, new_offdiagbasis .* new_grad_xd_diagbasis];

            % update evaluations of Psi0
            X0 = [X(:,1:end-1), zeros(size(X,1),1)];
            new_diagbasis0 = f.evaluate_diagbasis(X0, new_multi_idxs);
            self.eval_diagbasis0 = [self.eval_diagbasis0, new_diagbasis0];
            self.eval_basis0 = [self.eval_basis0, new_offdiagbasis .* new_diagbasis0];

            % update Psi(x) points for all non-zero quadrature points
            for l=1:self.max_levels
                % check if array is empty
                if isempty(self.quad_dxPsii{l})
                    continue
                end
                for i=1:length(self.quad_xi{l})
                    dxdPsi_i = zeros(size(self.quad_xi{l}{i},1), size(new_multi_idxs,1));
                    for j=1:size(new_multi_idxs,1)
                        dxdPsi_i(:,j) = f.basis{end}.grad_x(self.quad_xi{l}{i}, ...
                            new_multi_idxs(j,end), 1, f.is_orth);
                    end
                    self.quad_dxPsii{l}{i} = [self.quad_dxPsii{l}{i}, new_offdiagbasis .* dxdPsi_i];
                end
            end

        end %endFunction
        % -----------------------------------------------------------------
    end %endMethods
    
end %endClass

% -- END OF FILE --