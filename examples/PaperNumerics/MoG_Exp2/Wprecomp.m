classdef Wprecomp < matlab.mixin.Copyable%handle
    
    % Define struct to store pre-computed values for MultivariatePoly, 
    % ParametericPoly, and IntegratedPositiveFunction classes
    %
    % Author: Ricardo Baptista
    % Date: May 2020
    
    properties

        X                    % input evaluations
        basis                % define basis
        midxs                % multi-indices for basis
        Psi0                 % basis evaluations at X = 0
        Psi                  % basis evaluations at X
        grad_x_Psi           % gradient of basis evaluations

        quad_dxPsii          % evaluation quad points for Psi on [0,x_d]
        quad_xi              % evaluations of d_{xd} diagbasis on [0,x_d]
        quad_wi              % weights for integration

        pts_per_level        % points per integration level
        max_levels           % maximum number of levels
        num_levels           % number of pre-computed levels
        tol                  % integration tolerance
    end
    
    methods
        % -----------------------------------------------------------------
        function precomp = Wprecomp(basis)
            
            % assign basis
            precomp.basis = basis;
            
            % set defauts for integration
            precomp.max_levels = 10;
            precomp.num_levels = 2;
            precomp.pts_per_level = 4;
            precomp.tol = [1e-3,1e-3];
            
            % ddfine cells to store quadrature points
            precomp.quad_xi      = cell(precomp.max_levels,1);
            precomp.quad_wi      = cell(precomp.max_levels,1);
            precomp.quad_dxPsii  = cell(precomp.max_levels,1);
            
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function self = setup(self, X, midxs)

            % save inputs
            self.X = X;
            self.midxs = midxs;
            
            % evaluate basis at X0
            X0 = zeros(size(X,1),1);
            self.Psi0 = self.basis.grad_vandermonde(X0, midxs(:,1), midxs(:,2), 0);

            % evaluate basis
            self.Psi = self.basis.grad_vandermonde(X, midxs(:,1), midxs(:,2), 0);
            self.grad_x_Psi = self.basis.grad_vandermonde(X, midxs(:,1), midxs(:,2), 1);
            
            % evaluate integral points for the first two levels
            for l=1:self.num_levels
                n_pts = self.pts_per_level * 2^(l-1);
                [self.quad_dxPsii{l}, self.quad_wi{l}, self.quad_xi{l}] = ...
                    self.evaluate_quadrature(n_pts);
            end

        end %endFunction
        % -----------------------------------------------------------------
        % -----------------------------------------------------------------
        function self = update(self, new_midxs)
            
            % update multi indices
            self.midxs = [self.midxs; new_midxs];
            
            % evaluate new basis functions at X0
            X0 = zeros(size(self.X,1),1);
            Psi0_new = self.basis.grad_vandermonde(X0, new_midxs(:,1), new_midxs(:,2), 0);
            self.Psi0 = [self.Psi0, Psi0_new];

            % evaluate new basis functions
            Psi_new = self.basis.grad_vandermonde(self.X, new_midxs(:,1), new_midxs(:,2), 0);
            self.Psi = [self.Psi, Psi_new];

            % evaluate gradient of new basis functions
            dxPsi_new = self.basis.grad_vandermonde(self.X, new_midxs(:,1), new_midxs(:,2), 1);
            self.grad_x_Psi = [self.grad_x_Psi, dxPsi_new];

            % update Psi(x) points for all non-zero quadrature points
            for l=1:self.max_levels
                % check if array is empty
                if isempty(self.quad_dxPsii{l})
                    continue
                end
                for i=1:length(self.quad_xi{l})
                    dxPsii_new = self.basis.grad_vandermonde(self.quad_xi{l}{i}, ...
                        new_midxs(:,1), new_midxs(:,2), 1);
                    self.quad_dxPsii{l}{i} = [self.quad_dxPsii{l}{i}, dxPsii_new];
                end
            end

        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function [dxPsii, wi, xi] = evaluate_quadrature(self, N)

            % extract CC nodes and weights
            [xcc, wcc] = clenshaw_curtis(N);

            % sum contribution from each integration node
            xi = cell(N,1); wi = cell(N,1); dxPsii = cell(N,1);
            for j=1:N
                % extract a,b endpoints and rescale points to interval
                [xi{j}, wi{j}] = rescale_pts(zeros(size(self.X,1),1), self.X, xcc(j), wcc(j));
                % evaluate basis functions
                dxPsii{j} = self.basis.grad_vandermonde(xi{j}, self.midxs(:,1), self.midxs(:,2), 1);
            end

        end %endFunction
        % -----------------------------------------------------------------
    end %endMethods
    
end %endClass

% -- END OF FILE --