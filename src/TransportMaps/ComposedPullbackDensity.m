% Author: Ricardo Baptista
% Date:   June 2021
%
% See LICENSE.md for copyright information
%

classdef ComposedPullbackDensity
    % Define the pullback density via the composition of transport maps S
    % By convention: ComposedPullbackDensity({S_1,...,S_L},ref) builds
    %  S(x) = S_L \circ S_{L-1} ... \circ S_1(x)
    %  \pi = S_1^# ... S_{L-1}^# S_L^# \ref = \ref(S(x)) |\nabla_x S(x)|
    
    properties
        ref  % total number of total dimensions
        S    % {d x 1} cell of maps
    end
    
    methods
        function CM = ComposedPullbackDensity(S, ref)
            
            % assign maps and ref to CM
            CM.S = S;
            CM.ref = ref;
            
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function ncomp = ncomp(self)
            ncomp = length(self.S);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function nc = ncoeff(self)
            nc = 0;
            for k=1:length(self.S)
                nc = nc + self.S{k}.ncoeff;
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function Sx = evaluate(self, X, comp_idx)
            % if not specified, evaluate all components of maps
            if (nargin < 3)
                comp_idx = 1:self.S{1}.d;
            end
            % evaluate first d-1 maps at all inputs
            Sx = X;
            for j=1:(self.ncomp-1)
                Sx = self.S{j}.S.evaluate(Sx);
            end
            % evaluate comp_idx components of last map
            Sx = self.S{self.ncomp}.S.evaluate(Sx, comp_idx);
            % extract comp_idx components
            %Sx = Sx(:, comp_idx);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------                                                                                                                                                                           
        function gx_next = grad_x(self, X, grad_dim, comp_idx)
            % compute gradients of all components of triangular map
            %   \nabla_x S(x) \in R^{N x d x m}
            %     m = number of gradient terms
            % if not specified, evaluate all gradients
            %             if (nargin < 4)
            %                 comp_idx = 1:self.S{1}.d;
            %             end
            %             if (nargin < 3)
            %                 grad_dim = 1:self.S{1}.d;
            %             end
            gx=zeros(size(self.S{1}.grad_x(X)));
            for i=1:self.ref.dim
                gx(:,i,i)=1;
            end
            gx=permute(gx,[2 3 1]);
            for k=1:length(self.S)-1            
                gx_next=self.S{k}.grad_x(X);
                gx_next=permute(gx_next,[3 2 1]);
                gx=pagemtimes(gx,gx_next);
                X=self.S{k}.evaluate(X);
            end
            gx_last=self.S{end}.grad_x(X);
            gx_last=permute(gx_last,[2 3 1]);
            gx=pagemtimes(gx,gx_last);
            gx_next=permute(gx,[3 2 1]); 
        end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function Sz = inverse(self, Z, Xp, comp_idx)
            % if not specified, invert all components of maps
            if (nargin < 4) || isempty(Xp)
                % invert the maps in reverse order
                Sz = Z;
                for j=self.ncomp:-1:1
                    Sz = self.S{j}.inverse(Sz);
                end
            else
                % check comp_idx is lower part of map
                assert(all(comp_idx == min(comp_idx):max(comp_idx)))
                assert(max(comp_idx) == self.S{1}.d)
                % extract remaining component indices
                comp_idx_comp = setdiff(1:self.S{1}.d, comp_idx);
                assert(size(Z,2) == length(comp_idx));
                assert(size(Xp,2) == length(comp_idx_comp));
                % evaluate the upper part of maps at Xp
                SXp_eval = cell(self.ncomp-1,1);
                SXp = Xp;
                for j=1:(self.ncomp-1)
                    Sj_input = [SXp,zeros(size(SXp,1),length(comp_idx))];
                    SXp = self.S{j}.evaluate(Sj_input, comp_idx_comp);
                    SXp_eval{j} = SXp;
                end
                % invert bottom part of maps for bottom inputs
                Sz = Z;
                for j=self.ncomp:-1:2
                    Sz = self.S{j}.inverse(Sz, SXp_eval{j-1}, comp_idx);
                end
                Sz = self.S{1}.inverse(Sz, Xp, comp_idx);
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function [dJ, Sx] = logdet_Jacobian(self, X, comp_idx)
            % if not specified, evaluate all components of maps
            if (nargin < 3)
                comp_idx = 1:self.S{1}.d;
            end
            % evaluate S at all X and comp_idx components of Jacobian
            
            Sx = self.S{1}.evaluate(X);
            dJ = self.S{1}.logdet_Jacobian(X, comp_idx);
            for j=2:length(self.S)
                dJ = dJ + self.S{j}.logdet_Jacobian(Sx, comp_idx);
                Sx = self.S{j}.evaluate(Sx);
            end
            % extract comp_idx components
            %Sx = Sx(:, comp_idx);
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function log_pi = log_pdf(self, X, comp_idx)
            % if not specified, evaluate all components of maps
            if (nargin < 3)
                comp_idx = 1:self.S{1}.d;
            end
            % compute log-determinant and evaluate map
            
            [dJ, Sx] = self.logdet_Jacobian(X, comp_idx);
            % evaluate log_pi
            log_pi = self.ref.log_pdf(Sx, comp_idx) + dJ;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function grad_log_pi = grad_x_log_pdf(self, X, grad_dim, comp_idx)
            % if not specified, evaluate all components of maps
            if (nargin < 4)
                comp_idx = 1:self.S{1}.d;
            end
            % if not specified, compute gradients for all variables
            if (nargin < 3) || isempty(grad_dim)
                grad_dim = 1:self.S{1}.d;
            end
            
            PB=PullbackDensity(self.S{end},self.ref);
            for j=1:length(self.S)-1
                PB=PullbackDensity(self.S{end-j},PB);
            end
            grad_log_pi=PB.grad_x_log_pdf(X,grad_dim,comp_idx);
        end
        
        
        function [self, Sx] = optimize(self, XW, comp_idx)
            if isa(XW,'struct')
                X=XW.X;
                W=XW.W;
            else
                X=XW;
                W=(1/size(XW,1))*ones(size(XW,1),1);
            end
            % if not specified, optimize all components of transport map
            if (nargin < 3)
                comp_idx = 1:self.S{1}.d;
            end
            % optimize and push-forward samples for each layer
            Sx = X;
            for j=1:length(self.S)
                XW_s.X=Sx;
                XW_s.W=W;
                self.S{j} = self.S{j}.optimize(XW_s, comp_idx);
                Sx(:,comp_idx) = self.S{j}.evaluate(XW_s.X, comp_idx);
            end
        end %endFunction
        %------------------------------------------------------------------
    end %endMethods
    
end %endClass
