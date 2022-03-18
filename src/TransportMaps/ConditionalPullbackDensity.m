% Author: Ricardo Baptista and Olivier Zahm and Youssef Marzouk
% Date:   May 2020
%
% See LICENSE.md for copyright information
%

classdef ConditionalPullbackDensity  
    % Define a transport map S of dimension d that couples a target
    % and reference density by learning the transport map from samples

    properties
        S   % Triangular transport map
        ref % Global reference density
    end

    methods  
        %------------------------------------------------------------------
        function PB = ConditionalPullbackDensity(S, ref)
            % check if S is a transport map
            if ~isa(S, 'ConditionalTriangularTransportMap')
               error('define PB with ConditionalTriangularTransportMap')
            end
            % assign S and ref
            PB.S = S;
            PB.ref = ref;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function log_pi = log_pdf(self, X, Y)
        % evaluate log pullback density log S^#\eta (x)
            % evaluate S(x) and \partial_xd S(x)
            Sx = self.S.evaluate(X, Y);
            log_dxSx = self.S.logdet_Jacobian(X, Y);
            % evaluate pullback density log_pi(x)
            log_pi = self.ref.log_pdf(Sx) + log_dxSx;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dx_log_pi = grad_x_log_pdf(self, X, Y)
        % evaluate gradient of log pullback density log S^#\eta (x)
            % evaluate S, \nabla_x S, \nabla^2_x S, extract \partial_xd S
            Sx = self.S.evaluate(X, Y);
            dxSx = self.S.grad_x(X, Y);
            dx_logeta = self.ref.grad_x_log_pdf(Sx, 1:self.S.d);
            dxlogdS = self.S.grad_x_logdet_Jacobian(X);
            % evaluate gradient
            dx_log_pi = squeeze(sum(dx_logeta .* dxSx,2)) + dxlogdS;
        end %endFunction
        %------------------------------------------------------------------
    end %endMethods

end %endClass