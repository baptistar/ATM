classdef Rectifier

    % Function for continous rectifier
    %   x -> rec(x)
    %
    % where rec(\cdot) is a positive and monotonically 
    % increasing function (e.g., exponential, softplus, explinearunit)
    %
    % Author: 
    % Date:   January 2020

    properties
        type        % rectifier function (exponential or softplus)
    end

    methods
        function self = Rectifier(type)
            
            % set default rectifier to be softplus
            if nargin==1
                self.type = type;
            else
                self.type = 'softplus';
            end

        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function g = evaluate(self, x)
            switch self.type
                case 'squared'
                    g = (x).^2;
                case 'exponential'
                    g = exp(x);
                case 'softplus' % robust implementation for large/small x
                    a = log(2);
                    g = (log(1 + exp(-1*abs(a*x))) + max(a*x, 0))/a;
                case 'explinearunit' 
                    g = exp(x).*(x < 0) + (x+1).*(x >= 0);
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function x = inverse(self, g)
            % check positivity of inputs
            if any(g < 0)
                error('Inputs to inverse rectifier are negative')
            end
            switch self.type
                case 'squared'
                    error('squared rectifier is not invertible')
                case 'exponential'
                    x = log(g);
                case 'softplus' % robust implementation for large x
                    a = log(2);
                    x = min(log(exp(a*g) - 1)/a, g);
                case 'explinearunit' 
                    x = log(g).*(g < 1) + (g - 1).*(g >= 1);
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function dxg = grad_x(self, x)
            switch self.type
                case 'squared'
                    dxg = 2*x;
                case 'exponential'
                    dxg = exp(x);
                case 'softplus'
                    a = log(2);
                    dxg = 1./(1 + exp(-a*x));
                case 'explinearunit'
                    dxg = exp(x).*(x < 0) + ones(size(x)).*(x >= 0);
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function d2xg = hess_x(self, x)
            switch self.type
                case 'squared'
                    d2xg = 2;
                case 'exponential'
                    d2xg = exp(x);
                case 'softplus'
                    a = log(2);
                    d2xg = a./(2 + exp(a*x) + exp(-a*x));
                case 'explinearunit'
                    d2xg = exp(x).*(x < 0) + zeros(size(x)).*(x >= 0);
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
    end %endMethods

end %endClass
