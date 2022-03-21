classdef LinearTriangularTransportMap < TriangularTransportMap
    
	% Defines a linear transport map of the form S(x) = L(x - c) where L is
	% a lower-triangular matrix and c is a column vector.
    % 
    % Author: Ricardo Baptista
    % Date:   June 2021

    properties
        diagonal  % flag for using diagonal or dense transformation
    end
    
    methods 
        %------------------------------------------------------------------
        function self = LinearTriangularTransportMap(d, diagonal)
            % define default properties
            if (nargin < 2)
                diagonal = true;
            end
            % define cells of map components that are 
            % initialized to the identity map
            default_c  = zeros(1,d);
            default_L  = eye(d);
            S    = cell(d,1);
            for k=1:d
                S{k} = LinearFunction(default_c(k), default_L(k,1:k));
            end
            % define TriangularTransportMap object 
            self@TriangularTransportMap(S);
            self.diagonal = diagonal;
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function L = L(self)
            % extract linear map from LinearMap components
            L = zeros(self.d, self.d);
            for k=1:self.d
                L(k,1:k) = self.S{k}.L;
            end
        end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function c = c(self)
            % extract constant from LinearMap components
            neg_Lc = zeros(self.d,1);
            for k=1:self.d
                neg_Lc(k) = self.S{k}.c;
            end
            L = self.L();
            c = -1*L\neg_Lc;
        end %endFunction
        %------------------------------------------------------------------
    end %endMethods
    
end %endClass