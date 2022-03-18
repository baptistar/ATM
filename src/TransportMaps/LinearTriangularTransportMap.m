classdef LinearTriangularTransportMap < TriangularTransportMap
    
	% Defines a linear transport map of the form S(x) = L(x - c) where L is
	% a lower-triangular matrix and c is a column vector.

    properties
        diagonal  % flag for using diagonal or dense transformation
    end
    
    methods 
        function self = LinearTriangularTransportMap(d, diagonal)
            % define default properties
            if (nargin < 2)
                diagonal = true;
            end
            % define cells of map components that are 
            % initialized to the identity map
            default_c    = zeros(1,d);
            default_L    = eye(d);
            S    = cell(d,1);
            for k=1:d
                S{k} = LinearFunction(default_c(k), default_L(k,1:k));
            end
            % define TriangularTransportMap object 
            self@TriangularTransportMap(S);
            self.diagonal = diagonal;
        end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function L = L(self)
            % extract linear map from LinearMap components
            L = zeros(self.d, self.d);
            for k=1:self.d
                L(k,1:k) = self.S{k}.L;
            end
        end
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
        end
        %------------------------------------------------------------------
        %------------------------------------------------------------------
%         function self = optimize(self, X, ref)
%         % Function finds optimal mean and covariance for all dimensions
%         % ignoring the third comp_idx input.
%         
%             % check reference
%         
%             % check dimensions of inputs
%             if size(X,2) ~= self.d
%                 error('Dimension mismatch of inputs and dimension')
%             end
%             
%             % compute mean of data
%             self.c = mean(X,1);
% 
%             % compute covariance or diagonal scaling of data
%             if self.diagonal == true
%                 std_data = std(X,[],1);
%                 self.Linv = diag(std_data);
%                 self.L = diag(1./std_data);
%             else
%                 C = cov(X);
%                 self.Linv = chol(C,'lower');
%                 self.L = inv(self.Linv);
%             end
%             
%         end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
    end %endMethods
    
end %endClass