function [Y] = map_f(X)
%UNTITLED10 Summary of this function goes here

Y=zeros(size(X));

Y(:,1)=X(:,1).^2+2.*X(:,1)+4;
Y(:,2)=X(:,2).^2+X(:,1).*X(:,2)+2*X(:,1)+X(:,2)+6;

end