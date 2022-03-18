function [F,G] = fun_f(X)
%UNTITLED10 Summary of this function goes here

xx=X(:,1);
y=X(:,2);


F=xx.^2+6*xx.*y-6*y+y.^2;

G=zeros(size(X));

G(:,1)=2*X(:,1)+6*X(:,2);
G(:,2)=2*X(:,2)+6*X(:,1)-6;

end
