function [G] = grad_x_map_f(X)
%UNTITLED10 Summary of this function goes here

G=zeros(size(X,1),2,2);

G(:,1,1)=2*X(:,1)+2;
G(:,1,2)=zeros(size(X(:,1)));
G(:,2,1)=X(:,2)+2;
G(:,2,2)=2*X(:,2)+X(:,1)+1;
end
