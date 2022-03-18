function [F,G] = truth_f(x)
%UNTITLED10 Summary of this function goes here

xx=x(:,1);
y=x(:,2);

XX=xx.^2+2*xx+4;
Y=y.^2+xx.*y+2*xx+y+6;


F=XX.^2+6*XX.*Y-6*Y+Y.^2;

G=zeros(size(x));

G(:,1)=2*(2*xx.^3+3*xx.^2.*(3*y+8)+xx.*(7*y.^2+22*y+76)+y.^3+9*y.^2+23*y+74);
G(:,2)=2*((xx+2*y+1).*(3*xx.^2+xx.*(y+8)+y.^2+y+15));

end

%gradient  (x^2+2x+4)^2+ 6*(x^2+2x+4)*(y^2+x*y+2x+y+6)-6*(y^2+x*y+2x+y+6)+(y^2+x*y+2x+y+6)^2