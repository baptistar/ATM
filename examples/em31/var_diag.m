function [var] = var_diag(ref,approx,XW)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
vals1=ref.log_pdf(XW.X);
vals2=approx.log_pdf(XW.X);

vals=vals1-vals2;
expect=sum(XW.W.*vals);
var=0.5*sum(XW.W.*(vals-expect).^2);

end

