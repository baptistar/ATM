function [xGL, wGL] = gauss_quad(n)
%--------------------------------------------------------------------------
% Returns Gauss-Legendre nodes and weights for n points using
% m-files of Walter Gautschi.
% https://www.cs.purdue.edu/archives/2002/wxg/codes/
%--------------------------------------------------------------------------

if n == 1
    xGL = 0;
    wGL = 1;
    return;
end
p   = n-1;
ab  = r_jacobi(p+1,0,0);
xw  = gauss(p+1,ab);
xGL = xw(:,1);
wGL = xw(:,2);

return;



%--------------------------------------------------------------------------
% R_JACOBI Recurrence coefficients for monic Jacobi polynomials.
%    ab=R_JACOBI(n,a,b) generates the first n recurrence
%    coefficients for monic Jacobi polynomials with parameters
%    a and b. These are orthogonal on [-1,1] relative to the
%    weight function w(t)=(1-t)^a(1+t)^b. The n alpha-coefficients
%    are stored in the first column, the n beta-coefficients in
%    the second column, of the nx2 array ab. The call ab=
%    R_JACOBI(n,a) is the same as ab=R_JACOBI(n,a,a) and
%    ab=R_JACOBI(n) the same as ab=R_JACOBI(n,0,0).

function ab = r_jacobi(N,a,b)

if nargin < 2
    a = 0;
elseif nargin < 3
    b = a;
end
if (N<=0 || a<=-1 || b<=-1)
    error('parameter(s) out of range');
end
nu = (b-a)/(a+b+2);
mu = 2^(a+b+1)*gamma(a+1)*gamma(b+1)/gamma(a+b+2);
if N == 1
    ab = [nu; mu];
else
    N   = N-1;
    n   = 1:N;
    nab = 2*n+a+b;
    A   = [nu (b^2-a^2)*ones(1,N)./(nab.*(nab+2))];
    n   = 2:N;
    nab = nab(n);
    B1  = 4*(a+1)*(b+1)/((a+b+2)^2*(a+b+3));
    B   = 4*(n+a).*(n+b).*n.*(n+a+b)./((nab.^2).*(nab+1).*(nab-1));
    ab  = [A' [mu; B1; B']];
end

return;



%----------------------------------------------------------------------
% GAUSS Gaussian quadrature rule.
%    Given a weight function w encoded by the nx2 array ab of the
%    first n recurrence coefficients for the associated orthogonal
%    polynomials, the first column of ab containing the n alpha-
%    coefficients and the second column the n beta-coefficients,
%    the call xw=GAUSS(n,ab) generates the nodes and weights xw of
%    the n-point Gauss quadrature rule for the weight function w.
%    The nodes, in increasing order, are stored in the first
%    column, the n corresponding weights in the second column, of
%    the nx2 array xw.

function xw = gauss(N, ab)

N0 = size(ab,1);
if N0 < N
    error('Input array ab too short');
end
J = zeros(N);
for n = 1:N
    J(n,n) = ab(n,1);
end
for n = 2:N
    J(n,n-1) = sqrt(ab(n,2));
    J(n-1,n) = J(n,n-1);
end
[V,D] = eig(J);
[D,I] = sort(diag(D));
V     = V(:,I);
xw    = [D, ab(1,2)*V(1,:)'.^2];

return;