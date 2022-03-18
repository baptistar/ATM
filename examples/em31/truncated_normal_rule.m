function truncated_normal_rule ( arg1, arg2, arg3, arg4, arg5, arg6, arg7 )

%*****************************************************************************80
%
%% MAIN is the main program for TRUNCATED_NORMAL_RULE.
%
%  Discussion:
%
%    This program computes a truncated normal quadrature rule
%    and writes it to a file.
%
%    The user specifies:
%    * option: 0/1/2/3 for none, lower, upper, double truncation.
%    * N, the number of points in the rule;
%    * MU, the mean of the original normal distribution;
%    * SIGMA, the standard deviation of the original normal distribution,
%    * A, the left endpoint (for options 1 or 3)
%    * B, the right endpoint (for options 2 or 3);
%    * FILENAME, the root name of the output files.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    07 February 2014
%
%  Author:
%
%    John Burkardt
%
  timestamp ( );
  fprintf ( 1, '\n' );
  fprintf ( 1, 'TRUNCATED_NORMAL_RULE\n' );
  fprintf ( 1, '  MATLAB version\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  For the (truncated) Gaussian probability density function\n' );
  fprintf ( 1, '    pdf(x) = exp(-0.5*((x-MU)/SIGMA)^2) / SIGMA / sqrt ( 2 * pi )\n' );
  fprintf ( 1, '  compute an N-point quadrature rule for approximating\n' );
  fprintf ( 1, '    Integral ( A <= x <= B ) f(x) pdf(x) dx\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  The value of OPTION determines the truncation interval [A,B]:\n' );
  fprintf ( 1, '  0: (-oo,+oo)\n' );
  fprintf ( 1, '  1: [A,+oo)\n' );
  fprintf ( 1, '  2: (-oo,B]\n' );
  fprintf ( 1, '  3: [A,B]\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  The user specifies OPTION, N, MU, SIGMA, A, B and FILENAME.\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  FILENAME is used to generate 3 files:\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '    filename_w.txt - the weight file\n' );
  fprintf ( 1, '    filename_x.txt - the abscissa file.\n' );
  fprintf ( 1, '    filename_r.txt - the region file, listing A and B.\n' );
%
%  Try to deal with arguments.
%
  if ( nargin < 7 )
    arg7 = [];
  end
  if ( nargin < 6 )
    arg6 = [];
  end
  if ( nargin < 5 )
    arg5 = [];
  end

  iarg = 0;
%
%  Get OPTION.
%
  iarg = iarg + 1;
  if ( nargin < iarg )
    option = input ( '  Enter OPTION, 0/1/2/3:  ' );
  elseif ( ischar ( arg1 ) )
    option = str2num ( arg1 );
  else
    option = arg1;
  end

  if ( option < 0 || 3 < option )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'TRUNCATED_NORMAL_RULE - Fatal error!\n' );
    fprintf ( 1, '  0 <= OPTION <= 3 was required.' );
    error ( 'TRUNCATED_NORMAL_RULE - Fatal error!' );
  end

  if ( option == 0 )
    arg6 = [];
    arg7 = [];
  elseif ( option == 1 || option == 2 )
    arg7 = [];
  end
%
%  Get N.
%
  iarg = iarg + 1;
  if ( nargin < iarg )
    n = input ( '  Enter the rule order N:  ' );
  elseif ( ischar ( arg2 ) )
    n = str2num ( arg2 );
  else
    n = arg2;
  end
%
%  Get MU.
%
  iarg = iarg + 1;
  if ( nargin < iarg )
    mu = input ( '  Enter MU, the mean value of the normal distribution:  ' );
  elseif ( ischar ( arg3 ) )
    mu = str2num ( arg3 );
  else
    mu = arg3;
  end
%
%  Get SIGMA.
%
  iarg = iarg + 1;
  if ( nargin < iarg )
    sigma = input ( '  Enter SIGMA, the standard deviation:  ' );
  elseif ( ischar ( arg4 ) )
    sigma = str2num ( arg4 );
  else
    sigma = arg4;
  end
%
%  Get A.
%
  if ( option == 1 || option == 3 )
    iarg = iarg + 1;
    if ( nargin < iarg )
      a = input ( '  Enter the left endpoint A:  ' );
    elseif ( ischar ( arg5 ) )
      a = str2num ( arg5 );
    else
      a = arg5;
    end
  else
    arg7 = arg6;
    arg6 = arg5;
    a = - r8_huge ( );
  end
%
%  Get B.
%
  if ( option == 2 || option == 3 )
    iarg = iarg + 1;
    if ( nargin < iarg )
      b = input ( '  Enter the right endpoint B:  ' );
    elseif ( ischar ( arg6 ) )
      b = str2num ( arg6 );
    else
      b = arg6;
    end
  else
    arg7 = arg6;
    b = r8_huge ( );
  end
%
%  Get FILENAME.
%
  iarg = iarg;
  if ( nargin < iarg )
    fprintf ( 1, '\n' );
    fprintf ( 1, '  FILENAME is the "root name" of the quadrature files.\n' );
    filename = input ( '  Enter FILENAME as a quoted string:  ' );
  else
    filename = arg7;
  end
%
%  Input summary.
%
  fprintf ( 1, '\n' );
  fprintf ( 1, '  OPTION = %d\n', option );
  fprintf ( 1, '  N = %d\n', n );
  fprintf ( 1, '  MU = %g\n', mu );
  fprintf ( 1, '  SIGMA = %g\n', sigma );
  if ( option == 1 || option == 3 )
    fprintf ( 1, '  A = %g\n', a );
  else
    fprintf ( 1, '  A = -oo\n' );
  end
  if ( option == 2 || option == 3 )
    fprintf ( 1, '  B = %g\n', b );
  else
    fprintf ( 1, '  B = +oo\n' );
  end
  fprintf ( 1, '  FILENAME = "%s"\n', filename );;
%
%  Compute the moments.
%
  if ( option == 0 )
    moments = moments_normal ( 2 * n + 1, mu, sigma );
  elseif ( option == 1 )
    moments = moments_truncated_normal_a ( 2 * n + 1, mu, sigma, a );
  elseif ( option == 2 )
    moments = moments_truncated_normal_b ( 2 * n + 1, mu, sigma, b );
  elseif ( option == 3 )
    moments = moments_truncated_normal_ab ( 2 * n + 1, mu, sigma, a, b );
  end
%
%  Compute the rule.
%
  [ x, w ] = moment_method ( n, moments );
%
%  Write the rule.
%
  r(1,1) = a;
  r(2,1) = b;

  rule_write ( n, filename, x, w, r );
%
%  Terminate.
%
  fprintf ( 1, '\n' );
  fprintf ( 1, 'TRUNCATED_NORMAL_RULE:\n' );
  fprintf ( 1, '  Normal end of execution.\n' );
  fprintf ( 1, '\n' );
  timestamp ( );

  return
end
function r = hankel_cholesky_upper ( n, h )

%*****************************************************************************80
%
%% HANKEL_CHOLESKY_UPPER computes the upper Cholesky factor of a Hankel matrix.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    30 January 2017
%
%  Author:
%
%    John Burkardt
%
%  Reference:
%
%    James Phillips,
%    The triangular decomposition of Hankel matrices,
%    Mathematics of Computation,
%    Volume 25, Number 115, July 1971, pages 599-602.
%
%  Parameters:
%
%    Input, integer N, the order of the matrix.
%
%    Input, real H(2*N-1), the values defining the antidiagonals.
%
%    Output, real R(N,N), the upper triangular Cholesky factor.
%
  h = h(:);
  a = zeros ( n, 1 );
  b = zeros ( n, 1 );
  c = zeros ( 2 * n - 1, 2 * n - 1 );

  c(1,1:2*n-1) = h(1:2*n-1)';

  for i = 1 : n - 1

    if ( i == 1 )
      a = c(1,2) / c(1,1);
      b = 0.0;
    else
      a = c(i,i+1) / c(i,i) - c(i-1,i) / c(i-1,i-1);
      b = c(i,i) / c(i-1,i-1);
    end

    for j = i + 1 : 2 * n - i - 1
      c(i+1,j) = c(i,j+1) - a * c(i,j);
      if ( 1 < i )
        c(i+1,j) = c(i+1,j) - b * c(i-1,j);
      end
    end
  end

  r(1:n,1:n) = c(1:n,1:n);
%
%  Normalize.
%
  for i = 1 : n
    t = sqrt ( r(i,i) );
    r(i,1:i-1) = 0.0;
    r(i,i:n) = r(i,i:n) / t;
  end

  return
end
function [ v, d, it_num, rot_num ] = jacobi_eigenvalue ( n, a, it_max )

%*****************************************************************************80
%
%% JACOBI_EIGENVALUE carries out the Jacobi eigenvalue iteration.
%
%  Discussion:
%
%    This function computes the eigenvalues and eigenvectors of a
%    real symmetric matrix, using Rutishauser's modfications of the classical
%    Jacobi rotation method with threshold pivoting. 
%
%  Modified:
%
%    17 September 2013
%
%  Parameters:
%
%    Input, integer N, the order of the matrix.
%
%    Input, real A(N,N), the matrix, which must be square, real,
%    and symmetric.
%
%    Input, integer IT_MAX, the maximum number of iterations.
%
%    Output, real V(N,N), the matrix of eigenvectors.
%
%    Output, real D(N), the eigenvalues, in descending order.
%
%    Output, integer IT_NUM, the total number of iterations.
%
%    Output, integer ROT_NUM, the total number of rotations applied.
%
  v = eye ( n, n );
  d = diag ( a );
  bw(1:n,1) = d(1:n,1);
  zw(1:n,1) = 0.0;
  it_num = 0;
  rot_num = 0;

  while ( it_num < it_max )

    it_num = it_num + 1;
%
%  The convergence threshold is based on the size of the elements in
%  the strict upper triangle of the matrix.
%
    thresh = sqrt ( sum ( sum ( triu ( a, 1 ).^2 ) ) ) / ( 4.0 * n );

    if ( thresh == 0.0 )
      break;
    end

    for p = 1 : n
      for q = p + 1 : n

        gapq = 10.0 * abs ( a(p,q) );
        termp = gapq + abs ( d(p) );
        termq = gapq + abs ( d(q) );
%
%  Annihilate tiny offdiagonal elements.
%
        if ( 4 < it_num && termp == abs ( d(p) ) && termq == abs ( d(q) ) )

          a(p,q) = 0.0;
%
%  Otherwise, apply a rotation.
%
        elseif ( thresh <= abs ( a(p,q) ) )

          h = d(q) - d(p);
          term = abs ( h ) + gapq;

          if ( term == abs ( h ) ) 
            t = a(p,q) / h;
          else
            theta = 0.5 * h / a(p,q);
            t = 1.0 / ( abs ( theta ) + sqrt ( 1.0 + theta * theta ) );
            if ( theta < 0.0 ) 
              t = - t;
            end
          end

          c = 1.0 / sqrt ( 1.0 + t * t );
          s = t * c;
          tau = s / ( 1.0 + c );
          h = t * a(p,q);
%
%  Accumulate corrections to diagonal elements.
%
          zw(p) = zw(p) - h;                  
          zw(q) = zw(q) + h;
          d(p) = d(p) - h;
          d(q) = d(q) + h;

          a(p,q) = 0.0;
%
%  Rotate, using information from the upper triangle of A only.
%
          for j = 1 : p - 1
            g = a(j,p);
            h = a(j,q);
            a(j,p) = g - s * ( h + g * tau );
            a(j,q) = h + s * ( g - h * tau );
          end

          for j = p + 1 : q - 1
            g = a(p,j);
            h = a(j,q);
            a(p,j) = g - s * ( h + g * tau );
            a(j,q) = h + s * ( g - h * tau );
          end

          for j = q + 1 : n
            g = a(p,j);
            h = a(q,j);
            a(p,j) = g - s * ( h + g * tau );
            a(q,j) = h + s * ( g - h * tau );
          end
%
%  Accumulate information in the eigenvector matrix.
%
          for j = 1 : n
            g = v(j,p);
            h = v(j,q);
            v(j,p) = g - s * ( h + g * tau );
            v(j,q) = h + s * ( g - h * tau );
          end

          rot_num = rot_num + 1;

        end

      end
    end

    bw(1:n,1) = bw(1:n,1) + zw(1:n,1);
    d(1:n,1) = bw(1:n,1);
    zw(1:n,1) = 0.0;

  end
%
%  Ascending sort the eigenvalues and eigenvectors.
%
  for k = 1 : n - 1

    m = k;

    for l = k + 1 : n
      if ( d(l) < d(m) )
        m = l;
      end
    end

    if ( m ~= k )

      t    = d(m);
      d(m) = d(k);
      d(k) = t;

      w        = v(1:n,m);
      v(1:n,m) = v(1:n,k);
      v(1:n,k) = w;

    end

  end

  return
end
function [ x, w ] = moment_method ( n, moments )

%*****************************************************************************80
%
%% MOMENT_METHOD computes a quadrature rule by the method of moments.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    26 January 2017
%
%  Author:
%
%    John Burkardt
%
%  Reference:
%
%    Gene Golub, John Welsch,
%    Calculation of Gaussian Quadrature Rules,
%    Mathematics of Computation,
%    Volume 23, Number 106, April 1969, pages 221-230.
%
%  Parameters:
%
%    Input, integer N, the order of the quadrature rule.
%
%    Input, real MOMENTS(2*N+1), moments 0 through 2*N.
%
%    Output, real X(N), W(N), the points and weights of the quadrature rule.
%
  debug = true;

  if ( debug )
    r8vec_print ( 2 * n + 1, moments, '  Moments:' );
  end
%
%  Define the N+1 by N+1 Hankel matrix H(I,J) = moments(I+J).
%
  h = zeros ( n + 1, n + 1 );
  for i = 1 : n + 1
    for j = 1 : n + 1
      h(i,j) = moments(i+j-1);
    end
  end

  if ( debug )
    r8mat_print ( n + 1, n + 1, h, '  Hankel matrix H:' )
  end
%
%  Compute R, the upper triangular Cholesky factor of H.
%
  if ( true )
    hanti = zeros ( 2 * n + 1, 1 );
    hanti(1:n+1,1) = h(1,1:n+1);
    hanti(n+2:2*n+1,1) = h(n+1,2:n+1);
    r = hankel_cholesky_upper ( n + 1, hanti );

  elseif ( false )

    fprintf ( 1, '\n' );
    fprintf ( 1, 'MOMENT_METHOD - Fatal error!\n' );
    fprintf ( 1, '  FALSE was mishandled.\n' );

    r = chol ( h );

  else
  
    [ r, flag ] = r8mat_cholesky_factor_upper ( n + 1, h );

    if ( flag ~= 0 )
      fprintf ( 1, '\n' );
      fprintf ( 1, 'MOMENT_METHOD - Fatal error!\n' );
      fprintf ( 1, '  R8MAT_CHOLESKY_FACTOR_UPPER returned FLAG = %d\n', flag );
      error ( 'MOMENT_METHOD - Fatal error!' )
    end
  
  end
  
  e = norm ( h - r' * r, 'fro' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  Frobenius norm of H-R''*R = %g\n', e );

  if ( debug )
    r8mat_print ( n + 1, n + 1, r, '  Cholesky factor R:' )
  end
%
%  Compute ALPHA and BETA from R, using Golub and Welsch's formula.
%
  alpha = zeros ( n, 1 );
  alpha(1) = r(1,2) / r(1,1);
  for i = 2 : n
    alpha(i) = r(i,i+1) / r(i,i) - r(i-1,i) / r(i-1,i-1);
  end

  beta = zeros ( n - 1, 1 );
  for i = 1 : n - 1
    beta(i) = r(i+1,i+1) / r(i,i);
  end
%
%  Compute the points and weights from the moments.
%
  jacobi = diag ( alpha, 0 ) + diag ( beta, -1 ) + diag ( beta, +1 );

  if ( debug )
    r8mat_print ( n, n, jacobi, '  Jacobi matrix J:' );
  end
%
%  Get the eigendecomposition of the Jacobi matrix.
%
  if ( false )

    [ v, eval ] = eig ( jacobi );
  
    if ( debug )
      r8mat_print ( n, n, v, '  Eigenvector matrix V:' );
    end

    x = diag ( eval );
    w = moments(1) * ( v(1,:).^2 )';

  else

    it_max = 100;

    [ v, x, it_num, rot_num ] = jacobi_eigenvalue ( n, jacobi, it_max );

    if ( debug )
      r8mat_print ( n, n, v, '  Eigenvector matrix V:' );
    end

    w = moments(1) * ( v(1,1:n) .^ 2 )';

  end

  return
end
function w = moments_normal ( m, mu, sigma )

%*****************************************************************************80
%
%% MOMENTS_NORMAL returns moments of the standard Normal distribution.
%
%  Discussion:
%
%    pdf(x) = exp ( -((x-mu)/sigma)^2/2 ) / sigma / sqrt ( pi * 2 )
%    mu(k) = integral ( -oo < x < +oo ) x^k pdf(x) dx
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    15 September 2013
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer M, the number of moments desired.
%
%    Input, real MU, SIGMA, the mean and standard deviation.
%
%    Output, real W(M), the weighted integrals of X^0 through X^(M-1).
%
  w = zeros ( m, 1 );

  for k = 1 : m
    n = k - 1;
    t = 0.0;
    for j = 0 : floor ( n / 2 )
      t = t + r8_choose ( n, 2 * j ) * r8_factorial2 ( 2 * j - 1 ) ...
        * sigma ^ ( 2 * j ) * mu ^ ( n - 2 * j );
    end
    w(k) = t;
  end

  return
end
function w = moments_truncated_normal_ab ( m, mu, s, a, b )

%*****************************************************************************80
%
%% MOMENTS_TRUNCATED_NORMAL_AB: moments of the truncated normal distribution.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    18 September 2013
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer M, the number of moments desired.
%
%    Input, real MU, S, the mean and standard deviation of the
%    parent Normal distribution.
%    0 < S.
%
%    Input, real A, B, the lower and upper truncation limits.
%    A < B.
%
%    Output, real W(M), the weighted integrals of X^0 through X^(M-1).
%
  w = zeros ( m, 1 );

  for order = 0 : m - 1
    w(order+1) = truncated_normal_ab_moment ( order, mu, s, a, b );
  end

  return
end
function w = moments_truncated_normal_a ( m, mu, s, a )

%*****************************************************************************80
%
%% MOMENTS_TRUNCATED_NORMAL_A: moments of the lower truncated normal distribution.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    19 September 2013
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer M, the number of moments desired.
%
%    Input, real MU, S, the mean and standard deviation of the
%    parent Normal distribution.
%    0 < S.
%
%    Input, real A, the lower truncation limit.
%
%    Output, real W(M), the weighted integrals of X^0 through X^(M-1).
%
  w = zeros ( m, 1 );

  for order = 0 : m - 1
    w(order+1) = truncated_normal_a_moment ( order, mu, s, a );
  end

  return
end
function w = moments_truncated_normal_b ( m, mu, sigma, b )

%*****************************************************************************80
%
%% MOMENTS_TRUNCATED_NORMAL_B: moments of the upper truncated normal.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    19 September 2013
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer M, the number of moments desired.
%
%    Input, real MU, SIGMA, the mean and standard deviation of the
%    parent Normal distribution.
%    0 < S.
%
%    Input, real B, the upper truncation limit.
%
%    Output, real W(M), the weighted integrals of X^0 through X^(M-1).
%
  w = zeros ( m, 1 );

  for order = 0 : m - 1
    w(order+1) = truncated_normal_b_moment ( order, mu, sigma, b );
  end

  return
end
function cdf = normal_01_cdf ( x )

%*****************************************************************************80
%
%% NORMAL_01_CDF evaluates the Normal 01 CDF.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    17 September 2004
%
%  Author:
%
%    John Burkardt
%
%  Reference:
%
%    A G Adams,
%    Areas Under the Normal Curve,
%    Algorithm 39,
%    Computer j.,
%    Volume 12, pages 197-198, 1969.
%
%  Parameters:
%
%    Input, real X, the argument of the CDF.
%
%    Output, real CDF, the value of the CDF.
%
  a1 = 0.398942280444E+00;
  a2 = 0.399903438504E+00;
  a3 = 5.75885480458E+00;
  a4 = 29.8213557808E+00;
  a5 = 2.62433121679E+00;
  a6 = 48.6959930692E+00;
  a7 = 5.92885724438E+00;
  b0 = 0.398942280385E+00;
  b1 = 3.8052E-08;
  b2 = 1.00000615302E+00;
  b3 = 3.98064794E-04;
  b4 = 1.98615381364E+00;
  b5 = 0.151679116635E+00;
  b6 = 5.29330324926E+00;
  b7 = 4.8385912808E+00;
  b8 = 15.1508972451E+00;
  b9 = 0.742380924027E+00;
  b10 = 30.789933034E+00;
  b11 = 3.99019417011E+00;
%
%  |X| <= 1.28.
%
  if ( abs ( x ) <= 1.28 )

    y = 0.5 * x * x;

    q = 0.5 - abs ( x ) * ( a1 - a2 * y / ( y + a3 ...
      - a4 / ( y + a5 ...
      + a6 / ( y + a7 ) ) ) );
%
%  1.28 < |X| <= 12.7
%
  elseif ( abs ( x ) <= 12.7 )

    y = 0.5 * x * x;

    q = exp ( - y ) ...
      * b0  / ( abs ( x ) - b1 ...
      + b2  / ( abs ( x ) + b3 ...
      + b4  / ( abs ( x ) - b5 ...
      + b6  / ( abs ( x ) + b7 ...
      - b8  / ( abs ( x ) + b9 ...
      + b10 / ( abs ( x ) + b11 ) ) ) ) ) );
%
%  12.7 < |X|
%
  else

    q = 0.0;

  end
%
%  Take account of negative X.
%
  if ( x < 0.0 )
    cdf = q;
  else
    cdf = 1.0 - q;
  end

  return
end
function pdf = normal_01_pdf ( x )

%*****************************************************************************80
%
%% NORMAL_01_PDF evaluates the Normal 01 PDF.
%
%  Discussion:
%
%    The Normal 01 PDF is also called the "Standard Normal" PDF, or
%    the Normal PDF with 0 mean and variance 1.
%
%  Formula:
%
%    PDF(x) = exp ( - 0.5 * x^2 ) / sqrt ( 2 * pi )
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    17 September 2004
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, real X, the argument of the PDF.
%
%    Output, real PDF, the value of the PDF.
%
  pdf = exp ( -0.5 * x * x ) / sqrt ( 2.0 * pi );

  return
end
function value = r8_choose ( n, k )

%*****************************************************************************80
%
%% R8_CHOOSE computes the binomial coefficient C(N,K).
%
%  Discussion:
%
%    The value is calculated in such a way as to avoid overflow and
%    roundoff.  The calculation is done in integer arithmetic.
%
%    The formula used is:
%
%      C(N,K) = N! / ( K! * (N-K)! )
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license. 
%
%  Modified:
%
%    31 March
%
%  Author:
%
%    John Burkardt
%
%  Reference:
%
%    ML Wolfson, HV Wright,
%    Algorithm 160:
%    Combinatorial of M Things Taken N at a Time,
%    Communications of the ACM,
%    Volume 6, Number 4, April 1963, page 161.
%
%  Parameters:
%
%    Input, integer N, K, are the values of N and K.
%
%    Output, real VALUE, the number of combinations of N
%    things taken K at a time.
%
  mn = min ( k, n - k );

  if ( mn < 0 )

    value = 0;

  elseif ( mn == 0 )

    value = 1;

  else

    mx = max ( k, n - k );
    value = mx + 1;

    for i = 2 : mn
      value = ( value * ( mx + i ) ) / i;
    end

  end

  return
end
function value = r8_factorial2 ( n )

%*****************************************************************************80
%
%% R8_FACTORIAL2 computes the double factorial function.
%
%  Discussion:
%
%    FACTORIAL2( N ) = Product ( N * (N-2) * (N-4) * ... * 2 )  (N even)
%                    = Product ( N * (N-2) * (N-4) * ... * 1 )  (N odd)
%
%    N  Value
%    -  -----
%    1      1
%    2      2
%    3      3
%    4      8
%    5     15
%    6     48
%    7    105
%    8    384
%    9    945
%   10   3840
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    19 February 2008
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer N, the argument of the double factorial function.
%    If N is less than 1, VALUE is returned as 1.
%
%    Output, real VALUE, the value of N!!.
%
  value = 1;

  if ( n < 1 )
    return
  end

  while ( 1 < n )
    value = value * n;
    n = n - 2;
  end

  return
end
function value = r8_factorial ( n )

%*****************************************************************************80
%
%% R8_FACTORIAL returns N factorial.
%
%  Discussion:
%
%    factorial ( N ) = Product ( 1 <= I <= N ) I
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    02 February 1999
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer N, the argument of the function.
%    0 <= N.
%
%    Output, real VALUE, the factorial of N.
%
  if ( n < 0 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'R8_FACTORIAL - Fatal error!\n' );
    fprintf ( 1, '  N < 0.\n' );
    error ( 'R8_FACTORIAL - Fatal error!' );
  end

  value = 1.0;

  for i = 2 : n
    value = value * i;
  end

  return
end

function value = r8_huge ( )

%*****************************************************************************80
%
%% R8_HUGE returns a "huge" real number.
%
%  Discussion:
%
%    The value returned by this function is NOT required to be the
%    maximum representable R8.  This value varies from machine to machine,
%    from compiler to compiler, and may cause problems when being printed.
%    We simply want a "very large" but non-infinite number.
%
%    MATLAB provides a built-in symbolic constant "inf" that can be used
%    if a huge number is really what you want!
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    27 January 2008
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Output, real VALUE, a huge number.
%
  value = 1.0E+30;

  return
end
function value = r8_mop ( i )

%*****************************************************************************80
%
%% R8_MOP returns the I-th power of -1 as an R8 value.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license. 
%
%  Modified:
%
%    07 November 2007
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer I, the power of -1.
%
%    Output, real R8_MOP, the I-th power of -1.
%
  if ( mod ( i, 2 ) == 0 )
    value = + 1.0;
  else
    value = - 1.0;
  end

  return
end
function [ c, flag ] = r8mat_cholesky_factor_upper ( n, a )

%*****************************************************************************80
%
%% R8MAT_CHOLESKY_FACTOR_UPPER: upper Cholesky factor of a symmetric matrix.
%
%  Discussion:
%
%    The matrix must be symmetric and positive semidefinite.
%
%    For a positive semidefinite symmetric matrix A, the upper Cholesky 
%    factorization is an upper triangular matrix L such that:
%
%      A = R * R
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    03 August 2013
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer N, the number of rows and columns of the matrix A.
%
%    Input, real A(N,N), the matrix.
%
%    Output, real C(N,N), the N by N upper triangular Cholesky factor.
%
%    Output, integer ( kind = 4 ) FLAG:
%    0, no error occurred.
%    1, the matrix is not positive definite.
%
  flag = 0;

  c(1:n,1:n) = a(1:n,1:n);

  for j = 1 : n

    c(j,1:j-1) = 0.0;

    for i = j : n

      sum2 = c(i,j) - c(1:j-1,j)' * c(1:j-1,i);

      if ( i == j )
        if ( sum2 <= 0.0 )
          if ( debug )
            fprintf ( 1, '\n' );
            fprintf ( 1, 'R8MAT_CHOLESKY_UPPER - Fatal error!\n' );
            fprintf ( 1, '  SUM2 = %g\n', sum2 );
          end
          flag = 1;
          return;
        else
          c(j,i) = sqrt ( sum2 );
        end
      else
        if ( c(j,j) ~= 0.0 )
          c(j,i) = sum2 / c(j,j);
        else
          c(j,i) = 0.0;
        end
      end

    end

  end

  return
end
function r8mat_print ( m, n, a, title )

%*****************************************************************************80
%
%% R8MAT_PRINT prints an R8MAT.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    06 September 2004
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer M, the number of rows in A.
%
%    Input, integer N, the number of columns in A.
%
%    Input, real A(M,N), the matrix.
%
%    Input, string TITLE, a title.
%
  r8mat_print_some ( m, n, a, 1, 1, m, n, title );

  return
end
function r8mat_print_some ( m, n, a, ilo, jlo, ihi, jhi, title )

%*****************************************************************************80
%
%% R8MAT_PRINT_SOME prints out a portion of an R8MAT.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    08 February 2010
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer M, N, the number of rows and columns of the matrix.
%
%    Input, real A(M,N), an M by N matrix to be printed.
%
%    Input, integer ILO, JLO, the first row and column to print.
%
%    Input, integer IHI, JHI, the last row and column to print.
%
%    Input, string TITLE, a title.
%
  incx = 5;

  fprintf ( 1, '\n' );
  fprintf ( 1, '%s\n', title );

  if ( m <= 0 || n <= 0 )
    fprintf ( 1, '\n' );
    fprintf ( 1, '  (None)\n' );
    return
  end

  for j2lo = max ( jlo, 1 ): incx : min ( jhi, n )

    j2hi = j2lo + incx - 1;
    j2hi = min ( j2hi, n );
    j2hi = min ( j2hi, jhi );
    
    fprintf ( 1, '\n' );
    fprintf ( 1, '  Col: ' );

    for j = j2lo : j2hi
      fprintf ( 1, '%7d       ', j );
    end

    fprintf ( 1, '\n' );
    fprintf ( 1, '  Row\n' );

    i2lo = max ( ilo, 1 );
    i2hi = min ( ihi, m );

    for i = i2lo : i2hi

      fprintf ( 1, '%7d :', i );
      
      for j = j2lo : j2hi
        fprintf ( 1, '%14g  ', a(i,j) );
      end

      fprintf ( 1, '\n' );

    end

  end

  return
end
function r8mat_write ( output_filename, m, n, table )

%*****************************************************************************80
%
%% R8MAT_WRITE writes an R8MAT file.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    11 August 2009
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, string OUTPUT_FILENAME, the output filename.
%
%    Input, integer M, the spatial dimension.
%
%    Input, integer N, the number of points.
%
%    Input, real TABLE(M,N), the points.
%

%
%  Open the file.
%
  output_unit = fopen ( output_filename, 'wt' );

  if ( output_unit < 0 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'R8MAT_WRITE - Error!\n' );
    fprintf ( 1, '  Could not open the output file.\n' );
    error ( 'R8MAT_WRITE - Error!' );
  end
%
%  Write the data.
%
%  For smaller data files, and less precision, try:
%
%     fprintf ( output_unit, '  %14.6f', table(i,j) );
%
  for j = 1 : n
    for i = 1 : m
      fprintf ( output_unit, '  %24.16f', table(i,j) );
    end
    fprintf ( output_unit, '\n' );
  end
%
%  Close the file.
%
  fclose ( output_unit );

  return
end
function r8vec_print ( n, a, title )

%*****************************************************************************80
%
%% R8VEC_PRINT prints an R8VEC.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    25 January 2004
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer N, the dimension of the vector.
%
%    Input, real A(N), the vector to be printed.
%
%    Input, string TITLE, a title.
%
  if ( 0 < length ( title ) )
    fprintf ( 1, '\n' );
    fprintf ( 1, '%s\n', title );
  end
  fprintf ( 1, '\n' );

  for i = 1 : n
    fprintf ( 1, '  %8d: %24.16g\n', i, a(i) );
  end

  return
end
function rule_write ( order, filename, x, w, r )

%*****************************************************************************80
%
%% RULE_WRITE writes a quadrature rule to a file.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    18 February 2010
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer ORDER, the order of the rule.
%
%    Input, string FILENAME, specifies the output files.
%    write files 'filename_w.txt', 'filename_x.txt', 'filename_r.txt' defining
%    weights, abscissas, and region.
%
%    Input, real X(ORDER), the abscissas.
%
%    Input, real W(ORDER), the weights.
%
%    Input, real R(2), the region.
%
  filename_x = strcat ( filename, '_x.txt' );
  filename_w = strcat ( filename, '_w.txt' );
  filename_r = strcat ( filename, '_r.txt' );

  fprintf ( 1, '\n' );
  fprintf ( 1,'  Creating quadrature files.\n' );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  "Root" file name is   "%s".\n', filename );
  fprintf ( 1, '\n' );
  fprintf ( 1, '  Weight file will be   "%s".\n', filename_w );
  fprintf ( 1, '  Abscissa file will be "%s".\n', filename_x );
  fprintf ( 1, '  Region file will be   "%s".\n', filename_r );

  r8mat_write ( filename_w, 1, order, w' );
  r8mat_write ( filename_x, 1, order, x' );
  r8mat_write ( filename_r, 1, 2,     r' );

  return
end
function timestamp ( )

%*****************************************************************************80
%
%% TIMESTAMP prints the current YMDHMS date as a timestamp.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    14 February 2003
%
%  Author:
%
%    John Burkardt
%
  t = now;
  c = datevec ( t );
  s = datestr ( c, 0 );
  fprintf ( 1, '%s\n', s );

  return
end
function moment_pdf = truncated_normal_ab_moment ( order, mu, s, a, b )

%*****************************************************************************80
%
%% TRUNCATED_NORMAL_AB_MOMENT returns a moment of the truncated Normal PDF.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    11 September 2013
%
%  Author:
%
%    John Burkardt
%
%  Reference:
%
%    Phoebus Dhrymes,
%    Moments of Truncated Normal Distributions,
%    May 2005.
%
%  Parameters:
%
%    Input, integer ORDER, the order of the moment.
%    0 <= ORDER.
%
%    Input, real MU, S, the mean and standard deviation of the
%    parent Normal distribution.
%    0 < S.
%
%    Input, real A, B, the lower and upper truncation limits.
%    A < B.
%
%    Output, real MOMENT_PDF, the moment of the PDF.
%
  if ( order < 0 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'TRUNCATED_NORMAL_AB_MOMENT - Fatal error!\n' );
    fprintf ( 1, '  ORDER < 0.\n' );
    error ( 'TRUNCATED_NORMAL_AB_MOMENT - Fatal error!\n' );
  end

  if ( s <= 0.0 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'TRUNCATED_NORMAL_AB_MOMENT - Fatal error!\n' );
    fprintf ( 1, '  S <= 0.0.\n' );
    error ( 'TRUNCATED_NORMAL_AB_MOMENT - Fatal error!\n' );
  end

  if ( b <= a )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'TRUNCATED_NORMAL_AB_MOMENT - Fatal error!\n' );
    fprintf ( 1, '  B <= A.\n' );
    error ( 'TRUNCATED_NORMAL_AB_MOMENT - Fatal error!\n' );
  end

  a_h = ( a - mu ) / s;
  a_pdf = normal_01_pdf ( a_h );
  a_cdf = normal_01_cdf ( a_h );

  if ( a_cdf == 0.0 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'TRUNCATED_NORMAL_AB_MOMENT - Fatal error!\n' );
    fprintf ( 1, '  PDF/CDF ratio fails, because A_CDF is too small.\n' );
    fprintf ( 1, '  A_PDF = %g\n', a_pdf );
    fprintf ( 1, '  A_CDF = %g\n', a_cdf );
    error ( 'TRUNCATED_NORMAL_AB_MOMENT - Fatal error!\n' );
  end

  b_h = ( b - mu ) / s;
  b_pdf = normal_01_pdf ( b_h );
  b_cdf = normal_01_cdf ( b_h );

  if ( b_cdf == 0.0 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'TRUNCATED_NORMAL_AB_MOMENT - Fatal error!\n' );
    fprintf ( 1, '  PDF/CDF ratio fails, because B_CDF too small.\n' );
    fprintf ( 1, '  B_PDF = %g\n', b_pdf );
    fprintf ( 1, '  B_CDF = %g\n', b_cdf );
    error ( 'TRUNCATED_NORMAL_AB_MOMENT - Fatal error!\n' );
  end

  moment_pdf = 0.0;
  irm2 = 0.0;
  irm1 = 0.0;

  for r = 0 : order

    if ( r == 0 )
      ir = 1.0;
    elseif ( r == 1 )
      ir = - ( b_pdf - a_pdf ) / ( b_cdf - a_cdf );
    else
      ir = ( r - 1 ) * irm2 ...
        - ( b_h ^ ( r - 1 ) * b_pdf - a_h ^ ( r - 1 ) * a_pdf ) ...
        / ( b_cdf - a_cdf );
    end

    moment_pdf = moment_pdf + r8_choose ( order, r ) ...
      * mu ^ ( order - r ) ...
      * s ^ r * ir;

    irm2 = irm1;
    irm1 = ir;

  end

  return
end
function moment_pdf = truncated_normal_a_moment ( order, mu, s, a )

%*****************************************************************************80
%
%% TRUNCATED_NORMAL_A_MOMENT returns a moment of the lower truncated Normal PDF.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    11 September 2013
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer ORDER, the order of the moment.
%    0 <= ORDER.
%
%    Input, real MU, S, the mean and standard deviation of the
%    parent Normal distribution.
%
%    Input, real A, the lower truncation limit.
%
%    Output, real MOMENT_PDF, the moment of the PDF.
%
  moment_pdf = r8_mop ( order ) ...
    * truncated_normal_b_moment ( order, -mu, s, -a );

  return
end
function moment_pdf = truncated_normal_b_moment ( order, mu, s, b )

%*****************************************************************************80
%
%% TRUNCATED_NORMAL_B_MOMENT returns a moment of the upper truncated Normal PDF.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    02 September 2013
%
%  Author:
%
%    John Burkardt
%
%  Reference:
%
%    Phoebus Dhrymes,
%    Moments of Truncated Normal Distributions,
%    May 2005.
%
%  Parameters:
%
%    Input, integer ORDER, the order of the moment.
%    0 <= ORDER.
%
%    Input, real MU, S, the mean and standard deviation of the
%    parent Normal distribution.
%
%    Input, real B, the upper truncation limit.
%
%    Output, real MOMENT_PDF, the moment of the PDF.
%
  debug = true;
  
  if ( order < 0 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'TRUNCATED_NORMAL_B_MOMENT - Fatal error!\n' );
    fprintf ( 1, '  ORDER < 0.\n' );
    error ( 'TRUNCATED_NORMAL_B_MOMENT - Fatal error!\n' );
  end
  
  if ( debug )
    fprintf ( 1, '  ORDER = %d, B = %g, MU = %g, S = %g\n', ...
      order, b, mu, s );
  end
  
  h = ( b - mu ) / s;
  h_pdf = normal_01_pdf ( h );
  h_cdf = normal_01_cdf ( h );

  if ( debug )
    fprintf ( 1, '  ORDER = %d, H = %g, H_PDF = %g, H_CDF = %g\n', ...
      order, h, h_pdf, h_cdf );
  end
  
  if ( h_cdf == 0.0 )
    fprintf ( 1, '\n' );
    fprintf ( 1, 'TRUNCATED_NORMAL_B_MOMENT - Fatal error!\n' );
    fprintf ( 1, '  PDF/CDF ratio fails, because CDF too small.\n', h_cdf );
    fprintf ( 1, '  H_PDF = %g\n', h_pdf );
    fprintf ( 1, '  H_CDF = %g\n',h_cdf );
    error ( 'TRUNCATED_NORMAL_B_MOMENT - Fatal error!\n' );
  end

  f = h_pdf / h_cdf;

  moment_pdf = 0.0;
  irm2 = 0.0;
  irm1 = 0.0;

  for r = 0 : order

    if ( r == 0 )
      ir = 1.0;
    elseif ( r == 1 )
      ir = - f;
    else
      ir = - h ^ ( r - 1 ) * f + ( r - 1 ) * irm2;
    end

    moment_pdf = moment_pdf + r8_choose ( order, r ) ...
      * mu ^ ( order - r ) ...
      * ( s ^ r ) * ir;

    irm2 = irm1;
    irm1 = ir;

  end

  if ( debug )
    fprintf ( 1, '  MOMENT_PDF = %g\n', moment_pdf );
  end
  
  return
end

