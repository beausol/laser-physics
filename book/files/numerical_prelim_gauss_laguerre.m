%% gauss_laguerre
% Compute the abscissae (x) and weights (w) for Gauss-Laguerre quadrature.
%% Syntax
%# [x, w] = gauss_laguerre(n, a);
%# [x, w] = gauss_laguerre(n, a, b);
%% Description
% The function gauss_laguerre computes the abscissae (x) and weights (w)
% needed for Gauss-Laguerre quadrature using the symmetric tridiagonal
% Jacobi matrix J of the Laguerre polynomials. The values of x are derived
% from the eigenvalues of J, and the weights from the first elements of the
% eigenvectors. This method is numerically more reliable than using the
% MATLAB function roots(...), since the symmetry of J guarantees that the
% abscissae and weights will be real even for large n.
% * n - An integer > 0 giving the number of abscissae and weights.
% * a, b - Real numbers defining the generalized weight function
% x^a*exp(-b*x); a must be specified, but if b is absent, then the default
% value b = 1 is used.
% * x - A column vector containing the abscissae.
% * w - A row vector containing the weights.
%% Example
% Integral over [0, infinity] with W(x) = (x^0)*exp(-x); analytic result
% computed using Mma.
%# npts = 15;
%# f = inline('1./(1 + x.^2)', 'x');
%# analytic = 0.6214496242358134;
%# [x, w] = gauss_laguerre(npts, 0);
%# quad = w * f(x);
%# fprintf('Gauss-Laguerre Result = %f; error = %e\n', quad, quad - analytic);
%% References
% * G. H. Golub and J. H. Welsch, "Calculation of Gauss Quadrature Rules,"
% Mathematics of Computation 23, 221 (1969).
% * W. H. Press, S. A. Teukolsky, W. T. Vetterling, and B. P. Flannery,
% Numerical Recipes: The Art of Scientific Computing (Cambridge University
% Press, 2007), Sections 4.6.1 and 4.6.2.
%
function [x, w] = gauss_laguerre(n, a, b)

% Construct the symmetric tridiagonal Jacobi matrix J.
alpha = 2*(1:n) + a - 1;
beta = sqrt((1:n-1).*((1:n-1) + a));
J = diag(alpha) + diag(beta,1) + diag(beta,-1);

% Specify the integral of the weight function W(x) over the exact interval.
mu_0 = gamma(a + 1);

% The abscissae will be the (sorted) real eigenvalues of J, and the weights
% can be found from the first element of the corresponding eigenvectors.
[v, lambda] = eig(J);
[x, ix] = sort(diag(lambda));
v = v(:,ix);
w = mu_0 * v(1,:).^2;

% If the interval was specified, then rescale both the abscissae and
% weights.
if nargin > 2
    x = x/b;
    w = w/b^(a+1);
end
