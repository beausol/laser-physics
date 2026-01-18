%% gauss_legendre
% Compute the abscissae (x) and weights (w) for Gauss-Legendre quadrature.
%% Syntax
%# [x, w] = gauss_legendre(n);
%# [x, w] = gauss_legendre(n, a, b);
%% Description
% The function gauss_legendre computes the abscissae (x) and weights (w)
% needed for Gauss-Legendre quadrature using the symmetric tridiagonal
% Jacobi matrix J of the Legendre polynomials. The values of x are derived
% from the eigenvalues of J, and the weights from the first elements of the
% eigenvectors. This method is numerically more reliable than using the
% MATLAB function roots(...), since the symmetry of J guarantees that the
% abscissae and weights will be real even for large n.
% * n - An integer > 0 giving the number of abscissae and weights; if n is
% odd, the midpoint of the interval will be included.
% * a, b - Real numbers giving the integration interval [a, b]; if these
% arguments are absent, then the default interval [-1, +1] is used.
% * x - A column vector containing the abscissae.
% * w - A row vector containing the weights.
%% Example
% Integral over [-1, 1] with W(x) = 1; analytic result computed using Mma.
%# npts = 15;
%# f = inline('1./(1 + x.^2)', 'x');
%# analytic = 1.570796326794897;
%# [x, w] = gauss_legendre(npts);
%# quad = w * f(x);
%# fprintf('Gauss-Legendre Result = %f; error = %e\n', quad, quad - analytic);
%% References
% * G. H. Golub and J. H. Welsch, "Calculation of Gauss Quadrature Rules,"
% Mathematics of Computation 23, 221 (1969).
% * W. H. Press, S. A. Teukolsky, W. T. Vetterling, and B. P. Flannery,
% Numerical Recipes: The Art of Scientific Computing (Cambridge University
% Press, 2007), Sections 4.6.1 and 4.6.2.
%
function [x, w] = gauss_legendre(n, a, b)

% Construct the symmetric tridiagonal Jacobi matrix J.
j = 1:n-1;
beta = j./sqrt(4*j.^2-1);
J = diag(beta,1) + diag(beta,-1);

% Specify the integral of the weight function W(x) over the exact interval.
mu_0 = 2;

% The abscissae will be the (sorted) real eigenvalues of J, and the weights
% can be found from the first element of the corresponding eigenvectors.
[v, lambda] = eig(J);
[x, ix] = sort(diag(lambda));
v = v(:,ix);
w = mu_0 * v(1,:).^2;

% If the interval was specified, then rescale both the abscissae and
% weights.
if nargin > 1
    x = (b - a)*x/2 + (a + b)/2;
    w = (b - a)*w/2;
end
