%% gauss_hermite
% Compute the abscissae (x) and weights (w) for Gauss-Hermite quadrature.
%% Syntax
%# [x, w] = gauss_hermite(n);
%# [x, w] = gauss_hermite(n, m, sigma);
%% Description
% The function gauss_hermite computes the abscissae (x) and weights (w)
% needed for Gauss-Hermite quadrature using the symmetric tridiagonal
% Jacobi matrix J of the Hermite polynomials. The values of x are derived
% from the eigenvalues of J, and the weights from the first elements of the
% eigenvectors. This method is numerically more reliable than using the
% MATLAB function roots(...), since the symmetry of J guarantees that the
% abscissae and weights will be real even for large n.
% * n - An integer > 0 giving the number of abscissae and weights.
% * m, sigma - Real numbers defining the generalized weight function
% exp(-(x - m)^2/sigma^2); if these arguments are absent, then the default
% values m = 0 and sigma = 1 are used.
% * x - A column vector containing the abscissae.
% * w - A row vector containing the weights.
%% Example
% Integral over [-infinity, infinity] with W(x) = exp(-x^2); analytic
% result computed using Mma.
%# npts = 15;
%# f = inline('1./(1 + x.^2)', 'x');
%# analytic = 1.343293421646735;
%# [x, w] = gauss_hermite(npts);
%# quad = w * f(x);
%# fprintf('Gauss-Hermite Result = %f; error = %e\n', quad, quad - analytic);
%% References
% * G. H. Golub and J. H. Welsch, "Calculation of Gauss Quadrature Rules,"
% Mathematics of Computation 23, 221 (1969).
% * W. H. Press, S. A. Teukolsky, W. T. Vetterling, and B. P. Flannery,
% Numerical Recipes: The Art of Scientific Computing (Cambridge University
% Press, 2007), Sections 4.6.1 and 4.6.2.
%
function [x, w] = gauss_hermite(n, m, sigma)

% Construct the symmetric tridiagonal Jacobi matrix J.
j = 1:n-1;
beta = sqrt(j/2);
J = diag(beta,1) + diag(beta,-1);

% Specify the integral of the weight function W(x) over the exact interval.
mu_0 = sqrt(pi);

% The abscissae will be the (sorted) real eigenvalues of J, and the weights
% can be found from the first element of the corresponding eigenvectors.
[v, lambda] = eig(J);
[x, ix] = sort(diag(lambda));
v = v(:,ix);
w = mu_0 * v(1,:).^2;

% If the weight function parameters were specified, then rescale both the
% abscissae and weights.
if nargin > 1
    x = sigma*x + m;
    w = sigma*w;
end
