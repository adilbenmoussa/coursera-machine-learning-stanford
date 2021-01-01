function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
grad_not_regularized = zeros(size(theta));

% Skip biad θ(0)
temp_theta = theta;
temp_theta(1) = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Calculate the hypothesis
hypothesis = X * theta;

% Calculate the square error 
sqrError = sum((hypothesis - y).^2);
J_not_regularized = (1/ (2 * m)) * sqrError;

% Calculate the regularization
regularization = (lambda/ (2 * m)) * sum(temp_theta.^2);

% Calculate the total cost
J = J_not_regularized + regularization;

% Calculate the grad
 for i = 1:length(theta)
    hypothisisThetaError = (hypothesis - y) .* X(:, i);
    grad_regularized(i) = (lambda/m) * temp_theta(i);
    grad(i) = ((1 / m) * sum(hypothisisThetaError)) + grad_regularized(i);
end

grad = grad(:);

end
