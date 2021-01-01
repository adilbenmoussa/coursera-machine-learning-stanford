function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.


% Cost function J(theta) = J(\theta) =\frac{1}{m}\sum_{i=1}^m{\left[-y^{(i)} \log(h_{\theta}(x^{(i)}))- (1 -y^{(i)}) \log(1- h_{\theta}(x^{(i)}))\right];
% Gradient descent = \frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^m{\left( h_\theta(x^{(i)})-y^{(i)}\right)x_j^{(i)}

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% J = 0;
featuresSize = size(theta);
grad = zeros(featuresSize);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Calculate the cost
hypothisis = sigmoid(X*theta);
sqrError = (-y .* log(hypothisis)) - ((1 - y) .* log(1 - hypothisis));
J = (1/m) * sum(sqrError);


% Calculate the gradient
    for featuresSizeIter = 1:length(theta)
        hypothisisThetaError = (hypothisis - y) .* X(:, featuresSizeIter);
        grad(featuresSizeIter) = (1 / m) * sum(hypothisisThetaError);
    end
% =============================================================

end
