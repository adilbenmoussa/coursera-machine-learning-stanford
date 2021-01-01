function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% J = 0;
grad = zeros(size(theta));
thetaPopFirst = theta(2:end);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



% Calculate the cost
hypothisis = sigmoid(X*theta);
sqrError = (-y .* log(hypothisis)) - ((1 - y) .* log(1 - hypothisis));
tempJ = (1/m) * sum(sqrError); % => tempJ = mean(sqrError)
regularization = (lambda/(2*m)) * sum(thetaPopFirst.^2);

J = tempJ + regularization;


% Calculate the gradient
    for featuresSizeIter = 1:length(theta)
        hypothisisThetaError = (hypothisis - y) .* X(:, featuresSizeIter);
        deriveJ = (1 / m) * sum(hypothisisThetaError);
        if featuresSizeIter == 1
            grad(featuresSizeIter) = deriveJ;
        else 
            gradRegularization = (lambda/m) * theta(featuresSizeIter);
            grad(featuresSizeIter) = deriveJ + gradRegularization;
        end
    end



% =============================================================

end
