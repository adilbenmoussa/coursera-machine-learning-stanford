function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

%{
    To test this run first the comments in computeCost.m
    and then this:
    
    theta = gradientDescent(X, y, theta, alpha, iterations);

    fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2))

    hold on; % keep previous plot visible
    plot(X(:,2), X*theta, '-')
    legend('Training data', 'Linear regression')
    hold off % don't overlay any more plots on this figure

    % Predict values for population sizes of 35,000 and 70,000
    predict1 = [1, 3.5] *theta;
    fprintf('For population = 35,000, we predict a profit of %f\n', predict1*10000);
    predict2 = [1, 7] * theta;
    fprintf('For population = 70,000, we predict a profit of %f\n', predict2*10000);
%}

% Rever lecture 4!!
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % ============================================================

    hypothisis = X*theta;
    hypothisisTheta1Error = (hypothisis - y); %  = (hypothisis - y).* X(:, 1);
    hypothisisTheta2Error = (hypothisis - y) .* X(:, 2);
    temp_theta1 = theta(1) - (alpha / m) * sum(hypothisisTheta1Error);
    temp_theta2 = theta(2) - (alpha / m) * sum(hypothisisTheta2Error);

    theta(1) = temp_theta1;
    theta(2) = temp_theta2;

    % fprintf("%f", theta);

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
