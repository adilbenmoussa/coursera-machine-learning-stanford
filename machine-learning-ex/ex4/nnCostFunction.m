function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Y = eye(num_labels); % dim = num_labelsxnum_labels and y dim = 5000x1
X = [ones(m, 1), X];
sqr_error = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% 1. Implement FP to get hypothisis(x^i) for any x^i
a1 = X;
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3); % dim = 5000x10

% For loop version
% for i = 1:m
%     H_k = a3(i, :); % grab every row => will output: [0.1, 0.2 .... 0.1] num_labels times
%     label_val = y(i); % label_val dim => 1x1 => will output: 1, 2 ... or 10
%     Y_k = Y(label_val, :); % dim = 10x1 => will output: vectors like [1;0;0..0] or [0;1;0;0..0] etc...
%     sqr_error += (-Y_k .* log(H_k)) - ((1 - Y_k) .* log(1 - Y_k));
% end
% J = (1/m) * sum(sqr_error);

% Verctorized version
Y_k = Y(y,:);
H_k = a3;
sqr_error = (-Y_k .* log(H_k)) - ((1 - Y_k) .* log(1 - H_k));
% inner sum on each sample in the row
cost_sum1 = sum(sqr_error);
% outer sum on each weight in the column
cost_sum2 = sum(cost_sum1);

% Regularization:
temp_Theta1 = Theta1;
temp_Theta1(:, 1)  = zeros(size(Theta1, 1), 1);

temp_Theta2 = Theta2;
temp_Theta2(:, 1)  = zeros(size(Theta2, 1), 1);

reg_theta1_sum1 = sum(temp_Theta1.^2);
reg_theta1_sum2 = sum(reg_theta1_sum1);

reg_theta2_sum1 = sum(temp_Theta2.^2);
reg_theta2_sum2 = sum(reg_theta2_sum1);

regularization = (lambda/(2*m)) * (reg_theta1_sum2 + reg_theta2_sum2);

% Const function:
J = (1/m) * cost_sum2 + regularization;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% BP for loop version:
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

% 1. Set the input layer's values to the t-th training example . Perform a feedforward pass (Figure 2),
% computing the activations for layers 2 and 3. Note that you need to add a term to
% ensure that the vectors of activations for layers and also include the bias unit. In MATLAB, if a_1
% is a column vector, adding one corresponds to a_1 = [1; a_1].

for i = 1:m
    % Calculate activation layers  %
    % Input layer (1)
    a1_i = X(i, :);
    y_i = Y_k(i, :);

    % Hidden layer (2)
    z2_i = a1_i * Theta1';
    a2_i = sigmoid(z2_i);
    a2_i = [1, a2_i];

    % Output layer (3)
    z3_i = a2_i * Theta2';
    a3_i = sigmoid(z3_i);

    % 2. For each output unit k in layer 3 (the output layer), set where indicates
    % whether the current training example belongs to class k , or if it belongs to a different class
    % .You may find logical arrays helpful for this task (explained in the previous programming
    % exercise).

    % Calculate netword error %
    % Output layer error (3)
    delta3_i = a3_i - y_i;
    
    % 3. For the hidden layer , set delta =...
    % Output layer error (3)
    delta2_i_part1_with_bias = Theta2' * delta3_i';
    % remove the delta0 (bias)
    delta2_i_part1_without_bias = delta2_i_part1_with_bias(2:end, :);
    delta2_i_part2 = sigmoidGradient(z2_i);
    delta2_i = delta2_i_part1_without_bias' .* delta2_i_part2;

    % 4. Accumulate the gradient from this example using the following formula: .
    % Note that you should skip or remove . In MATLAB, removing corresponds to delta_2 =
    % delta_2(2:end).
    Delta1 = Delta1 + delta2_i'*a1_i;
    Delta2 = Delta2 + delta3_i'*a2_i;
end

    % 5. Obtain the (unregularized) gradient for the neural network cost function by dividing the accumulated
    % gradients by formula

    Theta1_regularization = (lambda/m) * temp_Theta1;
    Theta2_regularization = (lambda/m) * temp_Theta2;
    Theta1_grad = ((1/m)*Delta1) + Theta1_regularization;
    Theta2_grad = ((1/m)*Delta2) + Theta2_regularization;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
