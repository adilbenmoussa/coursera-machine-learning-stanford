function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
n = size(X, 2);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% fprintf('\nTheta1 size: [%d x %d]', size(Theta1, 1), size(Theta1, 2));
% fprintf('\nTheta2 size: [%d x %d]', size(Theta2,1), size(Theta2, 2));
% fprintf('\nX size: [%d x %d]\n', m, n);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones(size(z2,1), 1) sigmoid(z2)]; % fprintf('Z2 size: [%d x %d]\n', size(z2,1), size(z2,2));
z3 = a2 * Theta2'; %fprintf('Z3 size: [%d x %d]\n', size(z3,1), size(z3,2));
a3 = sigmoid(z3); %fprintf('A3 size: [%d x %d]\n', size(a3,1), size(a3,2));

[~, p] = max(a3, [], 2);
% =========================================================================


end
