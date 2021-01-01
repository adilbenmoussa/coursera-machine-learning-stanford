function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

C_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
error_map = zeros(1, 3);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


i = 1;
for cIter = 1:length(C_vec)
    for sIter = 1:length(sigma_vec)
        C_local = C_vec(cIter);
        sigma_local = sigma_vec(sIter);
        error_map(i, 1) = C_local;
        error_map(i, 2) = sigma_local;
        model= svmTrain(X, y, C_local, @(x1, x2) gaussianKernel(x1, x2, sigma_local)); 
        predictions = svmPredict(model, Xval);
        error_map(i, 3) = mean(double(predictions ~= yval));
        i = i+1;
    end
end;

% Get the error vector
error_val = error_map(:, 3);
C_map = error_map(:, 1);
sigma_map = error_map(:, 2);
[~, minIndex] = min(error_val);

C = C_map(minIndex);
sigma = sigma_map(minIndex);



% =========================================================================

end
