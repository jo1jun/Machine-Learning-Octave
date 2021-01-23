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

% landmark 선택은 x data 그자체로 시작
##x1 = X;
##x2 = X;
##
##values = [0.01 0.03 0.1 0.3 1 3 10 30];
##means = zeros(64, 1);
##
##k = 1;
##for i=1:8,
##  for j=1:8,
##    C = values(i);
##    sigma = values(j);
##    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
##    predictions = svmPredict(model, Xval);
##    means(k,1) = mean(double(predictions ~= yval));
##    k++;
##  endfor
##endfor
##
##[val, idx] = min(means);
##
##if idx <= 8,
##  i = 1;
##  j = idx;
##elseif idx <= 16,
##  i = 2;
##  j = idx - 8;
##  elseif idx <= 24,
##  i = 3;
##  j = idx - 16;
##  elseif idx <= 32,
##  i = 4;
##  j = idx - 24;
##  elseif idx <= 40,
##  i = 5;
##  j = idx - 32;
##   elseif idx <= 48,
##  i = 6;
##  j = idx - 40;
##   elseif idx <= 56,
##  i = 7;
##  j = idx - 48;
##   elseif idx <= 64,
##  i = 8;
##  j = idx - 56;
##endif
##
##
##C = values(i);
##sigma = values(j);

C = 1;
sigma = 0.1;
% =========================================================================

end
