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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(size(X,1),1), X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(X,1),1), a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
h = a3;

ymatrix = zeros(size(y,1) , num_labels);

for i=1:m,
  index = y(i);
  ymatrix(i,index) = 1;
endfor

J = (1/m)*sum(sum((-ymatrix.*log(h) - (1-ymatrix).*(log(1-h)))));

% bias 는 제거!! (401 중 맨 앞 1 떼어내고 제곱)
J += lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% cost 함수에서 y 는 1~10 의 값이 아닌 0과 1로 표현된 값이므로 vector 을 matrix 로.
% y^(i) 는 10차원 벡터여야 하므로(신경망 학습을 위해) 전체 dataset 은 5000*10 matrix.

%위는 잘 동작. 그러나 back propagation 에서 잘 안된다. 일단 추천한 대로 for loop 써보자.
%아니야.... 행렬 계산 실수해서 그래... 피드백하라.

delta3 = h - ymatrix;
delta2 = delta3*Theta2(:,2:end).*sigmoidGradient(z2);
%theta2 (10 * 26) 의 bias 부분을 떼어내서(10 * 25) 계산.
%bias 까지 다 포함시키고 싶으면(거의 안 할듯) 떼어내지 말고 sigmoid 도함수에 1 붙여야 한다.


Theta1_grad = (delta2'*a1)/m;
Theta2_grad = (delta3'*a2)/m;

Theta1_grad += lambda/m*Theta1;
Theta1_grad(:,1) -= lambda/m*Theta1(:,1);
Theta2_grad += lambda/m*Theta2;
Theta2_grad(:,1) -= lambda/m*Theta2(:,1);


##r1 = 0;
##r2 = 0;
##
##for i=1:m,
##  a1 = X(i,:)';
##  a1 = [1; a1];
##  z2 = Theta1*a1;
##  a2 = sigmoid(z2);
##  a2 = [1; a2];
##  z3 = Theta2*a2;
##  a3 = sigmoid(z3);
##  
##  tempy = zeros(num_labels, 1);
##  tempy(y(i)) = 1; % y 가 정수값으로 이루어져있으므로 벡터로 mapping.
##  d3 = a3 - tempy;
##  d2 = Theta2'*d3.*[1; sigmoidGradient(z2)]; %여기서 의문. 다시 이해하자.
##  d2 = d2(2:end); %delta^(l) 중에서 0번째는 없다. (bias 의 delta 없음)
##  r1 += d2*a1';
##  r2 += d3*a2'; 
##endfor
##
##Theta1_grad = r1/m + lambda/m*Theta1;
##Theta1_grad(:,1) -= lambda/m*Theta1(:,1);
##Theta2_grad = r2/m + lambda/m*Theta2;
##Theta2_grad(:,1) -= lambda/m*Theta2(:,1);

##행렬 덧셈, 뺄셈할 때 크기비교 똑바로 하자. 10x3 +/- 10x1 하면 왼쪽의 모든 열마다
##뒤의 벡터를 덧셈/뺄셈 하게된다... 이것때문에 시간 너무 많이 잡아먹음 조심하자.
##행렬 연산할 때 크기를 써 내려가면서 계산하자.
##
##그리고 역시 for loop 보다 vertorizing 이 훨~씬 효율적이다. 계산 시간만 봐도 알 수 있다.
##벡터화에 친숙해지자.

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
