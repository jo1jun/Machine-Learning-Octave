function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J = -y'* log(sigmoid(X*theta)) -(1-y)'*log(1-sigmoid(X*theta));
J/=m;

fprintf('%f', J)

tempJ = 0;
for i=2:size(theta),
  tempJ += theta(i)^2;
endfor
tempJ *= lambda/(2*m);
J += tempJ;

  
%for i=2:size(theta),
%  grad(i) = (sigmoid(X*theta) - y)'*X(:,i);
%  grad(i) += lambda*theta(i);
%endfor

%�ξ� �����ϰ� ȿ����!
grad = X'*(sigmoid(X*theta) - y) + lambda*theta;
grad(1) = (sigmoid(X*theta) - y)'*X(:,1);

grad /= m;

% vectorizing �� ���� ȿ�����̴�! �ݵ�� �����ص� ��.

% =============================================================

end
