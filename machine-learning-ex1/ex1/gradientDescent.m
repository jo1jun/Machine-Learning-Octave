function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
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
    
    %delta = zeros(2,1);
    %for i=1:m,
    %  delta += (X(i,:)*theta - y(i))*X(i,:)'; %����� delta �� n+1 ����!
    %endfor
    %delta /= m;
    %theta -= alpha*delta;
    
    %�� vectorizing �غ���.
    theta -= alpha*X'*(X*theta - y)/m;
    %����! ��κ��� for ���� vectorerize �� �ذ��� �� �ִ�! �Դٰ� �ξ� ȿ����.
    %�ؼ��� ���� �ɸ��� ������.
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
