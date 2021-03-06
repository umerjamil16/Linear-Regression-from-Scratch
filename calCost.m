function J = calCost(X, Y, theta, iter)
    m = length(Y); % number of training examples
    J = (1/(2*m))*sumFunc(((mulFunc(X, theta)) - Y).^2); %Implementation of cost function for multivariate linear regression
    fprintf('Current Epoch: %i and Corresponding CostFunction val: %f\n', iter, J);
end
