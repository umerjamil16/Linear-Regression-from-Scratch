function [theta, J_values] = GD(X, Y, theta, epochs, alpha)
%mulFunc(A, B) multiple two matrices A and B and returns the resulting Matrice

m = length(Y); % number of training examples
J_values = createMatrix(epochs, 1, 0);  % To log values of Cost Function J

for iter = 1:epochs
    theta = theta - (alpha/m)*mulFunc((X'), (mulFunc(X, theta) - Y));%theta update equation for multivariate linear regression
	J_values(iter) = calCost(X, Y, theta, iter); %To compute cost function
end
end