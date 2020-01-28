function [r2, RMSE] = modelEval(Predicted, Actual)

RMSE = sqrt(meanFunc(( Actual - Predicted).^2));  % Root Mean Squared Error

%Implementation of formulae of R-Squared
a=sumFunc((Actual-Predicted).^2);
b=sumFunc((Actual-meanFunc(Actual)).^2);    
r2=1 - a/b; 

end