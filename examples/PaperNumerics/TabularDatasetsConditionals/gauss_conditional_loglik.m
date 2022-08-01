function [mean_ll, std_ll] = gauss_conditional_loglik(Xtrain, Xtest, Ytrain, Ytest)

% determine the dimension
[N,d] = size(Xtest);
 
% standardize data
mean_x = mean(Xtrain,1); sigma_x = std(Xtrain,[],1);
Xtrain = (Xtrain - mean_x) ./ sigma_x;
Xtest  = (Xtest - mean_x) ./ sigma_x;
mean_y = mean(Ytrain,1); sigma_y = std(Ytrain,[],1);
Ytrain = (Ytrain - mean_y) ./ sigma_y;
Ytest  = (Ytest - mean_y) ./ sigma_y;

% compute gaussian approximation from training data
mu_y = mean(Ytrain,1);
mu_x = mean(Xtrain,1);
C_y  = cov(Ytrain,1);
C_x  = cov(Xtrain,1);
C_xy = cov([Xtrain, Ytrain]); C_xy = C_xy(1,2:end);
mu_cond = mu_x + (C_xy*(C_y\((Ytest - mu_y)')))';
C_cond  = C_x - C_xy*(C_y\C_xy');

% extract log-likelihood
test_loglik = zeros(size(Xtest,1),1);
for k=1:size(Xtest,1)
    test_loglik(k) = -0.5*(d*log(2*pi) + log(det(C_cond)) + (Xtest(k,:) - mu_cond(k,:))*(C_cond\(Xtest(k,:) - mu_cond(k,:))'));
end

% compute statistics of test_loglik
mean_ll = mean(test_loglik) + sum(log(1./sigma_x));
std_ll  = 1.96*std(test_loglik)/sqrt(size(Xtest,1));

end
