function [mean_ll, std_ll] = gauss_loglik(Xtrain, Xtest)

% determine the dimension
[N,d] = size(Xtest);

% compute gaussian approximation from training data
mu = mean(Xtrain,1);
C  = cov(Xtrain);

% extract each log-likelihood
test_loglik = zeros(N,1);
for k=1:N
    test_loglik(k) = -0.5*(d*log(2*pi) + log(det(C)) + ...
        (Xtest(k,:) - mu)*(C\(Xtest(k,:) - mu)'));
end

% compute statistics of test_loglik
mean_ll = mean(test_loglik);
std_ll  = std(test_loglik);

end
