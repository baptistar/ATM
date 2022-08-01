%clear; close all; clc
addpath(genpath('../../../src'))
sd = 1; rng(sd);

%% --- Define test case ---

% define problems
problems = {'Housing','Yacht','Energy','Concrete'};

% define orders, sample size, and MC iterations
n_folds = 10;

% run each test case
for p=1:length(problems)
    
    % evalute log-likelihood on test set
    tmap_loglik_mean  = zeros(n_folds, 1);
    tmap_loglik_stde  = zeros(n_folds, 1);
    gauss_loglik_mean = zeros(n_folds, 1);
    gauss_loglik_stde = zeros(n_folds, 1);

    for j=1:n_folds

        %% load training and test data
        load(['data/' problems{p} '_cv_fold' num2str(j)],'points','CM');

        % extract data
        Xtrain = points.training(:,end);
        Xtest  = points.test(:,end);
        Ytrain = points.training(:,1:end-1);
        Ytest  = points.test(:,1:end-1); 
        
        % evaluate gaussian log-likelihood
        [gauss_loglik_mean(j), gauss_loglik_stde(j)] = gauss_conditional_loglik(Xtrain, Xtest, Ytrain, Ytest);

        % evaluate transport map log-lik
        component = size([Ytest, Xtest],2);
        test_loglik = CM.log_pdf([Ytest, Xtest],component);
        tmap_loglik_mean(j) = mean(test_loglik);
        tmap_loglik_stde(j) = 1.96*std(test_loglik)/sqrt(size(Xtest,1));

    end

    disp(problems{p})
    fprintf('d = %d, n = %d\n', size(Xtrain,2)+size(Ytrain,2), size(Xtrain,1)+size(Xtest,1));
    fprintf('Gaussian: $%.2f \\pm %.2f$\n', -1*mean(gauss_loglik_mean), 1.96*std(gauss_loglik_mean)/sqrt(n_folds))
    fprintf('ATM: $%.2f \\pm %.2f$\n', -1*mean(tmap_loglik_mean), 1.96*std(tmap_loglik_mean)/sqrt(n_folds))

end

% -- END OF FILE --
