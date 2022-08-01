%clear; close all; clc
addpath(genpath('../../../src'))
addpath(genpath('testProblems'))

%% --- Define test case ---

% define problems
problems = {@Housing, @RedWine, @WhiteWine, @Parkinsons};
data_folder = './data';
folds = 1:10;
n_folds = length(folds);
cv_type = '_cv';
orders  = 1:2;

% run each test case
for p=1:length(problems)

    P = problems{p}();
    fprintf('Processing adaptive map: %s\n', P.name);
    
    % define array to store results
    test_err_atm      = zeros(n_folds,1);
    test_err_gauss    = zeros(n_folds,1);
    test_err_nonadapt = zeros(length(orders),n_folds);
    ncoeffs_atm       = zeros(n_folds,1);
    ncoeffs_nonadapt  = zeros(length(orders),n_folds);

    % run each fold    
    for nfi = 1:length(folds)
        nf = folds(nfi);
        
        % load data and results
        test_case = [data_folder '/' P.name cv_type '_fold' num2str(nf) '.mat']; 
        load(test_case,'output','CM','points');

        % compute ATM log-lik at optimal map
        test_log_pdf = CM.log_pdf(points.test);
        test_log_pdf = test_log_pdf(test_log_pdf > -1e4);
        test_err_atm(nf) = mean(test_log_pdf);
        ncoeffs_atm(nf)  = CM.S{2}.n_coeff;

        % compute Gaussian log-lik
        [test_err_gauss(nf),~] = gauss_loglik(points.training, points.test);

        % compute log-lik with non-adaptive maps
        for j=1:length(orders)
            test_case = [data_folder '/' P.name '_fold' num2str(nf) '_nonadapt_order' num2str(orders(j)) '.mat']; 
            load(test_case,'CM');
            test_log_pdf = CM.log_pdf(points.test);
            test_err_nonadapt(j,nf) = mean(test_log_pdf);
            ncoeffs_nonadapt(j,nf) = CM.S{2}.n_coeff;
        end

    end

    stderror = @(data) 1.96*std(data)/sqrt(length(data));

    % plot results
    fprintf('Gauss %.3f pm %.3f\n', mean(test_err_gauss), stderror(test_err_gauss))
    fprintf('ATM %.3f pm %.3f, coeffs: %.1f pm %.1f\n', mean(test_err_atm), stderror(test_err_atm), mean(ncoeffs_atm), stderror(ncoeffs_atm));
    for j=1:length(orders)
        fprintf('Nonadapt-order %d, %.3f pm %.3f, coeffs: %.1f pm %.1f\n', orders(j), mean(test_err_nonadapt(j,:)), stderror(test_err_nonadapt(j,:)), mean(ncoeffs_nonadapt(j,:)), stderror(ncoeffs_nonadapt(j,:)));
    end

end

% -- END OF FILE --
