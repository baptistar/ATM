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
    
    % print timing and average params
    tmap_timing    = zeros(n_folds, 1);
    tmap_tottiming = zeros(n_folds, 1);
    tmap_nparams   = zeros(n_folds, 1);

    for j=1:n_folds

        %% load training and test data
        load(['data_timing/' problems{p} '_cv_fold' num2str(j)],'data');

        % extract data
        tmap_timing(j)    = data.times_opt(end);
        tmap_tottiming(j) = sum(data.times_cv(end,:)) + data.times_opt(end);
        tmap_nparams(j)   = data.n_params;

    end

    disp(problems{p})
    fprintf('Time %.2f \\pm %.2f$\n', mean(tmap_timing), 1.96*std(tmap_timing)/sqrt(n_folds))
    fprintf('Total time %.2f \\pm %.2f$\n', mean(tmap_tottiming), 1.96*std(tmap_tottiming)/sqrt(n_folds))
    fprintf('Nparams %.2f \\pm %.2f$\n', mean(tmap_nparams), 1.96*std(tmap_nparams)/sqrt(n_folds))

end

% -- END OF FILE --
