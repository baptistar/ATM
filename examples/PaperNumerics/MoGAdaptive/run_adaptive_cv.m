%clear; close all; clc
sd = 1; rng(sd);

%% --- Define test case ---

% define grid of tests
[testD, testN, testP] = meshgrid(d_vect, N_vect, MCruns);

% setup parallel for loop
n_proc = 5;
c = parcluster('local');
c.NumWorkers = n_proc;
parpool(n_proc);

parfor i = 1 : numel(testP)

    % extract d, P and N
    d  = testD(i);
    N  = testN(i);
    j  = testP(i);
    
    % define name for output file
    test_case = ['data/MoG_d' num2str(d) '_N' num2str(N) '_run' num2str(j) '.mat'];
    if isfile(test_case)
        continue
    end

    % load training and test samples
    file_name = ['samples/MoG_d' num2str(d) '_N' num2str(N) '_run' num2str(j)];
    Xtrain = parLoad(file_name);

    % learn map and save results
    [CM, output] = buildComposedMap(Xtrain);
    parSave(test_case, CM, output);
    
end

delete(gcp('nocreate'))
    
%% -- Helper functions --

function Xtrain = parLoad(file_name)
    load(file_name, 'Xtrain');
end

function parSave(file_name, CM, output)
    save(file_name, 'CM', 'output');
end

function [CM, output] = buildComposedMap(Xtrain, Xvalid)
    if nargin < 2
        Xvalid = zeros(0,size(Xtrain,2));
    end

    % define reference
    d = size(Xtrain,2);
    ref = IndependentProductDistribution(repmat({Normal()},1,d));

    % define Gaussian map
    G = GaussianPullbackDensity(d, true);
    G = G.optimize(Xtrain);
    Ztrain = G.S.evaluate(Xtrain);
    Zvalid = G.S.evaluate(Xvalid);

    % set max_iter and max_patience
    max_iter = min(200,ceil(size(Xtrain,1)));
    max_patience = 20;

    % define bounds
    basis = cell(d,1);
    for k=1:d
        basis{k} = HermiteProbabilistPolyWithLinearization();
        basis{k}.bounds = quantile(Ztrain(:,k),[0.01,0.99]).';
    end
    
    % define TM
    TM = identity_map(1:d, basis);
    PB = PullbackDensity(TM, ref);
 
    % optimize map
    %[PB, output] = PB.greedy_optimize(Z, Ztest, max_iter, 'kFold');
    output = cell(d,1);
    for k=1:d
        % define cross-validation splits of data
        n_folds = 5;
        cv = cvpartition(size(Ztrain,1), 'kFold', n_folds);
        % run greedy approximation on S_valid
        S_validk = PB.S.S{k};
        % define matrix to store results
        train_error_fold = nan(n_folds, max_iter);
        valid_error_fold = nan(n_folds, max_iter);
        % run greedy_fit on each parition of data
        for fold=1:n_folds
            Sfold = S_validk;
            fprintf('Fold %d/%d\n', fold, n_folds);
            Ntrainfold = sum(cv.training(fold));
            Ntestfold = sum(cv.test(fold));
            ZWtrainkfold.X  = Ztrain(cv.training(fold),1:k); ZWtrainkfold.W = ones(Ntrainfold,1)/Ntrainfold;
            ZWvalidkfold.X  = Ztrain(cv.test(fold),1:k); ZWvalidkfold.W = ones(Ntestfold,1)/Ntestfold;
            [~, options] = greedy_fit(Sfold, ref.factors{k}, ...
                max_iter, ZWtrainkfold, ZWvalidkfold, max_patience);
            train_error_fold(fold, 1:length(options.train_err)) = options.train_err;
            valid_error_fold(fold, 1:length(options.valid_err)) = options.valid_err;
        end
        % find optimal number of terms
        % remove one to account for initial condition
        mean_valid_error = mean(valid_error_fold, 1);
        [~, n_added_terms] = min(mean_valid_error);
        n_added_terms = n_added_terms(1) - 1;
        opt_terms = PB.S.S{k}.n_coeff + n_added_terms;
        % run greedy_fit up to opt_terms with all data
        fprintf('Final run\n');
        ZWtrain.X = Ztrain(:,1:k); ZWtrain.W = ones(size(Ztrain,1),1)/size(Ztrain,1);
        ZWvalid.X = Zvalid(:,1:k); ZWvalid.W = ones(size(Zvalid,1),1)/size(Zvalid,1);
        [PB.S.S{k}, output{k}] = greedy_fit(PB.S.S{k}, ref.factors{k}, ...
            opt_terms, ZWtrain, ZWvalid, []);
        output{k}.train_err_fold = train_error_fold;
        output{k}.valid_err_fold = valid_error_fold;
    end
    
    % compose map
    CM = ComposedPullbackDensity({G, PB}, ref);

end

% -- END OF FILE --
