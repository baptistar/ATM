%clear; close all; clc
addpath(genpath('../../../src'))
addpath(genpath('testProblems'))
sd = 1; rng(sd);

% define folder to save data
if ~exist('./data', 'dir')
  mkdir('data');
end

%% --- Define test case ---

% define problems
problems = {@Housing, @RedWine, @WhiteWine, @Parkinsons};

% define reference, basis, and reg param
n_folds  = 10;
max_iter = [60,60,60,80];

% set parallel processing
n_proc = 5;
c = parcluster('local');
c.NumWorkers = n_proc;
parpool(n_proc);

for i = 1 : length(problems)
     
    % set problem
    P = problems{i}();

    % normalize and split data
    data = P.normalize_data(P.data);

    % split data
    cv = cvpartition(size(data,1), 'kFold', n_folds);
    parfor k=1:n_folds
        fprintf('Data fold %d/%d\n', k, n_folds);
        % split into training/validation + testing data
        Xtrain = data(cv.training(k),:);
        Xtest = data(cv.test(k),:);
        % learn map and save results
        [CM, output] = buildComposedMap(Xtrain, max_iter(i));
        test_case = ['data/' P.name '_cv_fold' num2str(k) '.mat'];
        % save results
        points = struct;
        points.training = Xtrain; 
        points.test = Xtest;
        parSave(test_case, CM, output, points)
    end
    
end

delete(gcp('nocreate'))

function parSave(file_name, CM, output, points)
    save(file_name, 'CM', 'output', 'points');
end

%% --- Helper Functions ---

function [CM, output] = buildComposedMap(Xtrain, max_iter)

    % define reference
    d = size(Xtrain,2);
    ref_d = IndependentProductDistribution(repmat({Normal()},1,d));

    % define Gaussian map
    G = GaussianPullbackDensity(d, true);
    G = G.optimize(Xtrain);
    Ztrain = G.S.evaluate(Xtrain);

    % set max_iter and max_patience
    max_iter = min(max_iter,ceil(size(Xtrain,1)));
    max_patience = 20;
 
    % define basis
    basis = cell(d,1);
    for k=1:d
        basis{k} = HermiteProbabilistPolyWithLinearization();
        quantiles = quantile(Ztrain(:,k),[0.01,0.99]);
        basis{k}.bounds = quantiles.';
    end

    % define TM
    TM = identity_map(1:d, basis);
    PB = PullbackDensity(TM, ref_d);
    
    % optimize each component using Xtrain and Xvalid
    %[PB, output] = PB.greedy_optimize(Ztrain, Zvalid, max_iter, 'Split');
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
            ZWtrainkfold.X = Ztrain(cv.training(fold),1:k); ZWtrainkfold.W = ones(Ntrainfold,1)/Ntrainfold;
            ZWvalidkfold.X = Ztrain(cv.test(fold),1:k); ZWvalidkfold.W = ones(Ntestfold,1)/Ntestfold;
            [~, options] = greedy_fit(Sfold, ref_d.factors{k}, ...
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
        ZWvalid.X = zeros(0,k);    ZWvalid.W = zeros(0,1);
        [PB.S.S{k}, output{k}] = greedy_fit(PB.S.S{k}, ref_d.factors{k}, ...
            opt_terms, ZWtrain, ZWvalid, []);
        output{k}.train_err_fold = train_error_fold;
        output{k}.valid_err_fold = valid_error_fold;
    end

    % compose map
    CM = ComposedPullbackDensity({G, PB}, ref_d);

end

% -- END OF FILE --
