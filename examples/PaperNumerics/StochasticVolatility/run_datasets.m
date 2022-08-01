%clear; close all; clc
addpath(genpath('../../../src'))
sd = 1; rng(sd);

%% --- Define test case ---

% define problems
N          = [2000];
Nvalid     = 0;
dx         = 40;
d          = dx+2;
order_vect = [1,2];

% set cv_type
cv_type = 'kFold';

% define model
SV = stoc_volatility(d);

if ~exist('./data', 'dir')
       mkdir('./data')
end

%% -- Build maps --

for Ntrain=N

    % generating training samples
    Xtrain = SV.sample(Ntrain);
    Xvalid = SV.sample(Nvalid);
    
    %% --- Learn maps ---
    
    % learn dense map
    [CM, output] = buildComposedMap(Xtrain, Xvalid, cv_type);
    
    % learn sparse map
    active_variables = cell(d,1); 
    active_variables{1} = 1;
    active_variables{2} = 2;
    for k=3:d
        active_variables{k} = [1,2,k-1:k];
    end
    [CM_sparse, output_sparse] = buildComposedMap(Xtrain, Xvalid, cv_type, active_variables);
    
    % learn maps of fixed order
    CM_totorder = cell(length(order_vect),1);
    output_totorder = cell(length(order_vect),1);
    for j=1:length(order_vect)
        order = order_vect(j);
        [CM_totorder{j}, output_totorder{j}] = buildComposedMap_fixedorder(Xtrain, order);
    end
    
    % save results
    file_name = ['data/map_results_N' num2str(Ntrain)];
    save(file_name)

end

%% --- Helper functions ---

function [CM, output] = buildComposedMap(Xtrain, Xvalid, cv_type, active_variables)

    % extract d
    d = size(Xtrain,2);

    % set active variables
    if nargin < 4
        active_variables = cell(d,1);
        for k=1:d
            active_variables{k} = 1:k;
        end
    end
    
    % set cv_type
    if nargin < 3
        cv_type = 'Split';
    end

    % define reference
    ref = IndependentProductDistribution(repmat({Normal()},1,d));
    
    % define Gaussian map
    G = GaussianPullbackDensity(d, true);
    G = G.optimize(Xtrain);
    Ztrain = G.S.evaluate(Xtrain);
    Zvalid = G.S.evaluate(Xvalid);

    % set max_iter and max_patience
    max_iter = ceil(size(Xtrain,1));
    max_patience = 20;

    % define basis
    basis = HermiteProbabilistPolyWithLinearization();
    basis_k = repmat({basis},1,d);
    for k=1:d
        basis_k{k}.bounds = quantile(Ztrain(:,k),[0.01,0.99]).';
    end
    
    % define TM
    TM = cell(d,1);
    for k = 1:d
        % define polynomial and integrated function
        ninputsk = length(active_variables{k});
        P = ParametericPoly(basis_k(active_variables{k}), zeros(1,ninputsk));
        Sk = IntegratedPositiveFunction(P);
        TM{k} = Sk.set_coeff(0);
    end
    %TM = identity_map(1:d, basis);
    PB = PullbackDensity(TM, ref);
    
    % optimize each component using Xtrain and Xvalid
    %[PB, output] = PB.greedy_optimize(Ztrain, Zvalid, max_iter, 'Split');
    output = cell(d,1);
    for k=1:d
        fprintf('Optimizing component %d\n',k)
        % run greedy approximation on S_valid
        S_validk = PB.S.S{k};
        % extract data
        ZWtraink.X = Ztrain(:,active_variables{k}); ZWtraink.W = ones(size(Ztrain,1),1)/size(Ztrain,1);
        ZWvalidk.X = Zvalid(:,active_variables{k}); ZWvalidk.W = ones(size(Zvalid,1),1)/size(Zvalid,1);
        if strcmp(cv_type, 'Split')
            % run fit
            [S_validk, output{k}] = greedy_fit(S_validk, ref.factors{k}, ...
                            max_iter, ZWtraink, ZWvalidk, max_patience);
            % find optimal number of terms (adding terms originally in S)
            % remove one to account for initial condition
            [~, n_added_terms] = min(output{k}.valid_err);
            opt_terms = PB.S.S{k}.n_coeff + n_added_terms;
            fprintf('Final map: %d terms\n', opt_terms);
            % extract optimal multi-indices
            midx_opt = S_validk.multi_idxs();
            midx_opt = midx_opt(1:opt_terms,:);
            PB.S.S{k} = PB.S.S{k}.set_multi_idxs(midx_opt);
            % run greedy_fit up to opt_terms with all data
            a0 = zeros(opt_terms,1);
            PB.S.S{k} = optimize_component(PB.S.S{k}, ref.factors{k}, a0, ZWtraink, []);
        elseif strcmp(cv_type, 'kFold')
            % define cross-validation splits of data
            n_folds = 5;
            cv = cvpartition(size(Ztrain,1), 'kFold', n_folds);
            % define matrix to store results
            train_error_fold = nan(n_folds, max_iter);
            valid_error_fold = nan(n_folds, max_iter);
            % run greedy_fit on each parition of data
            for fold=1:n_folds
                Sfold = S_validk;
                fprintf('Fold %d/%d\n', fold, n_folds);
                Ntrain_fold = sum(cv.training(fold));
                Ntest_fold  = sum(cv.test(fold)); 
                ZWtrainkfold.X = ZWtraink.X(cv.training(fold),:); ZWtrainkfold.W = ones(Ntrain_fold,1)/Ntrain_fold;
                ZWvalidkfold.X = ZWtraink.X(cv.test(fold),:);     ZWvalidkfold.W = ones(Ntest_fold,1)/Ntest_fold;
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
            [PB.S.S{k}, output{k}] = greedy_fit(PB.S.S{k}, ref.factors{k}, ...
                opt_terms, ZWtraink, ZWvalidk, []);
            output{k}.train_err_fold = train_error_fold;
            output{k}.valid_err_fold = valid_error_fold;
        end
    end
    
    % compose map
    CM = ComposedPullbackDensity({G, PB}, ref);

end

function [CM, output] = buildComposedMap_fixedorder(Xtrain, order)

    % extract d
    d = size(Xtrain,2);

    % define reference
    ref = IndependentProductDistribution(repmat({Normal()},1,d));
    
    % define Gaussian map
    G = GaussianPullbackDensity(d, true);
    G = G.optimize(Xtrain);
    Ztrain = G.S.evaluate(Xtrain);

    % define basis
    basis = HermiteProbabilistPolyWithLinearization();
    basis_k = repmat({basis},1,d);
    for k=1:d
        basis_k{k}.bounds = quantile(Ztrain(:,k),[0.01,0.99]).';
    end
    
    % define TM
    TM = total_order_map(1:d, basis, order);
    PB = PullbackDensity(TM, ref);
    
    % optimize all components using Xtrain
    [PB, output] = PB.optimize(Ztrain);
    
    % compose map
    CM = ComposedPullbackDensity({G, PB}, ref);

end

% -- END OF FILE --
