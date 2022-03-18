function [S, output] = greedy_basis_selection(S, ref, XW, XW_test, ...
                            max_terms, stopping, alpha)
                        
    X=XW.X;
    W=XW.W;
    
    % set default alpha, stopping, and Xvalid
    if (nargin < 7) || isempty(alpha)
        alpha = 0;
    end
    if (nargin < 6) || isempty(stopping)
        stopping = 'max_terms';
    end
    if (nargin < 5) || isempty(max_terms)
        max_terms = size(X,1) - 1;
    end
    if (nargin < 4) || isempty(XW_test)
        X_test = zeros(0,size(X,2));
        W_test=(1/size(X_test,1))*ones(size(X_test,1),1);
        XW_test.X=X_test;
        XW_test.W=W_test;
    end

    % run forward greedy approximations up to max_terms
    if strcmp(stopping, 'max_terms')

        [S, output] = greedy_fit(S, ref, max_terms, XW, XW_test, [], alpha);

    elseif strcmp(stopping, 'kFold')

        % define cross-validation splits of data
        n_folds = 5;
        cv = cvpartition(size(X,1), 'kFold', n_folds);

        % define matrix to store results
        train_error_fold = nan(n_folds, max_terms);
        valid_error_fold = nan(n_folds, max_terms);

        % run greedy_fit on each parition of data
        for fold=1:n_folds
            S_cv = S;
            fprintf('Fold %d/%d\n', fold, n_folds);

            XW_train.X=X(cv.training(fold),:);
            XW_train.W=W(cv.training(fold),:);
            
            XW_valid.X=X(cv.test(fold),:);
            XW_valid.W=W(cv.test(fold),:);

            [~, options] = greedy_fit(S_cv, ref, max_terms, XW_train, XW_valid, [], alpha);
            train_error_fold(fold, 1:length(options.train_err)) = options.train_err;
            valid_error_fold(fold, 1:length(options.valid_err)) = options.valid_err;
        end

        % find optimal number of terms
        % remove one to account for initial condition
        mean_valid_error = mean(valid_error_fold, 1);
        [~, n_added_terms] = min(mean_valid_error);
        n_added_terms = n_added_terms(1) - 1;
        opt_terms = S.n_coeff + n_added_terms;

        % run greedy_fit up to opt_terms with all data
        fprintf('Final run\n');
        [S, output] = greedy_fit(S, ref, opt_terms, XW, XW_test, [], alpha);
        output.train_error_fold = train_error_fold;
        output.valid_error_fold = valid_error_fold;

    elseif strcmp(stopping, 'Split')

        % set defaults
        train_valid_split = 0.8;
        max_patience = 20;

        % split data into training and validation
        N_train   = floor(train_valid_split*size(X,1));
        train_idx = randperm(size(X,1),N_train);
        valid_idx = setdiff(1:size(X,1),train_idx);
        X_train   = X(train_idx,:);
        X_valid   = X(valid_idx,:);
        W_train=W(train_idx,:);
        W_valid=W(valid_idx,:);
        
        XW_train.X=X_train;
        XW_train.W=W_train;
        
        XW_valid.X=X_valid;
        XW_valid.W=W_valid;
        

        % run greedy approximation on S_valid
        S_valid = S;
        [S_valid, output] = greedy_fit(S_valid, ref, max_terms, ...
                                XW_train, XW_valid, max_patience, alpha);

        % find optimal number of terms (adding terms originally in S)
        % remove one to account for initial condition
        [~, n_added_terms] = min(output.valid_err);
        n_added_terms = n_added_terms(1) - 1;
        opt_terms = S.n_coeff + n_added_terms;
        fprintf('Final map: %d terms\n', opt_terms);

        % extract optimal multi-indices
        midx_opt = S_valid.multi_idxs();
        midx_opt = midx_opt(1:opt_terms,:);
        S = S.set_multi_idxs(midx_opt);

        % run greedy_fit up to opt_terms with all data
        a0 = zeros(opt_terms,1);
        S = optimize_component(S, ref, a0, XW_train, [], alpha);

    else
        error('TMC: stopping is not recognized')
    end
    
end

% -- END OF FILE --