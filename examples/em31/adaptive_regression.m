function [S, output] = greedy_fit(S, ref, max_terms, XWtrain, XWvalid, ...
                            max_patience, alpha, max_degree, verbose)
    % Inputs:  TMc          - object for transport map component
    %          max_terms    - maximum number of iterations to run
    %          Xtrain       - (N x d) array of training samples
    %          Xvalid       - (optional) array of validation samples
    %          max_patience - (optional) patience for stopping algorithm
    %          max_degree   - (optional) maximum degree for basis functions
    % Outputs: TMc          - component with optimal coefficients
    %          output       - struct containing training and test error

    %% Initialize algorithm
    
    Xtrain=XWtrain.X;
    
    % set verbosity level (default is true)
    if (nargin < 10)
        verbose = true;
    end
    % set criteria for greedy_fit
    if (nargin < 9)
        criteria = 'gradient';
    end
    % set maximum degree (default: inf)
    if (nargin < 8) || isempty(max_degree)
        max_degree = inf;
    end
    % set default alpha
    if (nargin < 7) || isempty(alpha)
        alpha = 0;
    end
    % set maximum patience (termination criteria) to be infinite if not specified
    if (nargin < 6) || isempty(max_patience)
        max_patience = Inf;
    end
    % set validation samples
    if (nargin < 5) || isempty(XWvalid)
        Xvalid = zeros(0, size(Xtrain,2));
        Wvalid=(1/size(Xvalid,1))*ones(size(Xvalid,1),1);
        XWvalid.X=Xvalid;
        XWvalid.W=Wvalid;

    end
    
    Xvalid=XWvalid.X;    
    
    % check max_terms
    if isempty(max_terms) || ~isnumeric(max_terms)
        error('greedy_fit: max_terms must be specified')
    end
    max_terms = min(max_terms, size(Xtrain,1));
    % check validation samples
    if max_patience < Inf && size(Xvalid,1) == 0
        error('gredy_fit: Xvalid samples must be specified for CV')
    end

    % initialize best validation error and patience
    best_valid_err = Inf;
    patience = 0;
    
    % measure generalization error on all data sets with initial map
    train_err = negative_log_likelihood(S, ref, XWtrain);
    valid_err = negative_log_likelihood(S, ref, XWvalid);

    % define output_msg
    if verbose == true
        output_msg = 'Term %3d - Training error: %f, Validation error: %f \n';
        fprintf(output_msg, S.n_coeff, train_err, valid_err);
    end

    % determine the reduced margin
    reduced_margin = getReducedMargin(S.multi_idxs());
    
    % define precompute structures
    precomp_Xtrain = PPprecomp();
    precomp_Xtrain = precomp_Xtrain.evaluate(S.f, Xtrain);
    precomp_Xvalid = PPprecomp();
    precomp_Xvalid = precomp_Xvalid.evaluate(S.f, Xvalid);

    %% Run adaptive procedure

    while(S.n_coeff < max_terms)

        % check max degree and remove reduced margin items above boundary
        invalid_idx = (sum(reduced_margin,2) >= max_degree);
        reduced_margin(invalid_idx,:) = [];
        if isempty(reduced_margin)
            break
        end
        
        % determine new multi-index and reduced margin
        [midx_new, reduced_margin] = update_component(S, ref, reduced_margin, XWtrain, precomp_Xtrain, criteria);

        % update multi-indices in S
        S_new = S;
        S_new = S_new.set_multi_idxs(midx_new);
        
        % update precomputed structures
        precomp_Xtrain = precomp_Xtrain.update_precomp(S_new.f, Xtrain, midx_new(end,:));
        precomp_Xvalid = precomp_Xvalid.update_precomp(S_new.f, Xvalid, midx_new(end,:));
            
        % extract coefficients a0 and optimize coefficients 
        [a0,~,~] = update_coeffs(S, S_new);
        [S_new,flag] = optimize_component(S_new, ref, a0, XWtrain, precomp_Xtrain, alpha);
        if flag <= -1
            % if optimization wasn't successful, return map
            train_err = [train_err, nan*ones(1,max_terms-S.n_coeff)];
            valid_err = [valid_err, nan*ones(1,max_terms-S.n_coeff)];
            break
        else
            % measure generalization error on both data sets
            S = S_new;
            train_err = [train_err, negative_log_likelihood(S, ref, XWtrain, precomp_Xtrain)];
            valid_err = [valid_err, negative_log_likelihood(S, ref, XWvalid, precomp_Xvalid)];
        end
        
        % print output
        if verbose == true
            fprintf(output_msg, S.n_coeff, train_err(end), valid_err(end));
        end

        % update patience
        if valid_err(end) >= best_valid_err
            patience = patience + 1;
        else
            best_valid_err = valid_err(end);
            patience = 0;
        end
        % check if patience exceeded maximum patience
        if patience >= max_patience
            break
        end
        
    end
        
    % save training and validation error in output
    output = struct;
    output.train_err    = train_err;
    output.valid_err    = valid_err;
    output.max_terms    = max_terms;
    output.max_patience = max_patience;
    output.alpha        = alpha;
    
end %endFunction

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

function [midx_new, reduced_margin] = update_component(S, ref, reduced_margin, XW, precomp, criteria)
    
    % extract old and new multi-indices
    midx_old = S.multi_idxs();
    midx_new = [midx_old; reduced_margin];

    % define new component
    S_new = S;
    S_new = S_new.set_multi_idxs(midx_new);
    
    % set coefficients based on previous optimal solution
    [a_new, idx_added, midx_added] = update_coeffs(S, S_new);

    % add new elements to precomp_margin
    precomp_margin = copy(precomp);
    precomp_margin = precomp_margin.update_precomp(S_new.f, XW.X, reduced_margin);
    S_new = S_new.set_coeff(a_new);
    

        [~, dJ] = regression_objective(S_new, ref, XW, precomp_margin, idx_added);
        grad_criteria = abs(dJ);

    % find function in the reduced margin most correlated with the residual
    [~, opt_dJ_idx] = max( grad_criteria );
    opt_midx = midx_added(opt_dJ_idx, :);
    % update multi-indices and the reduced margins based on opt_idx
    ind = ismember(reduced_margin, opt_midx, 'rows');
    if isempty(midx_old)
        midx_old = zeros(0, S.dim);
    end
    [midx_new, reduced_margin] = UpdateReducedMargin(midx_old, reduced_margin, ind);

end %endFunction

% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

function [c_new, idx_added, midx_added] = update_coeffs(S_old, S_new)

    % extract old and new multi-indices and old coefficients
    midx_old = S_old.multi_idxs();
    midx_new = S_new.multi_idxs();
    c_old = S_old.coeff;
    
    % declare vectors for new coefficients and to track added terms
    c_new = zeros(S_new.n_coeff, 1);
    c_add = ones(S_new.n_coeff, 1);

    % update coefficients
    for i=1:size(midx_old,1)
        [~, idx_i] = ismember(midx_old(i,:), midx_new, 'rows');
        c_new(idx_i) = c_old(i);
        c_add(idx_i) = 0;
    end

    % find indices of added coefficients and corresponding multi_indices
    idx_added  = find(c_add);
    midx_added = midx_new(idx_added,:);

end %endFunction

% -- END OF FILE --