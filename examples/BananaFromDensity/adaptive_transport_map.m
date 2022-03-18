function [S, output] = adaptive_transport_map(S, ref, Xtrain, Xvalid, ...
                            max_iterations, max_patience, alpha)
    % Inputs:  S            - TriangularTransportMap object
    %          max_terms    - maximum number of iterations to run
    %          Xtrain       - (N x d) array of training samples
    %          Xvalid       - (optional) array of validation samples
    %          max_patience - (optional) patience for stopping algorithm
    % Outputs: TMc          - component with optimal coefficients
    %          output       - struct containing training and test error

    %% Initialize algorithm
    
    % check inputs
    assert(isa(S, 'TriangularTransportMap'))
    assert(size(Xtrain,2) == S.d)
    
    
    
    % set default alpha
    if (nargin < 7) || isempty(alpha)
        alpha = 0;
    end
    % set maximum patience (termination criteria) to be infinite if not specified
    if (nargin < 6) || isempty(max_patience)
        max_patience = Inf;
    end
    % check max_iterations
    if isempty(max_iterations) || ~isnumeric(max_iterations)
        error('ATM: max_iterations must be specified')
    end
    max_iterations = min(max_iterations, size(Xtrain,1));
    % set validation samples
    if (nargin < 5) || isempty(Xvalid)
        Xvalid = zeros(0, size(Xtrain,2));
    end    
    % check validation samples
    if max_patience < Inf && size(Xvalid,1) == 0
        error('ATM: Xvalid samples must be specified for CV')
    end

    % initialize best validation error and patience
    best_valid_err = Inf;
    patience = 0;
    
    % define precompute structures
    precomp_Xtrain = set_precomp(S, Xtrain);
    precomp_Xvalid = set_precomp(S, Xvalid);

    % measure generalization error on all data sets with initial map
    XWtrain.X=Xtrain;
    XWtrain.W=(1/size(Xtrain,1))*ones(size(Xtrain,1),1);
    
    [S,~] = optimize_map(S, ref, S.coeff, XWtrain, precomp_Xtrain, alpha);
    train_err = negative_log_likelihood(S, ref, Xtrain, precomp_Xtrain);
    valid_err = negative_log_likelihood(S, ref, Xvalid, precomp_Xvalid);

    % define output_msg
    output_msg = 'Term %4d - Training error: %8.4f, Validation error: %8.4f \n';
    fprintf(output_msg, S.n_coeff, train_err, valid_err);

    %% Greedy procedure

    while(S.n_coeff < max_iterations && patience < max_patience)

        % determine the reduced margin for each component
        reduced_margin = get_full_reduced_margin(S);
        
        % find new multi-index and define updated map S
        [S_new, new_midx] = update_map(S, ref, reduced_margin, Xtrain, precomp_Xtrain);

        % update precomputed structures
        precomp_Xtrain = update_precomp(precomp_Xtrain, S_new, Xtrain, new_midx);
        precomp_Xvalid = update_precomp(precomp_Xvalid, S_new, Xvalid, new_midx);
            
        % extract coefficients a0 and optimize coefficients 
        a0 = S_new.coeff;
        XWtrain.X=Xtrain;
        XWtrain.W=(1/size(Xtrain,1))*ones(size(Xtrain,1),1);
        [S_new, flag] = optimize_map(S_new, ref, a0, XWtrain, precomp_Xtrain, alpha);

        if flag <= -1
            % if optimization wasn't successful, return map
            train_err = [train_err, nan*ones(1,max_terms-S.n_coeff)];
            valid_err = [valid_err, nan*ones(1,max_terms-S.n_coeff)];
            break
        else
            % measure generalization error on both data sets
            S = S_new;
            train_err = [train_err, negative_log_likelihood(S, ref, Xtrain, precomp_Xtrain)];
            valid_err = [valid_err, negative_log_likelihood(S, ref, Xvalid, precomp_Xvalid)];
        end
        
        % print output
        fprintf(output_msg, S.n_coeff, train_err(end), valid_err(end));

        % update patience
        if valid_err(end) >= best_valid_err
            patience = patience + 1;
        else
            best_valid_err = valid_err(end);
            patience = 0;
        end
        
    end

    % save training and validation error in output
    output = struct;
    output.train_err      = train_err;
    output.valid_err      = valid_err;
    output.max_iterations = max_iterations;
    output.max_patience   = max_patience;
    output.alpha          = alpha;

end %endFunction

%--------------------------------------------------------------------------

function [S, new_midx] = update_map(S, ref, reduced_margin, X, precomp)

    % define list to store new coefficient and component indices
    counter = 0;
    margin_coeff_idx = [];
    comp_coeff_idx = [];
    component_idx = [];
    
    % add new basis funtions to each component
    S_margin = S;
    for k=1:S_margin.d
        % update multi-indices
        midx_k   = [S_margin.S{k}.multi_idxs; reduced_margin{k}];
        S_margin.S{k} = S_margin.S{k}.set_multi_idxs(midx_k);
        % update coefficients
        size_RMk = size(reduced_margin{k},1);
        coeff_k  = [S_margin.S{k}.coeff, zeros(1,size_RMk)];
        S_margin.S{k} = S_margin.S{k}.set_coeff(coeff_k);
        % update lists
        counter  = counter + S_margin.S{k}.n_coeff;
        margin_coeff_idx = [margin_coeff_idx, (counter-size_RMk+1):counter];
        comp_coeff_idx = [comp_coeff_idx, 1:size_RMk];
        component_idx = [component_idx, k*ones(1,size_RMk)];
    end
    
    % compute gradient after adding the new elements
    precomp_margin = cell(S_margin.d,1);
    for k=1:S_margin.d
        precomp_margin{k} = copy(precomp{k});
    end
    precomp_margin = update_precomp(precomp_margin, S_margin, X, reduced_margin);
    
    % compute gradient with respect to the reduced margin
    [~, dJ] = negative_log_likelihood(S_margin, ref, X, precomp_margin);
    grad_criteria = abs(dJ);
    %Psi_norm = sum(precomp_margin.eval_basis.^2,1);
    %grad_criteria = abs(dJ).^2 ./ Psi_norm(idx_added);
    
    % find function in the reduced margin most correlated with the residual
    [~, opt_subset_idx] = max( grad_criteria(margin_coeff_idx) );
    opt_component = component_idx(opt_subset_idx);
    opt_comp_coeff_idx = comp_coeff_idx(opt_subset_idx);
        
    % update new_midx
    new_midx = cell(S.d,1);
    new_midx{opt_component} = reduced_margin{opt_component}(opt_comp_coeff_idx,:);

    % update S
    midx_opt = [S.S{opt_component}.multi_idxs; new_midx{opt_component}];
    S.S{opt_component} = S.S{opt_component}.set_multi_idxs(midx_opt);
    coeff_opt = [S.S{opt_component}.coeff, 0];
    S.S{opt_component} = S.S{opt_component}.set_coeff(coeff_opt);

end %endFunction

% -------------------------------------------------------------------------

function [L, dcL] = negative_log_likelihood(S, ref, X, precomp)
    if isempty(precomp)
        precomp = set_precomp(S, X);
    end
     
    % define delta (regularization term) - add small diagonal term
    delta = 1e-9;

    % evaluate objective
    Sx = S.evaluate(X, precomp) + delta*X;
    dxdS = S.grad_xd(X, precomp) + delta;
    % evaluate log_pi(x)
    L = ref.log_pdf(Sx) + sum(log(dxdS),2);
    L = -1 * mean(L,1);
    
    % evaluate gradient
    if (nargout > 1)
        % evaluate \nabla_c S, \nabla_c_xd S
        dcS = S.grad_coeff(X, precomp);
        dcdxdS = S.grad_coeff_grad_xd(X, precomp);
        % evaluate \nabla_c log_pi(x)
        dcL = squeeze(sum(ref.grad_x_log_pdf(Sx) .* dcS + dcdxdS ./ dxdS, 2));
        dcL = -1 * mean(dcL,1);
    end
        
end %endFunction

% -------------------------------------------------------------------------

function RM = get_full_reduced_margin(S)
    RM = cell(S.d,1);
    for k=1:S.d
        RM{k} = getReducedMargin(S.S{k}.multi_idxs);
    end
end %endFunction

% -------------------------------------------------------------------------

function precomp = set_precomp(S, X)
    precomp = cell(S.d,1);
    for k=1:S.d
       precomp{k} = PPprecomp();
       precomp{k} = precomp{k}.evaluate(S.S{k}.f, X(:,1:k));
    end
end %endFunction

% -------------------------------------------------------------------------

function precomp = update_precomp(precomp, S, X, new_midx)
    assert(length(new_midx) == S.d)
    for k=1:S.d
        % only update precomp if multi-index has changed
        if ~isempty(new_midx{k})
            precomp{k} = precomp{k}.update_precomp(S.S{k}.f, X(:,1:k), new_midx{k});
        end
    end
end %endFunction

% -- END OF FILE --