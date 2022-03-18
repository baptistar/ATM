function [S, output] = greedy_regression(S, Sref, XWtrain,max_terms, tol)
    % Inputs:  S            - object for transport map component
    %          Sref         - reference map evaluated at XWtrain.X 
    %          max_terms    - maximum number of iterations to run
    %          Xtrain       - Reference map evaluated at quadrature points
    %          Xvalid       - (optional) array of validation samples
    %          tol          - (optional) tolerance on the objective
    % Outputs: TMc          - component with optimal coefficients
    %          output       - struct containing training and test error

    %% Initialize algorithm
    
    % set maximum patience (termination criteria) to be infinite if not specified
    if (nargin < 5) || isempty(tol)
        tol = 0;
    end
    
        % set maximum patience (termination criteria) to be infinite if not specified
    if (nargin < 7) 
        criteria = 'gradient';
    end

    % check max_terms
    if isempty(max_terms) || ~isnumeric(max_terms)
        error('greedy_fit: max_terms must be specified')
    end
    max_terms = min(max_terms, size(XWtrain.X,1));


    % determine the reduced margin
    reduced_margin = getReducedMargin(S.multi_idxs());
    
    % define precompute structures
    precomp_Xtrain = PPprecomp();
    precomp_Xtrain = precomp_Xtrain.evaluate(S.f, XWtrain.X);
    
    
    % measure generalization error on all data sets with initial map
    train_err = regression_objective(S, Sref, XWtrain,precomp_Xtrain);

    % define output_msg
    output_msg = 'Term %3d - Training error: %f \n';
    fprintf(output_msg, S.n_coeff, train_err);
    
    %% Run adaptive procedure

    while(S.n_coeff < max_terms && tol < train_err(end) )
        
        % determine new multi-index and reduced margin
        [midx_new, reduced_margin] = update_component(S, Sref, reduced_margin, XWtrain, precomp_Xtrain, criteria);

        % update multi-indices in S
        S_new = S;
        S_new = S_new.set_multi_idxs(midx_new);
        
        % update precomputed structures
        precomp_Xtrain = precomp_Xtrain.update_precomp(S_new.f, XWtrain.X, midx_new(end,:));
            
        % extract coefficients a0 and optimize coefficients 
        [a0,~,~] = update_coeffs(S, S_new);
        [S_new,flag] = optimize_component_regression(S_new, Sref, a0, XWtrain, precomp_Xtrain);
        if flag <= -1
            % if optimization wasn't successful, return map
            train_err = [train_err, nan*ones(1,max_terms-S.n_coeff)];
            break
        else
            % measure generalization error on both data sets
            S = S_new;
            train_err = [train_err, regression_objective(S, Sref, XWtrain, precomp_Xtrain)];
        end
        
        % print output

       fprintf(output_msg, S.n_coeff, train_err(end));
        
        
    end
        
    % save training and validation error in output
    output = struct;
    output.train_err    = train_err;
    output.max_terms    = max_terms;
end %endFunction

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

function [midx_new, reduced_margin] = update_component(S, Sref, reduced_margin, XW, precomp, criteria)
    
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
    
    % compute criteria after adding the new elements
    if strcmp(criteria,'gradient')
        [~, dJ] = regression_objective(S_new, Sref, XW, precomp_margin, idx_added);
        grad_criteria = abs(dJ);
    elseif strcmp(criteria,'gradient_normalized')
        [~, dJ] = regression_objective(S_new, Sref, XW, precomp_margin, idx_added);
        Psi_norm = sum(precomp_margin.eval_basis.^2,1);
        grad_criteria = abs(dJ).^2 ./ Psi_norm(idx_added);
    else
        error('ATM criteria is not recognized')
    end
    
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