clear; close all; clc
sd = 1; rng(sd);

% define true Ld 
nz_coeff = 10;
Ld = [randn(1, nz_coeff-1), rand()];

%% Run cross-validation

% define sample sizes
d_vect = floor(logspace(1,3,10));
N_vect = 1000; 
Ntest  = 1e4;
MCruns = 10;
n_folds = 5;
max_terms = floor(2*nz_coeff);

% define arrays to store results
KL_best        = zeros(length(d_vect), length(N_vect), MCruns);
KL_kfold       = zeros(length(d_vect), length(N_vect), MCruns);
ncoeff_kfold   = zeros(length(d_vect), length(N_vect), MCruns);

for i=1:length(d_vect)
    
    % generate true L and Linv (cholesky factor of precision)
    d = d_vect(i);
    L = eye(d); L(d,d-nz_coeff+1:d) = Ld;
    Linv = inv(L);

    % true logpdf and sample function
    logpdf = @(X) -0.5*diag(X*(L'*L)*X.') + sum(log(diag(L))) - d/2*log(2*pi);
    sample = @(N) (Linv*randn(size(L,1),N)).';

    % define test set
    Xtest  = sample(Ntest);

    for j=1:length(N_vect)
        for k=1:MCruns

            % define Ntrain
            Ntrain = N_vect(j);
            fprintf('d = %d, N = %d, run = %d\n', d, Ntrain, k)

            % sample from Xtrain
            Xtrain = sample(Ntrain);

            % find coeff with true sparsity pattern
            beta = optimize_component(find(L(d,:)), Xtrain);
            Lbest = eye(d); Lbest(d,:) = beta;
            logpdf_best = @(X) -0.5*diag(X*(Lbest'*Lbest)*X.') + sum(log(diag(Lbest))) - d/2*log(2*pi);
            KL_best(i,j,k) = KL_divergence(logpdf, logpdf_best, Xtest);

            % run ATM with CV for each number of folds
            [beta,~] = ATM_CV(Xtrain, n_folds, max_terms, 1);
            Lapprox = eye(d); Lapprox(d,:) = beta;
            logpdf_approx = @(X) -0.5*diag(X*(Lapprox'*Lapprox)*X.') + sum(log(diag(Lapprox))) - d/2*log(2*pi);
            KL_kfold(i,j,k) = KL_divergence(logpdf, logpdf_approx, Xtest);
            ncoeff_kfold(i,j,k) = nnz(beta);

        end
    end
end

%% plot errors

figure('position',[0,0,1600,400])
hold on
boxplot(squeeze(KL_kfold).', 'labels', d_vect);
plot(1:length(d_vect), mean(KL_best,3), '-b');
set(gca,'YScale','log')
xlabel('$n$')
ylabel('$D_{KL}(\pi,\tilde{\pi})$')
ylim([1e-5,5])
hold off
%print('-depsc','linearmaps_kl_vs_sample_size_sparse');

figure('position',[0,0,1600,400])
hold on
boxplot(squeeze(ncoeff_kfold).', 'labels', d_vect);
plot(1:length(d_vect), nz_coeff*ones(length(d_vect),1), '-b');
xlabel('$n$')
ylabel('Number of coeffs')
ylim([0,20])
hold off
print('-depsc','linearmaps_number_coeff_vs_sample_size_sparse');

%% ------------------------------------------------------------------------

function [beta, m_idxs, train_err, valid_err] = ATM(Xtrain, Xvalid, max_terms, grad_approx)

    % initialize beta
    dim = size(Xtrain,2);
    m_idxs = dim;
    beta = optimize_component(m_idxs, Xtrain);
    
    % measure generalization error on all data sets with initial map
    train_err = negative_log_likelihood(beta, Xtrain);
    valid_err = negative_log_likelihood(beta, Xvalid);
    
    % Run adaptive procedure
    while(nnz(beta) < max_terms && nnz(beta) < dim)

        % determine new multi-index
        m_idxs_new = new_index(beta, Xtrain, grad_approx);
        m_idxs = [m_idxs, m_idxs_new];
        
        % extract coefficients a0 and optimize coefficients 
        beta = optimize_component(sort(m_idxs), Xtrain);

        % measure generalization error on both data sets
        train_err = [train_err, negative_log_likelihood(beta, Xtrain)];
        valid_err = [valid_err, negative_log_likelihood(beta, Xvalid)];

    end

end %endFunction

function [J, dJ, d2J] = negative_log_likelihood(beta, X, grad_dim)
    assert(all(size(beta) == [size(X,2),1]))
    S = X*beta;
    dkS = beta(end);
    % evaluate objective
    J = 0.5*S.^2 - log(dkS);
    J = mean(J,1);
    % evaluate gradient
    if nargout > 1
        dJ = S.*X(:,grad_dim); 
        diag_idx = intersect(grad_dim, size(X,2));
        dJ(:,diag_idx) = dJ(:,diag_idx) - 1./dkS;
        dJ = mean(dJ,1);
    end
    if nargout > 2
        d2J = X(:,grad_dim).'*X(:,grad_dim); 
        diag_idx = intersect(grad_dim, size(X,2));
        d2J(diag_idx,diag_idx) = d2J(diag_idx,diag_idx) + 1./dkS^2;
    end
end %endFunction

function beta = optimize_component(m_idxs, X)
    offdiag_idxs = setdiff(m_idxs, size(X,2));
    Xoffdiag = X(:,offdiag_idxs);
    alpha = [-1*(Xoffdiag.'*Xoffdiag)\(Xoffdiag.'*X(:,end)); 1];
    betad = 1/sqrt(mean((X(:,m_idxs)*alpha).^2));
    beta  = zeros(size(X,2),1); beta(m_idxs) = alpha.'*betad;
end %endFunction

function midx_new = new_index(beta, X, grad_approx)

    % evaluate objective and gradients
    zero_indices = find(beta == 0);
    [~, dJ, d2J] = negative_log_likelihood(beta, X, zero_indices);

    % find entry in the reduced margin most correlated with the residual
    if grad_approx == 1
        greedy_crit = abs(dJ);
    elseif grad_approx == 2
        greedy_crit = abs(dJ).^2 ./ diag(d2J).';
    else
        error('grad_approx should be 1 or 2')
    end
    
    % constraint gradient to non-zero indices
    [~, opt_zero_idx] = max( greedy_crit );
    midx_new = zero_indices(opt_zero_idx);
    
end %endFunction

function [beta,m_idxs] = ATM_CV(Xtrain, n_folds, max_terms, grad_approx)

    % define cross-validation splits of data
    cv = cvpartition(size(Xtrain,1), 'kFold', n_folds);

    % redefine max_terms
    max_terms = min(size(Xtrain,2)-1, max_terms);
    
    % define matrix to store results
    valid_error = nan(n_folds, max_terms);

    % run greedy_fit on each parition of data
    for fold=1:n_folds
        Xtrain_train  = Xtrain(cv.training(fold),:);
        Xtrain_valid  = Xtrain(cv.test(fold),:);
        [~,~,~,valid_error(fold,:)] = ATM(Xtrain_train, Xtrain_valid, max_terms, grad_approx);
    end

    % find optimal number of terms
    % remove one to account for initial condition
    mean_valid_error = mean(valid_error, 1);
    [~, opt_terms] = min(mean_valid_error);

    % run greedy_fit up to opt_terms with all data
    [beta,m_idxs] = ATM(Xtrain, zeros(0,size(Xtrain,2)), opt_terms, grad_approx);

end %endFunction

function DKL = KL_divergence(logpdf, logpdf_approx, X)
    logpdf_x = logpdf(X);
    logpdf_approx_x = logpdf_approx(X);
    DKL = mean(logpdf_x - logpdf_approx_x);
end %endFunction

%  -- END OF FILE --