clear; close all; clc

addpath(genpath('../../src'))
sd = 1; rng(sd);

%% --- Define test cases ---

% define problems
Ntrain = 1e3;
Ntest  = 1e4;
d_vect = [5,10,15,20,25];
MCruns = 5;

% define reference, basis, and order
ref        = Normal();
max_terms  = 1000;
alpha      = 0;

% define basis
basis = HermiteProbabilistPoly();

% set parameters
mu = 0; phi = 1; sigma = 0.25;

%% --- Learn maps (Hermite polynomial basis) ---

neg_loglik = zeros(length(d_vect), MCruns);

for i = 1:length(d_vect)
    for j=1:MCruns
    
        d = d_vect(i);
        % generating training samples
        Xtrain_valid = zeros(Ntrain,d);
        Xtrain_valid(:,1) = randn(Ntrain,1);
        for k=2:d
            Xtrain_valid(:,k) = mu + phi .* (Xtrain_valid(:,k-1) - mu) + sigma*randn(Ntrain,1);
        end

        % seperate training and validation
        train_idx = randi(Ntrain, floor(0.8*Ntrain), 1);
        valid_idx = setdiff(1:Ntrain, train_idx);
        Xtrain = Xtrain_valid(train_idx,:);
        Xvalid = Xtrain_valid(valid_idx,:);

        % generate test samples
        Xtest = zeros(Ntest,d);
        Xtest(:,1) = randn(Ntest,1);
        for k=2:d
            Xtest(:,k) = mu + phi .* (Xtest(:,k-1) - mu) + sigma*randn(Ntest,1);
        end

        % learn dense map
        [CM_HP, output_HP] = buildComposedMap(Xtrain, Xvalid, basis, ref);

        % comptue test log-likelihood
        test_loglik = CM_HP.log_pdf(Xtest);
        test_loglik = test_loglik(~isinf(test_loglik));
        q = quantile(test_loglik, [0.01, 0.99]);
        test_loglik = test_loglik(test_loglik > q(1));
        neg_loglik(i,j) = -1*mean(test_loglik);
        fprintf('Hermite polynomials: test loglik = %f\n', mean(test_loglik))

        % plot Hermite polynomial dependence
        polylinext_dep = zeros(d,d);
        for k=1:d
            midx = CM_HP.S{2}.S{k}.multi_idxs;
            midx(:,k) = midx(:,k) + 1;
            polylinext_dep(k,1:k) = max(midx,[],1);
        end
        figure;
        niceSpy(polylinext_dep)
        print('-depsc',['sparsity_d' num2str(d) '_run' num2str(j)])
        
    end
end

%% Plot results

figure
errorbar(d_vect, mean(neg_loglik,2), 1.96*std(neg_loglik,[],2)/sqrt(MCruns), ...
    '-o','linewidth',2);
xlabel('Dimension of state, $d$','FontSize',16)
ylabel('Negative log-likelihood','FontSize',16)
print('-depsc','negloglik_vs_d')

%% --- Helper functions ---

function [CM, output] = buildComposedMap(X, Xvalid, basis, ref, alpha)

    % set alpha
    if nargin < 5
        alpha = 0;
    end

    % define reference
    d = size(X,2);
    ref_d = IndependentProductDitribution(repmat({ref},1,d));

    % define Gaussian map
    G = GaussianPullbackDensity(d, true);
    G = G.optimize(X);
    Z = G.evaluate(X);
    Zvalid = G.evaluate(Xvalid);

    % set max_iter and max_patience
    max_iter = ceil(size(X,1));
    max_patience = 20;

    % define TM
    TM = identity_map(1:d, basis);
    PB = PullbackDensity(TM, ref_d);
    
    % optimize each component using Xtrain and Xvalid
    %[PB, output] = PB.greedy_optimize(Z, [], max_iter, 'Split');
    output = cell(d,1);
    for k=1:d
        disp(['Component ' num2str(k) '/' num2str(d)])
        % run greedy approximation on S_valid
        S_valid = PB.S{k};
        % run fit
        [S_valid, output{k}] = greedy_fit(S_valid, ref_d.factors{k}, ...
            max_iter, Z(:,1:k), Zvalid(:,1:k), max_patience, alpha);
        % find optimal number of terms (adding terms originally in S)
        % remove one to account for initial condition
        [~, n_added_terms] = min(output{k}.valid_err);
        n_added_terms = n_added_terms(1) - 1;
        opt_terms = PB.S{k}.n_coeff + n_added_terms;
        % extract optimal multi-indices
        midx_opt = S_valid.multi_idxs();
        midx_opt = midx_opt(1:opt_terms,:);
        PB.S{k} = PB.S{k}.set_multi_idxs(midx_opt);
        % run greedy_fit up to opt_terms with all data
        a0 = zeros(opt_terms,1);
        PB.S{k} = optimize_component(PB.S{k}, ref_d.factors{k}, a0, Z(:,1:k), [], alpha);
    end
    
    % compose map
    CM = ComposedPullbackDensity({G, PB}, ref_d);

end

% -- END OF FILE --