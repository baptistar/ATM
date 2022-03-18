clear; close all; clc

addpath(genpath('../../src'))
sd = 1; rng(sd);

%% --- Define test case ---

% define problems
d = 10;
Ntrain = 1e3;
Ntest  = 1e4;
 
% define reference, and other map parameters
ref        = Normal();
max_terms  = 100;
alpha      = 0;

% set parameters
mu = 0; phi = 1; sigma = 0.25;

% generating training samples
Xtrain_valid = zeros(Ntrain,d);
Xtrain_valid(:,1) = randn(Ntrain,1);
for i=2:d
    Xtrain_valid(:,i) = mu + phi .* (Xtrain_valid(:,i-1) - mu) + sigma*randn(Ntrain,1);
end

% seperate training and validation
train_idx = randi(Ntrain, floor(0.8*Ntrain), 1);
valid_idx = setdiff(1:Ntrain, train_idx);
Xtrain = Xtrain_valid(train_idx,:);
Xvalid = Xtrain_valid(valid_idx,:);

% generate test samples
Xtest = zeros(Ntest,d);
Xtest(:,1) = randn(Ntest,1);
for i=2:d
    Xtest(:,i) = mu + phi .* (Xtest(:,i-1) - mu) + sigma*randn(Ntest,1);
end

%% --- Learn maps (Hermite Functions basis) ---

% define basis
basis = ProbabilistHermiteFunction();

% learn dense map
[CM_HF, output_HF] = buildComposedMap(Xtrain, Xvalid, basis, ref);

% comptue test log-likelihood
test_loglik = CM_HF.log_pdf(Xtest);
test_loglik = test_loglik(~isinf(test_loglik));
q = quantile(test_loglik, [0.01, 0.99]);
test_loglik = test_loglik(test_loglik > q(1));
fprintf('Hermite functions: test loglik = %f\n', mean(test_loglik))

% check gaussianity
push_forward_marginals(CM_HF.S{2}, CM_HF.S{1}.evaluate(Xtest));
print('-depsc','quantiles_hermitefunctions')

%% --- Learn maps (Hermite polynomial basis) ---

% define basis
basis = HermiteProbabilistPoly();

% learn dense map
[CM_HP, output_HP] = buildComposedMap(Xtrain, Xvalid, basis, ref);

% comptue test log-likelihood
test_loglik = CM_HP.log_pdf(Xtest);
test_loglik = test_loglik(~isinf(test_loglik));
q = quantile(test_loglik, [0.01, 0.99]);
test_loglik = test_loglik(test_loglik > q(1));
fprintf('Hermite polynomials: test loglik = %f\n', mean(test_loglik))

% check gaussianity
push_forward_marginals(CM_HP.S{2}, CM_HP.S{1}.evaluate(Xtest));
print('-depsc','quantiles_hermitepoly')

%% --- Learn maps (Hermite lienear extension basis) ---

% define basis
basis = HermiteProbabilistPolyWithLinearization();

% learn dense map
[CM_HPL, output_HPL] = buildComposedMap(Xtrain, Xvalid, basis, ref);

% comptue test log-likelihood
test_loglik = CM_HPL.log_pdf(Xtest);
test_loglik = test_loglik(~isinf(test_loglik));
q = quantile(test_loglik, [0.01, 0.99]);
test_loglik = test_loglik(test_loglik > q(1));
fprintf('Hermite polynomials: test loglik = %f\n', mean(test_loglik))

% check gaussianity
push_forward_marginals(CM_HPL.S{2}, CM_HPL.S{1}.evaluate(Xtest));
print('-depsc','quantiles_hermitepolylinearextension')

%% Plot map dependence

% true dependence (chain graph)
true_dep = eye(d) + spdiags(ones(d,1),-1,d,d);
figure;
niceSpy(true_dep)
print('-depsc','true_map_sparsity')

% Hermite function dependence
function_dep = zeros(d,d);
for k=1:d
    midx = CM_HF.S{2}.S{k}.multi_idxs;
    midx(:,k) = midx(:,k) + 1;
    function_dep(k,1:k) = max(midx,[],1);
end
figure;
niceSpy(function_dep)
print('-depsc','Hermite_function_sparsity')

% Hermite polynomial dependence
poly_dep = zeros(d,d);
for k=1:d
    midx = CM_HP.S{2}.S{k}.multi_idxs;
    midx(:,k) = midx(:,k) + 1;
    poly_dep(k,1:k) = max(midx,[],1);
end
figure;
niceSpy(poly_dep)
print('-depsc','Hermite_polynomial_sparsity')

% Hermite polynomial dependence
polylinext_dep = zeros(d,d);
for k=1:d
    midx = CM_HP.S{2}.S{k}.multi_idxs;
    midx(:,k) = midx(:,k) + 1;
    polylinext_dep(k,1:k) = max(midx,[],1);
end
figure;
niceSpy(polylinext_dep)
print('-depsc','Hermite_hermitepolylinearextension_sparsity')

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