clear; close all; clc

addpath(genpath('../../src'))
sd = 1; rng(sd);

%% --- Define test case ---

% define problems
Ntrain = 1e4;
Ntest  = 1e4;
d      = 5;%2+5;
 
% define reference, basis, and order
ref        = Normal();
basis      = ProbabilistHermiteFunction();
max_terms  = 1000;
alpha      = 0;%1e-12;

% generating training samples
%Xtrain = stoc_volatility_model(Ntrain, d);
%Xtest  = stoc_volatility_model(Ntest, d);

% sample auto-regressively
Xtrain = zeros(Ntrain,d); mu = 0; phi = 1; sigma = 0.25;
Xtrain(:,1) = randn(Ntrain,1);
for i=2:d
    Xtrain(:,i) = mu + phi .* (Xtrain(:,i-1) - mu) + sigma*randn(Ntrain,1);
end
% subselect parameters
%Xtrain = Xtrain(:,3:end);
%Xtest  = Xtest(:,3:end);

Xtest = zeros(Ntest,d);
Xtest(:,1) = randn(Ntest,1);
for i=2:d
    Xtest(:,i) = mu + phi .* (Xtest(:,i-1) - mu) + sigma*randn(Ntest,1);
end

%% --- Learn maps ---

% learn dense map
[CM, output] = buildComposedMap(Xtrain, Xtest, basis, ref);

%% TEMPORARY LEARNING OF COMPONENT FIVE

% scale samples
Z = (Xtrain - mean(Xtrain)) ./ std(Xtrain);
Zvalid = (Xtest - mean(Xtrain)) ./ std(Xtrain);

% define TM
ref_d = IndependentProductDitribution(repmat({ref},1,d));
TM = identity_map(1:d, basis);
PB = PullbackDensity(TM, ref_d);

% run greedy approximation on S_valid
comp = 5;
max_patience = 10;
[PB.S{comp}, output_c] = greedy_fit(PB.S{comp}, ref_d.factors{comp}, ...
                max_terms, Z, Zvalid, max_patience, alpha);

%%

alpha_vect = [0,1e-12,1e-10,1e-8,1e-6,1e-4];
max_patience = 25;

output_vs_alpha = cell(length(alpha_vect),1);
idx_vs_alpha = cell(length(alpha_vect),1);
for i=1:length(alpha_vect)
    TM = identity_map(1:d, basis);
    PB = PullbackDensity(TM, ref_d);
    [PB.S{comp}, output_vs_alpha{i}] = greedy_fit(PB.S{comp}, ref_d.factors{comp}, ...
                max_terms, Z, Zvalid, max_patience, alpha_vect(i));
    [~, n_added_terms] = min(output_vs_alpha{i}.valid_err);
    n_added_terms = n_added_terms(1) - 1;
    opt_terms = 1 + n_added_terms;
    % extract optimal multi-indices
    midx_opt = PB.S{comp}.multi_idxs();
    idx_vs_alpha{i} = midx_opt(1:opt_terms,:);
end
%%
figure
subplot(1,2,1)
hold on
for i=1:length(alpha_vect)
    plot(output_vs_alpha{i}.train_err,'-','DisplayName',['$\alpha = ' num2str(alpha_vect(i)) '$']);
end
hold off
xlabel('Number of iterations')
ylabel('Training error')
legend('show')

subplot(1,2,2)
hold on
for i=1:length(alpha_vect)
    plot(output_vs_alpha{i}.valid_err,'-');
end
hold off
xlabel('Number of iterations')
ylabel('Validation error')

figure
sum_idx = zeros(length(alpha_vect),comp);
for i=1:length(alpha_vect)
    sum_idx(i,:) = sum(idx_vs_alpha{i});
end
bar(sum_idx.')
xlabel('Index')
ylabel('Count of indices')
%set(gca,'xticklabel',alpha_vect)

%%

% learn sparse map
active_variables = cell(d,1); active_variables{1} = 1;
for k=2:d
    active_variables{k} = k-1:k;
end
[CM_sparse, output_output] = buildComposedMap(Xtrain, Xtest, basis, ref, active_variables);

%% Plot errors

figure
for k=2:d-2
    subplot(1,d-2,k)
    idx_k = CM.S{2}.S{k}.multi_idxs;
    idx_correct = find(sum(idx_k(:,active_variables{k}),2));
    hold on 
    plot(output{k}.valid_err)
    plot(idx_correct, output{k}.valid_err(idx_correct), 'or')
    hold off
    %plot(output_output{k}.valid_err)
    xlabel('Number of iterations')
    ylabel('Negative log-likelihood')
    set(gca,'Yscale','log')
end

%% --- Helper functions ---

function [CM, output] = buildComposedMap(X, Xvalid, basis, ref, active_variables, alpha)

    % extract d
    d = size(X,2);

    % set active variables
    if nargin < 6
        alpha = 0;
    end
    if nargin < 5
        active_variables = cell(d,1);
        for k=1:d
            active_variables{k} = 1:k;
        end
    end

    % define reference
    ref_d = IndependentProductDitribution(repmat({ref},1,d));

    % define Gaussian map
    G = GaussianPullbackDensity(d, true);
    G = G.optimize(X);
    Z = G.evaluate(X);
    Zvalid = G.evaluate(Xvalid);

    % set max_iter and max_patience
    max_iter = ceil(size(X,1));
    max_patience = 10;

    % define TM
    TM = identity_map(1:d, basis);
    PB = PullbackDensity(TM, ref_d);
    
    % optimize each component using Xtrain and Xvalid
    %[PB, output] = PB.greedy_optimize(Z, [], max_iter, 'Split');
    output = cell(d,1);
    for k=2:d%1:d
        disp(['Component ' num2str(k) '/' num2str(d)])
        % run greedy approximation on S_valid
        S_valid = PB.S{k};
        % extract data
        Zk = Z(:,active_variables{k});
        Zvalidk = Zvalid(:,active_variables{k});
        % run fit
        [S_valid, output{k}] = greedy_fit(S_valid, ref_d.factors{k}, ...
                        max_iter, Zk, Zvalidk, max_patience, alpha);
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