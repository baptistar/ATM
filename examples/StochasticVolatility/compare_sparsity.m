clear; close all; clc

addpath(genpath('../../src'))
sd = 1; rng(sd);

%% --- Define test case ---

% define problems
Ntrain = 2e3;
Ntest  = 1e4;
d      = 5;
 
% define reference, basis, and order
ref        = Normal();
basis      = ProbabilistHermiteFunction();
max_terms  = 1000;
alpha      = 0;

% generating training samples
Xtrain = zeros(Ntrain,d); mu = 0; phi = 1; sigma = 0.25;
Xtrain(:,1) = randn(Ntrain,1);
for i=2:d
    Xtrain(:,i) = mu + phi .* (Xtrain(:,i-1) - mu) + sigma*randn(Ntrain,1);
end

% generate test samples
Xtest = zeros(Ntest,d);
Xtest(:,1) = randn(Ntest,1);
for i=2:d
    Xtest(:,i) = mu + phi .* (Xtest(:,i-1) - mu) + sigma*randn(Ntest,1);
end

%% --- Setup learning ---

% define component
comp = 5;

% define active variables
active_vars = comp-1:comp;

% define max patience
max_patience = 10;

% scale samples
Z = (Xtrain - mean(Xtrain)) ./ std(Xtrain);
Zvalid = (Xtest - mean(Xtrain)) ./ std(Xtrain);

%% --- Learn dense map component ---

% define TM
ref_d = IndependentProductDitribution(repmat({ref},1,d));
TM = identity_map(1:d, basis);
PB = PullbackDensity(TM, ref_d);

% run greedy approximation on S_valid
[S_dense, output_dense] = greedy_fit(PB.S{comp}, ref_d.factors{comp}, ...
	max_terms, Z(:,1:comp), Zvalid(:,1:comp), max_patience, alpha);

% remove one to account for initial condition
[~, n_added_terms] = min(output_dense.valid_err);
n_added_terms = n_added_terms(1) - 1;
opt_terms = PB.S{comp}.n_coeff + n_added_terms;
% extract optimal multi-indices
midx_opt = S_dense.multi_idxs();
midx_opt = midx_opt(1:opt_terms,:);
PB.S{comp} = PB.S{comp}.set_multi_idxs(midx_opt);
% run greedy_fit up to opt_terms with all data
a0 = zeros(opt_terms,1);
PB.S{comp} = optimize_component(PB.S{comp}, ref_d.factors{comp}, ...
    a0, Z(:,1:comp), [], alpha);
S_dense = PB.S{comp};

%% --- Learn sparse map component ---

% define TM
ref_d = IndependentProductDitribution(repmat({ref},1,2));
TM = identity_map(1:2, basis);
PB = PullbackDensity(TM, ref_d);

% run greedy approximation on S_valid
max_patience = 10;
[S_sparse, output_sparse] = greedy_fit(PB.S{2}, ref_d.factors{2}, ...
	max_terms, Z(:,active_vars), Zvalid(:,active_vars), max_patience, alpha);

% remove one to account for initial condition
[~, n_added_terms] = min(output_sparse.valid_err);
n_added_terms = n_added_terms(1) - 1;
opt_terms = PB.S{2}.n_coeff + n_added_terms;
% extract optimal multi-indices
midx_opt = S_sparse.multi_idxs();
midx_opt = midx_opt(1:opt_terms,:);
PB.S{2} = PB.S{2}.set_multi_idxs(midx_opt);
% run greedy_fit up to opt_terms with all data
a0 = zeros(opt_terms,1);
PB.S{2} = optimize_component(PB.S{2}, ref_d.factors{2}, ...
    a0, Z(:,active_vars), [], alpha);
S_sparse = PB.S{2};

%% Plot errors

figure
subplot(1,2,1)
hold on
plot(output_dense.train_err,'-')
plot(output_sparse.train_err,'-')
hold off
xlabel('Number of iterations')
ylabel('Training error')
legend('Dense map','Known sparsity')
legend('show')

subplot(1,2,2)
hold on
plot(output_dense.valid_err,'-')
plot(output_sparse.valid_err,'-')
hold off
xlabel('Number of iterations')
ylabel('Validation error')
legend('Dense map','Known sparsity')
legend('show')

%% Plot random conditionals

plot_random_conditionals(S_dense, Z)
plot_random_conditionals(S_sparse, Z(:,active_vars))

%% Compare dense and sparse maps log-likelihood

% define multi-indices
m_idxs = zeros(3,d); m_idxs(2,d-1) = 1; m_idxs(3,d) = 1;
a0 = zeros(3,1);

% compute dense map
ref_d = IndependentProductDitribution(repmat({ref},1,d));
TM = identity_map(1:d, basis);
PB_dense = PullbackDensity(TM, ref_d);
PB_dense.S{comp} = PB_dense.S{comp}.set_multi_idxs(m_idxs);
PB_dense.S{comp} = PB_dense.S{comp}.set_coeff(zeros(size(m_idxs,1),1));
PB_dense.S{comp} = optimize_component(PB_dense.S{comp}, ref_d.factors{comp}, ...
    a0, Z(:,1:comp), [], alpha);
valid_err = negative_log_likelihood(PB_dense.S{comp}, ref_d.factors{comp}, Zvalid(:,1:comp));
fprintf('Dense map validation error: %f\n', valid_err);

f = PB_dense.S{comp}.f.evaluate(Zvalid(:,1:comp));
Psi = PB_dense.S{comp}.f.evaluate_basis(Zvalid(:,1:comp));
fprintf('f_dense(1) evaluation: %f\n', f(1,1));
fprintf('Psi_dense(1) evaluation: %f\n', Psi(1,1));

% compute sparse map
TM = identity_map(1:d, basis);
PB_sparse = PullbackDensity(TM, ref_d);
PB_sparse.S{2} = PB_sparse.S{2}.set_multi_idxs(m_idxs(:,4:5));
PB_sparse.S{2} = PB_sparse.S{2}.set_coeff(zeros(size(m_idxs,1),1));
PB_sparse.S{2} = optimize_component(PB_sparse.S{2}, ref_d.factors{2}, ...
    a0, Z(:,active_vars), [], alpha);
valid_err = negative_log_likelihood(PB_sparse.S{2}, ref_d.factors{2}, Zvalid(:,active_vars));
fprintf('Sparse map validation error: %f\n', valid_err);

f2 = PB_sparse.S{2}.f.evaluate(Zvalid(:,active_vars));
Psi2 = PB_sparse.S{2}.f.evaluate_basis(Zvalid(:,active_vars));
fprintf('f_sparse(1) evaluation: %f\n', f2(1,1));
fprintf('Psi_sparse(1) evaluation: %f\n', Psi2(1,1));

%% Compare different basis 

basis_cell = {@HermiteProbabilistPoly, @ProbabilistHermiteFunction, ...
    @PhysicistHermiteFunction, @ConstExtProbabilistHermiteFunction, ...
    @LinearExtProbabilistHermiteFunction};
basis_str = {'Hermite Poly','Hermite Function',...
    'Physicist Hermite Functions','Const Ext Hermite Functions',...
    'Linear Ext Hermite Functions'};
S_dense_cell       = cell(length(basis_cell),1);
S_sparse_cell      = cell(length(basis_cell),1);
output_dense_cell  = cell(length(basis_cell),1);
output_sparse_cell = cell(length(basis_cell),1);

for i=1:length(basis_cell)
    
    % set basis
    basis = basis_cell{i}();
    
    % define TM
    ref_d = IndependentProductDitribution(repmat({ref},1,d));
    TM = identity_map(1:d, basis);
    PB = PullbackDensity(TM, ref_d);

    % run greedy approximation on S_valid
    [S_dense, output_dense] = greedy_fit(PB.S{comp}, ref_d.factors{comp}, ...
        max_terms, Z(:,1:comp), Zvalid(:,1:comp), max_patience, alpha);

    % remove one to account for initial condition
    [~, n_added_terms] = min(output_dense.valid_err);
    n_added_terms = n_added_terms(1) - 1;
    opt_terms = PB.S{comp}.n_coeff + n_added_terms;
    % extract optimal multi-indices
    midx_opt = S_dense.multi_idxs();
    midx_opt = midx_opt(1:opt_terms,:);
    PB.S{comp} = PB.S{comp}.set_multi_idxs(midx_opt);
    % run greedy_fit up to opt_terms with all data
    a0 = zeros(opt_terms,1);
    PB.S{comp} = optimize_component(PB.S{comp}, ref_d.factors{comp}, ...
        a0, Z(:,1:comp), [], alpha);
    
    % save
    S_dense_cell{i} = PB.S{comp};
    output_dense_cell{i} = output_dense;
    
    % define TM
    ref_d = IndependentProductDitribution(repmat({ref},1,2));
    TM = identity_map(1:2, basis);
    PB = PullbackDensity(TM, ref_d);

    % run greedy approximation on S_valid
    max_patience = 10;
    [S_sparse, output_sparse] = greedy_fit(PB.S{2}, ref_d.factors{2}, ...
        max_terms, Z(:,active_vars), Zvalid(:,active_vars), max_patience, alpha);

    % remove one to account for initial condition
    [~, n_added_terms] = min(output_sparse.valid_err);
    n_added_terms = n_added_terms(1) - 1;
    opt_terms = PB.S{2}.n_coeff + n_added_terms;
    % extract optimal multi-indices
    midx_opt = S_sparse.multi_idxs();
    midx_opt = midx_opt(1:opt_terms,:);
    PB.S{2} = PB.S{2}.set_multi_idxs(midx_opt);
    % run greedy_fit up to opt_terms with all data
    a0 = zeros(opt_terms,1);
    PB.S{2} = optimize_component(PB.S{2}, ref_d.factors{2}, ...
        a0, Z(:,active_vars), [], alpha);
    
    % save
    S_sparse_cell{i} = PB.S{2};
    output_sparse_cell{i} = output_sparse;
    
end

%%

figure
subplot(1,2,1)
hold on
for i=1:length(basis_cell)-1
    plot(output_dense_cell{i}.train_err,'-','DisplayName',basis_str{i})
end
hold off
xlabel('Number of iterations')
ylabel('Training error - dense map')
legend('show')

subplot(1,2,2)
hold on
for i=1:length(basis_cell)-1
    plot(output_dense_cell{i}.valid_err,'-','DisplayName',basis_str{i})
end
hold off
xlabel('Number of iterations')
ylabel('Validation error - dense map')

figure
subplot(1,2,1)
hold on
for i=1:length(basis_cell)
    plot(output_sparse_cell{i}.train_err,'-','DisplayName',basis_str{i})
end
hold off
xlabel('Number of iterations')
ylabel('Training error - sparse map')

subplot(1,2,2)
hold on
for i=1:length(basis_cell)
    plot(output_sparse_cell{i}.valid_err,'-','DisplayName',basis_str{i})
end
hold off
xlabel('Number of iterations')
ylabel('Validation error - sparse map')


% -- END OF FILE --