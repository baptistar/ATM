%clear; close all; clc
addpath(genpath('../../../src'))
sd = 10; rng(sd);

%% --- Define test case ---

% define problems
N      = [2000];
Ntest  = 1e4;
Nvalid = 0;
dx     = 40;
d      = dx+2;
order_vect = [1,2];

% define model
SV = stoc_volatility(d);

% generate test samples
Xtest  = SV.sample(Ntest);

% evaluate true NLL
logpi = SV.log_pdf(Xtest);

% define active variables
active_variables = cell(d,1); 
active_variables{1} = 1;
active_variables{2} = 2;
for k=3:d
    active_variables{k} = [1,2,k-1:k];
end

for Ntrain=N
    
    % load results
    file_name = ['data/map_results_N' num2str(Ntrain)];
    load(file_name,'CM','CM_sparse','CM_totorder')
   
    % define reference 
    ref = IndependentProductDistribution(repmat({Normal()},1,d));
    
    % measure conditional NLL on test set
    cond_NLL_atm             = zeros(d,2);
    cond_NLL_totorder        = zeros(length(order_vect),d,2);
    cond_NLL_sparse          = zeros(d,2);
    margcond_NLL_atm         = zeros(d,2);
    margcond_NLL_totorder    = zeros(length(order_vect),d,2);
    margcond_NLL_sparse      = zeros(d,2);
    cond_KL_atm              = zeros(d,2);
    cond_KL_totorder         = zeros(length(order_vect),d,2);
    cond_KL_sparse           = zeros(d,2);
    margcond_KL_atm          = zeros(d,2);
    margcond_KL_totorder     = zeros(length(order_vect),d,2);
    margcond_KL_sparse       = zeros(d,2);
    running_tot_atm          = zeros(size(Xtest,1),1);
    running_tot_totorder     = zeros(length(order_vect),size(Xtest,1));
    running_tot_sparse       = zeros(size(Xtest,1),1);
        
    for k=1:d
        fprintf('Component %d\n',k)
        
        % evaluate NLL for ATM map
        cond_NLL_evals = CM.log_pdf(Xtest,k);
        cond_NLL_atm(k,1) = mean(cond_NLL_evals);
        cond_NLL_atm(k,2) = 1.96*std(cond_NLL_evals)/sqrt(Ntest);
        running_tot_atm = running_tot_atm + cond_NLL_evals;
        margcond_NLL_atm(k,1) = mean(running_tot_atm);
        margcond_NLL_atm(k,2) = 1.96*std(running_tot_atm)/sqrt(Ntest);

        % evaluate KL for ATM map
        cond_KL_atm(k,1) = mean(logpi(:,k) - cond_NLL_evals);
        cond_KL_atm(k,2) = 1.96*std(logpi(:,k) - cond_NLL_evals)/sqrt(Ntest);
        margcond_KL_atm(k,1) = mean(sum(logpi(:,1:k),2) - running_tot_atm);
        margcond_KL_atm(k,2) = 1.96*std(sum(logpi(:,1:k),2) - running_tot_atm)/sqrt(Ntest);

        % evaluate NLL for sparse ATM map
        Sx = CM_sparse.S{1}.evaluate(Xtest);
        dJ = CM_sparse.S{1}.logdet_Jacobian(Xtest, k);
        dJ = dJ + log(CM_sparse.S{2}.S{k}.grad_xd(Sx(:,active_variables{k})));
        Sxk = CM_sparse.S{2}.S{k}.evaluate(Sx(:,active_variables{k}));
        cond_NLL_evals = ref.factors{k}.log_pdf(Sxk) + dJ;
        cond_NLL_sparse(k,1) = mean(cond_NLL_evals);
        cond_NLL_sparse(k,2) = 1.96*std(cond_NLL_evals)/sqrt(Ntest);
        running_tot_sparse = running_tot_sparse + cond_NLL_evals;
        margcond_NLL_sparse(k,1) = mean(running_tot_sparse);
        margcond_NLL_sparse(k,2) = 1.96*std(running_tot_sparse)/sqrt(Ntest);
    
        % evaluate KL for sparse ATM map
        cond_KL_sparse(k,1) = mean(logpi(:,k) - cond_NLL_evals);
        cond_KL_sparse(k,2) = 1.96*std(logpi(:,k) - cond_NLL_evals)/sqrt(Ntest);
        margcond_KL_sparse(k,1) = mean(sum(logpi(:,1:k),2) - running_tot_sparse);
        margcond_KL_sparse(k,2) = 1.96*std(sum(logpi(:,1:k),2) - running_tot_sparse)/sqrt(Ntest);

        % evaluate NLL + KL for non-adaptive map
        for j=1:length(order_vect)
            Sx = CM_totorder{j}.S{1}.evaluate(Xtest);
            dJ = CM_totorder{j}.S{1}.logdet_Jacobian(Xtest, k);
            dJ = dJ + log(batch_grad_xd(CM_totorder{j}.S{2}.S{k}, Sx(:,1:k)));
            Sxk = batch_eval(CM_totorder{j}.S{2}.S{k}, Sx(:,1:k));
            cond_NLL_evals = ref.factors{k}.log_pdf(Sxk) + dJ;
            
            cond_NLL_totorder(j,k,1) = mean(cond_NLL_evals);
            cond_NLL_totorder(j,k,2) = 1.96*std(cond_NLL_evals)/sqrt(Ntest);
            running_tot_totorder(j,:) = running_tot_totorder(j,:) + cond_NLL_evals';
            margcond_NLL_totorder(j,k,1) = mean(running_tot_totorder(j,:));
            margcond_NLL_totorder(j,k,2) = 1.96*std(running_tot_totorder(j,:))/sqrt(Ntest);
            
            cond_KL_totorder(j,k,1) = mean(logpi(:,k) - cond_NLL_evals);
            cond_KL_totorder(j,k,2) = 1.96*std(logpi(:,k) - cond_NLL_evals)/sqrt(Ntest);
            margcond_KL_totorder(j,k,1) = mean(sum(logpi(:,1:k),2) - running_tot_totorder(j,:).');
            margcond_KL_totorder(j,k,2) = 1.96*std(sum(logpi(:,1:k),2) - running_tot_totorder(j,:).')/sqrt(Ntest);
        end
    
    end
    
    % save results
    file_name = ['data/postprocess_N' num2str(Ntrain)];
    save(file_name)

end

%% -- Helper Functions --

function Sx = batch_eval(S, X, batch_size)
    if nargin < 3
        batch_size = 1000;
    end
    nbatches = ceil(size(X,1)/batch_size);
    Sx = zeros(size(X,1),1);
    start_idx = 1;
    % compute for each batch
    for ii=1:nbatches
        end_idx = min(start_idx + batch_size - 1, size(X,1));
        batch_idx = start_idx : end_idx;
        Sx(batch_idx) = S.evaluate(X(batch_idx,:));
        start_idx = end_idx + 1;
    end
end

function dxSx = batch_grad_xd(S, X, batch_size)
    if nargin < 3
        batch_size = 1000;
    end
    nbatches = ceil(size(X,1)/batch_size);
    dxSx = zeros(size(X,1),1);
    start_idx = 1;
    % compute for each batch
    for ii=1:nbatches
        end_idx = min(start_idx + batch_size - 1, size(X,1));
        batch_idx = start_idx : end_idx;
        dxSx(batch_idx) = S.grad_xd(X(batch_idx,:));
        start_idx = end_idx + 1;
    end
end

% -- END OF FILE --
