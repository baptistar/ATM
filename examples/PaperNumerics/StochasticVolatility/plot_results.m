%clear; close all; clc
addpath(genpath('../../../src'))
sd = 1; rng(sd);

%% --- Define test case ---

% define problems
N  = [2000];
dx = 40;
d  = dx+2;
order_vect = [1,2];

% define active variables
active_variables = cell(d,1); 
active_variables{1} = 1;
active_variables{2} = 2;
for k=3:d
    active_variables{k} = [1,2,k-1:k];
end

%% Plot sparsity of maps 

for Ntrain=N

    % load results
    file_name = ['data/map_results_N' num2str(Ntrain)];
    load(file_name,'CM','CM_sparse','CM_totorder')

    % true dependence (chain graph)
    true_dep = zeros(dx,dx);
    for k=1:dx
        true_dep(k,active_variables{k}) = 1;
    end
    figure;
    niceSpy(true_dep)
    title('True map')
    % approximate map dependence
    approxmap_dep = zeros(dx,dx);
    for k=1:dx
        midx = CM.S{2}.S{k}.multi_idxs;
        midx(:,k) = midx(:,k) + 1;
        approxmap_dep(k,1:k) = max(midx,[],1);
    end
    figure;
    niceSpy(approxmap_dep)
    title('ATM')
    % compare sparsity
    figure
    map_sparsity(approxmap_dep, true_dep)
    print('-depsc',['ATM_sparsity_N' num2str(Ntrain)])

    % load results
    file_name = ['data/postprocess_N' num2str(Ntrain)];
    load(file_name,'cond_NLL_atm','cond_NLL_sparse','cond_NLL_totorder',...
        'margcond_NLL_atm','margcond_NLL_sparse','margcond_NLL_totorder',...
        'cond_KL_atm','cond_KL_sparse','cond_KL_totorder',...
        'margcond_KL_atm','margcond_KL_sparse','margcond_KL_totorder')

    figure;
    hold on;
    errorbar(1:d, margcond_KL_atm(:,1), margcond_KL_atm(:,2), '-', 'LineWidth', 3, 'MarkerSize',8,'DisplayName', 'ATM')
    errorbar(1:d, margcond_KL_sparse(:,1), margcond_KL_sparse(:,2), '-','LineWidth', 3, 'MarkerSize',8,'DisplayName', 'Sparsity-aware ATM')
    for j=1:length(order_vect)
        errorbar(1:d, margcond_KL_totorder(j,:,1), margcond_KL_totorder(j,:,2), '-', 'LineWidth', 3, 'MarkerSize',8,'DisplayName', ['Non-adaptive: $p=' num2str(order_vect(j)) '$'])
    end
    set(gca,'FontSize',24)
    xlabel('$\mathbf{X}$ dimension, $d$','FontSize',22)
    ylabel('$D_{KL}(\pi(\mathbf{x}_{1:d})||S_{1:d}^\sharp\eta)$','FontSize',22)
    set(gca,'LineWidth',2)
    set(gca,'YScale','log')
    legend('show','location','southeast')
    print('-depsc',['marginals_KL_N' num2str(Ntrain)])
    
end

% -- END OF FILE --
