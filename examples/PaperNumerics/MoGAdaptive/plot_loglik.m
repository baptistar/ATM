%clear; close all; clc
sd = 1; rng(sd);

%% --- Define test case ---

% define orders, sample size, and MC iterations
d = 3;
N_vect = floor(logspace(1,4,25));
N_vect = N_vect(1:2:end);
MCruns = 1:20;
order_vect = [1,3,5];
batch_size = 10000;

% define test_loglik cell
test_loglik_gauss     = nan(length(N_vect), length(MCruns));
test_loglik_adapt     = nan(length(N_vect), length(MCruns));
test_loglik_nonadapt  = nan(length(N_vect), length(MCruns), length(order_vect));
num_coeffs_gauss      = nan(length(N_vect), length(MCruns));
num_coeffs_adapt      = nan(length(N_vect), length(MCruns));
num_coeffs_nonadapt   = nan(length(N_vect), length(MCruns), length(order_vect));
kldivergence_gauss    = nan(length(N_vect), length(MCruns));
kldivergence_adapt    = nan(length(N_vect), length(MCruns));
kldivergence_nonadapt = nan(length(N_vect), length(MCruns), length(order_vect));

% load test data
load(['samples/MoG_d' num2str(d)],'Xtest','P');
% estimate entropy
Xtest = Xtest(randperm(size(Xtest,1)),:);
neg_entropy = P.log_pdf(Xtest);

for i=1:length(N_vect)
    for j=MCruns

        % load samples
        N = N_vect(i);
        load(['samples/MoG_d' num2str(d) '_N' num2str(N) '_run' num2str(j)],'Xtrain');

        % compute gaussian loglikelihood
        G = GaussianPullbackDensity(d, true);
        G = G.optimize(Xtrain);
        gauss_log_pdf = G.log_pdf(Xtest);
        test_loglik_gauss(i,j) = mean(gauss_log_pdf);
        kldivergence_gauss(i,j) = mean(neg_entropy - gauss_log_pdf);

        % compute ATM results
        try
            load(['data/MoG_d' num2str(d) '_N' num2str(N) '_run' num2str(j)],'CM');
            atm_log_pdf = batch_eval(CM, Xtest, batch_size);
            atm_log_pdf = atm_log_pdf(~isinf(atm_log_pdf));
            test_loglik_adapt(i,j) = mean(atm_log_pdf);
            kldivergence_adapt(i,j) = mean(neg_entropy - atm_log_pdf);
            num_coeffs_adapt(i,j) = CM.S{2}.n_coeff;
       catch
       end

        % compute non-adaptive results
        for k=1:length(order_vect)
            try
                load(['data/MoG_nonadapt_d' num2str(d) '_N' num2str(N) ...
                    '_run' num2str(j) '_order' num2str(order_vect(k))],'CM');
                nonadapt_log_pdf = batch_eval(CM, Xtest, batch_size);
                test_loglik_nonadapt(i,j,k) = mean(nonadapt_log_pdf);
                kldivergence_nonadapt(i,j,k) = mean(neg_entropy - nonadapt_log_pdf);
                num_coeffs_nonadapt(i,j,k) = CM.S{2}.n_coeff;
            catch
            end
        end
    
    end
end

save('data/post_processed_results')

%% Make plots

% correct inf in negloglik_nonadapt
kldivergence_nonadapt(isinf(kldivergence_nonadapt)) = nan;

% define mean and ste
pmean = @(x) (nanmean(x,2)).';
pste  = @(x) (1.96*nanstd(x,[],2)/sqrt(size(x,2))).';

% plot data
figure()
hold on
H(1) = shadedErrorBar(N_vect, kldivergence_adapt, {pmean, pste}, 'lineprops',{'-o','linewidth',3,'markersize',8},'patchsaturation',0.1);
for k=1:length(order_vect)
    H(k+1) = shadedErrorBar(N_vect, squeeze(kldivergence_nonadapt(:,:,k)), {pmean, pste}, 'lineprops',{'-o','linewidth',3,'markersize',8},'patchSaturation',0.1);
end
set(gca,'FontSize',24)
uistack(H(1).mainLine,'top')
uistack(H(1).edge,'top')
uistack(H(1).patch,'top')
xlabel('Training samples, $n$','FontSize',22)
ylabel('$D_{KL}(\pi||\widehat{S}^\sharp\eta)$','FontSize',22)
set(gca,'Yscale','log')
set(gca,'Xscale','log')
legend([H.mainLine],{'ATM','Non-adaptive: $p=1$', 'Non-adaptive: $p=3$', ...
    'Non-adaptive: $p=5$'},'location','southwest','FontSize', 21)
set(gcf,'Units','inches');
set(gca,'LineWidth',2)
screenposition = get(gcf,'Position');
set(gcf, 'PaperPosition',[0 0 screenposition(3:4)], ...
    'PaperSize',[screenposition(3:4)]);
ylim([1e-2,1e1])
set(gca,'YTick',[0.01,0.1,1,10])
set(gca,'YTickLabels',{'$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$'})
hold off
print -dpdf -painters 'figures/KL_vs_N'

figure()
hold on
Hh = shadedErrorBar(nanmean(kldivergence_adapt,2), num_coeffs_adapt, {pmean, pste}, 'lineprops',{'-o','linewidth',3,'markersize',8},'patchsaturation',0.1);
Lh(1) = Hh(1).mainLine;
for k=1:length(order_vect)
    Lh(k+1) = plot(min(nanmean(kldivergence_nonadapt(:,:,k),2)), mean(num_coeffs_nonadapt(:,:,k),'all'), '*', 'Markersize',15, 'Color', H(k+1).mainLine.Color); %8},'patchsaturation',0.1);
end
set(gca,'FontSize',24)
xlabel('$D_{KL}(\pi||\widehat{S}^\sharp\eta)$','fontsize',22)
ylabel('Map parameters, $\#\Lambda_{m^*}$','fontsize',22)
set(gca,'xscale','log')
set(gca,'XTick',[0.01,0.1,1,10])
set(gca,'XTickLabels',{'$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$'})
%xlim([0.0198,8.0239])%xlim([min(nanmean(kldivergence_adapt,2)), max(nanmean(kldivergence_adapt,2))])%[0.0944, 10])
legend(Lh,{'ATM','Non-adaptive: $p=1$', 'Non-adaptive: $p=3$', ...
    'Non-adaptive: $p=5$'},'location','northeast','fontsize', 22)
ylim([0,100])
set(gca,'LineWidth',2)
set(gcf,'units','inches');
screenposition = get(gcf,'position');
set(gcf, 'paperposition',[0 0 screenposition(3:4)], ...
    'papersize',[screenposition(3:4)]);
hold off
print -dpdf -painters 'figures/ncoeffs_vs_KLopt'

%% -- Helper Functions --

function loglik = batch_eval(TM, X, batch_size)
    nbatches = ceil(size(X,1)/batch_size);
    loglik = zeros(size(X,1),1);
    start_idx = 1;
    % compute for each batch
    for ii=1:nbatches
        end_idx = min(start_idx + batch_size - 1, size(X,1));
        batch_idx = start_idx : end_idx;
        loglik(batch_idx) = TM.log_pdf(X(batch_idx,:));
        start_idx = end_idx + 1;
    end
end

% -- END OF FILE --
