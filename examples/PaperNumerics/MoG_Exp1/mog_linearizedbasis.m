%clear; close all; clc
addpath(genpath('../../../src'))

% define parameters
N_vect = 1e4;
orders = 1:10;

% define density
mu = [-2; 2];
sigma = cat(3,0.5,2);
p = ones(1,2)/2;
gm = gmdistribution(mu,sigma,p);

%% Approximate map

% define cell arrays to store map
G  = cell(length(N_vect),1);
CM = cell(length(N_vect),length(orders));

for i=1:length(N_vect)
    
    % generate samples from gumbel density
    N = N_vect(i);
    X = gm.random(N);
    
    % find the Gaussian approximation
    G{i} = GaussianPullbackDensity(1, true);
    G{i} = G{i}.optimize(X);
    Z = G{i}.S.evaluate(X);

    % find the order k approximation 
    for k=1:length(orders)
        fprintf('N = %d, order %d\n', N, orders(k));
        basis = HermiteProbabilistPolyWithLinearization();
        basis.bounds = quantile(Z,[0.01,0.99]).';%[-4;6];
        S = total_order_map(1, basis, orders(k));
        ref = IndependentProductDistribution({Normal()});
        PB = PullbackDensity(S, ref);
        PB = PB.optimize(Z);
        CM{i,k} = ComposedPullbackDensity({G{i}.S, PB.S}, ref);
    end
    
end

%% Compute errors

% compute KL divergence
kldivergence = nan(length(N_vect),length(orders), 2);
KRmap_error  = nan(length(N_vect),length(orders), 2);

% generate test samples
Ntest = 1e7;
Xtest = gm.random(Ntest);

% compute entropy
ent_pi = log(gm.pdf(Xtest));

% evaluate KR map
SKR_test = sqrt(2)*erfinv(2*gm.cdf(Xtest)-1);
%%
for i=1:length(N_vect)
    for k=1:length(orders)
        if isempty(CM{i,k})
            continue
        end
        
        % interpolate log-pdf at interpolation points
        log_pdf_diff = ent_pi - batch_eval_log_pdf(CM{i,k}, Xtest);
        log_pdf_diff(isinf(log_pdf_diff)) = []; log_pdf_diff(isnan(log_pdf_diff)) = [];

        % evaluate kl-divergence and error
        kldivergence(i,k,1) = mean(log_pdf_diff);
        kldivergence(i,k,2) = 1.96*std(log_pdf_diff)/sqrt(length(Xtest));

        % evaluate error in map
        S_err = (SKR_test - batch_eval_S(CM{i,k}, Xtest)).^2;
        KRmap_error(i,k,1) = mean(S_err);
        KRmap_error(i,k,2) = 1.96*std(S_err)/sqrt(length(Xtest));

    end
    
end

clear pdf_test ent_pi Xtest Xnorm SKR_test 
save('results_mog1')

%% Plot results

% define domain
dom = [-4,6];
xx = linspace(dom(1),dom(2),10000);

% evaluate true pdf and map
true_pdf = gm.pdf(xx');
true_cdf = gm.cdf(xx');
true_KR  = sqrt(2)*erfinv(2*true_cdf - 1);

% plot results
figure
hold on
plot(xx, true_pdf, '-k', 'DisplayName', 'True PDF', 'LineWidth', 4)
for k=2:2:length(orders)	
	plot(xx, exp(CM{end,k}.log_pdf(xx')), 'DisplayName', ['Degree ' num2str(k)], 'Linewidth', 3)
end
xlim(dom)
ylim([0,0.33])
set(gca,'FontSize',24)
xlabel('$x$','FontSize', 24)
ylabel('$\pi(x)$','FontSize', 24)
h=legend('show','location','northeast');
set(h, 'FontSize', 22)
set(gca,'LineWidth',2)
hold off
print('-depsc','pdf_approx')
close all

% plot results
figure()
hold on
for i=1:length(N_vect)
    errorbar(orders, kldivergence(i,:,1), kldivergence(i,:,2), '-s', 'LineWidth', 3, 'MarkerSize', 8)
    errorbar(orders, 0.5*KRmap_error(i,:,1), KRmap_error(i,:,2), '-s', 'LineWidth', 3, 'MarkerSize', 8)
end
xlim([min(orders),max(orders)])
set(gca,'YScale','log')
ylim([5e-5,0.5])
set(gca,'FontSize',24)
xlabel('Maximum polynomial degree, $p$','FontSize',24)
h=legend({'$D_{KL}(\pi||\widehat{S}^{\sharp}\eta)$','$\frac{1}{2}E_{\pi}\|\widehat{S}(x) - S_{KR}(x)\|^2$'},'location','northeast');
set(h, 'FontSize', 22)
set(gca,'LineWidth',2)
hold off
print('-depsc','kl_divergence')
close all

%% --- Helper Functions ---

function log_pi = batch_eval_log_pdf(PB, X, batch_size)
    if nargin < 3
        batch_size = 1e5;
    end
    log_pi = zeros(size(X,1),1);
    % define batches
    n_batches = ceil(size(X,1)/batch_size);
    start_idx = 1;
    for i=1:n_batches
        % extract data
        end_idx = min(start_idx + batch_size, size(X,1));
        Xi = X(start_idx:end_idx,:);
        % evaluate density
        log_pi(start_idx:end_idx) = PB.log_pdf(Xi);
        % update starting index
        start_idx = end_idx + 1;
    end
end

function Sx = batch_eval_S(S, X, batch_size)
    if nargin < 3
        batch_size = 1e5;
    end
    Sx = zeros(size(X));
    % define batches
    n_batches = ceil(size(X,1)/batch_size);
    start_idx = 1;
    for i=1:n_batches
        % extract data
        end_idx = min(start_idx + batch_size, size(X,1));
        Xi = X(start_idx:end_idx,:);
        % evaluate map
        Sx(start_idx:end_idx,:) = S.evaluate(Xi);
        % update starting index
        start_idx = end_idx + 1;
    end
end

% -- END OF FILE --
